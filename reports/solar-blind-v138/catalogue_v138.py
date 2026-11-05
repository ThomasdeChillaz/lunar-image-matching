"""Image-only shared catalogue-light and camera search from reference geometry.

This is a discrete, bounded physical-template initializer, not a continuous Sun
estimator or independent solar-visibility validation. It never reads query truth.
"""
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter

import numpy as np

_path = Path(__file__).resolve().parents[1]/"solar-warp-v125/fast_projection.py"
_spec = spec_from_file_location("blind_catalogue_projection_v138", _path)
fast = module_from_spec(_spec); _spec.loader.exec_module(fast)
coarse = fast.original
MAX_LIGHTS = 24
MAX_SEEDS = 8


@dataclass(frozen=True)
class Catalogue:
    images: np.ndarray
    sun_world: np.ndarray
    light_ids: tuple

    def validate(self):
        a, sun = np.asarray(self.images), np.asarray(self.sun_world, float)
        if (not isinstance(self.images,np.ndarray) or a.ndim != 3 or not 1 <= len(a) <= MAX_LIGHTS
                or any(n < 256 or n > 512 or n % 32 for n in a.shape[1:])
                or a.dtype.kind not in 'fiu'
                or np.isinf(a).any() or (a[np.isfinite(a)] < 0).any()
                or sun.shape != (len(a), 3) or not np.isfinite(sun).all()
                or not np.allclose(np.linalg.norm(sun, axis=1), 1., rtol=0, atol=1e-10)
                or not isinstance(self.light_ids, tuple) or len(self.light_ids) != len(a)
                or any(type(n) is not int or not 0 <= n < 24 for n in self.light_ids)
                or len(set(self.light_ids)) != len(a)):
            raise ValueError("Bounded reference-light images, distinct IDs and unit reference-world Suns required")
        return self


def digest(*arrays):
    h = sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def reference_cloud(catalogue, depth, k, pose, valid):
    """Identical geometry and reference support for every light, including darkness."""
    catalogue.validate()
    images = np.asarray(catalogue.images, float)
    depth, valid = np.asarray(depth, float), np.asarray(valid)
    if depth.shape != images.shape[1:] or valid.shape != depth.shape or valid.dtype != bool:
        raise ValueError("Aligned reference axial depth and explicit Boolean coverage required")
    masked_depth = np.where(valid, depth, np.nan)
    # Geometry is independent of the illumination and its image brightness.
    cloud = coarse.build_reference_cloud(np.ones_like(depth), masked_depth, k, pose)
    stride = 8
    y, x = np.mgrid[4:depth.shape[0]:stride, 4:depth.shape[1]:stride]
    centre = masked_depth[y, x]
    depth_tiles = coarse._tiles(masked_depth, stride)
    support = (np.isfinite(depth_tiles) & (depth_tiles > 0)
        & (np.abs(depth_tiles-centre[:, None, :, None]) <= .05+.01*centre[:, None, :, None]))
    # One common mask across catalogue entries: a light cannot win by silently
    # dropping poorly predicted cells. Zero radiance remains known evidence.
    common = np.isfinite(images).all(axis=0)
    support &= coarse._tiles(common, stride)
    counts = support.sum(axis=(1, 3))
    rows = []
    source_valid = np.isfinite(centre) & (centre > 0)
    for image in images:
        samples = coarse._tiles(np.where(common, image, 0.), stride)
        values = np.where(support, samples, 0.).sum(axis=(1, 3))/np.maximum(counts, 1)
        values[counts < 32] = np.nan
        rows.append(values[source_valid])
    return cloud, np.array(rows)


class ProjectionCache:
    """One reference-geometry cache, never a query/ranking/camera-output cache."""
    def __init__(self):
        self.key = self.bank = None

    def get(self, cloud, k, poses, shape):
        key = digest(cloud.world, cloud.flat_world, np.asarray(k), poses, np.array(shape))
        hit = key == self.key
        tick = perf_counter()
        if not hit:
            self.bank = fast.FastProjectionBank(cloud, k, poses, shape, 4)
            self.key = key
        self.bank.validate(cloud, k, poses, shape, 4)
        return self.bank, dict(hit=hit, key=key, seconds=perf_counter()-tick,
                               scope="Reference geometry only; all query scores recomputed")


def batch_losses(observed, predictions, weights, floor=1e-9):
    """Positive-gain Huber loss on a fixed observation vector for every light."""
    observed = coarse._unit(observed, weights, floor)
    if observed is None:
        return [None]*len(predictions)
    # Use the unchanged scalar metric. Batched rasterization avoids repeating
    # projection and cell-support work, rather than approximating the metric.
    return [coarse._loss(observed, row, weights, floor) for row in predictions]


def select_seeds(scores, light_ids):
    """At most two global lights and four camera seeds per light (eight total)."""
    per_light = []
    for slot, row in enumerate(scores):
        order = sorted((i for i,v in enumerate(row) if v is not None), key=lambda i:(row[i],i))
        if order:
            per_light.append((row[order[0]], light_ids[slot], slot, order))
    per_light.sort(key=lambda r:(r[0],r[1]))
    result = []
    for _, light_id, slot, order in per_light[:2]:
        result.extend(dict(light_slot=slot, light_id=light_id, pose_index=i,
                           coarse_loss=float(scores[slot][i])) for i in order[:4])
    return sorted(result, key=lambda r:(r["coarse_loss"],r["light_id"],r["pose_index"]))[:MAX_SEEDS]


def rank(catalogue, query, depth, k, reference_pose, query_k, *, reference_valid,
         poses=None, cache=None):
    """Jointly rank reference-only camera and *one shared* catalogue Sun.

    The optional camera bank supports generated controls and must be frozen by
    the experiment protocol. It is not an initial true query pose input.
    """
    tick = perf_counter()
    cloud, radiance = reference_cloud(catalogue, depth, k, reference_pose, reference_valid)
    query = np.asarray(query, float)
    if (query.shape != catalogue.images.shape[1:] or np.isinf(query).any()
            or (query[np.isfinite(query)] < 0).any()):
        raise ValueError("Native query radiance with explicit unknown pixels required")
    cameras = coarse.reference_pose_bank(reference_pose) if poses is None else np.asarray(poses,float)
    if cameras.ndim != 3 or cameras.shape[1:] != (4,4) or not 1 <= len(cameras) <= 2048:
        raise ValueError("One to2048 declared reference-side camera hypotheses required")
    for camera in cameras: coarse.camera_check(query_k,camera)
    cells = coarse._cell_means(query, 32, 32**2)
    fixed, weights, _, _ = coarse._region_and_weights(cells.shape, None, 2)
    losses = [[None]*len(cameras) for _ in radiance]
    details = dict(light_count=len(radiance), pose_count=len(cameras), fixed_cells=int(fixed.sum()),
        fixed_region="Central half in each detector dimension; all native content retained downstream",
        common_support_across_lights=True, global_light_per_hypothesis=True,
        query_truth_used=False, query_scores_cached=False, cameras_scored=0,
        shared_solar_visibility_certified=False)
    def finish(reason):
        return dict(scores=losses, seeds=select_seeds(losses,catalogue.light_ids),
            cameras=cameras, diagnostics=dict(details,reason=reason,seconds=perf_counter()-tick))
    if not len(cloud.world): return finish("no_reference_geometry")
    if not np.isfinite(cells[fixed]).all(): return finish("unknown_query_on_fixed_cells")
    if coarse._unit(cells[fixed],weights,1e-9) is None: return finish("uniform_query_on_fixed_cells")
    detector = coarse._cell_means(query,4,16)
    cache = ProjectionCache() if cache is None else cache
    projected, details["projection_cache"] = cache.get(cloud,query_k,cameras,query.shape)
    for index in range(len(cameras)):
        winners = projected.get(index)[0]  # Full3D z-buffer, never a flat-terrain proxy.
        safe = np.maximum(winners,0)
        predicted = radiance[:,safe]
        common = ((winners >= 0) & np.isfinite(detector)
                  & np.isfinite(predicted).all(axis=0))
        counts = coarse._tiles(common,8).sum(axis=(1,3))[fixed]
        if (counts < 8).any(): continue
        obs = coarse._tiles(np.where(common,detector,0.),8).sum(axis=(1,3))[fixed]/counts
        templates = np.array([coarse._tiles(np.where(common,row,0.),8).sum(axis=(1,3))[fixed]/counts
                              for row in predicted])
        values = batch_losses(obs,templates,weights)
        for slot,value in enumerate(values): losses[slot][index] = value
        details["cameras_scored"] += 1
    return finish("complete_bounded_catalogue_search")
