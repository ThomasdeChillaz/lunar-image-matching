"""Reference-only coarse pose proposals for a separate known-Sun ceiling."""

from dataclasses import dataclass
import math

import numpy as np

from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import project_z, world_points_z
from north_camera_variation_v59 import rolled_pose
from north_render_plan_v31 import camera_to_world


MAX_POSES = 2048
MAX_SOURCE_SAMPLES = 4096
ARM_NAMES = ("geometry_3d", "planar", "gradient")


@dataclass(frozen=True)
class ReferenceCloud:
    world: np.ndarray
    flat_world: np.ndarray
    radiance: np.ndarray


def _halton(count):
    samples = np.zeros((count, 6))
    for axis, base in enumerate((2, 3, 5, 7, 11, 13)):
        indices = np.arange(1, count + 1)
        place = 1. / base
        while np.any(indices):
            samples[:, axis] += (indices % base) * place
            indices //= base
            place /= base
    return samples


def reference_pose_bank(reference_pose, *, count=MAX_POSES, datum_z=0.):
    """Construct a deterministic local-ENU simulator-prior pose list.

    The reference pose comes first. The remaining centres span ±500 m per
    axis, look targets ±250 m in the datum plane, and relative roll ±55°.
    Above-datum and off-nadir feasibility filters can shorten the list.
    """
    if type(count) is not int or not 1 <= count <= MAX_POSES or not np.isfinite(datum_z):
        raise ValueError("Bounded bank size and finite reference datum required")
    _, pose = camera_check(np.eye(3), reference_pose)
    centre, direction = pose[:3, 3], pose[:3, 2]
    min_down = math.cos(math.radians(52.))
    if centre[2] <= datum_z or direction[2] > -min_down:
        raise ValueError("Reference must face the local datum from inside the simulator cone")
    target = centre + (datum_z - centre[2]) / direction[2] * direction
    basis = np.asarray(camera_to_world(centre, target))[:3, :3]
    right = pose[:3, 0]
    roll = math.degrees(math.atan2(right @ basis[:, 1], right @ basis[:, 0]))
    offsets = (2. * _halton(count - 1) - 1.) * [500., 500., 500., 250., 250., 55.]
    candidates = [pose.copy()]
    for delta in offsets:
        position = centre + delta[:3]
        look = target + [delta[3], delta[4], 0.]
        down = position[2] - datum_z
        if down <= 0 or down / np.linalg.norm(position - look) < min_down:
            continue
        candidates.append(np.asarray(rolled_pose(position, look, roll + delta[5])))
    return np.stack(candidates)


def _image(image):
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or any(not 0 < size <= 512 for size in array.shape):
        raise ValueError("Expected a bounded native scene-linear luminance image")
    return array


def _tiles(array, size):
    rows, cols = array.shape
    if rows % size or cols % size:
        raise ValueError("Cell size must divide both detector dimensions")
    return array.reshape(rows // size, size, cols // size, size)


def fitted_plane(world):
    """Fit local ENU elevation with fixed robust reweighting on reference points."""
    if len(world) < 3:
        return None
    origin = np.median(world, axis=0)
    xy = world[:, :2] - origin[:2]
    scale = np.max(np.linalg.norm(xy, axis=1))
    if scale <= 0:
        return None
    design = np.column_stack((xy / scale, np.ones(len(world))))
    target = world[:, 2] - origin[2]
    weights = np.ones(len(world))
    for _ in range(4):
        coeffs, _, rank, _ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
        if rank != 3:
            return None
        residual = np.abs(target - design @ coeffs)
        cutoff = max(1e-6, 1.5 * 1.4826 * np.median(residual))
        weights = np.sqrt(np.minimum(1., cutoff / np.maximum(residual, 1e-12)))
    normal = np.array([-coeffs[0] / scale, -coeffs[1] / scale, 1.])
    offset = normal @ origin + coeffs[2]
    return normal, offset


def build_reference_cloud(luminance, axial_depth, intrinsics, pose, *, stride=8):
    """Bind sampled radiance to native reference rays with positive axial depth.

    The fitted-plane control shares the reference rays and radiance. Geometry with
    unavailable photometry still participates in occlusion in both arms.
    """
    image, depth = _image(luminance), _image(axial_depth)
    if depth.shape != image.shape or type(stride) is not int or stride <= 0:
        raise ValueError("Aligned depth and a bounded reference grid required")
    tile = _tiles(image, stride)
    if tile.shape[0] * tile.shape[2] > MAX_SOURCE_SAMPLES:
        raise ValueError("Reference sample limit exceeded")
    y, x = np.mgrid[stride//2:image.shape[0]:stride, stride//2:image.shape[1]:stride]
    anchors = depth[y, x]
    valid = np.isfinite(anchors) & (anchors > 0)
    depth_tiles = _tiles(depth, stride)
    centre_depth = anchors[:, None, :, None]
    supported = (np.isfinite(tile) & (tile >= 0) & np.isfinite(depth_tiles) & (depth_tiles > 0)
                 & (np.abs(depth_tiles - centre_depth) <= .05 + .01 * centre_depth))
    counts = supported.sum(axis=(1, 3))
    means = np.where(supported, tile, 0.).sum(axis=(1, 3)) / np.maximum(counts, 1)
    means[counts < (stride**2 + 1) // 2] = np.nan

    xy = np.column_stack((x[valid], y[valid]))
    intrinsics, pose = camera_check(intrinsics, pose)
    world = world_points_z(xy, anchors[valid], intrinsics, pose)
    flat = np.full_like(world, np.nan)
    plane = fitted_plane(world)
    if plane is not None:
        normal, offset = plane
        unit_rays = world_points_z(xy, np.ones(len(xy)), intrinsics, pose) - pose[:3, 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            planar_depth = (offset - normal @ pose[:3, 3]) / (unit_rays @ normal)
        forward = np.isfinite(planar_depth) & (planar_depth > 0)
        flat[forward] = world_points_z(xy[forward], planar_depth[forward], intrinsics, pose)
    return ReferenceCloud(world, flat, means[valid])


def detector_winners(world, intrinsics, pose, shape, stride, *, footprint=False):
    """Nearest positive axial depth wins in each bounded detector subbin."""
    height, width = shape
    world = np.asarray(world, float)
    if (world.ndim != 2 or world.shape[1] != 3 or len(world) > MAX_SOURCE_SAMPLES
            or type(stride) is not int or stride <= 0 or height % stride or width % stride):
        raise ValueError("Detector stride must divide both detector dimensions")
    xy, z, front = project_z(world, intrinsics, pose)
    inside = (front & np.isfinite(xy).all(axis=1) & (xy >= -.5).all(axis=1)
              & (xy < [width - .5, height - .5]).all(axis=1))
    source = np.flatnonzero(inside)
    low_shape = height // stride, width // stride
    low_height, low_width = low_shape
    if footprint:
        # The 8-pixel source tile integrates a patch. Assign its projected
        # sample to the surrounding 2x2 subbin centres, at most one stride away.
        origin = np.floor((xy[source] + .5) / stride - .5).astype(int)
        bins = (origin[:, None, :] + [[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(-1, 2)
        source = np.repeat(source, 4)
        valid = ((bins >= 0) & (bins < [low_width, low_height])).all(axis=1)
        bins, source = bins[valid], source[valid]
    else:
        bins = np.floor((xy[source] + .5) / stride).astype(int)
    detector = bins[:, 1] * low_width + bins[:, 0]
    order = np.lexsort((source, z[source], detector))
    _, first = np.unique(detector[order], return_index=True)
    indices = np.full(low_shape, -1, dtype=np.int16)
    indices.flat[detector[order[first]]] = source[order[first]]
    return indices


def rasterize(winners, radiance):
    values = np.full(winners.shape, np.nan)
    valid = winners >= 0
    values[valid] = radiance[winners[valid]]
    return values


def splat_detector(world, radiance, intrinsics, pose, shape, stride):
    """Single-bin splat used for precise native pixel centre controls."""
    radiance = np.asarray(radiance, float)
    if radiance.shape != (len(world),):
        raise ValueError("Radiance and reference sample counts differ")
    winners = detector_winners(world, intrinsics, pose, shape, stride)
    return rasterize(winners, radiance)


class ProjectionBank:
    """Reuse reference-only z-buffer winners across seeds and scales of one case."""

    def __init__(self, cloud, intrinsics, poses, shape, stride):
        self.worlds = cloud.world.copy(), cloud.flat_world.copy()
        self.intrinsics = np.array(intrinsics, copy=True)
        self.poses = np.array(poses, copy=True)
        self.shape, self.stride = shape, stride
        self.indices = {}

    def validate(self, cloud, intrinsics, poses, shape, stride):
        if (not np.array_equal(self.worlds[0], cloud.world, equal_nan=True)
                or not np.array_equal(self.worlds[1], cloud.flat_world, equal_nan=True)
                or not np.array_equal(self.intrinsics, intrinsics)
                or not np.array_equal(self.poses, poses)
                or self.shape != shape or self.stride != stride):
            raise ValueError("Projection cache must bind the same reference geometry and cameras")

    def get(self, index):
        if index not in self.indices:
            self.indices[index] = tuple(
                detector_winners(world, self.intrinsics, self.poses[index], self.shape, self.stride, footprint=True)
                for world in self.worlds)
        return self.indices[index]

def _cell_means(image, size, minimum_bins):
    tiles = _tiles(image, size)
    valid = np.isfinite(tiles) & (tiles >= 0)
    counts = valid.sum(axis=(1, 3))
    means = np.where(valid, tiles, 0.).sum(axis=(1, 3)) / np.maximum(counts, 1)
    means[counts < minimum_bins] = np.nan
    return means


def _region_and_weights(shape, region, min_cells_per_quadrant):
    rows, cols = shape
    y, x = np.indices(shape)
    if region is None:
        region = ((y >= rows//4) & (y < rows-rows//4) & (x >= cols//4) & (x < cols-cols//4))
    region = np.asarray(region)
    if region.shape != shape or region.dtype != bool:
        raise ValueError("Expected a predeclared Boolean detector-cell region")
    quadrant = (2 * (y >= rows//2) + (x >= cols//2))[region]
    counts = np.bincount(quadrant, minlength=4)
    represented = counts > 0
    if represented.sum() < 3 or np.any(counts[represented] < min_cells_per_quadrant):
        raise ValueError("Fixed detector region lacks support across at least three quadrants")
    weights = 1. / (represented.sum() * counts[quadrant])
    plane = np.column_stack((np.ones(len(weights)), x[region] / cols, y[region] / rows))
    root = np.sqrt(weights)
    projector = np.linalg.pinv(plane * root[:, None]) * root
    return region, weights, plane, projector


def _unit(signal, weights, floor):
    centred = signal - signal[0]
    centred -= weights @ centred
    rms = np.sqrt(weights @ (centred**2))
    floor = max(floor, 64 * np.finfo(float).eps * np.abs(signal).max())
    return centred / rms if rms > floor else None


def _loss(observed, predicted, weights, floor):
    template = _unit(predicted, weights, floor)
    if template is None:
        return None
    gain = weights @ (observed * template)
    if gain <= 0.:
        return None
    error = np.abs(observed - gain * template)
    return float(weights @ np.where(error <= 1., .5*error**2, error - .5))


def common_cells(observed, actual, flat, size, fixed, minimum):
    """Average the exact same detector subbins for the observation and both controls."""
    images = observed, actual, flat
    common = np.logical_and.reduce([np.isfinite(image) & (image >= 0) for image in images])
    counts = _tiles(common, size).sum(axis=(1, 3))[fixed]
    if (counts < minimum).any():
        return None
    means = [_tiles(np.where(common, image, 0.), size).sum(axis=(1, 3))[fixed] / counts
             for image in images]
    return *means, counts


def rank_pose_bank(cloud, query_luminance, query_intrinsics, poses, *, cell_pixels=32,
                   detector_stride=4, region=None, min_cells_per_quadrant=2,
                   min_occupancy=.125, min_spatial_rms=1e-9, evidence=None, projection_bank=None):
    """Rank 3D, plane and affine trend on identical measured and predicted subbins."""
    query = _image(query_luminance)
    bank = np.asarray(poses, dtype=float)
    if (bank.ndim != 3 or bank.shape[1:] != (4, 4) or not 1 <= len(bank) <= MAX_POSES
            or not isinstance(cloud, ReferenceCloud) or len(cloud.radiance) > MAX_SOURCE_SAMPLES
            or type(cell_pixels) is not int or cell_pixels not in (16, 32)
            or type(detector_stride) is not int or detector_stride <= 0
            or cell_pixels % detector_stride or type(min_cells_per_quadrant) is not int or min_cells_per_quadrant < 1
            or not 0 < min_occupancy <= 1 or not np.isfinite(min_spatial_rms) or min_spatial_rms < 0
            or (evidence is not None and (not isinstance(evidence, dict) or evidence))):
        raise ValueError("Invalid bounded pose bank or detector sampling policy")
    query_cells = _cell_means(query, cell_pixels, cell_pixels**2)
    fixed, weights, plane, projector = _region_and_weights(query_cells.shape, region, min_cells_per_quadrant)
    n, count = len(bank), int(fixed.sum())
    losses = {name: [None] * n for name in ARM_NAMES}
    report = dict(cell_pixels=cell_pixels, candidate_count=n, fixed_cells=count,
                  common_supported_candidates=0, scores=losses, orders={name: [] for name in ARM_NAMES})
    query_fixed = query_cells[fixed]
    kept = []
    snapshots = {"observed": [], "geometry_3d": [], "planar": [], "counts": []}

    def finish(reason):
        if evidence is not None:
            evidence.update(query_fixed=query_fixed.copy(), indices=np.array(kept, dtype=np.int64))
            for name, values in snapshots.items():
                dtype = np.int16 if name == "counts" else np.float64
                evidence[name] = np.array(values, dtype=dtype).reshape(len(kept), count)
        return dict(report, reason=reason)

    if not np.isfinite(query_fixed).all():
        return finish("invalid_observation_on_fixed_cells")
    if _unit(query_fixed, weights, min_spatial_rms) is None:
        return finish("uniform_observation_on_fixed_cells")
    detector_query = _cell_means(query, detector_stride, detector_stride**2)
    cell_bins = cell_pixels // detector_stride
    minimum = math.ceil(min_occupancy * cell_bins**2)
    shared = projection_bank if projection_bank is not None else ProjectionBank(
        cloud, query_intrinsics, bank, query.shape, detector_stride)
    shared.validate(cloud, query_intrinsics, bank, query.shape, detector_stride)
    for index in range(len(bank)):
        predicted = [rasterize(indices, cloud.radiance) for indices in shared.get(index)]
        measurements = common_cells(detector_query, *predicted, cell_bins, fixed, minimum)
        if measurements is None:
            continue
        observed, real, flat, counts = measurements
        kept.append(index)
        if evidence is not None:
            for name, array in zip(snapshots, measurements, strict=True):
                snapshots[name].append(array)
        observed = _unit(observed, weights, min_spatial_rms)
        if observed is None:
            continue
        signals = real, flat, plane @ (projector @ real)
        for name, signal in zip(ARM_NAMES, signals, strict=True):
            losses[name][index] = _loss(observed, signal, weights, min_spatial_rms)
    report["common_supported_candidates"] = len(kept)
    for name, scores in losses.items():
        report["orders"][name] = sorted((i for i, loss in enumerate(scores) if loss is not None),
                                        key=lambda i: (scores[i], i))
    if not kept:
        reason = "insufficient_common_candidate_support"
    elif not report["orders"]["geometry_3d"]:
        reason = "no_informative_3d_template"
    else:
        reason = None
    return finish(reason)
