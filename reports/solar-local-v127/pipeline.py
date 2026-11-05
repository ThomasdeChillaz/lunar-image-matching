"""Compare native measurement corrections with the full Direct reference search."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V126 = ROOT / "reports/solar-parallax-v126"
for directory in (ROOT, V126):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
V125 = ROOT / "reports/solar-warp-v125"
if str(V125) not in sys.path:
    sys.path.insert(0, str(V125))

import measure_v126 as previous
from direct_render_adapter_v73 import SiftLibraryMatcher
from localization_robust_pose_v4 import CONFIGS, estimate
from lunar_geometry_v5 import map_points_z
from masked_patch import search_patch, original
from parallax_policy import select
from fast_projection import FastProjectionBank, original as coarse


METHODS = ("legacy", "halo", "pooled_uniform", "pooled_parallax", "direct")
SOLVER = next(item for item in CONFIGS if item["name"] == "ap3p_reclassify")


def initialize(reference, query, depth, k, reference_pose, query_k):
    cloud = coarse.build_reference_cloud(reference, depth, k, reference_pose)
    bank = coarse.reference_pose_bank(reference_pose)
    projected = FastProjectionBank(cloud, query_k, bank, reference.shape, 4)
    ranked = coarse.rank_pose_bank(cloud, query, query_k, bank, cell_pixels=32, projection_bank=projected)
    indices = ranked["orders"]["planar"][:8]
    return bank[indices], dict(indices=indices, candidate_count=len(bank), reason=ranked["reason"])


def encode_image(image, valid=None):
    mask = np.isfinite(image) & (image >= 0.)
    if valid is not None:
        mask &= valid
    if not mask.any():
        raise ValueError("Photometry is unavailable")
    low, high = np.quantile(image[mask], [.01, .99])
    if high <= low:
        high = low + max(abs(low) * 1e-6, 1e-12)
    encoded = np.rint(255 * np.clip((image - low) / (high - low), 0., 1.)).astype(np.uint8)
    return encoded, dict(low=float(low), high=float(high))


def common_images(reference, query, valid):
    """Give every method the same independent native linear image encodings."""
    a, pa = encode_image(reference, valid)
    b, pb = encode_image(query)
    return a, b, dict(reference=pa, query=pb, source="independent_image_1_and_99_percentiles")


def solve(p, q, depth, k, reference_pose, query_k):
    p, q = np.asarray(p, float).reshape(-1, 2), np.asarray(q, float).reshape(-1, 2)
    world, valid = map_points_z(p, depth, k, reference_pose)
    p, q = p[valid], q[valid]
    pose, info = estimate(world, q, query_k, SOLVER)
    return dict(reference_xy=p.tolist(), query_xy=q.tolist(), reference_world=world.tolist(),
                pose=None if pose is None else np.asarray(pose).tolist(), estimator=info)


def patch_measure(plane, query, indices, search, depth, k, reference_pose, query_k):
    failures, p, q, traces = [], [], [], []
    for index in sorted(indices):
        centre = plane.predicted_xy[index]
        measured = search(plane.raster, query, centre, centre, radius=16,
                          reference_valid=plane.detector_valid)
        traces.append(dict(anchor_index=index, search=measured))
        if measured["query_xy"] is None:
            failures.append(dict(anchor_index=index, reason=measured["reason"]))
        else:
            p.append(previous.ANCHORS[index].tolist())
            q.append(measured["query_xy"])
    return dict(**solve(p, q, depth, k, reference_pose, query_k), failures=failures, searches=traces,
                selected_anchors=sorted(indices), attempted_patches=len(indices))


def direct(reference, query, library, depth, k, reference_pose, query_k):
    """Published SIFT/ranking/PnP settings, with the matched-light reference added."""
    matcher = SiftLibraryMatcher(dict(nfeatures=2048, ratio_threshold=.75))
    query_features = matcher.query_features(query)
    proposals = []
    for i, image in enumerate([*(encode_image(image)[0] for image in library), reference]):
        p, q, appearance = matcher.match(image, query_features)
        proposals.append(dict(index=i, p=p, q=q, appearance=appearance))
    proposals.sort(key=lambda r: (-len(r["p"]), -r["appearance"], r["index"]))
    candidates = []
    for row in proposals[:8]:
        solved = solve(row["p"], row["q"], depth, k, reference_pose, query_k)
        candidates.append(dict(**solved, library_index=row["index"], matches=len(row["p"])))
    candidates.sort(key=lambda r: (r["pose"] is None, -r["estimator"].get("inliers", 0),
                                   r["estimator"].get("clipped_squared_reprojection_cost", float("inf")),
                                   -r["matches"], r["library_index"]))
    result = candidates[0]
    return dict(result, library_images_considered=len(proposals), pose_candidates=len(candidates),
                matched_reference_added=True, configuration=dict(nfeatures=2048, ratio_threshold=.75))


def run(reference, query, depth, k, reference_pose, query_k, seeds, library, *, reference_valid,
        initialization_seconds=0.):
    encoding_start = perf_counter()
    reference, query, encoding = common_images(reference, query, reference_valid)
    encoding_seconds = perf_counter() - encoding_start
    start = perf_counter()
    prepared = None
    if len(seeds):
        warper = previous.ReferencePatchWarper(reference.astype(float), depth, k, reference_pose,
                                              previous.ANCHORS, reference_valid=reference_valid)
        plane = warper.warp(query_k, seeds[0], reference.shape)["planar"]
        eligible = np.array([available and previous.template_available(plane.detector_valid, centre)
                             for available, centre in zip(plane.anchor_visible, plane.predicted_xy, strict=True)])
        selection = select(plane.reference_world, previous.ray_plane_points(warper), previous.ANCHORS,
                           query_k, seeds, eligible, budget=previous.BUDGET)
        prepared = plane, selection
    setup = perf_counter() - start
    outputs = {}
    for name in METHODS[:-1]:
        tick = perf_counter()
        if prepared is None:
            result = dict(**solve([], [], depth, k, reference_pose, query_k), failures=[], searches=[],
                          selected_anchors=[], attempted_patches=0)
        else:
            plane, selection = prepared
            indices = selection["active"] if name == "pooled_parallax" else selection["uniform"]
            if name == "legacy":
                search = original.search_patch
            elif name == "halo":
                search = lambda *a, **kw: search_patch(*a, **kw, mode="halo")
            else:
                search = search_patch
            result = patch_measure(plane, query.astype(float), indices, search, depth, k, reference_pose, query_k)
        outputs[name] = dict(result, inference_seconds=perf_counter() - tick + setup + initialization_seconds + encoding_seconds)
    tick = perf_counter()
    outputs["direct"] = dict(direct(reference, query, library, depth, k, reference_pose, query_k),
                             inference_seconds=perf_counter() - tick + encoding_seconds)
    return dict(methods=outputs, encoding=encoding, setup_seconds=setup,
                encoding_seconds=encoding_seconds,
                selection=None if prepared is None else prepared[1],
                reference_side_seeds=len(seeds), calibrated_acceptance=False)
