"""Turn independently measured patch offsets into raw camera proposals."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from localization_robust_pose_v4 import CONFIGS, estimate
from lunar_geometry_v5 import project_z

NATIVE = 512
ANCHOR_AXIS = (80, 197, 314, 431)
ANCHORS = np.asarray([(x, y) for y in ANCHOR_AXIS for x in ANCHOR_AXIS], dtype=float)
MAX_SEEDS = 8
MIN_MATCHES = 6
MIN_REGIONS = 3
REPROJECTION_PX = 3.
ARMS = {
    "wide_3d": ("geometry_3d", 128),
    "narrow_3d": ("geometry_3d", 16),
    "wide_planar": ("planar", 128),
}
COMMON_SOLVER = next(cfg for cfg in CONFIGS if cfg["name"] == "ap3p_reclassify")


@dataclass(frozen=True)
class Templates:
    """A pair of co-registered predicted detector images in one seed camera."""

    geometry_3d: np.ndarray
    planar: np.ndarray
    common_valid: np.ndarray
    reference_xy: np.ndarray
    reference_world: np.ndarray
    predicted_centres_xy: np.ndarray
    anchor_valid: np.ndarray


def quadrants(points):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    return len(np.unique(2 * (xy[:, 1] >= NATIVE / 2) + (xy[:, 0] >= NATIVE / 2)))


def in_image(point):
    return np.isfinite(point).all() and ((point >= 0.) & (point <= NATIVE - 1)).all()


def distinct_measurements(observations):
    """Keep one endpoint per 2-pixel neighbourhood in fixed anchor order."""
    kept = []
    for row in observations:
        xy = np.asarray(row["query_xy"])
        if all(np.linalg.norm(xy - previous["query_xy"]) > 2. for previous in kept):
            kept.append(row)
    return kept


def solve_observations(observations, intrinsics, *, solve=estimate):
    """Call the common AP3P estimator only with observed, distributed endpoints."""
    kept = distinct_measurements(observations)
    reference = np.asarray([r["reference_xy"] for r in kept]).reshape(-1, 2)
    query = np.asarray([r["query_xy"] for r in kept]).reshape(-1, 2)
    world = np.asarray([r["reference_world"] for r in kept]).reshape(-1, 3)
    out = dict(pose=None, measurements=kept, inliers=0, inlier_indices=[], score=None)
    if len(kept) < MIN_MATCHES or min(quadrants(reference), quadrants(query)) < MIN_REGIONS:
        return dict(out, reason="insufficient_distributed_measurements")
    pose, details = solve(world, query, intrinsics, COMMON_SOLVER)
    if pose is None:
        return dict(out, reason=details["reason"])
    proposed, _, front = project_z(world, intrinsics, pose)
    residual = np.linalg.norm(proposed - query, axis=1)
    inliers = front & np.isfinite(residual) & (residual <= REPROJECTION_PX)
    count = int(inliers.sum())
    out.update(inliers=count, inlier_indices=np.flatnonzero(inliers).tolist())
    if count < MIN_MATCHES or min(quadrants(reference[inliers]), quadrants(query[inliers])) < MIN_REGIONS:
        return dict(out, reason="insufficient_distributed_pose_inliers")
    # All 16 slots incur the same cap when missing, behind the camera,
    # or outside the reprojection limit.
    clipped = np.minimum(np.where(front & np.isfinite(residual), residual, np.inf)**2, REPROJECTION_PX**2)
    score = float(clipped.sum() + (len(ANCHORS) - len(kept)) * REPROJECTION_PX**2)
    return dict(out, pose=np.asarray(pose).tolist(), reason=None, score=score)


def validate_templates(templates):
    if (not isinstance(templates, Templates)
            or not np.array_equal(templates.reference_xy, ANCHORS)
            or templates.reference_world.shape != (len(ANCHORS), 3)
            or templates.predicted_centres_xy.shape != ANCHORS.shape
            or templates.anchor_valid.shape != (len(ANCHORS),)
            or templates.anchor_valid.dtype != bool
            or templates.common_valid.shape != (NATIVE, NATIVE)
            or templates.common_valid.dtype != bool
            or templates.geometry_3d.shape != templates.common_valid.shape
            or templates.planar.shape != templates.common_valid.shape):
        raise ValueError("Frozen anchors and a common native detector mask are required")
    for image in (templates.geometry_3d, templates.planar):
        if not np.isfinite(image[templates.common_valid]).all():
            raise ValueError("Common detector support contains an invalid radiance value")
    if (not np.isfinite(templates.reference_world[templates.anchor_valid]).all()
            or not np.isfinite(templates.predicted_centres_xy[templates.anchor_valid]).all()):
        raise ValueError("Eligible source anchors must have finite world and detector coordinates")


def localize(query, query_intrinsics, seeds, warp, search, query_valid=None, *, solve=estimate):
    """Evaluate paired measurement arms from the same at-most-eight frozen seeds.

    The warp callback uses reference-side geometry, while the search callback
    must return an independent query observation or an explicit failure.
    This entry point never accepts query truth or a camera pose prior other
    than the input seed list.
    """
    query = np.asarray(query, dtype=float)
    seeds = np.asarray(seeds, dtype=float)
    if query.shape != (NATIVE, NATIVE) or not np.isfinite(query).all():
        raise ValueError("Require a finite native scene-linear query")
    if seeds.ndim != 3 or seeds.shape[1:] != (4, 4) or len(seeds) > MAX_SEEDS or not np.isfinite(seeds).all():
        raise ValueError("At most eight frozen proper camera poses may seed the search")
    if len(seeds):
        rot = seeds[:, :3, :3]
        if (not np.allclose(seeds[:, 3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(rot), 1., atol=1e-6)):
            raise ValueError("Malformed camera seed")
    intrinsics = np.asarray(query_intrinsics, dtype=float)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.allclose(intrinsics[2], [0, 0, 1]) or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
        raise ValueError("Require calibrated native query intrinsics")
    mask = np.ones(query.shape, dtype=bool) if query_valid is None else np.asarray(query_valid)
    if mask.shape != query.shape or mask.dtype != bool:
        raise ValueError("Query radiometric support mask must be native Boolean")
    arms = {name: [] for name in ARMS}
    shared_geometry_seconds = []
    for seed_index, pose in enumerate(seeds):
        tick = perf_counter()
        templates = warp(pose)
        shared_geometry_seconds.append(perf_counter() - tick)
        validate_templates(templates)
        for name, (model, radius) in ARMS.items():
            image = getattr(templates, model)
            observations, failures = [], []
            tick = perf_counter()
            for i, centre in enumerate(templates.predicted_centres_xy):
                if not templates.anchor_valid[i] or not in_image(centre):
                    failures.append(dict(anchor_index=i, reason="unsupported_reference_projection"))
                    continue
                result = search(image, query, centre, centre, radius=radius,
                                reference_valid=templates.common_valid, query_valid=mask)
                observed = result["query_xy"]
                if observed is None:
                    failures.append(dict(anchor_index=i, reason=result["reason"]))
                    continue
                observed = np.asarray(observed, dtype=float)
                if observed.shape != (2,) or not in_image(observed) or result["reason"] is not None:
                    raise ValueError("Unexpected measured native query position")
                observations.append(dict(anchor_index=i, reference_xy=ANCHORS[i].tolist(),
                                         reference_world=templates.reference_world[i].tolist(),
                                         seed_projection_xy=centre.tolist(), query_xy=observed.tolist(),
                                         patch_measurement=result))
            measurement_seconds = perf_counter() - tick
            tick = perf_counter()
            raw = solve_observations(observations, intrinsics, solve=solve)
            arms[name].append(dict(raw, seed_index=seed_index, anchor_slots=len(ANCHORS),
                                   measured_patch_count=len(observations), failures=failures,
                                   measurement_seconds=measurement_seconds, pose_seconds=perf_counter() - tick))
    output = {}
    for name, rows in arms.items():
        usable = [row for row in rows if row["pose"] is not None]
        best = min(usable, key=lambda row: (-row["inliers"], row["score"], row["seed_index"]), default=None)
        output[name] = dict(raw_pose=best["pose"] if best else None,
                            selected_seed=best["seed_index"] if best else None,
                            reason=None if best else ("no_pose_from_measured_matches" if rows else "empty_seed_bank"),
                            candidate_rows=rows)
    return dict(arms=output, raw_only=True, calibrated_acceptance=False,
                seed_count=len(seeds), shared_geometry_seconds_per_seed=shared_geometry_seconds,
                maximum_anchor_searches_per_arm=MAX_SEEDS * len(ANCHORS))
