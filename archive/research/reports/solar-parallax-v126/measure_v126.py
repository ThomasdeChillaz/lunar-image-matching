"""Measure native patches from one plane initializer under two equal budgets."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HANDOFF = Path(__file__).resolve().parents[1] / "solar-warp-handoff-v120"
if str(HANDOFF) not in sys.path:
    sys.path.insert(0, str(HANDOFF))

from handoff_localizer import MAX_SEEDS, NATIVE, in_image, solve_observations
from handoff_patch import _crop, _sample_layout, search_patch
from reference_patch_warp import ReferencePatchWarper, fitted_plane

from parallax_policy import select


AXIS = (72, 145, 218, 293, 366, 439)
ANCHORS = np.asarray([(x, y) for y in AXIS for x in AXIS], dtype=int)
BUDGET = 12
RADIUS = 16


def ray_plane_points(warper):
    """Intersect the same reference rays with the single fitted height plane."""
    grid = warper.world[4::8, 4::8]
    plane = fitted_plane(grid[np.isfinite(grid).all(axis=-1)])
    if plane is None:
        return np.full_like(warper.reference_world, np.nan)
    normal, offset = plane
    centre = warper.reference_pose[:3, 3]
    rays = warper.reference_world - centre
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = (offset - normal @ centre) / (rays @ normal)
    usable = np.isfinite(scale) & (scale > 0)
    return centre + rays * np.where(usable, scale, np.nan)[:, None]


def template_available(mask, centre):
    """Use the same valid taps and layout as the frozen reciprocal search."""
    if not in_image(centre):
        return False
    integer = np.floor(centre).astype(int)
    fx, fy = centre - integer
    if fx:
        mask = mask[:, :-1] & mask[:, 1:]
    if fy:
        mask = mask[:-1] & mask[1:]
    patch = _crop(mask, integer)
    return patch is not None and _sample_layout(patch) is not None


def observe(plane, query, query_mask, query_intrinsics, order):
    observations, failures, positions = [], [], 0
    tick = perf_counter()
    for index in order:
        centre = plane.predicted_xy[index]
        measured = search_patch(plane.raster, query, centre, centre, radius=RADIUS,
                                reference_valid=plane.detector_valid, query_valid=query_mask)
        positions += measured["evaluated_positions"]
        observed = measured["query_xy"]
        if observed is None:
            failures.append(dict(anchor_index=index, reason=measured["reason"], search=measured))
            continue
        if measured["reason"] is not None or not in_image(np.asarray(observed)):
            raise ValueError("Patch search did not return a valid observed native endpoint")
        observations.append(dict(anchor_index=index, reference_xy=ANCHORS[index].tolist(),
                                 reference_world=plane.reference_world[index].tolist(),
                                 query_xy=observed, seed_projection_xy=centre.tolist(),
                                 patch_measurement=measured))
    measurement_seconds = perf_counter() - tick
    tick = perf_counter()
    raw = solve_observations(observations, query_intrinsics)
    return dict(raw_pose=raw["pose"], reason=raw["reason"], measurements=raw["measurements"],
                inliers=raw["inliers"], inlier_indices=raw["inlier_indices"], pose_score=raw["score"],
                selected_anchors=order, failures=failures, patch_searches=len(order),
                evaluated_positions=positions, measured_patch_count=len(observations),
                measurement_seconds=measurement_seconds, pose_seconds=perf_counter() - tick)


def run(reference_luminance, reference_axial_depth, reference_intrinsics, reference_pose,
        query_luminance, query_intrinsics, seeds, *, reference_valid=None, query_valid=None):
    """Use the first plane camera for both arms and keep query radiance out of selection."""
    shape = (NATIVE, NATIVE)
    query = np.asarray(query_luminance, dtype=float)
    poses = np.asarray(seeds, dtype=float)
    intrinsics = np.asarray(query_intrinsics, dtype=float)
    if (np.asarray(reference_luminance).shape != shape or query.shape != shape
            or not np.isfinite(query).all() or poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or len(poses) > MAX_SEEDS or not np.isfinite(poses).all()):
        raise ValueError("Require native radiance and up to eight finite camera hypotheses")
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.array_equal(intrinsics[2], [0., 0., 1.])
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
        raise ValueError("Require calibrated native query intrinsics")
    query_mask = np.ones(shape, dtype=bool) if query_valid is None else np.asarray(query_valid)
    if query_mask.shape != shape or query_mask.dtype != bool:
        raise ValueError("Query sensor validity must be native Boolean")
    if not len(poses):
        def empty():
            return dict(raw_pose=None, reason="empty_plane_seed_bank", measurements=[], inliers=0,
                        inlier_indices=[], pose_score=None, selected_anchors=[], failures=[],
                        patch_searches=0, evaluated_positions=0, measured_patch_count=0,
                        measurement_seconds=0., pose_seconds=0.)

        return dict(arms={name: empty() for name in ("uniform", "parallax")}, selection=None,
                    shared_geometry_seconds=0., policy_seconds=0., seed_count=0,
                    raw_only=True, calibrated_acceptance=False, patch_budget=BUDGET, radius_pixels=RADIUS)

    tick = perf_counter()
    warper = ReferencePatchWarper(reference_luminance, reference_axial_depth, reference_intrinsics,
                                  reference_pose, ANCHORS, reference_valid=reference_valid)
    plane = warper.warp(intrinsics, poses[0], shape)["planar"]
    eligible = np.asarray([available and template_available(plane.detector_valid, centre)
                           for available, centre in zip(plane.anchor_visible, plane.predicted_xy, strict=True)])
    flat = ray_plane_points(warper)
    shared_geometry_seconds = perf_counter() - tick
    tick = perf_counter()
    selection = select(plane.reference_world, flat, ANCHORS, intrinsics, poses, eligible, budget=BUDGET)
    policy_seconds = perf_counter() - tick
    control, active = selection["uniform"], selection["active"]
    expected = min(BUDGET, int(eligible.sum()))
    for order in (control, active):
        if (len(order) != expected or len(set(order)) != expected
                or any(type(index) is not int or not 0 <= index < len(ANCHORS) or not eligible[index] for index in order)):
            raise ValueError("Both policies must use the same budget from their common eligible mask")
    # A shared template does not share measurements. Each arm performs its
    # own searches and keeps every failure in the equal-budget ledger.
    # Canonical measurement order prevents the priority order from changing
    # deduplication or the pose estimator when two schedules pick the same set.
    arms = {name: observe(plane, query, query_mask, intrinsics, sorted(order))
            for name, order in (("uniform", control), ("parallax", active))}
    return dict(arms=arms, selection=selection, shared_geometry_seconds=shared_geometry_seconds,
                policy_seconds=policy_seconds, seed_count=len(poses), raw_only=True,
                calibrated_acceptance=False, patch_budget=BUDGET, radius_pixels=RADIUS)
