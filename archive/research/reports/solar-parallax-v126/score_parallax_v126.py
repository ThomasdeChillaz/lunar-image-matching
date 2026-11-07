"""Independent native anchor and pose arithmetic for sealed v126 observations."""

import math
from pathlib import Path

import numpy as np

from archived_warp_worker import asset, roster
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import correspondences_z, median_reference_range_z, project_z, world_points_z


AXIS = (72, 145, 218, 293, 366, 439)
ANCHORS = np.asarray([(x, y) for y in AXIS for x in AXIS], dtype=float)
ARMS = ("parallax", "uniform")
BUDGET = 12
THRESHOLDS = (1, 2, 4, 8, 12)
CELLS = (ANCHORS[:, 1] // 128 * 4 + ANCHORS[:, 0] // 128).astype(int)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Finite nonnegative measured time required")
    return value


def geometry(root, reference, label):
    with np.load(asset(root, label["joint_scoring_geometry"]), allow_pickle=False) as original:
        z = {k: original[k].copy() for k in ("reference_depth", "query_depth", "reference_intrinsics",
                                             "reference_cam2world", "query_intrinsics", "query_cam2world")}
    with np.load(asset(root, reference["reference_geometry"]), allow_pickle=False) as r:
        need(np.array_equal(r["axial_depth"], z["reference_depth"], equal_nan=True)
             and np.array_equal(r["cam2world"], z["reference_cam2world"])
             and np.array_equal(r["intrinsics"], z["reference_intrinsics"]), "Original reference geometry differs")
    with np.load(asset(root, label["query_geometry"]), allow_pickle=False) as q:
        need(np.array_equal(q["axial_depth"], z["query_depth"], equal_nan=True)
             and np.array_equal(q["cam2world"], z["query_cam2world"])
             and np.array_equal(q["intrinsics"], z["query_intrinsics"])
             and np.array_equal(np.asarray(reference["query_intrinsics"]), z["query_intrinsics"]),
             "Original post-seal query geometry differs")
    need(z["query_depth"].shape == z["reference_depth"].shape == (512, 512), "Use original native scoring images")
    args = tuple(z[k] for k in ("reference_depth", "query_depth", "reference_intrinsics", "reference_cam2world",
                               "query_intrinsics", "query_cam2world"))
    _, compact, _, visible = correspondences_z(*args, points=ANCHORS)
    target = np.full((len(ANCHORS), 2), np.nan)
    target[visible] = compact
    index = ANCHORS.astype(int)
    depth = z["reference_depth"][index[:, 1], index[:, 0]]
    points = world_points_z(ANCHORS, depth, z["reference_intrinsics"], z["reference_cam2world"])
    return target, visible, points, z["query_cam2world"], z["query_intrinsics"], median_reference_range_z(*(
        z["reference_depth"], z["reference_intrinsics"],
    ))


def pose_metrics(raw_pose, truth, distance):
    if raw_pose is None:
        return dict(raw_pose_returned=False, correct=False, strict=False, wrong_raw_pose=False, pose_error=None)
    _, pose = camera_check(np.eye(3), np.asarray(raw_pose, dtype=float))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ truth[:3, :3]) - 1.) / 2., -1., 1.))))
    shift = float(np.linalg.norm(pose[:3, 3] - truth[:3, 3]))
    relative = shift / distance
    correct = angle <= 2. and relative <= .01
    return dict(raw_pose_returned=True, correct=correct, strict=angle <= .5 and relative <= .001,
                wrong_raw_pose=not correct, pose_error=dict(rotation_deg=angle, translation_m=shift,
                                                           translation_fraction_of_reference_range=relative))


def quadrants(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return len(set(((points[:, 1] >= 256).astype(int) * 2 + (points[:, 0] >= 256).astype(int)).tolist()))


def verify_patch(item, observed):
    patch = item["patch_measurement"]
    seed = np.asarray(item["seed_projection_xy"], dtype=float)
    forward, reverse = (patch[name]["fine"] for name in ("forward", "reverse"))
    fwd, rev = (np.asarray(row["offset_xy"], dtype=float) for row in (forward, reverse))
    need(seed.shape == fwd.shape == rev.shape == (2,) and np.isfinite(seed).all()
         and np.array_equal(fwd, np.rint(fwd)) and np.array_equal(rev, np.rint(rev))
         and (np.abs(fwd) <= 16).all() and (np.abs(rev) <= 16).all()
         and np.array_equal(seed + fwd, observed) and np.array_equal(patch["query_xy"], observed)
         and patch["reason"] is None and patch["radius_pixels"] == 16
         and forward["reason"] is None and reverse["reason"] is None
         and min(forward["score"], reverse["score"]) >= .55 and min(forward["margin"], reverse["margin"]) >= .015
         and patch["round_trip_error_px"] == np.linalg.norm(fwd + rev) <= 1.
         and patch["support_pixels"] >= 2048 and 0 < patch["evaluated_positions"] <= patch["max_total_positions"] == 500,
         "Require sealed reciprocal native image measurement rather than a projected anchor")


def score_arm(row, truth):
    target, visible, world, truth_pose, query_k, distance = truth
    selection = row["selected_anchors"]
    used = row["patch_searches"]
    measurements = row["measurements"]
    timing = {name: seconds(row[name]) for name in ("measurement_seconds", "pose_seconds")}
    need(type(used) is int and 0 <= used == len(selection) == len(set(selection)) <= BUDGET
         and selection == sorted(selection)
         and all(type(i) is int and 0 <= i < len(ANCHORS) for i in selection)
         and type(row["measured_patch_count"]) is int and len(measurements) <= row["measured_patch_count"] <= used,
         "Each arm must preserve its fixed selection and measurement budget")
    failure = [item["anchor_index"] for item in row["failures"]]
    need(len(failure) == len(set(failure)) and set(failure) <= set(selection)
         and row["measured_patch_count"] + len(failure) == used,
         "All selected attempts and refusals must remain in the denominator")
    indices, image, points = [], [], []
    anchor_scores = {i: dict(anchor_index=i, co_visible=bool(visible[i]), measured=False, error_px=None) for i in selection}
    for item in measurements:
        i = item["anchor_index"]
        point = np.asarray(item["query_xy"], dtype=float)
        need(i in anchor_scores and i not in indices and i not in failure
             and np.array_equal(item["reference_xy"], ANCHORS[i])
             and np.isfinite(world[i]).all() and np.allclose(item["reference_world"], world[i], atol=1e-6, rtol=0.)
             and point.shape == (2,) and np.isfinite(point).all() and ((point >= 0.) & (point < 512.)).all(),
             "Require actual measured native endpoint at an original reference anchor")
        verify_patch(item, point)
        need(all(np.linalg.norm(point - previous) > 2. for previous in image),
             "Independent native endpoint deduplication failed")
        indices.append(i)
        image.append(point)
        points.append(world[i])
        anchor_scores[i].update(measured=True, error_px=float(np.linalg.norm(target[i]-point)) if visible[i] else None)
    outcome = pose_metrics(row["raw_pose"], truth_pose, distance)
    inliers = row["inliers"]
    need(type(inliers) is int and 0 <= inliers <= len(indices), "Pose inlier count differs from native observations")
    if row["raw_pose"] is not None:
        projected, _, front = project_z(np.asarray(points), query_k, np.asarray(row["raw_pose"]))
        residual = np.linalg.norm(projected - image, axis=1)
        used_for_pose = np.flatnonzero(front & np.isfinite(residual) & (residual <= 3.)).tolist()
        need(len(used_for_pose) == inliers >= 6 and used_for_pose == row["inlier_indices"]
             and min(quadrants(ANCHORS[np.asarray(indices)[used_for_pose]]),
                     quadrants(np.asarray(image)[used_for_pose])) >= 3,
             "Saved pose support must agree with independent common three-pixel residual arithmetic")
    precise = [j for j, i in enumerate(indices) if anchor_scores[i]["error_px"] is not None
               and anchor_scores[i]["error_px"] <= 2.]
    precise_regions = [quadrants(ANCHORS[np.asarray(indices, dtype=int)[precise]]),
                       quadrants(np.asarray(image, dtype=float).reshape(-1, 2)[precise])]
    qualified = outcome["correct"] and len(precise) >= 6 and min(precise_regions) >= 3
    return dict(
        selected_anchors=selection, patch_searches=used, pose_inliers=inliers, measured_anchors=len(measurements),
        measured_patch_count=row["measured_patch_count"], failed_patch_count=len(failure),
        deduplicated_patch_count=row["measured_patch_count"] - len(measurements),
        cell_quotas=np.bincount(CELLS[selection], minlength=16).tolist(),
        qualifying_pose=qualified, two_pixel_measured_count=len(precise), two_pixel_quadrants=precise_regions,
        selected_co_visible=sum(item["co_visible"] for item in anchor_scores.values()),
        measured_co_visible=sum(item["co_visible"] and item["measured"] for item in anchor_scores.values()),
        correct_native_correspondences={str(t): sum(a["error_px"] is not None and a["error_px"] <= t
                                                      for a in anchor_scores.values()) for t in THRESHOLDS},
        anchors=list(anchor_scores.values()), reason=row["reason"], timing=timing, **outcome,
    )


def feasibility(rows):
    main = [row for row in rows if row["pairing"][0] in ("exposure-0", "exposure-1")]
    faint = [row for row in main if row["pairing"][0] == "exposure-0"]
    need(len(faint) == 4 and len(main) == 8, "Faint and bright must retain all original seed pairings")
    wins = [row["pairing"] for row in faint if row["arms"]["parallax"]["qualifying_pose"]
            and not row["arms"]["uniform"]["qualifying_pose"]]
    losses = [row["pairing"] for row in main if any(
        row["arms"]["uniform"][key] and not row["arms"]["parallax"][key] for key in ("correct", "qualifying_pose")
    )]
    return dict(passed=len(wins) >= 2 and not losses, faint_additional_qualifying_pairings=wins,
                faint_or_bright_paired_losses=losses, research_feasibility_only=True, automatic_advance=False)


def replay_scores(p, raw):
    """Reopen scoring data only after the calling controller verifies the blind seal."""
    root = Path.cwd().resolve()
    reference = {r["tag"]: r for r in roster(Path(p["inputs"]["path"]), p["inputs"]["sha256"], "inference")}
    labels = {r["tag"]: r for r in roster(Path(p["labels"]["path"]), p["labels"]["sha256"], "scoring_only")}
    rows, geometries = [], {}
    for row in raw["rows"]:
        tag, r, q = row["pairing"]
        if tag not in geometries:
            geometries[tag] = geometry(root, reference[tag], labels[tag])
        arms = {name: score_arm(row["arms"][name], geometries[tag]) for name in ARMS}
        elapsed = seconds(row["native_elapsed_seconds"])
        component = seconds(row["shared_geometry_seconds"]) + seconds(row["policy_seconds"]) + sum(
            sum(arm["timing"].values()) for arm in arms.values()
        )
        selection = row["selection"]
        quotas = [0] * 16 if selection is None else selection["cell_quotas"]
        need(arms["parallax"]["patch_searches"] == arms["uniform"]["patch_searches"]
             and arms["parallax"]["cell_quotas"] == arms["uniform"]["cell_quotas"] == quotas
             and row["seed_count"] == len(row["seed_indices"]) <= 8
             and row["patch_budget"] == BUDGET and row["radius_pixels"] == 16
             and row["raw_only"] is True and row["calibrated_acceptance"] is False and component <= elapsed,
             "Independent timing, pose-seed and equal-budget checks failed")
        rows.append(dict(case_id=labels[tag]["case_id"], pairing=[tag, r, q], arms=arms,
                         native_elapsed_seconds=elapsed, shared_geometry_seconds=row["shared_geometry_seconds"],
                         policy_seconds=row["policy_seconds"], seed_indices=row["seed_indices"],
                         reference_range_m=geometries[tag][-1]))
    return dict(schema="solar_plane_initialized_native_scores_v126", rows=rows, feasibility_gate=feasibility(rows),
                accepted_results=False,
                known_reference_illumination=True, calibration_or_final_access=False,
                scope="Independent saved native anchor and raw camera pose arithmetic on exposed archival positives only.")
