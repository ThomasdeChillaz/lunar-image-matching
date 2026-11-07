"""Known-target native controls for the separate v126 patch-selection pilot."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from measure_v126 import BUDGET, NATIVE, run


def generated(multidepth):
    """Render an exact independent horizontal translation on each visible layer."""
    rng = np.random.default_rng(20261126)
    reference = 1e-7 + 1e-6 * rng.random((64, 64)).repeat(8, 0).repeat(8, 1)
    reference += 1e-8 * rng.random(reference.shape)
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    truth = np.eye(4)
    truth[0, 3] = 20.
    strips = ((0, 180, 10), (180, 333, 8), (333, 512, 6)) if multidepth else ((0, 512, 8),)
    depth = np.empty_like(reference)
    query = np.full_like(reference, 1e-7)
    valid = np.zeros_like(reference, dtype=bool)
    for start, stop, shift in strips:
        depth[start:stop] = k[0, 0] * truth[0, 3] / shift
        query[start:stop, :-shift] = 1.2 * reference[start:stop, shift:] + 3e-8
        valid[start:stop, :-shift] = True
    seeds = np.repeat(np.eye(4)[None], 5, axis=0)
    seeds[1:, :3, 3] = [[95., 60., 75.], [-90., 85., -80.], [-65., -100., 45.], [100., -80., -45.]]
    return (reference, depth, k, np.eye(4), query, k, seeds), valid, truth


def angular_error(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.) / 2., -1., 1.))))


def summarize_arm(arm, truth, depth, intrinsics):
    raw = arm["raw_pose"]
    pose = None if raw is None else np.asarray(raw)
    errors = []
    for row in arm["measurements"]:
        x, y = row["reference_xy"]
        shift = intrinsics[0, 0] * truth[0, 3] / depth[y, x]
        errors.append(float(np.linalg.norm(np.asarray(row["query_xy"]) - [x - shift, y])))
    rotation = None if pose is None else angular_error(pose, truth)
    translation = None if pose is None else float(np.linalg.norm(pose[:3, 3] - truth[:3, 3]))
    passed = (pose is not None and rotation <= .05 and translation <= .25 and len(errors) >= 6
              and max(errors) <= .1 and arm["patch_searches"] == BUDGET)
    return dict(passed=passed, raw_pose=raw, measured_matches=len(errors), inliers=arm["inliers"],
                selected_anchors=arm["selected_anchors"], patch_searches=arm["patch_searches"],
                evaluated_positions=arm["evaluated_positions"],
                maximum_native_error_px=max(errors) if errors else None,
                translation_error_m=translation, rotation_error_deg=rotation, reason=arm["reason"])


def control():
    cv2.setNumThreads(1)
    rows = []
    for multidepth in (False, True):
        args, mask, truth = generated(multidepth)
        tick = perf_counter()
        raw = run(*args, query_valid=mask)
        elapsed = perf_counter() - tick
        arms = {name: summarize_arm(arm, truth, args[1], args[2]) for name, arm in raw["arms"].items()}
        changed = set(arms["uniform"]["selected_anchors"]) != set(arms["parallax"]["selected_anchors"])
        passed = all(row["passed"] for row in arms.values()) and changed == multidepth
        rows.append(dict(multidepth=multidepth, passed=passed, row_wall_seconds=elapsed, arms=arms,
                         shared_geometry_seconds=raw["shared_geometry_seconds"],
                         policy_seconds=raw["policy_seconds"], selected_set_differs=changed))
    return dict(schema="solar_parallax_procedural_v126", passed=all(row["passed"] for row in rows),
                procedural_only=True, archived_lunar_images_opened=False, native_shape=[NATIVE, NATIVE],
                row_wall_seconds=[row["row_wall_seconds"] for row in rows], rows=rows,
                scope="Generated known translation. Tests independent measurement and policy wiring.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    receipt = control()
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not receipt["passed"]:
        raise SystemExit(1)
