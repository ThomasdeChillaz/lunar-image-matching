"""Fresh native scene-linear warp, measurement and AP3P procedural control."""

import json

import numpy as np

from handoff_pipeline import run


def generated_trial():
    rng = np.random.default_rng(20261120)
    reference = 1e-7 + 1e-6 * rng.random((64, 64)).repeat(8, 0).repeat(8, 1)
    reference += 1e-8 * rng.random(reference.shape)
    query = np.full_like(reference, 1e-7)
    query[:, :-40] = reference[:, 40:]
    query[:, :-40] = query[:, :-40] * 1.2 + 3e-8
    measured_support = np.ones_like(query, dtype=bool)
    measured_support[:, -40:] = False
    depth = np.full_like(reference, 10.)
    intrinsics = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    pose = np.eye(4)
    return (reference, depth, intrinsics, pose, query, intrinsics, pose[None]), measured_support


def control():
    args, mask = generated_trial()
    output = run(*args, query_valid=mask)
    truth = np.array([40. / 440. * 10., 0., 0.])
    summary = {}
    for name, arm in output["arms"].items():
        p = np.asarray(arm["raw_pose"]) if arm["raw_pose"] is not None else None
        summary[name] = dict(
            pose_returned=p is not None,
            translation_error_m=None if p is None else float(np.linalg.norm(p[:3, 3]-truth)),
            inliers=arm["candidate_rows"][0]["inliers"],
            measured_matches=arm["candidate_rows"][0]["measured_patch_count"],
        )
    passed = (summary["wide_3d"]["pose_returned"] and summary["wide_3d"]["translation_error_m"] <= .03
              and summary["wide_planar"]["pose_returned"] and summary["wide_planar"]["translation_error_m"] <= .03
              and not summary["narrow_3d"]["pose_returned"])
    return dict(passed=passed, scope="Procedural same-light planar native512 translation. No lunar result.",
                true_native_pixel_shift_x=-40, raw_arms=summary)


if __name__ == "__main__":
    data = control()
    print(json.dumps(data, indent=2, allow_nan=False))
    if not data["passed"]:
        raise SystemExit(1)
