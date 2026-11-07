"""Exact synthetic camera, reciprocal search and preserved control checks."""
from types import SimpleNamespace

import cv2
import numpy as np

from lunar_geometry_v5 import map_points_z, project_z
from solar_direct_additive_windows_v96 import add_windows, measure_window
from solar_perspective_windows_v99 import add_surface_windows, measure_surface_window, surface_map


K = np.array([[410., 0, 255.5], [0, 410., 255.5], [0, 0, 1.]])
REFERENCE = np.eye(4)


def source_texture():
    rng = np.random.default_rng(20261101)
    texture = rng.integers(10, 246, size=(512, 512), dtype=np.uint8)
    return cv2.GaussianBlur(texture, (5, 5), .9)


def nonplanar_render(image, query_pose):
    """Ray-intersect analytic nonplanar terrain and render the common texture."""
    y, x = np.mgrid[:512, :512]
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(K).T

    def world(pose):
        d = rays @ pose[:3, :3].T
        c = pose[:3, 3]
        distance = (10. - c[2]) / d[..., 2]
        for _ in range(16):
            u, v = (c + distance[..., None] * d)[..., :2].transpose(2, 0, 1)
            z = 10. + .25 * np.sin(.9 * u + .4 * v) + .18 * np.cos(.3 * u - .8 * v)
            distance = (z - c[2]) / d[..., 2]
        return c + distance[..., None] * d

    a, b = world(REFERENCE), world(query_pose)
    h = b @ K.T
    ref_pixels = (h[..., :2] / h[..., 2, None]).astype(np.float32)
    query = cv2.remap(image, ref_pixels[..., 0], ref_pixels[..., 1], cv2.INTER_LINEAR)
    return a[..., 2], query


def trial_fixture():
    render = source_texture()
    rotation, _ = cv2.Rodrigues(np.array([.10, -.04, .55]))
    query_pose = np.eye(4)
    query_pose[:3, :3], query_pose[:3, 3] = rotation, [-.36, .45, -1.2]
    depth, query = nonplanar_render(render, query_pose)
    seed_pose = query_pose.copy()
    seed_pose[:3, 3] += [.026, -.032, 0.]
    geometry = SimpleNamespace(depth=depth, intrinsics=K, cam2world=REFERENCE)
    return render, query, query_pose, seed_pose, geometry


def check_terrain_projection():
    render, query, query_pose, seed_pose, geometry = trial_fixture()
    trial_errors, warp_errors, calls = [], [], []
    valid_points = [(x, y) for x in (140, 188, 236, 284, 332) for y in (140, 188, 236, 284, 332)]
    for point in valid_points:
        world, _ = map_points_z([point], geometry.depth, K, REFERENCE)
        actual, _, _ = project_z(world, K, query_pose)
        predicted, _, _ = project_z(world, K, seed_pose)
        actual, predicted = actual[0], predicted[0]
        plain, n_plain, _ = measure_window(render, query, point, predicted)
        warped, n_warped, _ = measure_surface_window(render, query, point, predicted, geometry, K, seed_pose)
        calls.append((n_plain, n_warped))
        trial_errors.append(np.inf if plain is None else np.linalg.norm(plain - actual))
        warp_errors.append(np.inf if warped is None else np.linalg.norm(warped - actual))
        if warped is not None:
            assert (np.abs(warped - np.rint(predicted)) <= 16).all()
    trial_errors, warp_errors = np.asarray(trial_errors), np.asarray(warp_errors)
    assert (warp_errors <= 1.0).sum() >= 20 and (trial_errors <= 1.0).sum() <= 5
    assert max(max(t) for t in calls) <= 2
    print("Nonplanar rough seed: plain and warped <=1px",
          int((trial_errors <= 1).sum()), int((warp_errors <= 1).sum()), "/", len(trial_errors),
          "warped median", round(float(np.median(warp_errors)), 3))


def check_fallbacks_and_paid_schedule():
    render, query, _, seed_pose, geometry = trial_fixture()
    seed_p = np.array([[110., 170.], [180., 160.], [240., 250.], [320., 170.], [280., 320.], [180., 290.]])
    world, _ = map_points_z(seed_p, geometry.depth, K, REFERENCE)
    seed_q, _, _ = project_z(world, K, seed_pose)
    seed = dict(reference_xy=seed_p, query_xy=seed_q, pose=seed_pose, reason=None)
    seen = []

    def fail(p, q):
        seen.append((p, q))
        return None, {"reason": "synthetic_refit_failure"}

    frozen, a = add_windows(seed, render, query, geometry, K, policy="uniform", budget=32, final_estimator=fail)
    fallback, b = add_surface_windows(seed, render, query, geometry, K, budget=32, final_estimator=fail)
    assert frozen is seed and fallback is seed and b["reason"] == "added_match_pose_failed"
    assert a["eligible_grid_ids"] == b["eligible_grid_ids"]
    assert [v["grid_id"] for v in a["attempted_windows"]] == [v["grid_id"] for v in b["attempted_windows"]]
    assert len(b["attempted_windows"]) == 32 and b["ncc_calls"] <= 64
    assert np.array_equal(seen[-1][0][:len(seed_p)], seed_p)
    assert np.array_equal(seen[-1][1][:len(seed_q)], seed_q)

    # Identity reduces to the same native search, and blank/crop/fold/missing
    # support all abstain within the paid measurement limit.
    identity_geometry = SimpleNamespace(depth=np.full((512, 512), 10.), intrinsics=K, cam2world=REFERENCE)
    p = (260, 260)
    simple = measure_window(render, render, p, p)
    projected = measure_surface_window(render, render, p, p, identity_geometry, K, REFERENCE)
    np.testing.assert_allclose(simple[0], projected[0], rtol=0, atol=1e-10)
    assert simple[1:] == projected[1:] == (2, None)
    blank = np.zeros_like(render)
    assert measure_surface_window(blank, render, p, p, identity_geometry, K, REFERENCE)[1:] == (0, "flat_template_or_ambiguous_forward_peak")
    assert measure_surface_window(render, query, (3, 3), p, geometry, K, seed_pose)[1:] == (0, "incomplete_native_crop")
    broken = identity_geometry.depth.copy()
    broken[260, 260] = np.nan
    broken_geometry = SimpleNamespace(depth=broken, intrinsics=K, cam2world=REFERENCE)
    assert surface_map(p, broken_geometry, K, REFERENCE)[1] == "incomplete_reference_surface"
    broken[260, 260] = 30.
    displaced = REFERENCE.copy()
    displaced[0, 3] = 0.7
    assert surface_map(p, broken_geometry, K, displaced)[1] == "folded_or_degenerate_projected_surface"
    no_pose = dict(seed, pose=None)
    untouched, trace = add_surface_windows(no_pose, None, None, None, None, budget=32, final_estimator=fail)
    assert untouched is no_pose and trace["reason"] == "no_direct_seed_pose" and trace["ncc_calls"] == 0


def check_common_pose():
    from solar_evaluation_v70 import common_pose

    render, query, truth, trial_pose, geometry = trial_fixture()
    p = np.array([[110., 170.], [180., 160.], [240., 250.], [320., 170.], [280., 320.], [180., 290.]])
    world, _ = map_points_z(p, geometry.depth, K, REFERENCE)
    q, _, _ = project_z(world, K, trial_pose)
    inference = SimpleNamespace(geometry=geometry, query_intrinsics=K)

    def estimate(p, q):
        return common_pose(p, q, inference)

    seed_pose, seed_info = estimate(p, q)
    seed = dict(reference_xy=p, query_xy=q, pose=seed_pose, reason=None, common_estimator=seed_info)
    control, control_trace = add_windows(seed, render, query, geometry, K, policy="uniform",
                                         budget=32, final_estimator=estimate)
    warped, warped_trace = add_surface_windows(seed, render, query, geometry, K, budget=32, final_estimator=estimate)
    for result in (seed, control, warped):
        replayed, metadata = estimate(result["reference_xy"], result["query_xy"])
        np.testing.assert_array_equal(replayed, result["pose"])
        assert metadata == result["common_estimator"]
    print("Synthetic common-estimator translation error m (seed / plain / warped):",
          [round(float(np.linalg.norm(r["pose"][:3, 3] - truth[:3, 3])), 4) for r in (seed, control, warped)],
          "added", [0, control_trace["accepted_additions"], warped_trace["accepted_additions"]])


if __name__ == "__main__":
    cv2.setNumThreads(1)
    check_terrain_projection()
    check_fallbacks_and_paid_schedule()
    check_common_pose()
    print("PASS: equal paid windows, identity, warp recovery, refit fallback, crop/fold/missing support")
