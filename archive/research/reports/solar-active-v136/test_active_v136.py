import inspect
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from matcher import camera_cloud, joint, public_result, run
from policy import (HEIGHTS, POLICIES, Pool, acquire, cell_schedule, observation_predictions,
                    rank_scores, separation)
from smoke_v126 import generated


def fixture_pool():
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    xy = np.array([[40., 40.], [63., 63.], [88., 88.]])
    rays = np.c_[xy, np.ones(len(xy))]@np.linalg.inv(k).T
    gradient = np.zeros((3, 9, 2)); gradient[0, :, 0] = 1.; gradient[1, :, 1] = 1.
    gradient[2, :, :] = [.05, .05]
    pool = Pool(xy, 1000*rays, xy.copy(), gradient, np.ones((3, 9), bool), np.ones(3, bool))
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, 0, 3] = 20.; poses[2, 1, 3] = 20.
    return pool, k, poses


def test_sequential_measurements_change_the_next_acquisition():
    pool, k, poses = fixture_pool()
    events = []
    def propose(state):
        events.append(("propose", len(state)))
        # The first observation eliminates horizontal-motion ambiguity. The next
        # useful measurement must distinguish the remaining vertical motion.
        return poses, np.array([.5, .5, 0.]) if not state else np.array([.5, 0., .5])
    def measure(index):
        events.append(("measure", index)); return dict(acquired=index)
    def update(state, index, observation):
        assert observation == dict(acquired=index)
        events.append(("update", index)); return state+[index]
    state, trace = acquire(pool, k, [0., 0., 1.], "active", 2, [], propose, measure, update)
    assert state == [0, 1]
    assert events == [("propose", 0), ("measure", 0), ("update", 0),
                      ("propose", 1), ("measure", 1), ("update", 1)]
    assert trace["attempted_extra_patches"] == 2
    assert trace["additional_query_fields_before_selection"] == 0


def test_unknown_height_coverage_cannot_become_selection_evidence():
    pool, k, poses = fixture_pool()
    photo, _, known = observation_predictions(pool, k, poses, [0., 0., 1.])
    nominal, count = separation(photo, known, np.array([.5, .5, 0.]))
    assert nominal[0] > 0 and count[0] == 1
    cropped = known.copy(); cropped[0, -1, 0] = False
    robust, count = separation(photo, cropped, np.array([.5, .5, 0.]))
    assert robust[0] == count[0] == 0
    assert len(HEIGHTS) == 7 and set(HEIGHTS) == {0., -5., 5., -10., 10., -20., 20.}


def test_unknown_pairs_are_not_renormalized_into_full_confidence():
    values = np.zeros((3, 7, 1, 2)); values[1] = 8.; values[2] = -8.
    known = np.ones((3, 7, 1), bool)
    full, _ = separation(values, known, np.full(3, 1/3))
    known[2] = False
    partial, evidence = separation(values, known, np.full(3, 1/3))
    assert evidence[0] == 1
    np.testing.assert_allclose(partial, full/3)


def test_uniform_and_active_share_spatial_slots_and_charge_failed_searches():
    pool, k, poses = fixture_pool()
    selected = {}
    for mode in POLICIES:
        calls = []
        def observe(index):
            calls.append(index)
            return None  # Every failed measurement still consumes its slot.
        _, trace = acquire(pool, k, [0., 0., 1.], mode, 3, None,
            lambda state: (poses, np.full(3, 1/3)), observe, lambda state, index, result: state)
        assert len(calls) == len(set(calls)) == trace["attempted_extra_patches"] == 3
        selected[mode] = trace["cell_schedule"]
    assert len({tuple(schedule) for schedule in selected.values()}) == 1


def test_flat_reference_prediction_has_same_tiebreak_as_uniform():
    pool, k, poses = fixture_pool(); pool = replace(pool, preview_gradients=np.zeros((3, 9, 2)))
    choices = []
    for mode in ("uniform", "active"):
        _, trace = acquire(pool, k, [0., 0., 1.], mode, 3, None,
            lambda state: (poses, np.full(3, 1/3)), lambda index: None, lambda state, index, result: state)
        choices.append(trace["selected_indices"])
    assert choices[0] == choices[1]


def test_invalid_policy_geometry_and_budgets_fail_before_patch_access():
    pool, k, poses = fixture_pool()
    for limit in (-1, 33, 1.5):
        with pytest.raises(ValueError): cell_schedule(pool, limit)
    with pytest.raises(ValueError): rank_scores(pool, k, poses, [0., 0., 1.], np.full(3, 1/3), "unknown")
    with pytest.raises(ValueError): observation_predictions(pool, k, poses, [0., 0., 0.])
    with pytest.raises(ValueError): replace(pool, reference_xy=np.repeat(pool.reference_xy[:1], 3, axis=0)).validate()
    with pytest.raises(ValueError): separation(np.ones((3, 7, 1, 2)), np.ones((3, 7, 1), bool), [-1., 1., 1.])


def test_cloud_is_rigid_and_does_not_use_query_truth():
    pool, k, _ = fixture_pool()
    world = np.vstack((pool.local_world, pool.local_world+[20., 10., 200.]))
    fields = dict(records=[dict(local_world=point) for point in world])
    cloud = camera_cloud(np.eye(4), fields, np.eye(4), k)
    assert cloud.shape == (7, 4, 4)
    np.testing.assert_allclose(cloud[0], np.eye(4), atol=1e-10)
    for pose in cloud:
        np.testing.assert_allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-10)
        assert np.linalg.det(pose[:3, :3]) > .99999
    assert not any(name in inspect.signature(run).parameters for name in ("query_depth", "true_query_pose", "true_sun"))


def test_zero_extra_budget_preserves_frozen_joint_measurements():
    cv2.setNumThreads(1)
    args, mask, _ = generated(False)
    ref, depth, k, pose, query, qk, _ = args
    kwargs = dict(reference_valid=np.ones_like(mask))
    original = joint.run(ref, query, depth, k, pose, qk, pose, **kwargs)["methods"]["joint"]
    result = run(ref, query, depth, k, pose, qk, pose, height_direction_reference=[0., 0., -1.], budget=0, **kwargs)
    for name in ("joint_no_extra", *POLICIES):
        for key in ("reference_xy", "query_xy", "pose", "estimator"):
            assert result["methods"][name][key] == original[key]
    assert public_result(result)["pose"] is None
    assert "raw_research" not in public_result(result)
    json.dumps(public_result(result, True), allow_nan=False)


def test_native_generated_images_have_lazy_extra_searches_and_real_endpoints():
    cv2.setNumThreads(1)
    args, mask, truth = generated(True)
    ref, depth, k, pose, query, qk, _ = args
    result = run(ref, query, depth, k, pose, qk, pose, reference_valid=np.ones_like(mask),
                 height_direction_reference=[0., 0., -1.], policies=("uniform", "active"), budget=2)
    for name in ("uniform", "active"):
        row = result["methods"][name]
        assert row["acquisition"]["attempted_extra_patches"] == 2
        assert row["pose"] is not None
        assert np.linalg.norm(np.array(row["pose"])[:3, 3]-truth[:3, 3]) < 2.
        p, q = np.array(row["reference_xy"]), np.array(row["query_xy"])
        expected = p.copy(); expected[:, 0] -= k[0, 0]*truth[0, 3]/depth[p[:, 1].astype(int), p[:, 0].astype(int)]
        assert np.median(np.linalg.norm(q-expected, axis=1)) < .2
        assert row["conditional_correspondences"] and not row["independent_landmark_validation"]
    json.dumps(public_result(result, True), allow_nan=False)


def test_missing_initializer_never_fabricates_a_pose():
    args, mask, _ = generated(False)
    ref, depth, k, pose, query, qk, _ = args
    result = run(ref, query, depth, k, pose, qk, None, reference_valid=np.ones_like(mask),
                 height_direction_reference=[0., 0., -1.])
    assert all(row["pose"] is None for row in result["methods"].values())
