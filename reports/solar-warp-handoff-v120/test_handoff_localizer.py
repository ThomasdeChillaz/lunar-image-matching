import numpy as np
import pytest

from handoff_localizer import ANCHORS, NATIVE, Templates, localize, solve_observations
from lunar_geometry_v5 import project_z, world_points_z


def setup():
    k = np.array([[443., 0., 255.5], [0., 443., 255.5], [0., 0., 1.]])
    source = np.eye(4)
    source[:, 3] = [1., -2., 3., 1.]
    world = world_points_z(ANCHORS, np.linspace(10., 12., len(ANCHORS)), k, source)
    scene = np.zeros((NATIVE, NATIVE))
    mask = np.ones(scene.shape, dtype=bool)
    return k, source, world, scene, mask


def test_raw_pose_requires_independent_search_and_preserves_matched_arm_denominators():
    k, seed, world, scene, mask = setup()
    actual_pose = seed.copy()
    actual_pose[0, 3] += 1.
    measured, _, _ = project_z(world, k, actual_pose)

    def warp(pose):
        centres, _, _ = project_z(world, k, pose)
        return Templates(scene, scene + 1, mask, ANCHORS.copy(), world.copy(), centres,
                         np.ones(len(ANCHORS), dtype=bool))

    def measured_search(image, observed, centre, initial, *, radius, reference_valid, query_valid):
        assert reference_valid is mask
        np.testing.assert_array_equal(centre, initial)
        i = int(np.argmin(np.linalg.norm(ANCHORS-centre, axis=1)))
        if radius == 16 or image[0, 0] == 1:
            return dict(query_xy=None, reason="no_measured_patch")
        return dict(query_xy=measured[i].tolist(), reason=None)

    out = localize(scene, k, seed[None], warp, measured_search, query_valid=mask)
    assert out["arms"]["wide_3d"]["selected_seed"] == 0
    pose = out["arms"]["wide_3d"]["raw_pose"]
    np.testing.assert_allclose(pose, actual_pose, atol=1e-6)
    for name in ("narrow_3d", "wide_planar"):
        rows = out["arms"][name]["candidate_rows"]
        assert rows[0]["measured_patch_count"] == 0 and len(rows[0]["failures"]) == 16
        assert out["arms"][name]["raw_pose"] is None
    assert out["raw_only"] and out["maximum_anchor_searches_per_arm"] == 128


def test_duplicate_and_clustered_observations_never_reach_estimator():
    k, _, world, _, _ = setup()
    observations = [dict(query_xy=[120., 120.], reference_xy=p.tolist(), reference_world=w.tolist())
                    for p, w in zip(ANCHORS, world)]

    def unexpected(*_):
        pytest.fail("The solver received duplicate observations")

    out = solve_observations(observations, k, solve=unexpected)
    assert out["pose"] is None and len(out["measurements"]) == 1


def test_invalid_seed_and_no_measured_pairs_are_rejected():
    k, seed, _, scene, _ = setup()
    bad = seed.copy()
    bad[0, 0] = 0.
    with pytest.raises(ValueError, match="Malformed camera seed"):
        localize(scene, k, bad[None], None, None)
    empty = localize(scene, k, np.empty((0, 4, 4)), None, None)
    assert all(row["raw_pose"] is None and row["reason"] == "empty_seed_bank"
               for row in empty["arms"].values())
