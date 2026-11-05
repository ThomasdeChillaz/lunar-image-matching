import json

import numpy as np
import pytest

from parallax_policy import HEIGHT_VARIANTS_M, select


K = np.array([[500., 0., 255.5], [0., 500., 255.5], [0., 0., 1.]])


def world(anchors, depth):
    depth = np.broadcast_to(depth, (len(anchors),))
    ray = (np.asarray(anchors) - K[:2, 2]) / 500.
    return np.column_stack((ray * depth[:, None], depth))


def cameras(*offsets):
    poses = np.repeat(np.eye(4)[None], len(offsets), axis=0)
    poses[:, 0, 3] = offsets
    return poses


def test_flat_elevated_anchor_retains_centre_parallax_after_height_offsets():
    anchors = [[255.5, 255.5], [260., 260.]]
    result = select(world(anchors, [800., 1000.]), world(anchors, 1000.), anchors,
                    K, cameras(0., 100.), [True, True], budget=1)
    assert result["scores"][0] == pytest.approx(50000 * (1/820 - 1/1020))
    assert result["nominal_scores"] == pytest.approx([12.5, 0.])
    assert result["known_pose_pairs"] == [1, 1]
    assert result["height_variants_m"] == list(HEIGHT_VARIANTS_M) == [0., -5., 5., -10., 10., -20., 20.]
    assert result["active"] == result["uniform"] == [0]


def test_overlapping_height_envelopes_remove_nominal_discrimination():
    anchors = [[430.5, 255.5]]
    result = select(world(anchors, 800.), world(anchors, 1000.), anchors, K,
                    cameras(100., 101.), [True], budget=1)
    assert result["nominal_scores"][0] == pytest.approx(.125)
    assert result["scores"] == [0.]
    assert result["active"] == result["uniform"] == [0]
    assert result["reason"] == "no_separable_parallax_uniform_order"


def test_active_and_uniform_use_same_slots_and_spatial_rounds():
    axis = [72, 145, 218, 293, 366, 439]
    anchors = np.array([(x, y) for y in axis for x in axis])
    # More relief at later slots so active choices differ within a spatial cell.
    points = world(anchors, 1000. - np.arange(36)*4)
    plane = world(anchors, 1000.)
    eligible = np.arange(36) % 5 != 0
    result = select(points, plane, anchors, K, cameras(0., 70.), eligible)
    repeated = select(points, plane, anchors, K, cameras(0., 70.), eligible)
    assert result == repeated
    assert result["scheduled_per_arm"] == len(result["active"]) == len(result["uniform"]) == 12
    assert result["active"] != result["uniform"]
    for arm in ("active", "uniform"):
        indices = result[arm]
        assert len(set(indices)) == len(indices)
        assert eligible[indices].all()
        cells = anchors[indices] // 128
        assert len(np.unique(cells, axis=0)) == len(cells)
        cell_ids = (4 * cells[:, 1] + cells[:, 0]).tolist()
        assert cell_ids == result["cell_schedule"]
        assert np.bincount(cell_ids, minlength=16).tolist() == result["cell_quotas"]
    assert all(result["scores"][i] is None for i in np.flatnonzero(~eligible))
    json.dumps(result, allow_nan=False)


def test_unknown_and_duplicate_hypotheses_keep_shared_eligibility():
    anchors = [[80, 80], [120, 100], [240, 240]]
    points = world(anchors, 800.)
    points[0] = np.nan
    plane = world(anchors, 1000.)
    result = select(points, plane, anchors, K, cameras(30., 30.), [True, False, True])
    assert result["scores"] == [None, None, 0.]
    assert result["known_pose_pairs"] == [0, 0, 1]
    assert result["active"] == result["uniform"] == [0, 2]
    assert result["reason"] == "no_separable_parallax_uniform_order"
    missing = select(points, plane, anchors, K, np.empty((0, 4, 4)), [False, False, False])
    assert missing["active"] == missing["uniform"] == []
    assert missing["reason"] == "no_shared_eligible_anchors"


def test_out_of_frame_height_envelope_is_unknown_not_apparent_evidence():
    anchors = [[5, 5]]
    result = select(world(anchors, 800.), world(anchors, 1000.), anchors,
                    K, cameras(0., 100.), [True], budget=1)
    assert result["scores"] == result["nominal_scores"] == [None]
    assert result["known_pose_pairs"] == [0]
    assert result["active"] == result["uniform"] == [0]
    with pytest.raises(ValueError, match="Boolean eligibility"):
        select(world(anchors, 800.), world(anchors, 1000.), anchors, K, cameras(0., 100.), [1])


def test_full_grid_spreads_to_all_corners_before_preferring_any_region():
    axis = [63.5, 191.5, 319.5, 447.5]
    anchors = [(x, y) for y in axis for x in axis]
    plane = world(anchors, 1000.)
    result = select(plane, plane, anchors, K, cameras(0., 70.), [True]*16)
    assert result["cell_schedule"][:4] == [0, 15, 3, 12]
    assert len(set(result["cell_schedule"])) == 12
    assert result["active"] == result["uniform"] == result["cell_schedule"]
    assert result["reason"] == "no_separable_parallax_uniform_order"


def test_active_picks_distinct_eligible_anchor_within_same_cell_quota():
    anchors = [[90., 80.], [63.5, 63.5], [110., 96.], [447.5, 447.5],
               [410., 420.], [447.5, 63.5], [63.5, 447.5]]
    plane = world(anchors, 1000.)
    points = world(anchors, [800., 1000., 750., 1000., 1000., 1000., 1000.])
    eligible = [True, True, False, True, True, True, True]
    result = select(points, plane, anchors, K, cameras(0., 70.), eligible, budget=6)
    assert result["cell_schedule"] == [0, 15, 3, 12, 0, 15]
    assert result["active"] == [0, 3, 5, 6, 1, 4]
    assert result["uniform"] == [1, 3, 5, 6, 0, 4]
    flat = select(plane, plane, anchors, K, cameras(0., 70.), eligible, budget=6)
    assert flat["cell_quotas"] == result["cell_quotas"]
    assert flat["active"] == flat["uniform"] == result["uniform"]
