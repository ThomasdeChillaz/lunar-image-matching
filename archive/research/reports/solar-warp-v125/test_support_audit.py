import numpy as np

from support_audit import attribution_rows, support_order


def test_support_order_rewards_weakest_cell_before_total_and_breaks_ties_by_prior():
    indices = np.array([1, 3, 7, 9, 12])
    counts = np.array([[3, 3], [2, 16], [3, 4], [3, 4], [4, 4]], dtype=np.int16)
    assert support_order(indices, counts) == [12, 7, 9, 1, 3]
    assert support_order(np.array([], dtype=int), np.empty((0, 2), dtype=np.int16)) == []


def test_scoring_keeps_support_control_and_arm_ranking_separate():
    rows = [{"tag": "exposure-0", "reference_seed": 0, "query_seed": 1, "cell_pixels": 32,
             "orders": {"geometry_3d": [9, 2], "planar": [2, 9], "gradient": [2]}}]
    near = np.zeros(10, dtype=bool)
    near[9] = True
    scored, = attribution_rows({"rows": rows}, [([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)],
                               {"exposure-0": ("test-case", near, 1000.)})
    assert scored["geometry_3d_first_hit"] == 1
    assert scored["planar_first_hit"] == 2
    assert scored["support_only_first_hit"] is None
    assert scored["reference_order_first_hit"] is None
    assert scored["all_photometric_arms_scoreable_candidates"] == 1
