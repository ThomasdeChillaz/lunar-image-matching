import numpy as np

from measure_v126 import BUDGET, NATIVE, RADIUS, run, template_available


def test_fractional_template_eligibility_rejects_edges_and_lost_support():
    mask = np.ones((NATIVE, NATIVE), dtype=bool)
    assert template_available(mask, np.array([80.5, 92.25]))
    assert not template_available(mask, np.array([15.5, 92.25]))
    mask[58:125, 48:81] = False
    assert not template_available(mask, np.array([80.5, 92.25]))


def test_empty_bank_retains_both_denominators_and_reports_no_search():
    shape = (NATIVE, NATIVE)
    image = np.zeros(shape)
    intrinsics = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    result = run(image, np.ones(shape), intrinsics, np.eye(4), image, intrinsics, np.empty((0, 4, 4)))
    assert result["raw_only"] is True and result["calibrated_acceptance"] is False
    assert result["patch_budget"] == BUDGET and result["radius_pixels"] == RADIUS
    assert result["selection"] is None
    assert set(result["arms"]) == {"uniform", "parallax"}
    for arm in result["arms"].values():
        assert arm["reason"] == "empty_plane_seed_bank" and arm["raw_pose"] is None
        assert arm["patch_searches"] == arm["inliers"] == 0
