import json

import numpy as np

from conditioning_diagnostic import sensitivity


def test_local_noise_sensitivity_has_consistent_physical_units():
    y, x = np.mgrid[-300:301:150, -300:301:150]
    world = np.c_[x.ravel(), y.ravel(), np.full(x.size, 1000.)]
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    projected = world@k.T
    pixels = projected[:, :2]/projected[:, 2, None]
    base = sensitivity(world, pixels, k, np.eye(4), 1100.)
    changed = sensitivity(world*3., pixels, k, np.eye(4), 3300.)
    assert base["rank"] == 6 and base["fitted_inliers"] == len(world)
    np.testing.assert_allclose(base["normalized_pose_jacobian_singular_values"],
                               changed["normalized_pose_jacobian_singular_values"], rtol=1e-12)
    assert np.isclose(changed["linearized_position_rms_m_for_independent_one_pixel_noise"],
                      3*base["linearized_position_rms_m_for_independent_one_pixel_noise"])


def test_degenerate_configuration_has_no_finite_uncertainty_claim():
    x = np.linspace(-300., 300., 6)
    world = np.c_[x, np.zeros(6), np.full(6, 1000.)]
    pixels = world[:, :2]/world[:, 2, None]
    result = sensitivity(world, pixels, np.eye(3), np.eye(4), 1000.)
    assert result["rank"] < 6
    assert result["condition_number"] is None
    assert result["linearized_position_rms_m_for_independent_one_pixel_noise"] is None
    json.dumps(result, allow_nan=False)
