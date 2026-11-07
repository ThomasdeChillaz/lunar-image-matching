import numpy as np

from audit_native_window_v130 import local_grid_reachable


def test_native_lattice_capture_includes_tolerance_beyond_window_but_excludes_occlusion():
    seed = np.zeros((5, 2))
    target = np.array([[16., 0.], [17.9, 0.], [18.1, 0.], [17.5, 17.5], [0., 0.]])
    visible = np.array([True, True, True, True, False])
    assert local_grid_reachable(seed, target, visible, 2.).tolist() == [True, True, False, False, False]


def test_unknown_seed_cannot_support_capture_and_fractional_grid_has_finite_precision():
    seed = np.array([[np.nan, 0.], [.2, .3]])
    target = np.array([[0., 0.], [.7, .8]])
    visible = np.ones(2, bool)
    assert local_grid_reachable(seed, target, visible, .6).tolist() == [False, False]
    assert local_grid_reachable(seed, target, visible, .8).tolist() == [False, True]
