import numpy as np

from masked_patch import original, search_patch


def scene(shift=9, noisy=False):
    rng = np.random.default_rng(20261105)
    signal = rng.uniform(.2, .8, (32, 32)).repeat(8, 0).repeat(8, 1)
    a = signal.copy()
    b = np.zeros_like(a)
    b[:, shift:] = signal[:, :-shift]
    if noisy:
        a += .75 * rng.standard_normal(a.shape)
        b += .75 * rng.standard_normal(b.shape)
        a -= min(a.min(), b.min())
        b -= min(a.min(), b.min())
    mask = np.ones(a.shape, dtype=bool)
    return a, b, mask


def test_comparison_halo_recovers_near_the_returned_radius():
    a, b, valid = scene(15)
    p = np.array([128., 128.])
    old = original.search_patch(a, b, p, p, radius=16)
    measured = search_patch(a, b, p, p, mode="halo")
    assert old["query_xy"] is None
    assert measured["reason"] is None
    assert np.array_equal(measured["query_xy"], p + [15., 0.])
    assert measured["evaluated_positions"] <= measured["max_total_positions"]


def test_masked_pooling_recovers_noise_averaged_structure():
    a, b, valid = scene(9, noisy=True)
    p = np.array([128., 128.])
    old = original.search_patch(a, b, p, p, radius=16)
    measured = search_patch(a, b, p, p)
    assert old["query_xy"] is None
    assert measured["query_xy"] is not None
    assert np.linalg.norm(np.asarray(measured["query_xy"]) - p - [9., 0.]) <= 1.
    assert measured["round_trip_error_px"] <= 1.


def test_partial_support_uses_matching_pixels_and_recovers_shift():
    a, b, valid = scene(9)
    p = np.array([128., 128.])
    valid[99:160:11, 95:163] = False
    measured = search_patch(a, b, p, p, reference_valid=valid, query_valid=valid)
    assert measured["query_xy"] is not None
    assert np.linalg.norm(np.asarray(measured["query_xy"]) - p - [9., 0.]) <= 1.


def test_blank_and_over_radius_queries_are_rejected():
    image = np.ones((256, 256))
    assert search_patch(image, image, [128, 128], [128, 128])["query_xy"] is None
    a, b, _ = scene(24)
    assert search_patch(a, b, [128, 128], [128, 128])["query_xy"] is None
