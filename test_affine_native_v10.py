import cv2
import numpy as np
import pytest
pytest.importorskip('torch')
from affine_native_v10 import estimate_affine, rectify, transform_points


def test_affine_native_centers_direction_shear_and_invalid_support():
    y, x = np.mgrid[:160, :160]
    image = (x+y).astype(np.uint8)
    a = np.array([[1., .1, 8.], [0., 1., -4.]])
    p = np.array([[40., 40.], [80., 40.], [120., 40.], [40., 80.], [80., 80.], [120., 80.], [40., 120.], [80., 120.]])
    q = transform_points(p, a)
    np.testing.assert_allclose(estimate_affine(p, q), a, atol=1e-5)
    np.testing.assert_allclose(transform_points(q, cv2.invertAffineTransform(a)), p, atol=1e-8)
    warped, support = rectify(image, a)
    assert support[80, 80] and not support[10, 80]
    assert abs(int(warped[80, 80])-(88+8+76)) <= 1


def test_degenerate_and_nonphysical_affine_do_not_create_proposals():
    p = np.column_stack([np.arange(8), np.zeros(8)])
    assert estimate_affine(p, p) is None
    p = np.random.default_rng(3).uniform(0, 100, (20, 2))
    assert estimate_affine(p, p*3) is None
    assert estimate_affine(p, p*[-1, 1]) is None
    with pytest.raises(ValueError):
        estimate_affine(p, np.full_like(p, np.nan))
