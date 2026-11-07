import numpy as np
from roma_matcher import normalized_to_native_centers


def test_dense_normalized_grid_uses_integer_native_pixel_centers():
    shape = (128, 192)
    native = np.array([[0., 0.], [191., 127.], [62.25, 31.75]])
    normalized = (native+.5)*2/np.array([192, 128])-1
    actual = normalized_to_native_centers(normalized, shape)
    assert np.allclose(actual, native, atol=1e-12)
    assert np.array_equal(normalized_to_native_centers([[-1., -1.], [1., 1.]], shape), [[-.5, -.5], [191.5, 127.5]])
