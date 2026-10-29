import os
import numpy as np
import pytest
from eloftr_matcher import EfficientLoFTRMatcher, native_pixel_centers


def test_internal_resize_uses_pixel_centers_without_half_pixel_bias():
    native = np.array([[0., 0.], [31.75, 62.25], [127., 127.]])
    enlarged = (native+.5)*2-.5
    assert np.array_equal(native_pixel_centers(enlarged, 2), native)
    assert np.array_equal(native_pixel_centers(native, 1), native)


@pytest.mark.skipif(os.environ.get('RUN_ELOFTR') != '1', reason='Pinned original source/weights and optional dependency target required')
def test_original_pretrained_eloftr_native_and_2x_forward():
    from synthetic import texture, make_pair
    from geometry import evaluation_points, endpoint_error
    image = texture(42, 128)
    query, truth = make_pair(image, dict(angle=12, dx=6, dy=-4, gain=.7, gamma=1.2, noise=.005), 43)
    for factor in [1, 2]:
        matcher = EfficientLoFTRMatcher(dependency_dir=os.environ.get('ELOFTR_DEPENDENCY_DIR'), factor=factor)
        result = matcher(image, query)
        error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
        assert error is not None and error < 1
        assert matcher(np.zeros_like(image), np.zeros_like(image)).reason == 'low_texture'
        assert matcher(image, image, np.zeros_like(image, bool), None).reason == 'no_valid_support'
