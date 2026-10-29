import os
import numpy as np
import pytest
from loftr_matcher import LoFTRMatcher, fetch_weights, valid_correspondences


def test_refined_matches_must_remain_finite_in_bounds_and_in_both_valid_masks():
    ma = np.ones((128, 128), bool)
    mb = ma.copy()
    mb[20, 20] = False
    a = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50], [60, 60]], float)
    b = np.array([[11, 12], [20, 20], [-.1, 30], [128, 40], [np.nan, 50], [60, 127.1]], float)
    aa, bb = valid_correspondences(a, b, ma, mb)
    assert np.array_equal(aa, [[10, 10]])
    assert np.array_equal(bb, [[11, 12]])


def test_corrupt_cached_loftr_weights_are_rejected_without_download(tmp_path):
    path = tmp_path/'data/models/loftr_outdoor.ckpt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        fetch_weights(tmp_path)


@pytest.mark.skipif(os.environ.get('RUN_LEARNED') != '1', reason='Pinned pretrained LoFTR inference requires downloaded weights')
def test_actual_loftr_translation_rotation_and_masked_failure():
    from geometry import endpoint_error, evaluation_points
    from synthetic import make_pair, texture
    image = texture(42, 128)
    query, truth = make_pair(image, dict(angle=12, dx=6, dy=-4, gain=.7, gamma=1.2, noise=.005), 43)
    matcher = LoFTRMatcher()
    result = matcher(image, query)
    error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
    assert error is not None and error < 1
    assert matcher(image, image, np.zeros_like(image, bool), None).reason == 'no_valid_support'
    blank = np.zeros_like(image)
    assert matcher(blank, blank).reason == 'low_texture'
