import numpy as np
import pytest
from geometry import endpoint_error, evaluation_points
from synthetic import texture, make_pair
from matchers import ncc, sift


@pytest.mark.parametrize('matcher', [ncc, sift])
@pytest.mark.parametrize('dx,dy', [(12, -8), (-9, 11), (0, 0)])
def test_translation_recovery(matcher, dx, dy):
    image = texture(7)
    query, truth = make_pair(image, dict(angle=0, dx=dx, dy=dy, gain=.7, gamma=1.2, noise=.005), 8)
    result = matcher(image, query)
    error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
    assert error is not None and error < 1


def test_sift_rotation():
    image = texture(21)
    query, truth = make_pair(image, dict(angle=12, dx=8, dy=-6, gain=1, gamma=1, noise=0), 8)
    error = endpoint_error(sift(image, query).matrix, truth, evaluation_points(image.shape, truth))
    assert error is not None and error < 1


@pytest.mark.parametrize('matcher', [ncc, sift])
def test_blank_is_explicit_failure(matcher):
    blank = np.zeros((256, 256), np.uint8)
    result = matcher(blank, blank)
    assert result.matrix is None
    assert result.reason
