import numpy as np
import pytest
from geometry import transform, endpoint_error, evaluation_points
from synthetic import texture, make_pair

I = np.array([[1., 0, 0], [0, 1, 0]])


def test_known_error_and_direction():
    shifted = I.copy(); shifted[:, 2] = [3, 4]
    points = np.array([[0, 0], [11, 20]])
    np.testing.assert_allclose(transform(points, shifted), [[3, 4], [14, 24]])
    assert endpoint_error(I, shifted, points) == 5
    assert endpoint_error(shifted, shifted, points) == 0


def test_common_support():
    shifted = I.copy(); shifted[:, 2] = [100, -50]
    points = evaluation_points((256, 256), shifted)
    target = transform(points, shifted)
    assert len(points) > 0
    assert ((target >= 16) & (target < 240)).all()


def test_failures_are_not_zero():
    assert endpoint_error(None, I, [[1, 2]]) is None
    assert endpoint_error(I, I, []) is None
    assert endpoint_error(np.full((2, 3), np.nan), I, [[1, 2]]) is None


def test_repeatability_and_identity():
    image = texture(42)
    np.testing.assert_array_equal(image, texture(42))
    case = dict(angle=0, dx=0, dy=0, gain=1, gamma=1, noise=0)
    query, truth = make_pair(image, case, 1)
    np.testing.assert_array_equal(image, query)
    np.testing.assert_allclose(truth, I)


def test_bad_input():
    with pytest.raises(ValueError):
        texture(42, 32)
    with pytest.raises(ValueError):
        transform([[1, 2]], np.eye(3))
