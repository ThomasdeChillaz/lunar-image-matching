import cv2
import numpy as np
from geometry import endpoint_error, evaluation_points
from similarity_search import similarity_search
from synthetic import texture


def test_image_only_similarity_search_recovers_unknown_rotation_scale_and_inversion():
    image = texture(48, 128)
    truth = cv2.getRotationMatrix2D((63.5, 63.5), 17.3, .83)
    truth[:, 2] += [6.2, -4.7]
    query = cv2.warpAffine(255-image, truth, (128, 128))
    valid = cv2.warpAffine(np.ones_like(image, np.float32), truth, (128, 128)) >= 1.-1e-6
    result = similarity_search(image, query, query_mask=valid)
    error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
    assert error is not None and error < 2.
    assert result.candidates_evaluated > 15


def test_similarity_search_has_explicit_blank_failure():
    blank = np.zeros((128, 128), np.uint8)
    assert similarity_search(blank, blank).reason == 'low_texture'
