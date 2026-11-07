import cv2
import numpy as np
from structural_matchers import multichannel_scores, self_similarity_features, structural_ncc
from synthetic import texture


def test_self_similarity_is_invariant_to_global_brightness_inversion():
    image = texture(42, 128)
    assert np.allclose(self_similarity_features(image), self_similarity_features(255-image), atol=1e-5)


def test_multichannel_correlation_matches_bruteforce_normalization():
    rng = np.random.default_rng(1)
    query = rng.normal(size=(12, 13, 4)).astype(np.float32)
    template = query[3:8, 4:10].copy()
    scores = multichannel_scores(template, query)
    for y, x in [(0, 0), (3, 4), (5, 7)]:
        a = template-template.mean(axis=(0, 1), keepdims=True)
        b = query[y:y+5, x:x+6]
        b = b-b.mean(axis=(0, 1), keepdims=True)
        expected = np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))
        assert np.isclose(scores[y, x], expected, atol=1e-5)


def test_structural_translation_survives_brightness_inversion_and_invalid_borders():
    image = texture(43, 128)
    matrix = np.array([[1., 0, 7], [0, 1, -5]])
    query = cv2.warpAffine(255-image, matrix, (128, 128))
    valid = cv2.warpAffine(np.ones_like(image), matrix, (128, 128)).astype(bool)
    result = structural_ncc(image, query, query_mask=valid)
    assert result.matrix is not None and np.allclose(result.matrix, matrix)
    assert structural_ncc(image, query, np.zeros_like(valid), valid).reason == 'no_valid_support'
