import cv2
import numpy as np
import pytest

from candidate_selection import Candidate
from modern_selection import FAMILIES, infer_pools, pool_scores
from synthetic import texture


def predictions(overrides):
    return [Candidate(name, family, overrides.get(name)) for name, family in FAMILIES.items()]


def test_legacy_control_cannot_use_new_model_predictions():
    image = texture(70, 128)
    matrix = np.array([[1., 0, 7], [0, 1, -5]])
    query = cv2.warpAffine(255-image, matrix, (128, 128))
    mask = cv2.warpAffine(np.ones_like(image, np.float32), matrix, (128, 128)) == 1
    wrong = np.array([[1., 0, -12], [0, 1, 11]])
    candidates = predictions(dict(ncc=wrong, matchanything_4x_mps=matrix))
    values = infer_pools(image, query, np.ones_like(mask), mask, candidates)
    assert values['legacy_global']['candidate'].name == 'ncc'
    assert values['modern_global']['candidate'].name == 'matchanything_4x_mps'
    assert values['expanded_global']['candidate'].name == 'matchanything_4x_mps'
    # Legacy consensus must be unchanged when an excluded model changes.
    changed = predictions(dict(ncc=wrong, matchanything_4x_mps=wrong))
    other = infer_pools(image, query, np.ones_like(mask), mask, changed)
    assert values['legacy_consensus']['quality'] == other['legacy_consensus']['quality']


def test_resolution_and_pretraining_variants_do_not_multiply_architecture_votes():
    matrix = np.array([[1., 0, 0], [0, 1, 0]])
    candidates = predictions({name: matrix for name in FAMILIES})
    quality = dict(global_correlation=1., patch_median_correlation=1., orientation_correlation=1.)
    scores = pool_scores(candidates, {name: quality for name in FAMILIES}, (128, 128))
    families = scores['ncc']['agreeing_other_families']
    assert families.count('eloftr') == 1 and len(families) == 6
    assert scores['ncc']['agreement_fraction'] == 1


def test_missing_predictor_is_an_error_and_explicit_failures_remain_failures():
    image = texture(70, 128)
    valid = np.ones_like(image, bool)
    candidates = predictions({})
    with pytest.raises(ValueError, match='Every declared predictor'):
        infer_pools(image, image, valid, valid, candidates[:-1])
    values = infer_pools(image, image, valid, valid, candidates)
    assert len(values) == 9 and all(v['candidate'] is None for v in values.values())
