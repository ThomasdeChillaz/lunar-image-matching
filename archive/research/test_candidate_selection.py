import cv2
import numpy as np
from candidate_selection import Candidate, alignment_features, choose, score_candidates
from synthetic import texture


def test_image_agreement_selects_correct_transform_despite_brightness_inversion():
    image = texture(48, 128)
    truth = np.array([[1., 0, 7], [0, 1, -5]])
    query = cv2.warpAffine(255-image, truth, (128, 128))
    valid = cv2.warpAffine(np.ones_like(image, np.float32), truth, (128, 128)) == 1
    candidates = [Candidate('good', 'structural', truth), Candidate('wrong', 'learned', np.array([[1., 0, -15], [0, 1, 11]]))]
    scores = score_candidates(image, query, np.ones_like(valid), valid, candidates)
    assert scores['good']['global_correlation'] > scores['wrong']['global_correlation']+.5
    for mode in ['global', 'balanced', 'consensus']:
        selected, diagnostics = choose(candidates, scores, mode)
        assert selected.name == 'good'


def test_multiple_variants_of_one_predictor_family_do_not_add_consensus_votes():
    image = texture(48, 128)
    valid = np.ones_like(image, bool)
    matrix = np.array([[1., 0, 0], [0, 1, 0]])
    candidates = [Candidate('one', 'structural', matrix), Candidate('net', 'disk', matrix), Candidate('adapted', 'disk', matrix)]
    scores = score_candidates(image, image, valid, valid, candidates)
    assert scores['one']['agreeing_other_families'] == ['disk']
    assert scores['one']['agreement_fraction'] == 1


def test_failed_or_unsupported_transform_is_not_silently_replaced():
    image = texture(48, 128)
    valid = np.ones_like(image, bool)
    assert alignment_features(image, image, valid, valid, None) is None
    assert alignment_features(image, image, valid, valid, [[0, 0, 0], [0, 0, 0]]) is None
    selected, diagnostic = choose([Candidate('failed', 'net', None)], {}, 'balanced')
    assert selected is None and diagnostic['reason'] == 'no_image_supported_candidate'
