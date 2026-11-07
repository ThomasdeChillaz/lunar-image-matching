import cv2
import numpy as np

from candidate_selection import Candidate
from terrain_selection import FAMILIES, infer_pools
from synthetic import texture


def test_initial_weights_cannot_influence_trained_or_modern_pools():
    image = texture(73, 128)
    truth = np.array([[1., 0, 7], [0, 1, -5]])
    wrong = np.array([[1., 0, -12], [0, 1, 11]])
    query = cv2.warpAffine(255-image, truth, (128, 128))
    mask = cv2.warpAffine(np.ones_like(image, np.float32), truth, (128, 128)) == 1
    def values(initial, trained):
        overrides = dict(matchanything_4x_mps=wrong, contrastive_initial=initial, contrastive_selected=trained)
        candidates = [Candidate(name, family, overrides.get(name)) for name, family in FAMILIES.items()]
        return infer_pools(image, query, np.ones_like(mask), mask, candidates)
    before, after = values(truth, wrong), values(wrong, wrong)
    for pool in ['modern_global', 'modern_terrain_global', 'expanded_global', 'expanded_terrain_global', 'terrain_global']:
        assert before[pool]['candidate'].name == after[pool]['candidate'].name
        assert before[pool]['quality'] == after[pool]['quality']
        assert 'contrastive_initial' not in before[pool]['candidate_quality']
    improved = values(wrong, truth)
    assert improved['modern_global']['candidate'].name == 'matchanything_4x_mps'
    assert improved['modern_terrain_global']['candidate'].name == 'contrastive_selected'
