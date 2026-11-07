import cv2
import numpy as np
from candidate_selection import Candidate
from geometry import endpoint_error, evaluation_points
from structural_refinement import AlignmentObjective, local_matrix, refine
from synthetic import texture


def test_multimodal_refinement_improves_a_perturbed_similarity():
    image = texture(52, 128)
    truth = cv2.getRotationMatrix2D((63.5, 63.5), 17.3, .91)
    truth[:, 2] += [6., -4.]
    query = cv2.warpAffine(255-image, truth, (128, 128))
    valid = cv2.warpAffine(np.ones_like(image, np.float32), truth, (128, 128)) == 1
    initial = local_matrix(truth, [1.1, .012, 1.2, -1.], image.shape)
    objective = AlignmentObjective(image, query, np.ones_like(valid), valid)
    result = refine(objective, Candidate('perturbed', 'structural', initial), max_evaluations=160)
    points = evaluation_points(image.shape, truth)
    before = endpoint_error(initial, truth, points)
    after = endpoint_error(result.candidate.matrix, truth, points)
    assert after < .65 and after < before*.5
    assert result.quality['selection_score'] >= objective.quality(initial)['global_correlation']


def test_refinement_never_fabricates_a_transform_without_image_support():
    image = texture(52, 128)
    valid = np.zeros_like(image, bool)
    objective = AlignmentObjective(image, image, valid, valid)
    identity = np.array([[1., 0, 0], [0, 1, 0]])
    assert refine(objective, Candidate('unsupported', 'structural', identity)).candidate is None
    assert refine(objective, None).candidate is None
