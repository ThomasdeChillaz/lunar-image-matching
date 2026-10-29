"""Image-only local similarity refinement inspired by multimodal registration.

No known motion, evaluation points or correctness labels enter this module.
Structural channels are recomputed after warping the grayscale query: rotating
the directional descriptor channels themselves would not be equivalent.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from candidate_selection import Candidate, MODES, correlation
from structural_matchers import self_similarity_features, orientation_features


@dataclass
class Refined:
    candidate: object
    quality: dict
    evaluations: int


class AlignmentObjective:
    def __init__(self, reference, query, reference_mask, query_mask):
        self.reference, self.query = reference, query
        self.ma, self.mb = np.asarray(reference_mask, bool), np.asarray(query_mask, bool)
        self.shape = reference.shape
        if query.shape != self.shape or self.ma.shape != self.shape or self.mb.shape != self.shape:
            raise ValueError('Image/mask shapes differ')
        self.a = self_similarity_features(np.where(self.ma, reference, 0))
        self.ao = orientation_features(np.where(self.ma, reference, 0))
        h, w = self.shape
        self.roi = np.zeros(self.shape, bool)
        self.roi[h//8:h-h//8, w//8:w-w//8] = True

    def quality(self, matrix):
        if matrix is None:
            return None
        matrix = np.asarray(matrix, float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            return None
        h, w = self.shape
        inverse = cv2.invertAffineTransform(matrix)
        aligned = cv2.warpAffine(np.where(self.mb, self.query, 0), inverse, (w, h), flags=cv2.INTER_LINEAR)
        fraction = cv2.warpAffine(self.mb.astype(np.float32), inverse, (w, h), flags=cv2.INTER_LINEAR)
        valid = self.ma & (fraction >= 1.-1e-6)
        # Orientation has blur radius2 + Sobel radius1 + normalization radius4.
        # Exclude its entire radius7 neighborhood around invalid pixels.
        valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        valid &= self.roi
        support = float(valid.sum()/self.roi.sum())
        if support < .2:
            return None
        b = self_similarity_features(aligned)
        score = correlation(self.a, b, valid)
        if score is None:
            return None
        patches = []
        for yy in np.array_split(np.arange(h//8, h-h//8), 4):
            for xx in np.array_split(np.arange(w//8, w-w//8), 4):
                location = np.ix_(yy, xx)
                mask = valid[location]
                if mask.mean() >= .5:
                    value = correlation(self.a[location], b[location], mask)
                    if value is not None:
                        patches.append(value)
        orientation = correlation(self.ao, orientation_features(aligned), valid)
        return dict(global_correlation=score, patch_median_correlation=float(np.median(patches)) if patches else -1.,
            orientation_correlation=-1. if orientation is None else orientation,
            geometric_support_fraction=support, valid_patch_count=len(patches),
            joint_nondark_fraction=float(np.mean((self.reference[valid] > 3) & (aligned[valid] > 3))))


def quality_score(quality, mode):
    if quality is None:
        return -2.
    if mode not in ('global', 'balanced'):
        raise ValueError('Refinement supports explicit global/balanced image objectives')
    weights = MODES[mode]
    return (weights['global_weight']*quality['global_correlation']+
            weights['patch_weight']*quality['patch_median_correlation']+
            weights['orientation_weight']*quality['orientation_correlation'])


def local_matrix(base, delta, shape):
    angle, logscale, dx, dy = np.asarray(delta, float)
    h, w = shape
    perturb = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, float(np.exp(logscale)))
    perturb[:, 2] += [dx, dy]
    return (np.vstack((np.asarray(base), [0, 0, 1]))@np.vstack((perturb, [0, 0, 1])))[:2]


def refine(objective, candidate, mode='global', max_evaluations=120):
    if mode not in ('global', 'balanced') or max_evaluations < 1:
        raise ValueError('Invalid declared refinement configuration')
    if candidate is None or candidate.matrix is None:
        return Refined(None, dict(reason='no_image_supported_candidate'), 0)
    first = objective.quality(candidate.matrix)
    if first is None:
        return Refined(None, dict(reason='insufficient_structural_support'), 1)
    best_matrix, best_quality = np.asarray(candidate.matrix, float).copy(), first
    best_score = quality_score(first, mode)
    minimum_support = max(.2, first['geometric_support_fraction']*.9)
    evaluated = 1
    def score_delta(delta):
        nonlocal best_matrix, best_quality, best_score, evaluated
        matrix = local_matrix(candidate.matrix, delta, objective.shape)
        quality = objective.quality(matrix)
        evaluated += 1
        if quality is None or quality['geometric_support_fraction'] < minimum_support:
            return -2.
        score = quality_score(quality, mode)
        if score > best_score:
            best_matrix, best_quality, best_score = matrix, quality, score
        return score
    position = np.zeros(4)
    limits = np.array([3., .04, 3., 3.])
    # Deterministic bounded coordinate search, independent of external solvers.
    # Each finer level revisits all four coupled similarity parameters.
    for step in ([1.5, .02, 1., 1.], [.75, .01, .5, .5], [.375, .005, .25, .25], [.1875, .0025, .125, .125]):
        for sweep in range(3):
            changed = False
            for dimension in range(4):
                winner, value = position, best_score
                for direction in (-1, 1):
                    trial = position.copy()
                    trial[dimension] += direction*step[dimension]
                    if evaluated >= max_evaluations or np.any(np.abs(trial) > limits+1e-10):
                        continue
                    score = score_delta(trial)
                    if score > value:
                        winner, value = trial, score
                if not np.array_equal(winner, position):
                    position, changed = winner, True
            if not changed or evaluated >= max_evaluations:
                break
    quality = dict(best_quality, selected_predictor=candidate.name, selection_score=best_score,
        refinement_evaluations=evaluated, minimum_support_fraction=minimum_support)
    return Refined(Candidate(candidate.name, candidate.family, best_matrix), quality, evaluated)
