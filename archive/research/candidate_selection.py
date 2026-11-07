"""Select competing transforms using images and predictions, never truth labels.

Self-similarity is inspired by medical registration. Agreement is counted by
predictor family so several variants of one network cannot create extra votes.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from structural_matchers import self_similarity_features, orientation_features


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    matrix: object


MODES = {
    'global': dict(global_weight=1., patch_weight=0., orientation_weight=0., agreement_weight=0.),
    'balanced': dict(global_weight=.5, patch_weight=.25, orientation_weight=.25, agreement_weight=0.),
    'consensus': dict(global_weight=.5, patch_weight=.25, orientation_weight=.25, agreement_weight=.1),
}


def correlation(a, b, valid):
    if valid.sum() < 32:
        return None
    aa, bb = a[valid].astype(np.float64), b[valid].astype(np.float64)
    aa -= aa.mean(axis=0)
    bb -= bb.mean(axis=0)
    denominator = np.sqrt(np.sum(aa**2)*np.sum(bb**2))
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.sum(aa*bb)/denominator, -1, 1))


def alignment_features(reference, query, reference_mask, query_mask, matrix):
    """Measure structural agreement after a proposed reference->query transform."""
    if matrix is None:
        return None
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        return None
    h, w = reference.shape
    inverse = cv2.invertAffineTransform(matrix)
    aligned = cv2.warpAffine(np.where(query_mask, query, 0), inverse, (w, h), flags=cv2.INTER_LINEAR)
    fraction = cv2.warpAffine(query_mask.astype(np.float32), inverse, (w, h), flags=cv2.INTER_LINEAR)
    valid = reference_mask & (fraction >= 1.-1e-6)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    roi = np.zeros_like(valid)
    roi[h//8:h-h//8, w//8:w-w//8] = True
    valid &= roi
    support = float(valid.sum()/roi.sum())
    if support < .2:
        return None
    a = self_similarity_features(np.where(reference_mask, reference, 0))
    b = self_similarity_features(aligned)
    global_score = correlation(a, b, valid)
    if global_score is None:
        return None
    patches = []
    for yy in np.array_split(np.arange(h//8, h-h//8), 4):
        for xx in np.array_split(np.arange(w//8, w-w//8), 4):
            location = np.ix_(yy, xx)
            mask = valid[location]
            if mask.mean() < .5:
                continue
            value = correlation(a[location], b[location], mask)
            if value is not None:
                patches.append(value)
    orientation = correlation(orientation_features(reference), orientation_features(aligned), valid)
    return dict(global_correlation=global_score,
        patch_median_correlation=float(np.median(patches)) if patches else -1.,
        orientation_correlation=-1. if orientation is None else orientation,
        geometric_support_fraction=support, valid_patch_count=len(patches),
        joint_nondark_fraction=float(np.mean((reference[valid] > 3) & (aligned[valid] > 3))))


def transform_distance(a, b, shape):
    h, w = shape
    probes = np.array([[x, y, 1.] for y in np.linspace(h/8, 7*h/8, 5) for x in np.linspace(w/8, 7*w/8, 5)])
    return float(np.median(np.linalg.norm(probes@np.asarray(a).T-probes@np.asarray(b).T, axis=1)))


def score_candidates(reference, query, reference_mask, query_mask, candidates):
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError('Duplicate predictor identity')
    scores = {}
    for candidate in candidates:
        quality = alignment_features(reference, query, reference_mask, query_mask, candidate.matrix)
        if quality is None:
            continue
        other_families = {c.family for c in candidates if c.family != candidate.family and c.matrix is not None}
        agree = {c.family for c in candidates if c.family in other_families and c.matrix is not None
                 and transform_distance(candidate.matrix, c.matrix, reference.shape) <= 3.}
        quality['agreement_fraction'] = len(agree)/max(len(other_families), 1)
        quality['agreeing_other_families'] = sorted(agree)
        scores[candidate.name] = quality
    return scores


def choose(candidates, scores, mode):
    if mode not in MODES:
        raise ValueError('Unknown image-only selection mode')
    weights = MODES[mode]
    ranking = []
    for candidate in candidates:
        q = scores.get(candidate.name)
        if q is None:
            continue
        value = (weights['global_weight']*q['global_correlation']+
                 weights['patch_weight']*q['patch_median_correlation']+
                 weights['orientation_weight']*q['orientation_correlation']+
                 weights['agreement_weight']*q['agreement_fraction'])
        ranking.append((value, candidate.name, candidate))
    ranking.sort(key=lambda x: (-x[0], x[1]))
    if not ranking:
        return None, dict(reason='no_image_supported_candidate')
    value, name, selected = ranking[0]
    return selected, dict(selected_predictor=name, selection_score=value,
        selection_margin=None if len(ranking) < 2 else value-ranking[1][0], **scores[name])
