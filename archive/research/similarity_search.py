"""Image-only coarse-to-fine similarity search using structural correlation.

Search ranges are a declared model prior, never a case's known motion. This
experimental method compares candidate image warps using structural similarity;
no evaluation points, errors, labels or illumination metadata enter inference.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from geometry import transform
from structural_matchers import CorrelationMatch, structural_ncc


@dataclass
class SimilarityMatch(CorrelationMatch):
    candidates_evaluated: int = 0


def similarity_search(reference, query, reference_mask=None, query_mask=None):
    if reference.shape != query.shape:
        raise ValueError('Similarity search needs equal image shapes')
    ma = np.ones_like(reference, bool) if reference_mask is None else reference_mask.astype(bool)
    mb = np.ones_like(query, bool) if query_mask is None else query_mask.astype(bool)
    if not ma.any() or not mb.any():
        return SimilarityMatch(reason='no_valid_support')
    if min(reference[ma].std(), query[mb].std()) < 1:
        return SimilarityMatch(reason='low_texture')
    h, w = reference.shape
    center = ((w-1)/2, (h-1)/2)
    best, best_parameters, best_geometry = None, None, None
    visited = set()
    def candidate(angle, scale):
        nonlocal best, best_parameters, best_geometry
        if not -40 <= angle <= 40 or not .65 <= scale <= 1.5:
            return
        key = (round(float(angle), 8), round(float(scale), 8))
        if key in visited:
            return
        visited.add(key)
        geometry = cv2.getRotationMatrix2D(center, angle, scale)
        inverse = cv2.invertAffineTransform(geometry)
        aligned = cv2.warpAffine(np.where(mb, query, 0), inverse, (w, h), flags=cv2.INTER_LINEAR)
        fraction = cv2.warpAffine(mb.astype(np.float32), inverse, (w, h), flags=cv2.INTER_LINEAR)
        valid = fraction >= 1.-1e-6
        aligned[~valid] = 0
        proposal = structural_ncc(reference, aligned, ma, valid)
        if proposal.matrix is None:
            return
        if best is None or (proposal.score, proposal.peak_margin) > (best.score, best.peak_margin):
            best, best_parameters, best_geometry = proposal, (angle, scale), geometry
    for angle in [-30., -15., 0., 15., 30.]:
        for scale in [.8, 1., 1.25]:
            candidate(angle, scale)
    if best is None:
        return SimilarityMatch(reason='no_structural_similarity_candidate', candidates_evaluated=len(visited))
    for angles, scales in [([-7.5, -3.75, 0., 3.75, 7.5], [-.1, -.05, 0., .05, .1]),
                          ([-1.875, 0., 1.875], [-.025, 0., .025])]:
        angle, scale = best_parameters
        for delta_angle in angles:
            for delta_scale in scales:
                candidate(angle+delta_angle, scale+delta_scale)
    g = np.vstack((best_geometry, [0, 0, 1]))
    d = np.vstack((best.matrix, [0, 0, 1]))
    matrix = (g@d)[:2]
    a = best.reference_points
    b = transform(a, matrix)
    inside = (b[:, 0] >= 0) & (b[:, 0] <= w-1) & (b[:, 1] >= 0) & (b[:, 1] <= h-1)
    valid = np.zeros(len(b), bool)
    xy = np.rint(b[inside]).astype(int)
    valid[inside] = mb[xy[:, 1], xy[:, 0]]
    if valid.sum() < 6:
        return SimilarityMatch(reason='insufficient_final_support', candidates_evaluated=len(visited))
    return SimilarityMatch(matrix=matrix, support=int(valid.sum()), reference_points=a[valid], query_points=b[valid],
        score=best.score, peak_margin=best.peak_margin, candidates_evaluated=len(visited))
