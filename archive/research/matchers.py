"""Classical baselines return a reference-to-query affine transform or failure."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Match:
    matrix: object = None
    reason: str = ''
    support: int = 0
    reference_points: object = None
    query_points: object = None


def preprocess(image, clahe=False):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image) if clahe else image


def ncc(reference, query, reference_mask=None, query_mask=None):
    if reference.shape != query.shape:
        raise ValueError('NCC benchmark expects equal image shapes')
    h, w = reference.shape
    x, y = w//4, h//4
    template = reference[y:h-y, x:w-x]
    if reference_mask is not None and not reference_mask[y:h-y, x:w-x].all():
        return Match(reason='invalid_template_support')
    if template.std() < 1 or query.std() < 1:
        return Match(reason='low_texture')
    scores = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
    if query_mask is not None:
        fractions = cv2.matchTemplate(query_mask.astype(np.float32), np.ones(template.shape, np.float32), cv2.TM_CCORR)/template.size
        scores[fractions < 0.9999] = -1
    _, score, _, location = cv2.minMaxLoc(scores)
    if not np.isfinite(score) or score < 0.2:
        return Match(reason='low_correlation')
    matrix = np.array([[1., 0, location[0]-x], [0, 1, location[1]-y]])
    return Match(matrix, support=int(template.size))


def sift(reference, query, reference_mask=None, query_mask=None):
    detector = cv2.SIFT_create(nfeatures=2000)
    ka, da = detector.detectAndCompute(reference, None if reference_mask is None else reference_mask.astype(np.uint8)*255)
    kb, db = detector.detectAndCompute(query, None if query_mask is None else query_mask.astype(np.uint8)*255)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return Match(reason='insufficient_features')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75*pair[1].distance]
    # Reject duplicate destination correspondences before robust estimation.
    good = sorted(good, key=lambda m: m.distance)
    unique = {}
    for match in good:
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 6:
        return Match(reason='insufficient_matches', support=len(good))
    a = np.float32([ka[m.queryIdx].pt for m in good])
    b = np.float32([kb[m.trainIdx].pt for m in good])
    return fit_geometry(a, b)


def fit_geometry(a, b):
    """Shared estimator for SIFT and learned matches; geometry settings stay fixed."""
    a,b=np.asarray(a,np.float32),np.asarray(b,np.float32)
    if len(a)<6 or a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:
        return Match(reason='insufficient_matches',support=len(a))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return Match(reason='nonfinite_matches')
    if np.linalg.matrix_rank(a-a.mean(axis=0))<2 or np.linalg.matrix_rank(b-b.mean(axis=0))<2:
        return Match(reason='degenerate_matches')
    matrix,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99)
    count=0 if mask is None else int(mask.sum())
    if matrix is None or count<6 or not np.isfinite(matrix).all():
        return Match(reason='geometry_failed',support=count)
    keep=mask.ravel().astype(bool)
    return Match(matrix,support=count,reference_points=a[keep],query_points=b[keep])
