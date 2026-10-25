"""Classical baselines return a reference-to-query affine transform or failure."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Match:
    matrix: object = None
    reason: str = ''
    support: int = 0


def preprocess(image, clahe=False):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image) if clahe else image


def ncc(reference, query):
    if reference.shape != query.shape:
        raise ValueError('NCC benchmark expects equal image shapes')
    h, w = reference.shape
    x, y = w//4, h//4
    template = reference[y:h-y, x:w-x]
    if template.std() < 1 or query.std() < 1:
        return Match(reason='low_texture')
    scores = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    if not np.isfinite(score) or score < 0.2:
        return Match(reason='low_correlation')
    matrix = np.array([[1., 0, location[0]-x], [0, 1, location[1]-y]])
    return Match(matrix, support=int(template.size))


def sift(reference, query):
    detector = cv2.SIFT_create(nfeatures=2000)
    ka, da = detector.detectAndCompute(reference, None)
    kb, db = detector.detectAndCompute(query, None)
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
    matrix, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000, confidence=0.99)
    count = 0 if mask is None else int(mask.sum())
    if matrix is None or count < 6 or not np.isfinite(matrix).all():
        return Match(reason='geometry_failed', support=count)
    return Match(matrix, support=count)
