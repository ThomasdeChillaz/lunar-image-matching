"""Affine transforms map reference (x,y) to query (x,y)."""
import numpy as np


def transform(points, matrix):
    points, matrix = np.asarray(points, dtype=float), np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Expected Nx2 points and a 2x3 affine matrix')
    return points @ matrix[:, :2].T + matrix[:, 2]


def evaluation_points(shape, truth, margin=16):
    height, width = shape
    xx, yy = np.meshgrid(np.linspace(margin, width-1-margin, 12), np.linspace(margin, height-1-margin, 12))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    target = transform(points, truth)
    valid = ((target[:, 0] >= margin) & (target[:, 0] < width-margin) & (target[:, 1] >= margin) & (target[:, 1] < height-margin))
    return points[valid]


def endpoint_error(estimate, truth, points):
    if estimate is None or not len(points):
        return None
    predicted = transform(points, estimate)
    if not np.isfinite(predicted).all():
        return None
    return float(np.median(np.linalg.norm(predicted-transform(points, truth), axis=1)))
