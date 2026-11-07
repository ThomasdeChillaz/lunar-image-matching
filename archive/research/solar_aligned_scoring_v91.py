"""Align compact projected labels to the full returned correspondence order."""
import numpy as np


def align_visible_projection(compact_xy, visible):
    """Scatter the projection routine's visible rows; invalid rows stay NaN."""
    compact_xy = np.asarray(compact_xy, dtype=float)
    visible = np.asarray(visible)
    if (visible.ndim != 1 or visible.dtype != np.bool_
            or compact_xy.shape != (int(visible.sum()), 2) or not np.isfinite(compact_xy).all()):
        raise ValueError("Compact projection must contain exactly the visible rows")
    target = np.full((len(visible), 2), np.nan)
    target[visible] = compact_xy
    return target


def load_aligned_scoring(case, points):
    from solar_evaluation_v70 import load_scoring

    truth, method = load_scoring(case, points)
    if method == "common_axial_depth_projection_at_returned_points":
        truth = dict(truth, query_xy=align_visible_projection(truth["query_xy"], truth["visible"]))
    return truth, method
