"""Couple recoverable-window supervision to competition in both score axes.

Prepared training hypothesis, not a measured localization improvement. Dual
softmax is standard matching machinery (LoFTR/EfficientLoFTR), not our novelty.
Historical v18 fitting/inference is untouched. No parameters or inference
thresholds are added; geometry labels enter training only.
"""
import math

import torch

from context_supervision_v15 import proposal_grid
from search_window_learning_v18 import objective as window_objective, window_mask


def log_assignment(scores, temperature=.1):
    """Log product of row/column softmax, without probability underflow."""
    if (scores.ndim != 2 or min(scores.shape) == 0 or not torch.isfinite(scores).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Finite nonempty score matrix and positive temperature required')
    logits = scores/temperature
    return logits.log_softmax(1)+logits.log_softmax(0)


def window_correction(scores, good, temperature=.1):
    """Replace a row-only positive-set likelihood with dual-softmax set mass.

    The nonnegative correction is log(row reachable mass / dual reachable mass).
    It penalizes a candidate that another reference location monopolizes even
    when both locations can reach it in native fine search. It does not prove
    that hard mutual-nearest-neighbour decoding or camera pose will improve.
    """
    dual = log_assignment(scores, temperature)
    if good.shape != scores.shape or good.dtype != torch.bool:
        raise ValueError('Boolean recoverable-window mask must match score matrix')
    positive = good.any(1)
    if not positive.any():return scores.sum()*0
    row = (scores/temperature).log_softmax(1)[positive]
    valid = good[positive]
    return (row.masked_fill(~valid, -torch.inf).logsumexp(1)-
            dual[positive].masked_fill(~valid, -torch.inf).logsumexp(1)).mean()


def objective(scores, targets, baseline_scores=None, anchor_weight=.5, coupled=True):
    """Keep v18 margins, anchors and labels; change only positive normalization."""
    original, labels = window_objective(scores, targets, baseline_scores, anchor_weight)
    if not coupled:return original, labels
    grid = torch.from_numpy(proposal_grid()).to(scores)
    forward = window_mask(targets['forward'], targets['forward_visible'], grid)
    reverse = window_mask(targets['reverse'], targets['reverse_visible'], grid)
    fc = window_correction(scores[0], forward)
    rc = window_correction(scores[0].T, reverse)
    labels = dict(labels, mutual_forward_correction=float(fc.detach()), mutual_reverse_correction=float(rc.detach()))
    return original+.5*(fc+rc), labels
