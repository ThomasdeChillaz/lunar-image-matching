"""Blind low-frequency reranking of existing lunar pose and Sun candidates."""

import numpy as np


BLOCK = 32
MIN_BLOCKS_PER_QUADRANT = 2
MIN_QUADRANTS = 3
HUBER_DELTA = 1.0
ROUND_OFF_TIE = 1e-12


def _fallback(order, reason):
    return dict(order=list(order), reason=reason, scores=None, quadrant_losses=None)


def _average_blocks(image, support, counts):
    height, width = support.shape
    blocks = np.where(support, image, 0.).reshape(height//BLOCK, BLOCK, width//BLOCK, BLOCK)
    return blocks.sum(axis=(1, 3)) / np.maximum(counts, 1)


def _unit_signal(values, weights):
    centered = values - values[0]
    centered -= weights @ centered
    rms = np.sqrt(weights @ (centered*centered))
    numerical_floor = 64*np.finfo(float).eps*np.abs(values).max()
    return centered / rms if rms > numerical_floor else None


def _remove_plane(unit, design, weights):
    if unit is None:
        return None
    root = np.sqrt(weights)
    plane = np.linalg.lstsq(design*root[:, None], unit*root, rcond=None)[0]
    residual = unit - design @ plane
    rms = np.sqrt(weights @ (residual*residual))
    return residual / rms if rms > ROUND_OFF_TIE else None


def _rank_mode(observed, predicted, weights, quadrants, order, no_signal_reason):
    if observed is None:
        return _fallback(order, no_signal_reason)
    active_quadrants = np.unique(quadrants)
    scores, losses = [], []
    for candidate in predicted:
        # Zero is the limiting nonnegative gain for anticorrelated or
        # spatially flat predictions. The offset was removed by centering.
        fitted = 0. if candidate is None else max(0., weights @ (observed*candidate))*candidate
        residual = np.abs(observed-fitted)
        huber = np.where(residual <= HUBER_DELTA, .5*residual**2,
                         HUBER_DELTA*(residual-.5*HUBER_DELTA))
        per_quadrant = [float(huber[quadrants == region].mean()) for region in active_quadrants]
        losses.append(per_quadrant)
        scores.append(float(np.mean(per_quadrant)))
    by_score = sorted(order, key=scores.__getitem__)
    positions = {candidate: index for index, candidate in enumerate(order)}
    ranked = []
    left = 0
    while left < len(by_score):
        right = left + 1
        while right < len(by_score) and scores[by_score[right]] - scores[by_score[left]] <= ROUND_OFF_TIE:
            right += 1
        ranked.extend(sorted(by_score[left:right], key=positions.__getitem__))
        left = right
    return dict(order=ranked, reason=None, scores=scores, quadrant_losses=losses)


def rerank_lunar_candidates(query_luminance, predictions, reference_valid, existing_order):
    """Rank frozen candidates using only query pixels and reference predictions.

    Predictions and masks have shape candidate × detector height × width.
    An action's map support must be obtained from its hypothesized pose and
    Sun, with reference-side data. Existing order is a permutation of the
    candidate indices. Query truth, camera depth and labels are unavailable.
    """
    query = np.asarray(query_luminance, dtype=float)
    atlas = np.asarray(predictions, dtype=float)
    valid = np.asarray(reference_valid)
    order = tuple(existing_order)
    if (query.ndim != 2 or min(query.shape) < 2*BLOCK or any(s % (2*BLOCK) for s in query.shape)
            or atlas.ndim != 3 or atlas.shape[1:] != query.shape or atlas.shape != valid.shape
            or valid.dtype != bool or len(atlas) == 0):
        raise ValueError("Require detector arrays divisible into 32-pixel blocks and explicit Boolean atlas support")
    if (not all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in order)
            or sorted(order) != list(range(len(atlas)))):
        raise ValueError("Existing order must contain every candidate index exactly once")
    order = tuple(map(int, order))
    if not np.isfinite(atlas[valid]).all():
        raise ValueError("Valid candidate predictions must be finite")

    support = valid.all(axis=0)
    height, width = query.shape
    rows, cols = height//BLOCK, width//BLOCK
    block_shape = (rows, BLOCK, cols, BLOCK)
    counts = support.reshape(block_shape).sum(axis=(1, 3))
    block_y, block_x = np.indices((rows, cols))
    regions = 2*(block_y >= rows//2) + (block_x >= cols//2)
    enough_pixels = counts >= BLOCK**2//2
    quadrant_counts = np.bincount(regions[enough_pixels], minlength=4)
    active_regions = quadrant_counts >= MIN_BLOCKS_PER_QUADRANT
    retained = enough_pixels & active_regions[regions]
    report = dict(
        common_pixels=int(support.sum()), retained_blocks=int(retained.sum()),
        retained_pixels=int(counts[retained].sum()),
        quadrant_blocks=quadrant_counts.tolist(), active_quadrants=np.flatnonzero(active_regions).tolist(),
    )

    def fallback(reason):
        return dict(**report, primary=_fallback(order, reason), residual_after_plane=_fallback(order, reason))

    if active_regions.sum() < MIN_QUADRANTS:
        return fallback("insufficient_shared_quadrant_support")
    # Query pixels cannot alter reference-defined support. Missing pixels
    # in retained cells trigger fallback instead of changing the comparison.
    retained_native = np.repeat(np.repeat(retained, BLOCK, axis=0), BLOCK, axis=1)
    support &= retained_native
    if not np.isfinite(query[support]).all():
        return fallback("nonfinite_query_on_retained_support")
    quadrants = regions[retained]
    weights = 1. / (active_regions.sum()*quadrant_counts[quadrants])
    query_blocks = _average_blocks(query, support, counts)[retained]
    candidate_blocks = [_average_blocks(image, support, counts)[retained] for image in atlas]
    observed = _unit_signal(query_blocks, weights)
    predicted = [_unit_signal(signal, weights) for signal in candidate_blocks]
    primary = _rank_mode(observed, predicted, weights, quadrants, order, "no_query_block_variation")

    # Average native coordinates on exactly the same support, so averaging
    # an affine plane with partially covered blocks preserves the plane.
    y, x = np.indices(query.shape)
    x, y = (_average_blocks(axis, support, counts)[retained] for axis in (x, y))
    design = np.column_stack([np.ones(len(x)), (x - weights @ x)/width, (y - weights @ y)/height])
    residual = _rank_mode(
        _remove_plane(observed, design, weights),
        [_remove_plane(signal, design, weights) for signal in predicted],
        weights, quadrants, order, "no_query_variation_after_plane",
    )
    return dict(**report, primary=primary, residual_after_plane=residual)
