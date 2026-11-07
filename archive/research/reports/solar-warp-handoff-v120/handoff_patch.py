"""Bounded reciprocal search on scene-linear native detector rasters."""

import numpy as np


PATCH = 64
COARSE_STEP = 8
FINE_RADIUS = 7
MIN_CELL_PIXELS = 32
MIN_CELLS_PER_QUADRANT = 4
MIN_PATCH_PIXELS = PATCH**2 // 2
MIN_CORRELATION = .55
MIN_PEAK_MARGIN = .015
ROUND_TRIP_MAX_PX = 1.
MIN_NUMERIC_RMS = 1e-12
MAX_IMAGE_SIDE = 512
CHUNK = 32


def _image(image, valid):
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or any(not PATCH <= size <= MAX_IMAGE_SIDE for size in array.shape):
        raise ValueError("Require a two-dimensional native raster between 64 and 512 pixels per side")
    available = np.isfinite(array) & (array >= 0.)
    if valid is not None:
        mask = np.asarray(valid)
        if mask.shape != array.shape or mask.dtype != bool:
            raise ValueError("Valid support must be a Boolean mask of the native raster")
        available &= mask
    return array, available


def _point(xy, shape):
    point = np.asarray(xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or not ((point >= 0) & (point <= np.array(shape[::-1])-1)).all():
        raise ValueError("Require an in-frame native (x, y) anchor")
    return point


def _lattice(image, valid, point):
    """Sample a translated integer lattice, with every bilinear tap valid."""
    anchor = np.floor(point).astype(int)
    fx, fy = point - anchor
    if fx == fy == 0.:
        return image, valid, anchor
    data = np.where(valid, image, 0.)
    if fx:
        data = (1.-fx) * data[:, :-1] + fx * data[:, 1:]
        valid = valid[:, :-1] & valid[:, 1:]
    if fy:
        data = (1.-fy) * data[:-1] + fy * data[1:]
        valid = valid[:-1] & valid[1:]
    return data, valid, anchor


def _crop(image, anchor):
    x, y = anchor - PATCH // 2
    if x < 0 or y < 0 or x + PATCH > image.shape[1] or y + PATCH > image.shape[0]:
        return None
    return image[y:y+PATCH, x:x+PATCH]


def _sample_layout(valid):
    y, x = np.indices((PATCH, PATCH))
    cell = (y // COARSE_STEP) * (PATCH // COARSE_STEP) + x // COARSE_STEP
    cells = np.bincount(cell[valid], minlength=(PATCH // COARSE_STEP)**2)
    active = cells.reshape(8, 8) >= MIN_CELL_PIXELS
    for dy, dx in ((0, 0), (0, 4), (4, 0), (4, 4)):
        if active[dy:dy+4, dx:dx+4].sum() < MIN_CELLS_PER_QUADRANT:
            return None
    valid = valid & active.ravel()[cell]
    if valid.sum() < MIN_PATCH_PIXELS:
        return None
    y, x = np.nonzero(valid)
    order = np.argsort(cell[y, x], kind="stable")
    y, x = y[order], x[order]
    _, starts, counts = np.unique(cell[y, x], return_index=True, return_counts=True)
    return y, x, starts, counts


def _centered_unit(signal):
    values = signal - signal[..., :1]
    values -= values.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.einsum("...i,...i->...", values, values))
    rms_floor = np.maximum(MIN_NUMERIC_RMS, 64*np.finfo(float).eps*np.abs(signal).max(axis=-1))
    usable = norm > rms_floor * np.sqrt(signal.shape[-1])
    return values / np.where(usable, norm, 1.)[..., None], usable


def _scores(template, target, target_valid, anchor, offsets, layout, *, pooled):
    y, x, starts, counts = layout
    values = template[y, x]
    if pooled:
        values = np.add.reduceat(values, starts) / counts
    ref, textured = _centered_unit(values)
    answer = np.full(len(offsets), -np.inf)
    if not textured:
        return answer, 0, False
    h, w = target.shape
    origins = anchor + offsets - PATCH // 2
    fits = ((origins >= 0) & (origins + PATCH <= [w, h])).all(axis=1)
    eligible = np.flatnonzero(fits)
    local = y * w + x
    data = target.ravel()
    valid = target_valid.ravel()
    for start in range(0, len(eligible), CHUNK):
        batch = eligible[start:start+CHUNK]
        ox, oy = origins[batch].T
        indices = (oy*w + ox)[:, None] + local
        supported = valid[indices].all(axis=1)
        if not supported.any():
            continue
        observed = data[indices[supported]]
        if pooled:
            observed = np.add.reduceat(observed, starts, axis=1) / counts
        obs, informative = _centered_unit(observed)
        answer[batch[supported][informative]] = np.clip(np.einsum("ij,j->i", obs[informative], ref), -1., 1.)
    return answer, len(eligible), True


def _grid(centre, radius, step):
    axis = np.arange(-radius, radius + 1, step)
    x, y = np.meshgrid(axis, axis)
    return np.column_stack((x.ravel(), y.ravel())) + centre


def _peak(offsets, scores, evaluated, textured, step, exclusion):
    finite = np.flatnonzero(np.isfinite(scores))
    result = dict(evaluated_positions=evaluated, scored_positions=len(finite), offset_xy=None,
                  score=None, margin=None, reason=None)
    if not textured:
        return dict(result, reason="uniform_template")
    if not len(finite):
        return dict(result, reason="insufficient_target_support_or_texture")
    dx, dy = offsets[finite].T
    order = np.lexsort((dx, dy, dx*dx+dy*dy, -scores[finite]))
    top = finite[order[0]]
    centre = offsets[top]
    result.update(offset_xy=centre.tolist(), score=float(scores[top]))
    if scores[top] < MIN_CORRELATION:
        return dict(result, reason="weak_peak")
    present = {tuple(point) for point in offsets[finite]}
    neighbours = centre + [[step, 0], [-step, 0], [0, step], [0, -step]]
    if any(tuple(point) not in present for point in neighbours):
        return dict(result, reason="boundary_peak")
    alternate = np.max(np.abs(offsets[finite] - centre), axis=1) > exclusion
    if not alternate.any():
        return dict(result, reason="insufficient_peak_comparison")
    margin = float(scores[top] - scores[finite[alternate]].max())
    result["margin"] = margin
    if margin < MIN_PEAK_MARGIN:
        result["reason"] = "ambiguous_peak"
    return result


def _search(template, target, target_valid, seed, layout, radius):
    coarse = _grid([0, 0], radius, COARSE_STEP)
    primary = _peak(coarse, *_scores(template, target, target_valid, seed, coarse, layout, pooled=True),
                    step=COARSE_STEP, exclusion=COARSE_STEP)
    result = dict(coarse=primary, fine=None, reason=primary["reason"])
    if primary["reason"] is not None:
        return result
    fine = _grid(primary["offset_xy"], FINE_RADIUS, 1)
    fine = fine[(np.abs(fine) <= radius).all(axis=1)]
    terminal = _peak(fine, *_scores(template, target, target_valid, seed, fine, layout, pooled=False),
                     step=1, exclusion=2)
    return dict(result, fine=terminal, reason=terminal["reason"])


def search_patch(reference, query, reference_xy, query_seed_xy, *, radius=128,
                 reference_valid=None, query_valid=None):
    """Measure one native query location from an observed reciprocal patch.

    The 64-pixel support spans integer offsets [-32, 31] about each anchor.
    Reference is a predicted detector raster under a hypothetical camera.
    Search uses its pixels and observed query luminance only, with no pose,
    query depth, labels or ground truth. Radius 16 is the matched control.
    """
    if type(radius) is not int or radius not in (16, 128):
        raise ValueError("The fixed wide and narrow radii are 128 and 16 native pixels")
    reference, reference_valid = _image(reference, reference_valid)
    query, query_valid = _image(query, query_valid)
    ref_xy = _point(reference_xy, reference.shape)
    query_xy = _point(query_seed_xy, query.shape)
    ref, ref_valid, ra = _lattice(reference, reference_valid, ref_xy)
    obs, obs_valid, qa = _lattice(query, query_valid, query_xy)
    coarse_limit = (2*radius//COARSE_STEP + 1)**2
    fine_limit = (2*FINE_RADIUS + 1)**2
    report = dict(query_xy=None, radius_pixels=radius, reason=None, score=None, margin=None,
                  round_trip_error_px=None, support_pixels=0, forward=None, reverse=None,
                  max_positions_per_direction=coarse_limit+fine_limit,
                  max_total_positions=2*(coarse_limit+fine_limit), evaluated_positions=0)
    template = _crop(ref, ra)
    if template is None:
        return dict(report, reason="incomplete_reference_crop")
    layout = _sample_layout(_crop(ref_valid, ra))
    if layout is None:
        return dict(report, reason="insufficient_fixed_reference_support")
    report["support_pixels"] = len(layout[0])
    forward = _search(template, obs, obs_valid, qa, layout, radius)
    report["forward"] = forward
    report["evaluated_positions"] += sum(stage["evaluated_positions"] for stage in (forward["coarse"], forward["fine"]) if stage is not None)
    if forward["reason"] is not None:
        return dict(report, reason="forward_" + forward["reason"])

    offset = np.asarray(forward["fine"]["offset_xy"], dtype=int)
    observed_template = _crop(obs, qa + offset)
    reverse = _search(observed_template, ref, ref_valid, ra + offset, layout, radius)
    report["reverse"] = reverse
    report["evaluated_positions"] += sum(stage["evaluated_positions"] for stage in (reverse["coarse"], reverse["fine"]) if stage is not None)
    if reverse["reason"] is not None:
        return dict(report, reason="reverse_" + reverse["reason"])
    error = float(np.linalg.norm(offset + reverse["fine"]["offset_xy"]))
    report["round_trip_error_px"] = error
    if error > ROUND_TRIP_MAX_PX:
        return dict(report, reason="nonreciprocal_peak")
    return dict(report, query_xy=(query_xy + offset).tolist(), score=forward["fine"]["score"],
                margin=forward["fine"]["margin"])
