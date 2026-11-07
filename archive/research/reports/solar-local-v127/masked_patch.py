"""Native masked patch correlation with pooled refinement and comparison halos."""

from pathlib import Path
import sys

import numpy as np


OLD = Path(__file__).resolve().parents[1] / "solar-warp-handoff-v120"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
import handoff_patch as original

PATCH = original.PATCH
CELL = original.COARSE_STEP
MIN_CORRELATION = original.MIN_CORRELATION
MIN_MARGIN = original.MIN_PEAK_MARGIN


def masked_scores(template, template_valid, target, target_valid, anchor, offsets):
    """Average the same valid pixels in both patches at each displacement."""
    scores = np.full(len(offsets), -np.inf)
    positions = anchor + offsets - PATCH // 2
    fits = ((positions >= 0) & (positions + PATCH <= target.shape[::-1])).all(axis=1)
    eligible = np.flatnonzero(fits)
    yy, xx = np.indices((PATCH, PATCH))
    textured = False
    for start in range(0, len(eligible), original.CHUNK):
        batch = eligible[start:start + original.CHUNK]
        x, y = positions[batch].T
        pixels_y, pixels_x = y[:, None, None] + yy, x[:, None, None] + xx
        observed = target[pixels_y, pixels_x]
        common = template_valid[None] & target_valid[pixels_y, pixels_x]
        tile = (len(batch), PATCH // CELL, CELL, PATCH // CELL, CELL)
        counts = common.reshape(tile).sum(axis=(2, 4))
        active = counts >= original.MIN_CELL_PIXELS
        regions = active.reshape(len(batch), 2, 4, 2, 4).sum(axis=(2, 4))
        supported = ((regions >= original.MIN_CELLS_PER_QUADRANT).all(axis=(1, 2))
                     & (np.where(active, counts, 0).sum(axis=(1, 2)) >= original.MIN_PATCH_PIXELS))
        ref = np.where(common, template[None], 0.).reshape(tile).sum(axis=(2, 4)) / np.maximum(counts, 1)
        obs = np.where(common, observed, 0.).reshape(tile).sum(axis=(2, 4)) / np.maximum(counts, 1)
        weight = active.reshape(len(batch), -1)
        ref, obs = ref.reshape(len(batch), -1), obs.reshape(len(batch), -1)
        count = np.maximum(weight.sum(axis=1), 1)
        ref = np.where(weight, ref - (ref * weight).sum(axis=1)[:, None] / count[:, None], 0.)
        obs = np.where(weight, obs - (obs * weight).sum(axis=1)[:, None] / count[:, None], 0.)
        ref_norm = np.sqrt(np.sum(ref * ref, axis=1))
        obs_norm = np.sqrt(np.sum(obs * obs, axis=1))
        informative = (ref_norm > original.MIN_NUMERIC_RMS * np.sqrt(count)) & (
            obs_norm > original.MIN_NUMERIC_RMS * np.sqrt(count))
        textured |= bool(np.any(supported & informative))
        keep = supported & informative
        scores[batch[keep]] = np.clip(np.sum(ref[keep] * obs[keep], axis=1) /
                                      (ref_norm[keep] * obs_norm[keep]), -1., 1.)
    return scores, int(len(eligible)), textured


def peak(offsets, scores, evaluated, textured, *, step, exclusion):
    return original._peak(offsets, scores, evaluated, textured, step, exclusion)


def search_direction(template, template_valid, target, target_valid, seed, radius, mode):
    halo = CELL
    coarse_offsets = original._grid([0, 0], radius + halo, CELL)
    if mode == "halo":
        layout = original._sample_layout(template_valid)
        if layout is None:
            return dict(coarse=None, fine=None, reason="insufficient_fixed_reference_support")
        coarse = peak(coarse_offsets, *original._scores(
            template, target, target_valid, seed, coarse_offsets, layout, pooled=True),
            step=CELL, exclusion=CELL)
    else:
        coarse = peak(coarse_offsets, *masked_scores(
            template, template_valid, target, target_valid, seed, coarse_offsets),
            step=CELL, exclusion=CELL)
    result = dict(coarse=coarse, fine=None, reason=coarse["reason"])
    if coarse["reason"] is not None:
        return result
    centre = np.asarray(coarse["offset_xy"], dtype=int)
    if np.any(np.abs(centre) > radius):
        return dict(result, reason="outside_returned_search_radius")
    fine_offsets = original._grid(centre, original.FINE_RADIUS, 1)
    if mode == "halo":
        fine = peak(fine_offsets, *original._scores(
            template, target, target_valid, seed, fine_offsets, layout, pooled=False),
            step=1, exclusion=2)
    else:
        # One pooled cell is one correlation basin. Native offsets inside
        # that basin are compared to each other for position, while the
        # ambiguity margin tests other basins at the same spatial scale.
        fine_offsets = np.unique(np.vstack((fine_offsets, coarse_offsets)), axis=0)
        fine = peak(fine_offsets, *masked_scores(
            template, template_valid, target, target_valid, seed, fine_offsets),
            step=1, exclusion=CELL)
    if fine["reason"] is None and np.any(np.abs(fine["offset_xy"]) > radius):
        fine["reason"] = "outside_returned_search_radius"
    return dict(result, fine=fine, reason=fine["reason"])


def search_patch(reference, query, reference_xy, query_seed_xy, *, radius=16,
                 reference_valid=None, query_valid=None, mode="pooled"):
    """Return image-measured offsets within the declared radius."""
    if radius != 16 or type(radius) is not int or mode not in ("pooled", "halo"):
        raise ValueError("The measured pilot declares a 16-pixel radius and pooled or halo mode")
    reference, reference_valid = original._image(reference, reference_valid)
    query, query_valid = original._image(query, query_valid)
    p, q = original._point(reference_xy, reference.shape), original._point(query_seed_xy, query.shape)
    ref, ref_valid, ra = original._lattice(reference, reference_valid, p)
    obs, obs_valid, qa = original._lattice(query, query_valid, q)
    template, template_valid = original._crop(ref, ra), original._crop(ref_valid, ra)
    report = dict(query_xy=None, radius_pixels=radius, mode=mode, reason=None,
                  score=None, margin=None, round_trip_error_px=None, support_pixels=0,
                  forward=None, reverse=None, evaluated_positions=0,
                  max_total_positions=646 if mode == "pooled" else 548)
    if template is None:
        return dict(report, reason="incomplete_reference_crop")
    report["support_pixels"] = int(template_valid.sum())
    forward = search_direction(template, template_valid, obs, obs_valid, qa, radius, mode)
    report["forward"] = forward
    report["evaluated_positions"] += sum(stage["evaluated_positions"] for stage in
                                         (forward["coarse"], forward["fine"]) if stage is not None)
    if forward["reason"] is not None:
        return dict(report, reason="forward_" + forward["reason"])
    offset = np.asarray(forward["fine"]["offset_xy"], dtype=int)
    observed_template = original._crop(obs, qa + offset)
    observed_valid = original._crop(obs_valid, qa + offset)
    reverse = search_direction(observed_template, observed_valid, ref, ref_valid, ra + offset, radius, mode)
    report["reverse"] = reverse
    report["evaluated_positions"] += sum(stage["evaluated_positions"] for stage in
                                         (reverse["coarse"], reverse["fine"]) if stage is not None)
    if reverse["reason"] is not None:
        return dict(report, reason="reverse_" + reverse["reason"])
    error = float(np.linalg.norm(offset + reverse["fine"]["offset_xy"]))
    report["round_trip_error_px"] = error
    if error > original.ROUND_TRIP_MAX_PX:
        return dict(report, reason="nonreciprocal_peak")
    return dict(report, query_xy=(q + offset).tolist(), score=forward["fine"]["score"],
                margin=forward["fine"]["margin"])
