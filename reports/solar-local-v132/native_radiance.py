"""Measure subpixel image displacements after a shared physical camera proposal.

Geometry predicts where to search; correlation of two measured rasters supplies
the endpoint. Query geometry and true camera parameters are not inputs.
"""

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solar-local-v131"))
import pose_consensus as control

PATCH = 64
RADIUS = 16
HALO = 8
SIGMA = 2.
MIN_CORRELATION = .55
MIN_MARGIN = .015
MAX_ROUND_TRIP = 1.
METHODS = ("native_plane_integer", "native_plane_subpixel", "native_terrain_subpixel")


def prepare(image, valid=None):
    image = np.asarray(image, float)
    if image.ndim != 2 or image.shape != (512, 512):
        raise ValueError("Require an unresized native 512-pixel scalar raster")
    support = np.isfinite(image) & (image >= 0)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != image.shape or valid.dtype != bool:
            raise ValueError("Native Boolean support required")
        support &= valid
    # One linear scale preserves faint image structure; there is no clipping,
    # histogram stretching or conversion to integer brightness codes.
    scale = max(float(np.sqrt(np.mean(image[support]**2))) if support.any() else 0., 1e-30)
    values = np.where(support, image / scale, 0.).astype(np.float32)
    numerator = cv2.GaussianBlur(values, (13, 13), SIGMA)
    weight = cv2.GaussianBlur(support.astype(np.float32), (13, 13), SIGMA)
    filtered = numerator / np.maximum(weight, 1e-12)
    # Missing terrain never becomes evidence through a smoothing operation.
    known = cv2.erode(support.astype(np.uint8), np.ones((13, 13), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return filtered, known


def sample(image, valid, center, side):
    axis = np.arange(side, dtype=np.float32) - side // 2
    x, y = np.meshgrid(axis + center[0], axis + center[1])
    values = cv2.remap(image, x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    weights = cv2.remap(valid.astype(np.float32), x, y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    return values, weights >= 1. - 1e-6


def correlation_surface(template, mask, observed, observed_valid):
    """Masked NCC of identical known taps in both candidate patches."""
    if mask.sum() < PATCH**2 // 2 or any(
            mask[y:y+32, x:x+32].sum() < 256 for y in (0, 32) for x in (0, 32)):
        return None, "insufficient_reference_support"
    weight = mask.astype(np.float32)
    centered = np.where(mask, template - template[mask].mean(), 0.).astype(np.float32)
    if float(np.sum(centered.astype(float)**2)) <= 1e-12 * weight.sum():
        return None, "uniform_template"
    known = observed_valid.astype(np.float32)
    values = observed * known
    count = cv2.matchTemplate(known, weight, cv2.TM_CCORR).astype(float)
    ref_sum = cv2.matchTemplate(known, centered, cv2.TM_CCORR).astype(float)
    ref_square = cv2.matchTemplate(known, centered**2, cv2.TM_CCORR).astype(float)
    sums = cv2.matchTemplate(values, weight, cv2.TM_CCORR).astype(float)
    squares = cv2.matchTemplate(values**2, weight, cv2.TM_CCORR).astype(float)
    product = cv2.matchTemplate(values, centered, cv2.TM_CCORR).astype(float)
    denominator = np.maximum(count, 1.)
    energy = np.maximum(ref_square-ref_sum**2/denominator, 0.)
    variance = np.maximum(squares-sums**2/denominator, 0.)
    covariance = product-ref_sum*sums/denominator
    supported = (count >= PATCH**2//2-.01) & (variance > 1e-12*count) & (energy > 1e-12*count)
    for y in (0, 32):
        for x in (0, 32):
            quadrant = np.zeros_like(weight)
            quadrant[y:y+32, x:x+32] = weight[y:y+32, x:x+32]
            supported &= cv2.matchTemplate(known, quadrant, cv2.TM_CCORR) >= 256.-.01
    score = np.full(variance.shape, -np.inf)
    score[supported] = np.clip(covariance[supported] / np.sqrt(energy[supported] * variance[supported]), -1., 1.)
    return score, None


def quadratic_offset(scores, y, x):
    """Fit the local peak, never average coordinates from competing modes."""
    patch = scores[y-1:y+2, x-1:x+2]
    if patch.shape != (3, 3) or not np.isfinite(patch).all():
        return None
    gx = (patch[1, 2] - patch[1, 0]) / 2
    gy = (patch[2, 1] - patch[0, 1]) / 2
    mixed = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4
    hessian = np.array([[patch[1, 2] - 2*patch[1, 1] + patch[1, 0], mixed],
                        [mixed, patch[2, 1] - 2*patch[1, 1] + patch[0, 1]]])
    if np.linalg.eigvalsh(hessian).max() >= -1e-8:
        return None
    delta = -np.linalg.solve(hessian, [gx, gy])
    return delta if np.max(np.abs(delta)) <= .75 else None


def direction(reference, reference_valid, query, query_valid, p, q, *, subpixel, template_mask=None):
    template, mask = sample(reference, reference_valid, p, PATCH)
    if template_mask is not None:
        mask &= template_mask
    reach = RADIUS + HALO
    observed, available = sample(query, query_valid, q, PATCH + 2*reach)
    scores, reason = correlation_surface(template, mask, observed, available)
    empty = dict(query_xy=None, reason=reason, score=None, margin=None)
    if scores is None:
        return empty
    if not np.isfinite(scores).any():
        return dict(empty, reason="insufficient_query_support_or_texture")
    # Stable coordinate ordering resolves exact score ties.
    y, x = np.unravel_index(np.argmax(scores), scores.shape)
    offset = np.array([x-reach, y-reach], float)
    score = float(scores[y, x])
    yy, xx = np.indices(scores.shape)
    alternate = np.maximum(np.abs(xx-x), np.abs(yy-y)) > HALO
    margin = float(score - np.max(scores[alternate]))
    report = dict(empty, score=score, margin=margin, integer_offset=offset.tolist())
    if score < MIN_CORRELATION:
        return dict(report, reason="weak_peak")
    if not np.isfinite(margin) or margin < MIN_MARGIN:
        return dict(report, reason="ambiguous_peak")
    if np.any(np.abs(offset) > RADIUS) or min(x, y) < 1 or max(x, y) >= 2*reach:
        return dict(report, reason="outside_returned_search_radius")
    if subpixel:
        delta = quadratic_offset(scores, y, x)
        if delta is None:
            return dict(report, reason="nonconcave_subpixel_peak")
        offset += delta
    if np.any(np.abs(offset) > RADIUS):
        return dict(report, reason="outside_returned_search_radius")
    return dict(report, query_xy=(q+offset).tolist(), reason=None)


def search(reference, query, reference_valid, query_valid, p, q, *, subpixel):
    forward = direction(reference, reference_valid, query, query_valid, p, q, subpixel=subpixel)
    result = dict(forward=forward, reverse=None, query_xy=None, reason=forward["reason"],
                  round_trip_error_px=None)
    if forward["query_xy"] is None:
        return result
    observed = np.asarray(forward["query_xy"])
    # Reverse search uses its own observed template and correlation surface.
    # The reverse template may use only the same known reference footprint.
    # Otherwise a query pixel corresponding to a hole in the reference map
    # would make every reverse candidate invalid, even at the correct location.
    peak_support = cv2.erode(reference_valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    _, paired_support = sample(reference, peak_support, p, PATCH)
    reverse = direction(query, query_valid, reference, reference_valid, observed, p,
                        subpixel=subpixel, template_mask=paired_support)
    result["reverse"] = reverse
    if reverse["query_xy"] is None:
        return dict(result, reason="reverse_"+reverse["reason"])
    error = float(np.linalg.norm(np.asarray(reverse["query_xy"])-p))
    result["round_trip_error_px"] = error
    if error > MAX_ROUND_TRIP:
        return dict(result, reason="nonreciprocal_peak")
    return dict(result, query_xy=observed.tolist(), reason=None)


def measure(reference, query, depth, k, reference_pose, query_k, seeds, *, reference_valid):
    tick = perf_counter()
    banks = None
    anchors = control.pipeline.previous.ANCHORS
    if len(seeds):
        warper = control.pipeline.previous.ReferencePatchWarper(
            reference, depth, k, reference_pose, anchors, reference_valid=reference_valid)
        banks = warper.warp(query_k, seeds[0], reference.shape)
    prepared_query = prepare(query)
    prepared_banks = {} if banks is None else {
        name: prepare(bank.raster, bank.detector_valid) for name, bank in banks.items()}
    setup = perf_counter()-tick
    outputs = {}
    for method in METHODS:
        tick = perf_counter()
        p, q, traces, selected = [], [], [], []
        name = "geometry_3d" if "terrain" in method else "planar"
        if banks is not None:
            bank = banks[name]
            image, valid = prepared_banks[name]
            for index, center in enumerate(bank.predicted_xy):
                if not bank.anchor_visible[index] or not np.isfinite(center).all():
                    continue
                selected.append(index)
                match = search(image, prepared_query[0], valid, prepared_query[1], center, center,
                               subpixel=not method.endswith("integer"))
                traces.append(dict(anchor_index=index, search=match))
                if match["query_xy"] is not None:
                    p.append(anchors[index].tolist())
                    q.append(match["query_xy"])
        row = control.pipeline.solve(p, q, depth, k, reference_pose, query_k)
        outputs[method] = dict(row, inference_seconds=perf_counter()-tick+setup,
                              searches=traces, selected_anchors=selected, attempted_patches=len(selected),
                              maximum_patch_budget=len(anchors),
                              failures=[dict(anchor_index=t["anchor_index"], reason=t["search"]["reason"])
                                        for t in traces if t["search"]["query_xy"] is None])
    return outputs


def run(reference, query, depth, k, reference_pose, query_k, seeds, library, **kwargs):
    # Rerun all seven controls, preserving their measurements and estimator.
    result = control.run(reference, query, depth, k, reference_pose, query_k, seeds, library, **kwargs)
    hypotheses = result["photometric_initialization"]
    chosen = np.asarray([row["pose"] for row in hypotheses[:1]]).reshape(-1, 4, 4)
    measured = measure(reference, query, depth, k, reference_pose, query_k, chosen,
                       reference_valid=kwargs["reference_valid"])
    for name, row in measured.items():
        row["inference_seconds"] += result["shared_camera_seconds"] + kwargs.get("initialization_seconds", 0.)
        result["methods"][name] = row
    return result
