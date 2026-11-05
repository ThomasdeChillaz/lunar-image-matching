"""After-seal geometric upper bound for the frozen v120 patch handoff."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np


V120_SHA256 = "1db8ddc2a165595300bc41949e6c6329da6dcfb6886f9fe8164d87ecfc58b4f9"
TRUTH_TOLERANCE_PX = 3.
MAX_ANCHORS = 16


def _patch_contract():
    source = Path(__file__).resolve().parents[1] / "solar-warp-handoff-v120" / "handoff_patch.py"
    if sha256(source.read_bytes()).hexdigest() != V120_SHA256:
        raise ValueError("The exact frozen v120 patch source is required")
    spec = spec_from_file_location("_frozen_handoff_patch_v120", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCH = _patch_contract()
CARDINAL = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


@dataclass(frozen=True)
class PreparedSupport:
    reference_xy: np.ndarray
    centres_xy: np.ndarray
    anchor_visible: np.ndarray
    common_reference_valid: np.ndarray
    query_valid: np.ndarray
    radius: int


def _frozen(array, dtype):
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _in_frame(xy, shape):
    return np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy <= np.asarray(shape[::-1])-1).all(axis=1)


def prepare_support(centres_xy, anchor_visible, common_reference_valid, query_valid, *, reference_xy, radius=128):
    """Freeze inputs available to v120 before opening the scoring geometry."""
    centres, reference = (np.asarray(x, float) for x in (centres_xy, reference_xy))
    visible, common, query = (np.asarray(x) for x in (anchor_visible, common_reference_valid, query_valid))
    if (type(radius) is not int or radius not in (16, 128)
            or centres.ndim != 2 or centres.shape[1] != 2 or not 1 <= len(centres) <= MAX_ANCHORS
            or reference.shape != centres.shape or visible.shape != (len(centres),)
            or common.ndim != 2 or query.shape != common.shape
            or any(m.dtype != bool for m in (visible, common, query))
            or any(not PATCH.PATCH <= n <= PATCH.MAX_IMAGE_SIDE for n in common.shape)):
        raise ValueError("Require frozen v120 centres, anchor and detector masks, and a declared radius")
    if (not _in_frame(reference, common.shape).all() or not np.array_equal(reference, np.rint(reference))
            or len(np.unique(reference, axis=0)) != len(reference)
            or not np.isfinite(centres[visible]).all()):
        raise ValueError("Reference anchors must be distinct native centres and eligible projections finite")
    return PreparedSupport(_frozen(reference, "<f8"), _frozen(centres, "<f8"), _frozen(visible, bool),
                           _frozen(common, bool), _frozen(query, bool), radius)


def support_digest(prepared):
    """Bind the complete centre and support bundle to a pre-scoring receipt."""
    digest = sha256(f"v122/{V120_SHA256}/{prepared.radius}/{TRUTH_TOLERANCE_PX}".encode())
    for array in (prepared.reference_xy, prepared.centres_xy, prepared.anchor_visible,
                  prepared.common_reference_valid, prepared.query_valid):
        digest.update(json.dumps((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def fractional_mask(mask, point):
    """Exact support of v120's translated integer lattice, without radiance."""
    anchor = np.floor(point).astype(int)
    fx, fy = point - anchor
    if fx:
        mask = mask[:, :-1] & mask[:, 1:]
    if fy:
        mask = mask[:-1] & mask[1:]
    return mask, anchor


class _CropSupport:
    """Check the fixed v120 footprint and every retained template sample."""

    def __init__(self, mask, layout):
        self.shape = mask.shape
        self.valid = mask.ravel()
        self.complete = self.valid.all()
        self.local = layout[0] * mask.shape[1] + layout[1]
        self.cached = {}

    def allowed(self, centres):
        keys = [tuple(xy) for xy in centres]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cached))
        if missing:
            origins = np.asarray(missing, dtype=int) - PATCH.PATCH // 2
            h, w = self.shape
            fits = ((origins >= 0) & (origins + PATCH.PATCH <= [w, h])).all(axis=1)
            if not self.complete:
                inside = np.flatnonzero(fits)
                for start in range(0, len(inside), PATCH.CHUNK):
                    indices = inside[start:start + PATCH.CHUNK]
                    ox, oy = origins[indices].T
                    fits[indices] = self.valid[(oy*w+ox)[:, None] + self.local].all(axis=1)
            self.cached.update(zip(missing, fits, strict=True))
        return np.array([self.cached[key] for key in keys], dtype=bool)


def _can_peak(candidate, valid_offsets, points, step, exclusion):
    """Necessary support for v120's cardinal-neighbour and rival checks."""
    if tuple(candidate) not in points or any(tuple(other) not in points for other in candidate + step*CARDINAL):
        return False
    return bool(np.any(np.max(np.abs(valid_offsets-candidate), axis=1) > exclusion))


def _paths(support, seed, desired, radius):
    """Return supported stride-8 / stride-1 paths that scores could select."""
    if not len(desired):
        return {}
    coarse = PATCH._grid([0, 0], radius, PATCH.COARSE_STEP)
    available = coarse[support.allowed(seed+coarse)]
    coarse_points = {tuple(xy) for xy in available}
    result = {}
    for candidate in available:
        inside = desired[(np.abs(desired-candidate) <= PATCH.FINE_RADIUS).all(axis=1)]
        if not len(inside) or not _can_peak(candidate, available, coarse_points, PATCH.COARSE_STEP, PATCH.COARSE_STEP):
            continue
        fine = PATCH._grid(candidate, PATCH.FINE_RADIUS, 1)
        fine = fine[(np.abs(fine) <= radius).all(axis=1)]
        fine = fine[support.allowed(seed+fine)]
        fine_points = {tuple(xy) for xy in fine}
        for offset in inside:
            if _can_peak(offset, fine, fine_points, 1, 2):
                result.setdefault(tuple(offset), candidate)
    return result


def _near_offsets(delta, tolerance, radius):
    low = np.maximum(np.ceil(delta-tolerance), -radius).astype(int)
    high = np.minimum(np.floor(delta+tolerance), radius).astype(int)
    x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
    offsets = np.column_stack((x.ravel(), y.ravel()))
    distance = np.linalg.norm(offsets-delta, axis=1)
    order = np.lexsort((offsets[:, 0], offsets[:, 1], distance))
    return offsets[order[distance[order] <= tolerance]]


def _probe_anchor(prepared, centre, target):
    delta = target-centre
    choices = _near_offsets(delta, TRUTH_TOLERANCE_PX, prepared.radius)
    if not len(choices):
        return dict(reason="no_truth_lattice_offset_within_radius", witness=None)
    ref, ra = fractional_mask(prepared.common_reference_valid, centre)
    query, qa = fractional_mask(prepared.query_valid, centre)
    template = PATCH._crop(ref, ra)
    if template is None:
        return dict(reason="incomplete_reference_crop", witness=None)
    layout = PATCH._sample_layout(template)
    if layout is None:
        return dict(reason="insufficient_fixed_reference_support", witness=None)
    forward = _paths(_CropSupport(query, layout), qa, choices, prepared.radius)
    if not forward:
        return dict(reason="no_feasible_forward_geometry_path", witness=None)

    reference = _CropSupport(ref, layout)
    for offset in choices:
        point = tuple(offset)
        if point not in forward:
            continue
        reverse_choices = _near_offsets(-offset, PATCH.ROUND_TRIP_MAX_PX, prepared.radius)
        reverse = _paths(reference, ra+offset, reverse_choices, prepared.radius)
        for back in reverse_choices:
            if tuple(back) not in reverse:
                continue
            possible = centre + choices
            native_shape = prepared.query_valid.shape
            regions = np.unique(_quadrant_indices(possible[_in_frame(possible, native_shape)], native_shape))
            return dict(reason="reachable_geometry_upper_bound", witness=dict(
                possible_query_xy=(centre+offset).tolist(), truth_error_px=float(np.linalg.norm(delta-offset)),
                round_trip_px=float(np.linalg.norm(offset+back)), support_pixels=len(layout[0]),
                forward_coarse_xy=forward[point].tolist(), forward_native_xy=offset.tolist(),
                reverse_coarse_xy=reverse[tuple(back)].tolist(), reverse_native_xy=back.tolist(),
            ), possible_query_quadrants_upper_bound=regions.tolist())
    return dict(reason="no_feasible_reciprocal_geometry_path", witness=None)


def _quadrant_indices(points, shape):
    sides = points >= np.asarray(shape[::-1])/2
    return 2*sides[:, 1] + sides[:, 0]


def _quadrants(points, shape):
    return np.bincount(_quadrant_indices(points, shape), minlength=4).tolist()


def score_after_seal(prepared, expected_sha256, truth_loader):
    """Verify frozen input support before invoking the independent truth loader.

    The loader returns native truth XY and its geometry-only co-visible mask.
    This digest is supplementary to the study's original prediction seal.
    """
    if expected_sha256 != support_digest(prepared):
        raise ValueError("Prepared support differs from its pre-scoring seal")
    truth_xy, truth_visible = truth_loader()
    target, co_visible = np.asarray(truth_xy, float), np.asarray(truth_visible)
    shape = prepared.query_valid.shape
    if (target.shape != prepared.centres_xy.shape or co_visible.shape != prepared.anchor_visible.shape
            or co_visible.dtype != bool or not _in_frame(target[co_visible], shape).all()):
        raise ValueError("Require aligned scoring-only native truth and co-visible flags")

    rows = []
    for index, centre in enumerate(prepared.centres_xy):
        if not co_visible[index]:
            row = dict(reason="truth_visibility_unavailable", witness=None)
        elif not prepared.anchor_visible[index] or not _in_frame(centre[None], shape)[0]:
            row = dict(reason="unsupported_reference_projection", witness=None)
        else:
            row = _probe_anchor(prepared, centre, target[index])
        rows.append(dict(anchor_index=index, **row))
    reached = np.array([row["witness"] is not None for row in rows])
    ref_regions = _quadrants(prepared.reference_xy[reached], shape)
    query_regions = _quadrants(target[reached], shape)
    possible_query_regions = sorted({region for row in rows for region in row.get("possible_query_quadrants_upper_bound", ())})
    count = int(reached.sum())
    return dict(schema="solar_warp_patch_geometry_upper_bound_v122", source_v120_sha256=V120_SHA256,
                support_sha256=expected_sha256, radius_pixels=prepared.radius, truth_tolerance_px=TRUTH_TOLERANCE_PX,
                fixed_anchors=len(rows), truth_co_visible=int(co_visible.sum()), reachable_anchors=count,
                reference_quadrants=ref_regions, true_query_quadrants=query_regions,
                query_quadrant_union_upper_bound=possible_query_regions,
                distributed_six_upper_bound=(count >= 6 and sum(n > 0 for n in ref_regions) >= 3
                                             and len(possible_query_regions) >= 3),
                photometric_feasibility_evaluated=False, rows=rows)
