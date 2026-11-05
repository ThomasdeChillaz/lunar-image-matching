"""Reference-only active patch selection by robust relative parallax."""

import numpy as np

from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import project_z
from solar_active_v63 import HEIGHT_VARIANTS_M


NATIVE = 512
MAX_ANCHORS = 64
MAX_HYPOTHESES = 8
GRID_CELLS = 4


def _deformation(points, plane, intrinsics, candidates):
    """Project 3D and matched plane points under the same ENU height offsets."""
    heights = np.asarray(HEIGHT_VARIANTS_M)
    xyz = np.stack((points, plane))
    xyz = np.where(np.isfinite(xyz).all(axis=-1, keepdims=True), xyz, np.nan)
    shifted = xyz[None] + heights[:, None, None, None] * np.array([0., 0., 1.])
    deformation, valid = [], []
    for pose in candidates:
        projected, _, front = project_z(shifted.reshape(-1, 3), intrinsics, pose)
        inside = (front & np.isfinite(projected).all(axis=1)
                  & (projected >= 0.).all(axis=1) & (projected <= NATIVE - 1).all(axis=1))
        xy = projected.reshape(len(heights), 2, len(points), 2)
        deformation.append(xy[:, 0] - xy[:, 1])
        valid.append(inside.reshape(len(heights), 2, len(points)).all(axis=1))
    return np.asarray(deformation), np.asarray(valid)


def _pair_scores(deformation, valid, eligible):
    """Use axiswise separation of all seven-height envelopes for each pair."""
    count = len(eligible)
    robust = np.zeros(count)
    nominal = np.zeros(count)
    known_pairs = np.zeros(count, dtype=int)
    size = len(deformation)
    total_pairs = size * (size - 1) // 2
    if total_pairs == 0:
        return robust, nominal, known_pairs, total_pairs

    low = np.where(valid[..., None], deformation, np.inf).min(axis=1)
    high = np.where(valid[..., None], deformation, -np.inf).max(axis=1)
    complete = valid.all(axis=1)
    for a in range(size):
        for b in range(a + 1, size):
            known = complete[a] & complete[b] & eligible
            gap = np.maximum(0., np.maximum(low[a, known] - high[b, known], low[b, known] - high[a, known]))
            raw = deformation[a, 0, known] - deformation[b, 0, known]
            robust[known] += np.sum(gap**2, axis=1)
            nominal[known] += np.sum(raw**2, axis=1)
            known_pairs[known] += 1
    return np.sqrt(robust/total_pairs), np.sqrt(nominal/total_pairs), known_pairs, total_pairs


def _cell_schedule(cells, pool, slots):
    """Balance rounds and spread the cells within each round without scores."""
    ids = np.arange(GRID_CELLS**2)
    coordinates = np.column_stack((ids % GRID_CELLS, ids // GRID_CELLS))
    capacity = np.bincount(cells[pool], minlength=len(ids))
    quotas = np.zeros(len(ids), dtype=int)
    schedule, served = [], []
    previous_round = -1
    for _ in range(slots):
        remaining = quotas < capacity
        current_round = quotas[remaining].min()
        available = ids[remaining & (quotas == current_round)]
        if current_round != previous_round:
            served = []
            previous_round = current_round
        if served:
            distances = coordinates[available, None, :] - coordinates[served]
            separation = (distances**2).sum(axis=2).min(axis=1)
            cell = int(available[np.argmax(separation)])
        else:
            cell = int(available[0])
        schedule.append(cell)
        served.append(cell)
        quotas[cell] += 1
    return schedule, quotas.tolist()


def _ordered_anchors(pool, cells, xy, scores, schedule):
    """Choose anchors within the same preallocated cell slots for both arms."""
    centre = (np.column_stack((cells % GRID_CELLS, cells // GRID_CELLS)) + .5) * (NATIVE//GRID_CELLS) - .5
    distance = ((xy-centre)**2).sum(axis=1)
    by_cell = {c: [i for i in pool if cells[i] == c] for c in set(schedule)}
    uniform = {c: iter(sorted(group, key=lambda i: (distance[i], i))) for c, group in by_cell.items()}
    active = {c: iter(sorted(group, key=lambda i: (-scores[i], distance[i], i))) for c, group in by_cell.items()}
    return [next(active[c]) for c in schedule], [next(uniform[c]) for c in schedule]


def select(points_world, plane_world, reference_xy, intrinsics, candidates, eligible, budget=12):
    """Schedule equal-budget parallax and uniform arms before any patch read.

    Each point and its fitted-plane counterpart represent one native reference
    ray. Candidate camera-to-world poses and detector intrinsics must be fixed
    before selection. The caller supplies the shared reference-side eligibility
    mask. The policy takes no query pixels, depth, visibility or true pose.
    """
    points, plane, xy, poses = (np.asarray(item, float) for item in (points_world, plane_world, reference_xy, candidates))
    eligible = np.asarray(eligible)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not 1 <= len(xy) <= MAX_ANCHORS
            or points.shape != (len(xy), 3) or plane.shape != points.shape
            or eligible.shape != (len(xy),) or eligible.dtype != bool
            or not np.isfinite(xy).all() or not ((xy >= 0.) & (xy <= NATIVE-1)).all()
            or len(np.unique(xy, axis=0)) != len(xy)
            or poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) > MAX_HYPOTHESES
            or type(budget) is not int or not 1 <= budget <= MAX_ANCHORS):
        raise ValueError("Require bounded native anchors, matched world points, frozen hypotheses and Boolean eligibility")
    k, _ = camera_check(intrinsics, np.eye(4))
    deformation, valid = _deformation(points, plane, k, poses)
    robust, nominal, known, total_pairs = _pair_scores(deformation, valid, eligible)
    axes = np.floor(xy/(NATIVE//GRID_CELLS)).astype(int)
    cells = GRID_CELLS * axes[:, 1] + axes[:, 0]
    pool = np.flatnonzero(eligible).tolist()
    slots = min(budget, len(pool))
    schedule, quotas = _cell_schedule(cells, pool, slots)
    active, uniform = _ordered_anchors(pool, cells, xy, robust, schedule)
    reason = None
    if not pool:
        reason = "no_shared_eligible_anchors"
    elif not np.any(robust > 0):
        reason = "no_separable_parallax_uniform_order"
    return dict(
        active=active, uniform=uniform, eligible_indices=pool, requested_budget=budget, scheduled_per_arm=slots,
        cell_schedule=schedule, cell_quotas=quotas,
        scores=[float(score) if number else None for score, number in zip(robust, known, strict=True)],
        nominal_scores=[float(score) if number else None for score, number in zip(nominal, known, strict=True)],
        known_pose_pairs=known.tolist(), hypothesis_pairs=total_pairs, reason=reason,
        height_variants_m=list(HEIGHT_VARIANTS_M),
        height_model="Equal deterministic ENU height offsets for terrain and fitted-plane points. No rerender or optical model.",
        score_model="RMS separation of the terrain-minus-plane projection envelopes over all frozen pose pairs. Unknown pairs add no evidence.",
    )
