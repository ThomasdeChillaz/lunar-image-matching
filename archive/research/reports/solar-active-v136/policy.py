"""Sequential, reference-only prediction scores for acquiring native patches.

Scores are robust separation proxies, not calibrated information gain. The
photometric prediction is a first-order patch model, not a physical rerender.
No query pixels or query geometry enter candidate ranking.
"""

from dataclasses import dataclass

import numpy as np

from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import project_z

HEIGHTS = np.array([0., -5., 5., -10., 10., -20., 20.])
POLICIES = ("uniform", "appearance", "geometry", "active")
MAX_CANDIDATES = 256
MAX_EXTRA = 32
MAX_HYPOTHESES = 8


@dataclass(frozen=True)
class Pool:
    reference_xy: np.ndarray
    local_world: np.ndarray
    template_centers: np.ndarray
    preview_gradients: np.ndarray
    preview_known: np.ndarray
    eligible: np.ndarray

    def validate(self):
        xy, world, centers, gradients, known, eligible = (
            np.asarray(getattr(self, name)) for name in self.__dataclass_fields__)
        n = len(xy)
        if (xy.shape != (n, 2) or not 1 <= n <= MAX_CANDIDATES
                or world.shape != (n, 3) or centers.shape != (n, 2)
                or gradients.shape != (n, 9, 2) or known.shape != (n, 9)
                or known.dtype != bool or eligible.shape != (n,) or eligible.dtype != bool
                or not np.isfinite(xy).all() or not ((xy >= 0) & (xy <= 511)).all()
                or not np.array_equal(xy, np.rint(xy)) or len(np.unique(xy, axis=0)) != n
                or not np.isfinite(world[eligible]).all() or not np.isfinite(centers[eligible]).all()
                or not np.isfinite(gradients[known]).all()):
            raise ValueError("Bounded distinct native anchors and reference-only previews required")
        return self


def observation_predictions(pool, intrinsics, poses, height_direction):
    """Predict nine linearized reference intensities under each trial camera.

    Terrain perturbations move reference 3D points along an explicitly supplied
    reference-frame direction. They do not certify query visibility or rerender
    the entire uncertain map. Off-detector/unknown coverage cannot support a score.
    """
    pool.validate()
    poses, up = np.asarray(poses, float), np.asarray(height_direction, float)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 1 <= len(poses) <= MAX_HYPOTHESES
            or up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1.)):
        raise ValueError("One to eight rigid cameras and an explicit unit height direction required")
    points = pool.local_world[None]+HEIGHTS[:, None, None]*up
    geometry, photometry, known = [], [], []
    for camera in poses:
        camera_check(intrinsics, camera)
        xy, _, front = project_z(points.reshape(-1, 3), intrinsics, camera)
        xy = xy.reshape(len(HEIGHTS), -1, 2)
        finite = np.isfinite(xy).all(axis=-1)
        supported = front.reshape(len(HEIGHTS), -1) & finite
        # Preview taps extend sixteen pixels either side of each centre.
        supported &= ((xy >= 17.) & (xy <= 494.)).all(axis=-1)
        supported &= pool.eligible[None] & pool.preview_known.all(axis=1)[None]
        delta = np.where(finite[..., None], xy-pool.template_centers[None], 0.)
        response = np.einsum("ntc,hnc->hnt", np.where(pool.preview_known[..., None], pool.preview_gradients, 0.), delta)
        geometry.append(delta)
        photometry.append(response)
        known.append(supported)
    return np.asarray(photometry), np.asarray(geometry), np.asarray(known)


def separation(values, known, weights):
    """Conservative interval separation, retaining all pose-pair prior mass.

    Unknown pairs supply zero evidence, without renormalizing the score over
    whatever small subset happens to be visible. Height envelopes are independent
    between competing hypotheses, a conservative first-order approximation.
    """
    values, known, weights = np.asarray(values, float), np.asarray(known), np.asarray(weights, float)
    if (values.ndim != 4 or known.shape != values.shape[:3] or known.dtype != bool
            or weights.shape != (len(values),) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1.)
            or not np.isfinite(values[np.broadcast_to(known[..., None], values.shape)]).all()):
        raise ValueError("Finite supported predictions and normalized hypothesis weights required")
    complete = known.all(axis=1)
    low = np.where(known[..., None], values, np.inf).min(axis=1)
    high = np.where(known[..., None], values, -np.inf).max(axis=1)
    score = np.zeros(values.shape[2]); evidence = np.zeros_like(score, dtype=int)
    denominator = 0.
    for first in range(len(values)):
        for second in range(first+1, len(values)):
            weight = weights[first]*weights[second]
            denominator += weight
            use = complete[first] & complete[second] & (weight > 0)
            gap = np.maximum(0., np.maximum(low[first, use]-high[second, use], low[second, use]-high[first, use]))
            distance2 = (gap**2).mean(axis=1)
            score[use] += weight*(-np.expm1(-.5*np.minimum(distance2, 100.)))
            evidence[use] += 1
    if denominator > 1e-15:
        score /= denominator
    return score, evidence


def rank_scores(pool, intrinsics, poses, height_direction, weights, mode):
    if mode not in POLICIES:
        raise ValueError("Unknown patch acquisition control")
    pool.validate()
    if mode == "uniform":
        return np.zeros(len(pool.reference_xy)), np.zeros(len(pool.reference_xy), int)
    if mode == "appearance":
        gradient = np.where(pool.preview_known[..., None], pool.preview_gradients, 0.)
        return (gradient**2).sum(axis=(1, 2))/9., pool.preview_known.sum(axis=1)
    photo, geometry, known = observation_predictions(pool, intrinsics, poses, height_direction)
    return separation(photo if mode == "active" else geometry, known, weights)


def cell_schedule(pool, budget):
    """All policies receive identical spatial slots from one eligible pool."""
    pool.validate()
    if type(budget) is not int or not 0 <= budget <= MAX_EXTRA:
        raise ValueError("Extra acquisition budget must be between zero and32")
    cells = (pool.reference_xy//128).astype(int)
    cells = 4*cells[:, 1]+cells[:, 0]
    capacity = np.bincount(cells[pool.eligible], minlength=16)
    quotas = np.zeros(16, int); schedule = []; served = []; previous_round = -1
    coordinates = np.c_[np.arange(16) % 4, np.arange(16)//4]
    for _ in range(min(budget, int(pool.eligible.sum()))):
        available = quotas < capacity
        current = quotas[available].min()
        options = np.flatnonzero(available & (quotas == current))
        if current != previous_round:
            served = []; previous_round = current
        if served:
            distance = ((coordinates[options, None]-coordinates[served])**2).sum(axis=2).min(axis=1)
            chosen = int(options[np.argmax(distance)])
        else:
            chosen = int(options[0])
        quotas[chosen] += 1; served.append(chosen); schedule.append(chosen)
    return cells, schedule


def choose(pool, cells, cell, selected, scores):
    remaining = pool.eligible & (cells == cell)
    remaining[np.asarray(selected, int)] = False
    candidates = np.flatnonzero(remaining)
    if not len(candidates) or np.asarray(scores).shape != (len(pool.reference_xy),):
        raise ValueError("Common spatial slot has no remaining candidate")
    centre = (np.array([cell % 4, cell//4])+.5)*128.-.5
    distance = ((pool.reference_xy[candidates]-centre)**2).sum(axis=1)
    # Deterministic spatial ordering breaks neutral or identical score ties.
    order = np.lexsort((candidates, distance, -np.asarray(scores)[candidates]))
    return int(candidates[order[0]])


def acquire(pool, intrinsics, height_direction, mode, budget, state, propose, measure, update):
    """Choose -> acquire one actual field -> update before choosing again.

    The callback interface makes the measurement boundary explicit. Ranking
    sees reference previews and hypotheses, while only `measure(index)` may
    read a full native query patch. Failed searches still consume a slot.
    """
    cells, schedule = cell_schedule(pool, budget)
    selected, trace = [], []
    for cell in schedule:
        poses, weights = propose(state)
        scores, evidence = rank_scores(pool, intrinsics, poses, height_direction, weights, mode)
        index = choose(pool, cells, cell, selected, scores)
        selected.append(index)
        observation = measure(index)
        state = update(state, index, observation)
        trace.append(dict(index=index, cell=cell, score=float(scores[index]),
                          known_pose_pairs=int(evidence[index]),
                          hypothesis_weights=np.asarray(weights).tolist(),
                          observation_present=observation is not None))
    return state, dict(policy=mode, requested_extra_patches=budget,
                       attempted_extra_patches=len(selected), selected_indices=selected,
                       cell_schedule=schedule, trace=trace, additional_query_fields_before_selection=0,
                       calibrated_information_gain=False)
