"""Reference-only selection for a persistent camera / catalogue-light beam.

The scores are conservative separation proxies, not calibrated probabilities or
certified physical shadow observations. Every branch keeps one shared light.
"""
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

MAX_BRANCHES = 8
MAX_SEARCHES = 128
HEIGHTS = np.array([0., -5., 5., -10., 10., -20., 20.])
WEIGHT_FLOOR_MASS = .10


@dataclass(frozen=True)
class CandidateQueue:
    """Reference-anchor scheduling only; contains no image / visibility prediction."""
    reference_xy: np.ndarray
    eligible: np.ndarray

    def validate(self):
        points, known = np.asarray(self.reference_xy), np.asarray(self.eligible)
        if (points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 256
            or known.shape != (len(points),) or known.dtype != bool or not np.isfinite(points).all()
            or not ((points >= 0) & (points <= 511)).all()
            or len(np.unique(points, axis=0)) != len(points)):
            raise ValueError("Distinct native reference anchors and explicit candidate availability required")
        return self


def candidate_queue(branches):
    if not branches:
        raise ValueError("At least one qualified branch required")
    points = branches[0]["pool"].reference_xy
    if any(not np.array_equal(b["pool"].reference_xy, points) for b in branches):
        raise ValueError("Every branch must refer to the same reference anchors")
    return CandidateQueue(points.copy(), np.logical_or.reduce([b["pool"].eligible for b in branches])).validate()


def common_field_budget(branches, cap):
    count = len(branches)
    capacity = int(candidate_queue(branches).eligible.sum())
    if type(cap) is not int or not 0 <= cap <= MAX_SEARCHES or not 1 <= count <= MAX_BRANCHES:
        raise ValueError("Bounded branch set and native-field cap required")
    return (min(cap, capacity)//count)*count, capacity


def normalized_weights(log_evidence):
    values = np.asarray(log_evidence, float)
    if values.ndim != 1 or not 1 <= len(values) <= MAX_BRANCHES or not np.isfinite(values).all():
        raise ValueError("One to eight finite branch evidence values required")
    weight = np.exp((values-values.max())/.25)
    weight /= weight.sum()
    # No branch is pruned by a provisional measurement. This exploration floor
    # is a declared search policy, never a calibrated acceptance probability.
    return (1-WEIGHT_FLOOR_MASS)*weight+WEIGHT_FLOOR_MASS/len(weight)


def choose_diverse_seeds(scores, light_ids, cameras, reference_world, query_k,
                         reference_pose, *, lights=4, branches=8, separation_px=8.):
    """Retain light diversity, then camera diversity, within the eight-seed cap.

    No query geometry is used: camera distances are projections of fixed
    reference anchors. Coarse image losses are the already computed observations.
    """
    from lunar_geometry_v5 import project_z
    if type(lights) is not int or not 1 <= lights <= 4 or branches != 8:
        raise ValueError("One to four lights and the declared eight-seed cap required")
    camera = np.asarray(cameras, float)
    reference_world = np.asarray(reference_world, float)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or len(scores) != len(light_ids):
        raise ValueError("Aligned catalogue scores and rigid camera proposals required")
    if (reference_world.ndim != 2 or reference_world.shape[1] != 3 or not len(reference_world)
        or not np.isfinite(reference_world).all()):
        raise ValueError("Finite reference anchors required for camera diversity")
    if any(len(row) != len(camera) for row in scores) or not np.isfinite(separation_px) or separation_px <= 0:
        raise ValueError("Aligned finite-score rows and positive diversity radius required")
    if any(v is not None and not np.isfinite(v) for row in scores for v in row):
        raise ValueError("Unavailable losses must be null")
    order = []
    for slot, row in enumerate(scores):
        indices = sorted((i for i, v in enumerate(row) if v is not None), key=lambda i: (row[i], i))
        if indices:
            order.append((row[indices[0]], light_ids[slot], slot, indices))
    order.sort(key=lambda x: (x[0], x[1]))
    chosen_lights = order[:lights]
    if not chosen_lights:
        return []
    projections = {}
    inverse = np.linalg.inv(reference_pose)
    def distance(first, second):
        for index in (first, second):
            if index not in projections:
                xy, _, front = project_z(reference_world, query_k, inverse @ camera[index])
                projections[index] = xy, front & np.isfinite(xy).all(axis=1)
        a, va = projections[first]; b, vb = projections[second]
        valid = va & vb
        return 0. if valid.sum() < min(6, len(reference_world)) else float(np.sqrt(np.mean((a[valid]-b[valid])**2)))
    retained = {row[2]: [] for row in chosen_lights}
    # Round-robin protects the fourth light before allocating a second camera.
    while sum(map(len, retained.values())) < branches:
        made_progress = False
        for _, _, slot, indices in chosen_lights:
            available = [i for i in indices if i not in retained[slot]]
            if not available or sum(map(len, retained.values())) == branches:
                continue
            diverse = [i for i in available if all(distance(i, old) >= separation_px for old in retained[slot])]
            index = (diverse or available)[0]
            retained[slot].append(index); made_progress = True
        if not made_progress:
            break
    return [dict(light_slot=slot, light_id=light_ids[slot], pose_index=index,
                 coarse_loss=float(scores[slot][index]))
            for _, _, slot, _ in chosen_lights for index in retained[slot]]


def preview_values(image, valid, centers):
    axis = np.array([-16., 0., 16.])
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    points = centers[:, None, :] + np.c_[xx.ravel(), yy.ravel()][None]
    safe = np.nan_to_num(points, nan=-100., posinf=-100., neginf=-100.)
    values = cv2.remap(image.astype(np.float32), safe[..., 0].astype(np.float32),
                       safe[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    known = cv2.remap(valid.astype(np.float32), safe[..., 0].astype(np.float32),
                      safe[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.) >= 1.-1e-6
    scale = np.maximum(np.std(values, axis=1), .05)
    return (values-values.mean(axis=1, keepdims=True))/scale[:, None], known


def predicted_observations(branches, query_k, reference_pose, up):
    from lunar_geometry_v5 import project_z
    if not 1 <= len(branches) <= MAX_BRANCHES:
        raise ValueError("Bounded persistent beam required")
    inverse = np.linalg.inv(reference_pose)
    photo, geometry, known = [], [], []
    taps = np.array([(x, y) for y in (-16., 0., 16.) for x in (-16., 0., 16.)])
    for branch in branches:
        pool = branch["pool"]
        points = pool.local_world[None] + HEIGHTS[:, None, None]*up
        xy, _, front = project_z(points.reshape(-1, 3), query_k, inverse @ branch["camera"])
        xy = xy.reshape(len(HEIGHTS), -1, 2)
        finite = np.isfinite(xy).all(axis=-1)
        support = front.reshape(len(HEIGHTS), -1) & finite & pool.eligible[None]
        support &= ((xy >= 17.) & (xy <= 494.)).all(axis=-1)
        sample_at = xy[:, :, None, :]+taps
        safe = np.nan_to_num(sample_at, nan=-100., posinf=-100., neginf=-100.)
        # Coverage is reference-map support, never a darkness / query-occlusion label.
        mapped = cv2.remap(branch["extra"]["available"].astype(np.float32),
            safe[..., 0].reshape(-1, 9).astype(np.float32),
            safe[..., 1].reshape(-1, 9).astype(np.float32), cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.).reshape(*support.shape, 9)
        support &= (mapped >= 1.-1e-6).all(axis=-1)
        support &= pool.preview_known.all(axis=1)[None] & branch["preview_known"].all(axis=1)[None]
        delta = np.where(finite[..., None], xy-pool.template_centers[None], 0.)
        response = branch["preview"][None] + np.einsum("ntc,hnc->hnt", pool.preview_gradients, delta)
        photo.append(response); geometry.append(np.where(finite[..., None], xy/8., 0.)); known.append(support)
    return np.asarray(photo), np.asarray(geometry), np.asarray(known)


def selection_scores(branches, query_k, reference_pose, up, weights, mode, separation):
    n = len(branches[0]["pool"].reference_xy)
    if mode == "uniform":
        return np.zeros(n), np.zeros(n, int)
    photo, geometry, known = predicted_observations(branches, query_k, reference_pose, up)
    geometric, count = separation(geometry, known, weights)
    if mode == "geometry":
        return geometric, count
    if mode != "active":
        raise ValueError("Declared uniform, geometry or active selection required")
    structural, photo_count = separation(photo, known, weights)
    return structural+.05*np.minimum(geometric, 25.), np.minimum(count, photo_count)


def search_schedule(pool, field_budget, branch_count, cell_schedule):
    if (type(field_budget) is not int or not 0 <= field_budget <= MAX_SEARCHES
        or type(branch_count) is not int or not 1 <= branch_count <= MAX_BRANCHES):
        raise ValueError("At most128 actual branch-specific patch searches required")
    rounds = field_budget // branch_count
    cells, schedule = cell_schedule(pool, min(rounds, 32))
    # The old helper's32-anchor cap does not restrict one-branch equal-work controls.
    if rounds > 32:
        cells = 4*(pool.reference_xy[:, 1]//128).astype(int)+(pool.reference_xy[:, 0]//128).astype(int)
        counts = Counter(); capacity = Counter(cells[pool.eligible].tolist()); schedule = []
        for _ in range(min(rounds, int(pool.eligible.sum()))):
            options = [c for c in capacity if counts[c] < capacity[c]]
            cell = min(options, key=lambda c: (counts[c], c))
            schedule.append(cell); counts[cell] += 1
    return cells, schedule
