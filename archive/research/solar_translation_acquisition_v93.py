"""Opt-in translation-information patch policy using existing geometry only."""

import numpy as np

from diagnose_direct_pose_geometry_v92 import projection_jacobian


# A 33x33 search has a 16px half-width. The 15.5px inscribed circle
# stays inside after any in-plane rotation and at most .5px centre rounding.
INSCRIBED_WINDOW_RADIUS_PX = 15.5


def directional_information_gain(previous, additions):
    """One 2x6 candidate per action, optimizing over nuisance rotation.

    Averages gain across a tied weakest translation eigenspace so that
    arbitrary singular-vector signs or rotations cannot choose the patch.
    Returns None before rotation is numerically observable.
    """
    rotation, translation = previous[:, :3], previous[:, 3:]
    if len(rotation) < 3:
        return None
    u, s, vt = np.linalg.svd(rotation, full_matrices=False)
    tolerance = np.finfo(float).eps * max(rotation.shape) * s[0]
    if s[-1] <= tolerance:
        return None
    # For a fixed translation d, original least-squares rotation is -A @ d.
    whiten = vt.T / s[None, :]
    a = whiten @ (u.T @ translation)
    profiled = translation - u @ (u.T @ translation)
    _, st, weak_axes = np.linalg.svd(profiled, full_matrices=False)
    tolerance_t = np.finfo(float).eps * max(profiled.shape) * max(st[0], s[0])
    weak = weak_axes[st <= st[-1] + tolerance_t].T

    # Additional least-squares energy after re-optimizing rotation is
    # e.T @ (I + R_new @ inv(R.T @ R) @ R_new.T)^-1 @ e.
    new_rotation = additions[:, :, :3]
    whitened = new_rotation @ whiten
    correction = np.eye(2) + whitened @ whitened.transpose(0, 2, 1)
    delta = (additions[:, :, 3:] - new_rotation @ a) @ weak
    gain = np.sum(delta * np.linalg.solve(correction, delta), axis=(1, 2)) / weak.shape[1]
    return np.maximum(gain, 0.)


def translation_patch_scores(graph, beams, actions, intrinsics, unavailable,
                             reference_range_m, crop_supported=None):
    """Worst-of-seven directional gain per action, averaged over beam weights.

    Beam assignments provide the already measured native candidates. Their
    nominal inlier set is recomputed with the original 3px rule at the current
    pose. No pixels, descriptors, query truth, or new measurements are used.
    The caller keeps its existing unavailability policy and patch budget.
    """
    actions = np.asarray(actions, int)
    unavailable = np.asarray(unavailable, bool)
    k = np.asarray(intrinsics, float)
    world = np.asarray(graph.world_variants, float)
    if (actions.ndim != 2 or actions.shape[1] != 2 or unavailable.shape != (len(actions),)
            or world.ndim != 3 or world.shape[1:] != (7, 3) or not np.isfinite(world).all()
            or k.shape != (3, 3) or not np.isfinite(k).all()
            or not np.isfinite(reference_range_m) or reference_range_m <= 0):
        raise ValueError('Native action masks, seven finite map variants, and positive reference range required')
    if crop_supported is None:
        crop_supported = np.ones(len(actions), bool)
    crop_supported = np.asarray(crop_supported, bool)
    if crop_supported.shape != unavailable.shape:
        raise ValueError('Crop support must describe the existing action grid')
    available = np.flatnonzero(~unavailable & crop_supported)
    scores = np.zeros(len(actions))
    if not len(available) or not beams:
        return scores
    anchor, candidate = actions[available].T
    centres = graph.query_xy[anchor, candidate]
    log_weights = np.asarray([beam['score'] for beam in beams], float)
    if not np.isfinite(log_weights).all():
        raise ValueError('Finite current beam scores required')
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    for beam, weight in zip(beams, weights, strict=True):
        pose = beam['pose']
        assignment = beam['assignment']
        measured = np.asarray(assignment.get('native_keep', []), int)
        if pose is None or not len(measured) or weight == 0:
            continue
        pose = np.asarray(pose, float)
        camera = (world - pose[:3, 3]) @ pose[:3, :3]
        h = camera @ k.T
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = h[..., :2] / h[..., 2, None]
        front = ((camera[..., 2] > 0) & np.isfinite(xy).all(-1)
                 & ((xy >= 0) & (xy <= 511)).all(-1))
        measured_q = graph.query_xy[measured, assignment['native_candidate'][measured]]
        inliers = measured[front[measured, 0] & (np.linalg.norm(xy[measured, 0] - measured_q, axis=1) <= 3.)]
        if len(inliers) < 2:
            continue
        reachable = (front[anchor].all(axis=1) & ~np.isin(anchor, inliers)
                     & (np.linalg.norm(xy[anchor] - centres[:, None], axis=2)
                        <= INSCRIBED_WINDOW_RADIUS_PX).all(axis=1))
        usable = np.flatnonzero(reachable)
        if not len(usable):
            continue
        unique, inverse = np.unique(anchor[usable], return_inverse=True)
        robust_gain = np.full(len(unique), np.inf)
        for variant in range(7):
            support = inliers[front[inliers, variant]]
            previous = projection_jacobian(camera[support, variant], k, reference_range_m)
            additions = projection_jacobian(camera[unique, variant], k, reference_range_m).reshape(-1, 2, 6)
            gain = directional_information_gain(previous, additions)
            if gain is None:
                robust_gain.fill(0.)
                break
            robust_gain = np.minimum(robust_gain, gain)
        scores[available[usable]] += weight * robust_gain[inverse]
    return scores


def choose_translation_patch(graph, beams, actions, intrinsics, unavailable,
                             reference_range_m, crop_supported=None):
    """Return one existing action, or None to request the original policy."""
    scores = translation_patch_scores(graph, beams, actions, intrinsics, unavailable,
                                      reference_range_m, crop_supported)
    if not len(scores) or scores.max() <= 0:
        return None
    return int(np.argmax(scores))
