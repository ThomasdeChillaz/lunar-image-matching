"""Quantization-aware INITIAL pose hypotheses, never accepted/final poses.

The original proposal grid has 16-native-pixel spacing. Its half-cell square
fits inside a radius sqrt(2)*8 disc, including when a rotation bank rotates the
grid. AP3P minimal-set roots are ranked by consensus with these discs, then
refined against cell centres using that consensus. This avoids selecting only
the small, biased subset that happens to fall within three pixels of a coarse
grid centre. Incorrect appearance proposals remain outliers, not known cells.

This is a matched engineering control, not a learned or novel contribution.
Native refinement and the common final 3-pixel pose estimator stay unchanged.
No query depth, pose, Sun direction or scoring labels are accepted here.
"""
import math
import cv2
import numpy as np

SEED = 20261101
HALF_GRID_NATIVE_PX = 8.
QUANTIZATION_DISC_NATIVE_PX = math.sqrt(2.) * HALF_GRID_NATIVE_PX


def errors(world, pixels, intrinsics, rvec, tvec):
    rotation = cv2.Rodrigues(rvec)[0]
    camera = world @ rotation.T + np.asarray(tvec).reshape(1, 3)
    homogeneous = camera @ intrinsics.T
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = np.linalg.norm(homogeneous[:, :2] / homogeneous[:, 2, None] - pixels, axis=1)
    return np.where((camera[:, 2] > 0) & np.isfinite(residual), residual, np.inf)


def quality(residual):
    """Robust cell-centre likelihood at the declared quantization noise scale.

    Counting disc support alone can favour a biased root over a lower-error
    refit that puts one boundary point just outside its disc. The truncated
    centre loss keeps every coarse observation's measured quantization scale.
    """
    support = residual <= QUANTIZATION_DISC_NATIVE_PX
    gap = np.maximum(residual - QUANTIZATION_DISC_NATIVE_PX, 0.)
    return (-float((np.minimum(residual, QUANTIZATION_DISC_NATIVE_PX)**2).sum()),
            int(support.sum()), -float((np.minimum(gap, 3.)**2).sum()))


def propose(world, pixels, intrinsics, iterations=128):
    """At most `iterations` four-point AP3P sets; keep one raw trial pose."""
    world = np.ascontiguousarray(world, dtype=np.float64)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,) or pixels.shape != (len(world), 2) or intrinsics.shape != (3, 3):
        raise ValueError('Paired Nx3/Nx2 native coordinates and camera intrinsics required')
    if type(iterations) is not int or not 1 <= iterations <= 1024:
        raise ValueError('One bounded initial-hypothesis allocation required')
    if not np.isfinite(intrinsics).all() or not np.isclose(intrinsics[2], [0, 0, 1]).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError('Finite calibrated camera intrinsics required')
    info = dict(stage='initial_quantization_disc_proposal_only',
        iterations_requested=iterations, minimal_sets_executed=0, roots_evaluated=0,
        half_grid_native_px=HALF_GRID_NATIVE_PX,
        quantization_disc_native_px=QUANTIZATION_DISC_NATIVE_PX,
        final_reprojection_threshold_native_px=3., accepted=False,
        inliers=0, reason=None)
    if len(world) < 6 or not np.isfinite(world).all() or not np.isfinite(pixels).all():
        return None, dict(info, reason='insufficient_or_invalid_coarse_matches')
    if ((pixels < 0) | (pixels > 511)).any():
        raise ValueError('Coarse proposal coordinates must use native512 pixels')
    center = world.mean(0)
    obj = np.ascontiguousarray(world - center)
    if np.linalg.matrix_rank(obj, tol=1e-7) < 2:
        return None, dict(info, reason='degenerate_coarse_geometry')
    rng = np.random.default_rng(SEED)
    best = None
    for _ in range(iterations):
        sample = rng.choice(len(obj), 4, replace=False)
        info['minimal_sets_executed'] += 1
        try:
            roots = cv2.solvePnPGeneric(obj[sample], pixels[sample], intrinsics, None, flags=cv2.SOLVEPNP_AP3P)
        except cv2.error:
            continue
        if not roots[0]:
            continue
        for rv, tv in zip(roots[1], roots[2]):
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                continue
            info['roots_evaluated'] += 1
            residual = errors(obj, pixels, intrinsics, rv, tv)
            score = quality(residual)
            if (residual <= QUANTIZATION_DISC_NATIVE_PX).sum() >= 6 and (best is None or score > best[0]):
                best = (score, rv.copy(), tv.copy())
    if best is None:
        return None, dict(info, reason='no_coarse_interval_consensus')
    # Refit the broad quantization consensus, rather than a lucky narrow
    # subset of rounded centres. Reclassification and roots use no labels.
    for _ in range(3):
        residual = errors(obj, pixels, intrinsics, best[1], best[2])
        use = residual <= QUANTIZATION_DISC_NATIVE_PX
        if use.sum() < 6:
            break
        try:
            rv, tv = cv2.solvePnPRefineLM(np.ascontiguousarray(obj[use]),
                np.ascontiguousarray(pixels[use]), intrinsics, None,
                best[1].copy(), best[2].copy())
        except cv2.error:
            break
        score = quality(errors(obj, pixels, intrinsics, rv, tv))
        if score > best[0]:
            best = (score, rv.copy(), tv.copy())
        else:
            break
    rv, tv = best[1:]
    rotation = cv2.Rodrigues(rv)[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = center - rotation.T @ tv.ravel()
    residual = errors(obj, pixels, intrinsics, rv, tv)
    info.update(inliers=int((residual <= QUANTIZATION_DISC_NATIVE_PX).sum()),
        strict3_support_diagnostic=int((residual <= 3).sum()),
        reason='raw_trial_pose_requires_native_refinement_and_final_estimator')
    return pose, info
