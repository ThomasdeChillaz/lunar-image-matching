"""Shared frozen-descriptor scales for a controlled native refinement study.

This is an engineering hypothesis, not a trained model. Every candidate remains
inside the same 33-original-pixel search. No query depth or true query pose is
accepted. The historical v11 implementation remains unchanged.
"""
import cv2
import numpy as np
import torch

from contrastive_terrain import sample_descriptors
from localization_diagnostics_v3 import refine_native
from localization_refiner_v3 import decode_peak

SCALES = (1, 2, 4)
# Radius12 low-resolution encoder + a full interpolation pixel and averaging
# footprint.56 original pixels conservatively covers all scales through4.
COMMON_SUPPORT_RADIUS = 56


def at_scale(points, scale):
    """Original integer-centre coordinates to a resized image, including .5."""
    return (points+.5)/scale-.5


def encode(encoder, image):
    if image.ndim != 2 or image.dtype != np.uint8 or any(n % 4 for n in image.shape):
        raise ValueError('Grayscale uint8 image with dimensions divisible by4 required')
    features = {}
    with torch.inference_mode():
        for scale in SCALES:
            resized = image if scale == 1 else cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale), interpolation=cv2.INTER_AREA)
            result = encoder(torch.from_numpy(resized.copy()).float()[None, None]/255)
            features[scale] = result[0] if isinstance(result, tuple) else result
    return features


def common_support(p, q, shape):
    p, q = [np.asarray(v, np.float32).reshape(-1, 2) for v in (p, q)]
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite paired Nx2 coordinates required')
    size = np.asarray(shape[::-1]); center = np.rint(q).astype(int)
    return ((p >= COMMON_SUPPORT_RADIUS) & (p < size-COMMON_SUPPORT_RADIUS)).all(1) & (
        (center-16 >= COMMON_SUPPORT_RADIUS) & (center+16 < size-COMMON_SUPPORT_RADIUS)).all(1)


def refine(p, q, reference, query, scales=(1, 2, 4), batch_size=32):
    """Equal-weight cosine scores at fixed native candidates, peak interpolation.

    Native-only decoding calls the historical routine exactly, after applying
    the common border. All profiles therefore have identical eligible points.
    Pixel support is conservative; surface warp validity must also be checked
    by the caller. No darkness-dependent eligibility is imposed.
    """
    scales = tuple(scales)
    if scales not in ((1,), (1, 2), (1, 2, 4)) or batch_size < 1:
        raise ValueError('Use a declared ordered scale profile and positive batch size')
    if set(reference) != set(query) or not set(scales) <= set(reference):
        raise ValueError('Paired descriptor pyramids required')
    h, w = reference[1].shape[-2:]
    for scale in scales:
        a, b = reference[scale], query[scale]
        if a.shape != b.shape or a.ndim != 4 or a.shape[0] != 1 or a.shape[-2:] != (h//scale, w//scale):
            raise ValueError('Descriptor pyramid shapes differ')
    p, q = [np.asarray(v, np.float32).reshape(-1, 2) for v in (p, q)]
    indices = np.where(common_support(p, q, (h, w)))[0]
    if scales == (1,):
        pp, qq, kept = refine_native(p[indices], q[indices], reference[1], query[1])
        return pp, qq, indices[kept]
    device = reference[1].device; offsets = torch.arange(-16, 17, device=device)
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    delta = torch.stack([xx.flatten(), yy.flatten()], -1)
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            keep = indices[start:start+batch_size]
            point = torch.from_numpy(p[keep]).to(device)
            center = torch.from_numpy(np.rint(q[keep]).astype(np.float32)).to(device)
            candidates = center[:, None]+delta[None]
            logits = torch.zeros((len(keep), 1089), device=device)
            for scale in scales:
                a = sample_descriptors(reference[scale], at_scale(point, scale)[None])[0]
                # One feature image, all candidate locations; bounded batches
                # avoid constructing a full dense native-sized pyramid tensor.
                b = sample_descriptors(query[scale], at_scale(candidates.reshape(-1, 2), scale)[None])[0]
                logits += torch.einsum('nc,nkc->nk', a, b.reshape(len(keep), 1089, -1))/len(scales)
            predicted.append((center+decode_peak(logits.reshape(-1, 33, 33))).cpu().numpy())
    return p[indices], np.concatenate(predicted) if predicted else np.empty((0, 2), np.float32), indices


def refine_surface(encoder, reference, query, reference_depth, reference_intrinsics,
                   reference_cam2world, query_intrinsics, initial, scales=(1, 2, 4), reference_features=None):
    """Two image-supported surface updates from an unchanged coarse pose.

    All scale profiles use the same conservative warp/image border and retain
    the last estimated pose if an update cannot be fit. No truth-based choice.
    """
    from geometry_warp_native_v11 import map_refined, surface_warp
    from localization_robust_pose_v4 import CONFIGS, estimate, residuals
    from lunar_geometry_v5 import map_points_z
    if initial is None or initial['pose'] is None:
        return None if initial is None else initial.copy()
    da = reference_features if reference_features is not None else encode(encoder, reference)
    current = initial.copy(); original = np.asarray(initial['p'], np.float32); history = []
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    for iteration in (1, 2):
        warped, complete, coordinates = surface_warp(query, reference_depth, reference_intrinsics,
            reference_cam2world, current['pose'], query_intrinsics)
        # Existing warp uses radius28. Additional44 yields radius72 total:
        # 56px feature support plus the16px candidate-search radius.
        complete = cv2.erode(complete.astype(np.uint8), np.ones((89, 89), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        centers = np.rint(original).astype(int)
        safe = centers.clip(0, np.asarray(complete.shape[::-1])-1)
        use = ((centers >= 0) & (centers < np.asarray(complete.shape[::-1]))).all(1)
        use &= complete[safe[:, 1], safe[:, 0]]
        db = encode(encoder, warped)
        p, rectified, _ = refine(original[use], original[use], da, db, scales)
        q = map_refined(rectified, coordinates)
        world, supported = map_points_z(p, reference_depth, reference_intrinsics, reference_cam2world)
        pose, info = estimate(world, q[supported], query_intrinsics, config)
        history.append(dict(iteration=iteration, supported_points=len(p), estimated=pose is not None,
            input_estimated_pose=current['pose'].tolist(), pose_info=info))
        if pose is None:break
        r = pose[:3, :3].T; t = -r@pose[:3, 3]
        error = residuals(world, q[supported], np.asarray(query_intrinsics), cv2.Rodrigues(r)[0], t)
        current = dict(p=p, q=q, pose=pose, info=info, quarter_turn=initial['quarter_turn'],
            median_inlier_residual=float(np.median(error[error <= 3])))
    current['geometry_history'] = history
    return current
