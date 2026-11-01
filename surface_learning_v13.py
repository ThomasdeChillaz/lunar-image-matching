"""Development-only surface-normalized fine matching.

Training may use true query geometry. This inference interface deliberately
accepts only reference geometry and an image-estimated initial query pose.
The frozen v11 algorithm remains unchanged for historical replay.
"""
import cv2
import numpy as np
import torch

from geometry_warp_native_v11 import map_refined, surface_warp
from localization_diagnostics_v3 import refine_native
from localization_robust_pose_v4 import CONFIGS, estimate, residuals
from lunar_geometry_v5 import correspondences_z, map_points_z
from native_refinement_v7 import refine
from train_balanced_localization_v3 import appearance

EXPERIMENTS = ('raw_encoder', 'normalized_encoder', 'normalized_pair')


def prepare_training(tiles, data):
    """Identical visible integer reference points in raw/normalized ablations.

    Integer reference centers avoid interpolating depth across different rays.
    Query crop centers are fractional; categorical labels retain that offset.
    No model-selection query pose is consumed here.
    """
    yy, xx = np.mgrid[50:451:16, 50:451:16]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    counts = {}
    for key, tile in tiles.items():
        if tile['split'] != 'train':
            continue
        z = data[key]; a, b = tile['cameras']
        _, targets, _, visible = correspondences_z(z['depth_a'], z['depth_b'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], b['cam2world'], points=points)
        full_target = np.zeros_like(points, dtype=float); full_target[visible] = targets
        normalized = []
        for light in range(3):
            image, complete, _ = surface_warp(z['images'][2*light+1], z['depth_a'],
                a['intrinsics'], a['cam2world'], b['cam2world'], b['intrinsics'])
            normalized.append(image)
        # 57px support plus +/-12.5px crop jitter and interpolation margin.
        common = cv2.erode(complete.astype(np.uint8), np.ones((27, 27), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        use = visible & ((full_target >= 44) & (full_target < 468)).all(1)
        use &= common[points[:, 1], points[:, 0]]
        if use.sum() < 8:
            raise ValueError('Insufficient common raw/normalized training support')
        z['surface_reference'] = points[use].astype(np.float32)
        z['surface_target'] = full_target[use].astype(np.float32)
        z['surface_normalized_images'] = np.stack(normalized)
        counts[key] = dict(component=tile['component'], points=int(use.sum()))
    return dict(tiles=len(counts), groups=len({c['component'] for c in counts.values()}),
                points=sum(c['points'] for c in counts.values()), per_tile=counts)


def training_batch(tiles, data, rng, size, normalized):
    groups = {}
    for tile in tiles.values():
        if tile['split'] == 'train':
            groups.setdefault(tile['component'], []).append(tile['tile_id'])
    names = sorted(groups); arrays = [[] for _ in range(5)]; metadata = []
    for _ in range(size):
        group = names[int(rng.integers(len(names)))]; ids = groups[group]
        key = ids[int(rng.integers(len(ids)))]; z = data[key]
        index = int(rng.integers(len(z['surface_reference'])))
        p, target = z['surface_reference'][index], z['surface_target'][index]
        la, lb = rng.integers(0, 3, 2)
        jitter = rng.integers(-12, 13, 2)+rng.uniform(-.5, .5, 2)
        exact = p if normalized else target
        center = exact+jitter
        query = z['surface_normalized_images'][lb] if normalized else z['images'][2*lb+1]
        ref_patch = cv2.getRectSubPix(z['images'][2*la], (27, 27), tuple(map(float, p)))
        query_patch = cv2.getRectSubPix(query, (57, 57), tuple(map(float, center)))
        if ((center < 28) | (center >= 484)).any():
            raise ValueError('Unsupported training crop')
        values = [appearance(ref_patch, rng)[None], appearance(query_patch, rng)[None],
                  np.array([13, 13], np.float32), (exact-center+16).astype(np.float32), np.float32(1)]
        for destination, value in zip(arrays, values):
            destination.append(value)
        metadata.append(dict(tile_id=key, component=group, reference_native=p.tolist(),
            target_query_native=target.tolist(), crop_center=center.tolist(), light_a=int(la), light_b=int(lb)))
    return [torch.from_numpy(np.stack(a)) for a in arrays], metadata


def refine_surface(model, reference, query, reference_depth, reference_intrinsics,
                   reference_cam2world, query_intrinsics, initial, reference_features=None):
    """Fixed two updates; no query depth, true pose, or truth-based selection."""
    if initial['pose'] is None:
        return initial.copy()
    model.eval(); current = initial.copy(); original = np.asarray(initial['p'], np.float32)
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    with torch.inference_mode():
        da = reference_features
        if da is None:
            da = model.encoder(torch.from_numpy(reference.copy()).float()[None, None]/255)
        history = []
        for iteration in (1, 2):
            warped, complete, coordinates = surface_warp(query, reference_depth, reference_intrinsics,
                reference_cam2world, current['pose'], query_intrinsics)
            center = np.rint(original).astype(int)
            valid = ((center >= 0) & (center < np.array(complete.shape[::-1]))).all(1)
            safe = center.clip(0, np.array(complete.shape[::-1])-1)
            valid &= complete[safe[:, 1], safe[:, 0]]
            db = model.encoder(torch.from_numpy(warped.copy()).float()[None, None]/255)
            if model.experiment_variant == 'encoder_3d':
                # Preserve the exact v11 deterministic decoder for this ablation.
                p, rectified, _ = refine_native(original[valid], original[valid], da, db)
            else:
                p, rectified, _ = refine(model, original[valid], original[valid], da, db)
            q = map_refined(rectified, coordinates)
            world, supported = map_points_z(p, reference_depth, reference_intrinsics, reference_cam2world)
            pose, info = estimate(world, q[supported], query_intrinsics, config)
            history.append(dict(iteration=iteration, input_estimated_pose=current['pose'].tolist(),
                supported_points=len(p), estimated=pose is not None, pose_info=info))
            if pose is None:
                break
            r = pose[:3, :3].T; t = -r@pose[:3, 3]
            errors = residuals(world, q[supported], np.asarray(query_intrinsics), cv2.Rodrigues(r)[0], t)
            current = dict(p=p, q=q, pose=pose, info=info, quarter_turn=initial['quarter_turn'],
                median_inlier_residual=float(np.median(errors[errors <= 3])))
        current['geometry_history'] = history
    return current
