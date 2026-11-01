"""Pose-guided reference-surface rectification, with no query depth/true pose.

This deterministic control tests whether explicit3D viewpoint normalization
helps the frozen image descriptors. It adds no learned parameters.
"""
import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer as initial_infer
from localization_diagnostics_v3 import refine_native
from localization_robust_pose_v4 import CONFIGS, estimate, residuals
from lunar_geometry_v5 import map_points_z


def surface_warp(query, depth, reference_k, reference_pose, estimated_query_pose, query_k):
    if query.ndim != 2 or query.dtype != np.uint8 or depth.shape != query.shape:
        raise ValueError('Same-size grayscale query and reference axial-depth surface required')
    h, w = depth.shape; y, x = np.mgrid[:h, :w]
    rays = np.column_stack([x.ravel(), y.ravel(), np.ones(h*w)])@np.linalg.inv(np.asarray(reference_k)).T
    ref = np.asarray(reference_pose); pose = np.asarray(estimated_query_pose)
    world = (rays*np.asarray(depth).ravel()[:, None])@ref[:3, :3].T+ref[:3, 3]
    camera = (world-pose[:3, 3])@pose[:3, :3]
    projected = camera@np.asarray(query_k).T
    with np.errstate(divide='ignore', invalid='ignore'):
        coordinates = (projected[:, :2]/projected[:, 2, None]).reshape(h, w, 2)
    valid = (np.isfinite(depth) & (depth > 0) & np.isfinite(coordinates).all(2)
        & (camera[:, 2].reshape(h, w) > 0) & (coordinates >= 0).all(2)
        & (coordinates < [w-1, h-1]).all(2))
    safe = np.where(valid[:, :, None], coordinates, -1).astype(np.float32)
    warped = cv2.remap(query, safe[:, :, 0], safe[:, :, 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Validity describes reference geometry and source-image support, not darkness.
    # Query-depth occlusion is unavailable to inference and is never consulted.
    complete = cv2.erode(valid.astype(np.uint8), np.ones((57, 57), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return warped, complete, safe


def map_refined(points, coordinates):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if not len(points):
        return np.empty((0, 2), dtype=float)
    return np.column_stack([cv2.remap(coordinates[:, :, k], points[:, 0, None], points[:, 1, None],
                                    cv2.INTER_LINEAR).ravel() for k in (0, 1)])


def infer(coarse, reference, query, reference_depth, reference_intrinsics,
          reference_cam2world, query_intrinsics, features=None, feature_ids=('a', 'b')):
    if features is None:
        features = {}
    selected, branches = initial_infer(coarse, coarse.model, reference, query, reference_depth,
        reference_intrinsics, reference_cam2world, query_intrinsics, features, feature_ids)
    if selected is None:
        return None, branches
    current = selected.copy(); original_points = selected['p'].copy(); history = []
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    da = features[(feature_ids[0], 0)]
    for iteration in (1, 2):
        normalized, complete, coordinates = surface_warp(query, reference_depth, reference_intrinsics,
            reference_cam2world, current['pose'], query_intrinsics)
        centers = np.rint(original_points).astype(int)
        use = ((centers >= 0) & (centers < np.array(complete.shape[::-1]))).all(1)
        safe = centers.clip(0, np.array(complete.shape[::-1])-1)
        use &= complete[safe[:, 1], safe[:, 0]]
        with torch.inference_mode():
            db = coarse.model(torch.from_numpy(normalized.copy()).float()[None, None]/255)[0]
            p, rectified, _ = refine_native(original_points[use], original_points[use], da, db)
        q = map_refined(rectified, coordinates)
        world, valid = map_points_z(p, reference_depth, reference_intrinsics, reference_cam2world)
        pose, info = estimate(world, q[valid], query_intrinsics, config)
        history.append(dict(iteration=iteration, input_estimated_pose=current['pose'].tolist(),
            supported_points=len(p), estimated=pose is not None, pose_info=info))
        if pose is None:
            break
        r = pose[:3, :3].T; t = -r@pose[:3, 3]
        errors = residuals(world, q[valid], np.asarray(query_intrinsics), cv2.Rodrigues(r)[0], t)
        median = float(np.median(errors[errors <= 3]))
        current = dict(p=p, q=q, pose=pose, info=info, quarter_turn=selected['quarter_turn'],
            median_inlier_residual=median)
    current['geometry_history'] = history
    return current, branches
