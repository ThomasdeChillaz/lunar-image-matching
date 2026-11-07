"""Broad pose proposals only for missing initial poses; strict final refinement.

The12px initializer is never returned as a final pose. Every recovered output
must come from the unchanged3px estimator inside surface refinement. These
remain uncalibrated research estimates, not accepted navigation outputs.
"""
import cv2
import numpy as np
import torch
from torch import nn

from coarse_pipeline_v5 import infer as initialize
from localization_robust_pose_v4 import residuals
from lunar_geometry_v5 import map_points_z
from surface_learning_v13 import refine_surface


class DescriptorOnly(nn.Module):
    def __init__(self, coarse):
        super().__init__(); self.base = coarse.model

    def forward(self, image):
        return self.base(image)[0]


class FrozenSurface(nn.Module):
    """Reuse the frozen coarse weights; no duplicate or trained parameters."""
    experiment_variant = 'encoder_3d'

    def __init__(self, coarse):
        super().__init__(); self.encoder = DescriptorOnly(coarse)


def broad_proposal(world, pixels, intrinsics):
    world = np.ascontiguousarray(world, dtype=np.float64).reshape(-1, 3)
    pixels = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 2)
    k = np.asarray(intrinsics, dtype=np.float64)
    if len(world) != len(pixels) or k.shape != (3, 3):
        raise ValueError('Paired3D/2D points and3x3 intrinsics required')
    invalid = dict(inliers=0, proposal_threshold_px=12., reason='insufficient_or_invalid_matches')
    if len(world) < 6 or not np.isfinite(world).all() or not np.isfinite(pixels).all():
        return None, invalid
    center = world.mean(0); obj = world-center
    if np.linalg.matrix_rank(obj, tol=1e-7) < 2:
        return None, dict(invalid, reason='degenerate_geometry')
    cv2.setRNGSeed(20261101)
    try:
        ok, rv, tv, indices = cv2.solvePnPRansac(obj, pixels, k, None,
            iterationsCount=2000, reprojectionError=12., confidence=.99, flags=cv2.SOLVEPNP_AP3P)
        if not ok or indices is None or len(indices) < 6:
            return None, dict(invalid, reason='broad_ransac_failed')
        use = indices.ravel(); rv, tv = cv2.solvePnPRefineLM(obj[use], pixels[use], k, None, rv, tv)
        error = residuals(obj, pixels, k, rv, tv); n = int((error <= 12).sum())
        if n < 6:
            return None, dict(invalid, reason='insufficient_front_facing_support')
        rotation = cv2.Rodrigues(rv)[0]; pose = np.eye(4)
        pose[:3, :3] = rotation.T; pose[:3, 3] = center-rotation.T@tv.ravel()
        return pose, dict(inliers=n, proposal_threshold_px=12., reason=None,
            median_proposal_residual=float(np.median(error[error <= 12])),
            clipped_squared_proposal_cost=float(np.minimum(error**2, 144).sum()))
    except cv2.error:
        return None, dict(invalid, reason='opencv_failure')


def recover(coarse, reference, query, reference_depth, reference_intrinsics,
            reference_cam2world, query_intrinsics, features=None, feature_ids=('a', 'b')):
    """Query truth is absent; the broad proposal only initializes image search."""
    selected, branches = initialize(coarse, coarse.model, reference, query, reference_depth,
        reference_intrinsics, reference_cam2world, query_intrinsics, features, feature_ids)
    fine = FrozenSurface(coarse); proposals = []
    if selected is not None:
        result = refine_surface(fine, reference, query, reference_depth, reference_intrinsics,
            reference_cam2world, query_intrinsics, selected)
        result['recovery'] = dict(attempted=False, recovered=False, final_threshold_px=3.)
        return result, branches
    for branch in branches:
        world, valid = map_points_z(branch['p'], reference_depth, reference_intrinsics, reference_cam2world)
        pose, info = broad_proposal(world, np.asarray(branch['q'])[valid], query_intrinsics)
        proposals.append(dict(branch, pose=pose, info=info))
    valid = [p for p in proposals if p['pose'] is not None]
    fallback = branches[0].copy(); fallback['pose'] = None
    metadata = dict(attempted=True, recovered=False, final_threshold_px=3.,
        broad_branches=[dict(quarter_turn=b['quarter_turn'], estimated=b['pose'] is not None,
            proposal_info=b['info']) for b in proposals])
    if not valid:
        fallback['recovery'] = dict(metadata, reason='no_broad_pose_proposal')
        return fallback, branches
    proposal = max(valid, key=lambda b: (b['info']['inliers'],
        -b['info']['median_proposal_residual'], -b['quarter_turn']))
    result = refine_surface(fine, reference, query, reference_depth, reference_intrinsics,
        reference_cam2world, query_intrinsics, proposal)
    history = result.get('geometry_history', [])
    strict_update = any(h['estimated'] for h in history)
    metadata.update(initial_proposal_cam2world=proposal['pose'].tolist(),
        initial_proposal_quarter_turn=proposal['quarter_turn'], strict_update=bool(strict_update))
    if not strict_update:
        # refine_surface keeps its initial pose if every update fails. That is
        # safe for the historical3px initializer, but forbidden for this12px one.
        result = dict(result, pose=None, info=dict(inliers=0, reason='strict_refinement_failed'))
    result['recovery'] = dict(metadata, recovered=bool(strict_update),
        reason=None if strict_update else 'strict_refinement_failed')
    return result, branches
