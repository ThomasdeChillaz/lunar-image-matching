"""Image-only affine preconditioning of the frozen native descriptor baseline.

No query camera pose/depth is accepted. The coarse image matches alone define
the warp. This is an engineering control, not a newly learned method.
"""
import cv2
import numpy as np
import torch

from localization_diagnostics_v3 import native_points, refine_native
from localization_robust_pose_v4 import CONFIGS, estimate, residuals
from lunar_geometry_v5 import map_points_z
from rotation_bank_pose_v4 import choose_branch, quarter_turn


def transform_points(points, affine):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return points@np.asarray(affine, dtype=float)[:, :2].T+np.asarray(affine)[:, 2]


def estimate_affine(p, q):
    p, q = [np.asarray(v, dtype=float).reshape(-1, 2) for v in (p, q)]
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite paired coarse points required')
    if len(p) < 6:
        return None
    cv2.setRNGSeed(20261101)
    affine, inliers = cv2.estimateAffine2D(p, q, method=cv2.RANSAC,
        ransacReprojThreshold=12., maxIters=2000, confidence=.99, refineIters=10)
    if affine is None or inliers.sum() < 6 or not np.isfinite(affine).all():
        return None
    singular = np.linalg.svd(affine[:, :2], compute_uv=False)
    if np.linalg.det(affine[:, :2]) <= 0 or singular.min() < .5 or singular.max() > 2:
        return None
    return affine


def rectify(query, affine):
    if query.ndim != 2 or query.dtype != np.uint8 or np.asarray(affine).shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError('Grayscale uint8 image and finite2x3 affine required')
    shape = query.shape[::-1]
    image = cv2.warpAffine(query, affine, shape, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(np.ones(query.shape, np.float32), affine, shape,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= .99999
    support = cv2.erode(mask.astype(np.uint8), np.ones((25, 25), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # Every candidate in a33px window must have a complete valid encoder support.
    complete = cv2.erode(support.astype(np.uint8), np.ones((33, 33), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return image, complete


def infer(coarse, reference, query, reference_depth, reference_intrinsics,
          reference_cam2world, query_intrinsics, features=None, feature_ids=('a', 'b')):
    if (reference.shape != query.shape or reference.ndim != 2 or reference.shape[0] != reference.shape[1]
            or reference.dtype != np.uint8 or query.dtype != np.uint8):
        raise ValueError('Equal square grayscale uint8 images required')
    if features is None:
        features = {}
    def encoded(image, identity):
        if identity not in features:
            with torch.inference_mode():
                features[identity] = coarse.model(torch.from_numpy(image.copy()).float()[None, None]/255)[0]
        return features[identity]
    da = encoded(reference, (feature_ids[0], 'reference'))
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    attempts = []
    for k in range(4):
        rotated = np.rot90(query, k).copy()
        low = [cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA) for im in (reference, rotated)]
        p, q = [native_points(v, reference.shape) for v in coarse.points(*low)]
        affine = estimate_affine(p, q)
        if affine is None:
            with torch.inference_mode():
                p, q, _ = refine_native(p, q, da, encoded(rotated, (feature_ids[1], k)))
        else:
            normalized, complete = rectify(rotated, affine)
            inverse = cv2.invertAffineTransform(affine)
            normalized_q = transform_points(q, inverse)
            centers = np.rint(normalized_q).astype(int)
            use = ((centers >= 0) & (centers < np.array(complete.shape[::-1]))).all(1)
            safe = centers.clip(0, np.array(complete.shape[::-1])-1)
            use &= complete[safe[:, 1], safe[:, 0]]
            with torch.inference_mode():
                # This pair-specific warp is never reused for another image pair.
                db = coarse.model(torch.from_numpy(normalized.copy()).float()[None, None]/255)[0]
                p, refined, _ = refine_native(p[use], normalized_q[use], da, db)
            q = transform_points(refined, affine)
        q = quarter_turn(q, k, query.shape[0], True)
        world, use = map_points_z(p, reference_depth, reference_intrinsics, reference_cam2world)
        pose, info = estimate(world, q[use], query_intrinsics, config)
        median = None
        if pose is not None:
            r = pose[:3, :3].T; t = -r@pose[:3, 3]
            error = residuals(world, q[use], np.asarray(query_intrinsics), cv2.Rodrigues(r)[0], t)
            median = float(np.median(error[error <= 3])) if (error <= 3).any() else float('inf')
        attempts.append(dict(p=p, q=q, pose=pose, info=info, quarter_turn=k,
            median_inlier_residual=median, affine=None if affine is None else affine.tolist()))
    return choose_branch(attempts), attempts
