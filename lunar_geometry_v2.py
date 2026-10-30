"""Independent ray-depth geometry; never infer reference coordinates from matches."""
import cv2
import numpy as np


def camera_check(k, pose):
    k, pose = np.asarray(k, float), np.asarray(pose, float)
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError('Finite 3x3 intrinsics and 4x4 camera-to-world pose required')
    if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Invalid perspective camera')
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError('Camera rotation must be proper and orthonormal')
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous camera pose')
    return k, pose


def world_points(points, ranges, k, pose):
    """EXR depth is distance along a ray, NOT optical-axis z depth."""
    k, pose = camera_check(k, pose)
    points, ranges = np.asarray(points, float), np.asarray(ranges, float)
    if points.ndim != 2 or points.shape[1] != 2 or ranges.shape != (len(points),):
        raise ValueError('Expected Nx2 pixels and N ray ranges')
    rays = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(k).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return (rays * ranges[:, None]) @ pose[:3, :3].T + pose[:3, 3]


def project(world, k, pose):
    k, pose = camera_check(k, pose)
    xyz = (np.asarray(world, float) - pose[:3, 3]) @ pose[:3, :3]
    xyh = xyz @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = xyh[:, :2] / xyh[:, 2:3]
    return pixels, np.linalg.norm(xyz, axis=1), xyz[:, 2] > 0


def correspondences(depth_a, depth_b, k_a, pose_a, k_b, pose_b, points=None, stride=4,
                    relative_tolerance=.01, absolute_tolerance=.05):
    """Reference-only unprojection followed by query-depth visibility testing.

    Tolerances concern simulated depth discretization, not a sensor accuracy claim.
    Target depth is bilinearly sampled only where all four finite neighbors exist.
    """
    da, db = np.asarray(depth_a, float), np.asarray(depth_b, float)
    if da.ndim != 2 or db.ndim != 2:
        raise ValueError('Single-channel depth maps required')
    if points is None:
        yy, xx = np.mgrid[0:da.shape[0]:stride, 0:da.shape[1]:stride]
        points = np.column_stack([xx.ravel(), yy.ravel()])
    points = np.asarray(points, float)
    ia = np.rint(points).astype(int)
    inside = ((ia >= 0) & (ia < np.array(da.shape[::-1]))).all(1)
    idx = ia.clip(0, np.array(da.shape[::-1])-1)
    ra = da[idx[:, 1], idx[:, 0]]
    valid = inside & np.isfinite(ra) & (ra > 0)
    world = world_points(points, np.where(valid, ra, 1), k_a, pose_a)
    target, predicted_range, front = project(world, k_b, pose_b)
    valid &= front & np.isfinite(target).all(1)
    valid &= ((target >= 0) & (target < np.array(db.shape[::-1])-1)).all(1)
    safe = np.where(np.isfinite(target), target, -1).astype(np.float32)
    sample = cv2.remap(np.where(np.isfinite(db), db, 0).astype(np.float32),
                       safe[:, 0, None], safe[:, 1, None], cv2.INTER_LINEAR).ravel()
    support = cv2.remap((np.isfinite(db) & (db > 0)).astype(np.float32),
                        safe[:, 0, None], safe[:, 1, None], cv2.INTER_LINEAR).ravel()
    valid &= support >= .9999
    valid &= np.abs(sample-predicted_range) <= absolute_tolerance+relative_tolerance*predicted_range
    return points[valid], target[valid], world[valid], valid


def resize_intrinsics(k, original_shape, resized_shape):
    """OpenCV pixel-center resize: x_new=(x_old+.5)*scale-.5."""
    sy, sx = np.array(resized_shape, float) / np.array(original_shape, float)
    affine = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
    return affine @ np.asarray(k, float)


def footprint_components(records, buffer_m=50.):
    """Join intersecting world XYZ bounds and shared trajectory/source keys."""
    n = len(records)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(records):
        ab = np.asarray(a['world_bounds'], float)
        for j in range(i):
            b = records[j]
            bb = np.asarray(b['world_bounds'], float)
            shared = bool(set(a.get('source_groups', [])) & set(b.get('source_groups', [])))
            overlap = bool(np.all(np.minimum(ab[1], bb[1])+buffer_m >= np.maximum(ab[0], bb[0])))
            if shared or overlap:
                parent[find(i)] = find(j)
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(find(i), []).append(row['pair_group'])
    return [sorted(group) for group in groups.values()]


def pose_error(estimated, reference):
    """Camera-to-world pose comparison; translation in source world units."""
    delta = estimated[:3, :3].T @ reference[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
    return dict(rotation_deg=float(angle), translation_m=float(np.linalg.norm(estimated[:3, 3]-reference[:3, 3])))


def estimate_pose(world, pixels, intrinsics):
    if len(world) < 6:
        return None
    # Center the map to keep large lunar world coordinates well conditioned.
    center = np.mean(world, axis=0)
    ok, r, t, inliers = cv2.solvePnPRansac(np.asarray(world-center, np.float64),
        np.asarray(pixels, np.float64), np.asarray(intrinsics, float), None,
        iterationsCount=2000, reprojectionError=3., confidence=.99, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None
    rotation = cv2.Rodrigues(r)[0]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = center-rotation.T @ t.ravel()
    return pose
