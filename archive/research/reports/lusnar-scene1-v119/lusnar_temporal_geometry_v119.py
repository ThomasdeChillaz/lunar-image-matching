"""Reference geometry hypotheses for the exposed LuSNAR Scene 1 gate."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from lunar_geometry_v5 import median_reference_range_z, project_z, world_points_z

NATIVE = 1024
FOCAL = 610.17784
LEFT_POSITION_BODY = (1., -.155, -1.5)
ORIGINS = {"rover": (0., 0., 0.), "imu_lidar": (1., 0., -1.5),
           "left_optical_centre": LEFT_POSITION_BODY}
# Columns are camera right, down and forward expressed in rover FRD axes.
OPTICAL_TO_ROVER = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
ABSENT_PX = 32.
ABSENT_LOG_DEPTH = .5


@dataclass(frozen=True)
class Convention:
    quaternion_direction: str
    position_origin: str
    pitch_deg: int
    principal_point: float

    @property
    def name(self):
        return f"{self.quaternion_direction}/{self.position_origin}/{self.pitch_deg}/{self.principal_point}"


CONVENTIONS = tuple(Convention(*parts) for parts in product(
    ("as_logged", "transposed"), tuple(ORIGINS), (-20, 20), (511.5, 512.0)))


def attitude(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError("Expected unit wxyz in the released convention")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def left_camera(pose, convention):
    """Interpret logged origin and attitude separately, without fitting parameters."""
    if convention not in CONVENTIONS:
        raise ValueError("Camera convention must be one of the original 24 alternatives")
    xyz = np.asarray(pose["position"], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite original position")
    r_body = attitude(pose["quaternion_wxyz"])
    if convention.quaternion_direction == "transposed":
        r_body = r_body.T
    pitch = np.deg2rad(convention.pitch_deg)
    c, s = np.cos(pitch), np.sin(pitch)
    rotate_pitch = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    offset = np.subtract(LEFT_POSITION_BODY, ORIGINS[convention.position_origin])
    camera = np.eye(4)
    camera[:3, :3] = r_body @ rotate_pitch @ OPTICAL_TO_ROVER
    camera[:3, 3] = xyz + r_body @ offset
    return camera


def intrinsics(centre):
    return np.array([[FOCAL, 0., centre], [0., FOCAL, centre], [0., 0., 1.]])


def half_intrinsics(k):
    """Convert original integer-pixel centres through an uncropped 2× resize."""
    return np.array([[.5, 0., -.25], [0., .5, -.25], [0., 0., 1.]]) @ k


def interpret_depth(storage, specification, k):
    row_order = specification["row_order"]
    if row_order not in ("bottom_to_top_storage", "top_to_bottom_storage"):
        raise ValueError("Expected a qualified PFM row order")
    factor = specification["scale_m"]
    if factor not in (1., .01, .001) or specification["depth"] not in ("axial", "ray"):
        raise ValueError("Expected qualified stereo units and axial or ray convention")
    values = storage[::-1] if row_order == "bottom_to_top_storage" else storage
    depth = values * factor
    if specification["depth"] == "ray":
        y, x = np.indices(depth.shape)
        ray_norm = np.sqrt(1. + ((x - k[0, 2]) / k[0, 0])**2 + ((y - k[1, 2]) / k[1, 1])**2)
        depth /= ray_norm
    return depth


def samples(depth, terrain, xy):
    """Sample a local bilinear stencil, rejecting mixed surfaces and unknown labels."""
    xy = np.asarray(xy, float)
    if (depth.ndim != 2 or depth.shape != terrain.shape or xy.ndim != 2 or xy.shape[1] != 2
            or not np.isfinite(xy).all()):
        raise ValueError("Aligned depth, terrain mask and finite native pixels required")
    h, w = depth.shape
    ix, iy = np.floor(xy).astype(int).T
    inside = (ix >= 0) & (iy >= 0) & (ix < w-1) & (iy < h-1)
    i = np.flatnonzero(inside)
    left, top = ix[i], iy[i]
    x = left[:, None] + [0, 1, 0, 1]
    y = top[:, None] + [0, 0, 1, 1]
    corners = depth[y, x]
    good = terrain[y, x].all(axis=1) & np.isfinite(corners).all(axis=1) & (corners > 0).all(axis=1)
    i, corners = i[good], corners[good]
    good = np.ptp(corners, axis=1) <= .05 + .02 * np.median(corners, axis=1)
    i, corners = i[good], corners[good]
    dx, dy = (xy[i] - np.column_stack((ix[i], iy[i]))).T
    weight = np.column_stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy))
    values = np.full(len(xy), np.nan)
    values[i] = np.sum(weight * corners, axis=1)
    return values


def predicted_alignment(ref_xy, query_xy, ref_z, query_z, k, ref_pose, query_pose):
    # A shared caller-provided observation set keeps methods from improving
    # their scores by silently dropping difficult tracked correspondences.
    n = len(ref_xy)
    points = world_points_z(ref_xy, ref_z, k, ref_pose)
    projected, projected_z, front = project_z(points, k, query_pose)
    valid = (front & np.isfinite(projected).all(axis=1) & np.isfinite(projected_z)
             & (projected >= -.5).all(axis=1) & (projected < NATIVE-.5).all(axis=1))
    pixel_error = np.full(n, ABSENT_PX)
    depth_error = np.full(n, ABSENT_LOG_DEPTH)
    pixel_error[valid] = np.minimum(ABSENT_PX, np.linalg.norm(projected[valid] - query_xy[valid], axis=1))
    depth_error[valid] = np.minimum(ABSENT_LOG_DEPTH, np.abs(np.log(projected_z[valid] / query_z[valid])))
    return dict(points=n, valid_projection_count=int(valid.sum()),
                median_pixel_error=float(np.median(pixel_error)),
                median_absolute_log_depth_error=float(np.median(depth_error)),
                at_2_native_pixels=int(np.count_nonzero(pixel_error <= 2.)),
                at_4_native_pixels=int(np.count_nonzero(pixel_error <= 4.)))


def compare_conventions(ref_xy, query_xy, ref_storage, query_storage, ref_terrain, query_terrain,
                        ref_log, query_log, specification):
    depths = {}
    for centre in (511.5, 512.0):
        k = intrinsics(centre)
        ref = interpret_depth(ref_storage, specification, k)
        query = interpret_depth(query_storage, specification, k)
        depths[centre] = samples(ref, ref_terrain, ref_xy), samples(query, query_terrain, query_xy)
    common = np.logical_and.reduce([np.isfinite(z) for pair in depths.values() for z in pair])
    eligible = int(common.sum())
    xy = ref_xy[common]
    quadrants = np.unique(2 * (xy[:, 1] >= NATIVE / 2) + (xy[:, 0] >= NATIVE / 2))
    if eligible < 24 or len(quadrants) < 3:
        return dict(status="insufficient_common_optical_and_depth_support",
                    observations=len(ref_xy), eligible=eligible, quadrants=len(quadrants), hypotheses={})
    hypotheses = {}
    for convention in CONVENTIONS:
        centre = convention.principal_point
        ref_z, query_z = depths[centre]
        hypotheses[convention.name] = predicted_alignment(
            xy, query_xy[common], ref_z[common], query_z[common], intrinsics(centre),
            left_camera(ref_log, convention), left_camera(query_log, convention))
    return dict(status="scored", observations=len(ref_xy), eligible=eligible,
                quadrants=len(quadrants), hypotheses=hypotheses)


def exact_geometry_pose(reference_depth, query_depth, reference_terrain, query_terrain,
                        reference_pose, query_pose, k):
    """Check the shared estimator with geometry-only target projections."""
    from localization_robust_pose_v4 import CONFIGS, estimate

    h, w = reference_depth.shape
    y, x = np.mgrid[16:h-16:32, 16:w-16:32]
    reference_xy = np.column_stack((x.ravel(), y.ravel()))
    source_z = samples(reference_depth, reference_terrain, reference_xy)
    valid_ref = np.isfinite(source_z)
    reference_xy, source_z = reference_xy[valid_ref], source_z[valid_ref]
    world = world_points_z(reference_xy, source_z, k, reference_pose)
    q, z, front = project_z(world, k, query_pose)
    valid = front & np.isfinite(q).all(axis=1) & np.isfinite(z)
    q, z, world, reference_xy = q[valid], z[valid], world[valid], reference_xy[valid]
    later = samples(query_depth, query_terrain, q)
    visible = np.isfinite(later) & (np.abs(later - z) <= .03 + .01 * later)
    q, world, ref = q[visible], world[visible], reference_xy[visible]
    quadrants = np.unique(2*(ref[:, 1] >= h/2)+(ref[:, 0] >= w/2))
    if len(ref) < 24 or len(quadrants) < 3:
        return dict(eligible=False, passed=False, reason="insufficient_visible_terrain", correspondences=len(ref))
    q_half = (q + .5)/2 - .5
    config = next(row for row in CONFIGS if row["name"] == "ap3p_reclassify")
    pose, detail = estimate(world, q_half, half_intrinsics(k), config)
    if pose is None:
        return dict(eligible=True, passed=False, reason=detail["reason"], correspondences=len(ref))
    ref_range = median_reference_range_z(np.where(reference_terrain, reference_depth, np.nan), k)
    trace = np.trace(pose[:3, :3].T @ query_pose[:3, :3])
    rotation = float(np.degrees(np.arccos(np.clip((trace-1)/2, -1, 1))))
    shift = float(np.linalg.norm(pose[:3, 3]-query_pose[:3, 3]))
    return dict(eligible=True, passed=bool(rotation <= 2. and shift <= .01*ref_range),
                reason=detail["reason"], correspondences=len(ref), rotation_deg=rotation,
                translation_m=shift, valid_reference_range_m=ref_range,
                pose_threshold_image_px=3., target="512 pixel coordinates")
