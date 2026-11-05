"""Reference-only native detector templates for a hypothesized query camera."""

from dataclasses import dataclass
from pathlib import Path
from runpy import run_path

import numpy as np

from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import project_z, world_points_z
# Preserve the exact published coarse experiment implementation. The source
# is in a separate report directory with a non-package hyphenated name.
fitted_plane = run_path(str(Path(__file__).resolve().parents[1] / "solar-warp-v117" / "warp_candidates.py"))["fitted_plane"]


ARMS = ("geometry_3d", "planar")
GRID_512 = (80, 197, 314, 431)
MAX_DIMENSION = 512
MAX_TRIANGLE_EXTENT = 8
RASTER_BATCH = 32768
DEPTH_ABSOLUTE_M = .05
DEPTH_RELATIVE = .01


@dataclass(frozen=True)
class PatchBank:
    reference_xy: np.ndarray
    reference_world: np.ndarray
    predicted_xy: np.ndarray
    template_centres_xy: np.ndarray
    anchor_visible: np.ndarray
    raster: np.ndarray
    detector_valid: np.ndarray
    templates: np.ndarray
    valid: np.ndarray
    diagnostics: dict


def native_anchors_512():
    """Reference x/y integer-centre grid fixed independently of query pixels."""
    return np.asarray([(x, y) for y in GRID_512 for x in GRID_512], dtype=int)


def detector_shape(shape):
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or not 1 < n <= MAX_DIMENSION for n in shape):
        raise ValueError("Native detector shape must have two dimensions of at most 512")
    return tuple(map(int, shape))


def depth_triangles(depth, valid):
    """Reject source quads that span holes or an axial depth discontinuity."""
    ids = np.arange(depth.size, dtype=np.int32).reshape(depth.shape)
    quad = np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]), axis=-1).reshape(-1, 4)
    use = valid.flat[quad].all(axis=1)
    quad = quad[use]
    values = depth.flat[quad]
    near = values.min(axis=1)
    smooth = values.max(axis=1) - near <= DEPTH_ABSOLUTE_M + DEPTH_RELATIVE * near
    quad = quad[smooth]
    return np.stack((quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]), axis=1).reshape(-1, 3)


def anchored_plane(rays, world, reference_pose, anchors, anchor_ids, plane):
    """One fitted normal, with parallel offsets through each fixed true anchor."""
    height, width = rays.shape[:2]
    if plane is None:
        return np.full_like(world, np.nan), np.full((height, width), -1, dtype=np.int16)
    y, x = np.indices((height, width))
    closest = np.full((height, width), np.inf)
    regions = np.full((height, width), -1, dtype=np.int16)
    for i, (ax, ay) in enumerate(anchors):
        distance = (x - ax)**2 + (y - ay)**2
        use = distance < closest
        closest[use] = distance[use]
        regions[use] = i
    normal, _ = plane
    origin = reference_pose[:3, 3]
    offsets = (world.reshape(-1, 3)[anchor_ids] - origin) @ normal
    with np.errstate(invalid="ignore", divide="ignore"):
        z = offsets[regions] / (rays @ normal)
    # Numerically unsafe or backward intersections must not support the control.
    good = np.isfinite(z) & (z > 0)
    planar = np.full_like(world, np.nan)
    planar[good] = rays[good] * z[good, None] + origin
    return planar, regions


def _winners(indices, z, triangle):
    order = np.lexsort((triangle, z, indices))
    sorted_pixels = indices[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    return order[first]


def _rasterize(xy, axial, triangles, shape):
    """Raster nearest known triangle using perspective-correct vertex weights."""
    h, w = shape
    axial = axial.copy()
    axial[~np.isfinite(axial) | (axial <= 0)] = np.nan
    depth = np.full(h * w, np.inf)
    winner = np.full(h * w, -1, dtype=np.int32)
    weights = np.zeros((h * w, 3), dtype=np.float32)
    skipped_extent = 0
    for start in range(0, len(triangles), RASTER_BATCH):
        chunk = triangles[start:start+RASTER_BATCH]
        points = xy[chunk]
        z = axial[chunk]
        good = np.isfinite(points).all(axis=(1, 2)) & np.isfinite(z).all(axis=1)
        ids = start + np.flatnonzero(good)
        points, z = points[good], z[good]
        p0, p1, p2 = points.transpose(1, 0, 2)
        e1, e2 = p1 - p0, p2 - p0
        area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        low = np.maximum(np.ceil(points.min(axis=1) - 1e-7), [0, 0])
        high = np.minimum(np.floor(points.max(axis=1) + 1e-7), [w-1, h-1])
        extent = high - low + 1
        front = (extent > 0).all(axis=1) & (area > 1e-10)
        limited = (extent <= MAX_TRIANGLE_EXTENT).all(axis=1)
        skipped_extent += int(np.count_nonzero(front & ~limited))
        keep = front & limited
        if not np.any(keep):
            continue
        ids, p0, e1, e2, area, z = ids[keep], p0[keep], e1[keep], e2[keep], area[keep], z[keep]
        low, extent = low[keep].astype(np.int32), extent[keep].astype(np.int32)
        for dy in range(int(extent[:, 1].max())):
            for dx in range(int(extent[:, 0].max())):
                trial = np.flatnonzero((dx < extent[:, 0]) & (dy < extent[:, 1]))
                pixel = low[trial] + [dx, dy]
                shift = pixel - p0[trial]
                b1 = (shift[:, 0] * e2[trial, 1] - shift[:, 1] * e2[trial, 0]) / area[trial]
                b2 = (e1[trial, 0] * shift[:, 1] - e1[trial, 1] * shift[:, 0]) / area[trial]
                barycentric = np.column_stack((1. - b1 - b2, b1, b2))
                inside = (barycentric >= -1e-7).all(axis=1)
                if not np.any(inside):
                    continue
                triangle = ids[trial[inside]]
                pixel = pixel[inside]
                inverse = np.maximum(0., barycentric[inside]) / z[trial[inside]]
                denominator = inverse.sum(axis=1)
                at_depth = 1. / denominator
                idx = pixel[:, 1] * w + pixel[:, 0]
                best = _winners(idx, at_depth, triangle)
                idx, at_depth, triangle = idx[best], at_depth[best], triangle[best]
                tied = np.abs(at_depth - depth[idx]) <= 1e-8
                better = (at_depth < depth[idx] - 1e-8) | (tied & (triangle < winner[idx]))
                dst, choice = idx[better], best[better]
                winner[dst], depth[dst] = triangle[better], at_depth[better]
                weights[dst] = inverse[choice] / denominator[choice, None]
    return winner, depth, weights, skipped_extent


def _project_surface(world, triangles, radiance, intrinsics, pose, shape, *, physical_visibility):
    xy, axial, front = project_z(world.reshape(-1, 3), intrinsics, pose)
    h, w = shape
    winner, zbuf, weights, skipped = _rasterize(xy, axial, triangles, shape)
    present = winner >= 0
    occluded = 0
    if physical_visibility:
        # Vertices near rejected source discontinuities still block farther
        # triangles in the corresponding detector bin. They supply no colour.
        inside = (front & np.isfinite(xy).all(axis=1) & (xy >= -.5).all(axis=1)
                  & (xy < [w-.5, h-.5]).all(axis=1))
        xy_int = np.floor(xy[inside] + .5).astype(np.int32)
        vertex_z = np.full(h * w, np.inf)
        np.minimum.at(vertex_z, xy_int[:, 1] * w + xy_int[:, 0], axial[inside])
        covered = np.flatnonzero(present & np.isfinite(vertex_z))
        near = vertex_z[covered]
        blocked = zbuf[covered] > near + DEPTH_ABSOLUTE_M + DEPTH_RELATIVE * near
        present[covered[blocked]] = False
        occluded = int(blocked.sum())
    raster = np.full(h * w, np.nan)
    sampled = np.flatnonzero(present)
    values = radiance[triangles[winner[sampled]]]
    visible = np.isfinite(values).all(axis=1)
    sampled = sampled[visible]
    raster[sampled] = np.sum(values[visible] * weights[sampled], axis=1)
    return raster.reshape(shape), zbuf.reshape(shape), {
        "source_triangles": len(triangles), "known_detector_pixels": len(sampled),
        "overlarge_projected_triangles_excluded": skipped, "nearer_vertex_blocks": occluded,
    }


def detector_templates(raster, common, centres, anchor_visible, patch_pixels):
    size = (len(centres), patch_pixels, patch_pixels)
    templates = np.full(size, np.nan)
    masks = np.zeros(size, dtype=bool)
    h, w = common.shape
    offsets = np.arange(patch_pixels) - patch_pixels//2
    for i in np.flatnonzero(anchor_visible):
        centre = centres[i]
        x, y = centre.astype(int)
        if not 0 <= x < w or not 0 <= y < h:
            continue
        xx, yy = x + offsets, y + offsets
        good_x, good_y = (xx >= 0) & (xx < w), (yy >= 0) & (yy < h)
        source, dest = np.ix_(yy[good_y], xx[good_x]), np.ix_(good_y, good_x)
        templates[i][dest], masks[i][dest] = raster[source], common[source]
    return templates, masks


def visible_anchors(centres, depths, common, true_z, plane_z):
    height, width = common.shape
    inside = (np.isfinite(centres).all(axis=1) & (centres >= 0).all(axis=1)
              & (centres < [width, height]).all(axis=1))
    valid = np.zeros(len(centres), dtype=bool)
    x, y = centres[inside].T.astype(int)
    z = depths[inside]
    tolerance = DEPTH_ABSOLUTE_M + DEPTH_RELATIVE * z
    valid[inside] = (common[y, x] & (np.abs(true_z[y, x] - z) <= tolerance)
                     & (np.abs(plane_z[y, x] - z) <= tolerance))
    return valid


class ReferencePatchWarper:
    """Warp full known support into a candidate detector without query samples.

    Only the later measured-search stage may make a correspondence. A patch
    center returned here is a pose-conditioned prediction, never evidence.
    """

    def __init__(self, reference_luminance, reference_axial_depth, reference_intrinsics,
                 reference_cam2world, anchors_xy, *, patch_pixels=64, reference_valid=None):
        image, depth = np.asarray(reference_luminance, float), np.asarray(reference_axial_depth, float)
        shape = detector_shape(image.shape)
        anchors = np.asarray(anchors_xy, float)
        if (depth.shape != shape or anchors.ndim != 2 or anchors.shape[1] != 2
                or not 1 <= len(anchors) <= 64 or not np.isfinite(anchors).all()
                or not np.array_equal(anchors, np.rint(anchors))
                or not ((anchors >= 0) & (anchors < [shape[1], shape[0]])).all()
                or len(np.unique(anchors, axis=0)) != len(anchors)
                or type(patch_pixels) is not int or not 3 <= patch_pixels <= 64):
            raise ValueError("Aligned reference geometry, distinct integer native anchors and bounded template required")
        if reference_valid is None:
            reference_valid = np.ones(shape, bool)
        mask = np.asarray(reference_valid)
        if mask.shape != shape or mask.dtype != bool:
            raise ValueError("Reference geometry validity must be a native Boolean mask")
        valid = mask & np.isfinite(depth) & (depth > 0)
        self.k, self.reference_pose = camera_check(reference_intrinsics, reference_cam2world)
        self.anchors = anchors.astype(int)
        self.patch_pixels = patch_pixels
        y, x = np.indices(shape)
        xy = np.column_stack((x.ravel(), y.ravel()))
        world_rays = world_points_z(xy, np.ones(len(xy)), self.k, self.reference_pose) - self.reference_pose[:3, 3]
        rays = world_rays.reshape(*shape, 3)
        self.world = rays * np.where(valid, depth, np.nan)[..., None] + self.reference_pose[:3, 3]
        anchor_ids = self.anchors[:, 1] * shape[1] + self.anchors[:, 0]
        self.reference_world = self.world.reshape(-1, 3)[anchor_ids].copy()
        self.triangles = depth_triangles(depth, valid)
        plane_samples = self.world[4::8, 4::8]
        plane = fitted_plane(plane_samples[np.isfinite(plane_samples).all(axis=-1)])
        self.planar_world, regions = anchored_plane(rays, self.world, self.reference_pose, self.anchors, anchor_ids, plane)
        tri_region = regions.flat[self.triangles]
        self.planar_triangles = self.triangles[(tri_region == tri_region[:, :1]).all(axis=1)]
        self.seam_triangles = len(self.triangles) - len(self.planar_triangles)
        self.radiance = np.where(np.isfinite(image) & (image >= 0), image, np.nan).ravel()

    def warp(self, query_intrinsics, hypothesized_cam2world, query_shape):
        shape = detector_shape(query_shape)
        k, pose = camera_check(query_intrinsics, hypothesized_cam2world)
        predicted, anchor_depth, front = project_z(self.reference_world, k, pose)
        predicted[~front | ~np.isfinite(predicted).all(axis=1)] = np.nan
        centre = np.floor(predicted + .5)
        true, true_z, true_info = _project_surface(
            self.world, self.triangles, self.radiance, k, pose, shape, physical_visibility=True)
        plane, plane_z, plane_info = _project_surface(
            self.planar_world, self.planar_triangles, self.radiance, k, pose, shape, physical_visibility=False)
        common = np.isfinite(true) & np.isfinite(plane)
        anchor_visible = visible_anchors(centre, anchor_depth, common, true_z, plane_z)
        banks = {}
        for arm, raster, diagnostics in zip(ARMS, (true, plane), (true_info, plane_info)):
            raster[~common] = np.nan
            template, valid = detector_templates(raster, common, centre, anchor_visible, self.patch_pixels)
            banks[arm] = PatchBank(
                self.anchors.copy(), self.reference_world.copy(), predicted.copy(), centre.copy(),
                anchor_visible.copy(), raster, common.copy(), template, valid,
                {**diagnostics, "shared_detector_pixels": int(common.sum()), "source_seam_triangles_excluded": self.seam_triangles,
                 "geometric_anchors": int(np.isfinite(self.reference_world).all(axis=1).sum()),
                 "shared_visible_anchors": int(anchor_visible.sum())},
            )
        return banks
