"""Reference-only one-bounce diffuse atlas with explicit visibility support."""

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


MAX_PATCHES = 1024


class Visibility(IntEnum):
    UNKNOWN = 0
    CLEAR = 1
    BLOCKED = 2


def _points(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be an N×3 finite array")
    return result


def _boolean(values, size, name):
    result = np.asarray(values)
    if result.shape != (size,) or result.dtype != bool:
        raise ValueError(f"{name} must be a Boolean mask over terrain patches")
    return result


def _readonly(values):
    result = values.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TerrainPatches:
    """Coarse tangent quadrature patches in a shared reference world frame.

    A provisional patch may have geometry but lack map certification. Its
    geometry can still occlude a supported patch in the candidate camera.
    """

    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    albedo: np.ndarray
    mapped: np.ndarray
    transport_coverage: np.ndarray

    def __post_init__(self):
        centers = _points(self.centers, "centers")
        normals = _points(self.normals, "normals")
        count = len(centers)
        if not 0 < count <= MAX_PATCHES or normals.shape != centers.shape:
            raise ValueError(f"Require between 1 and {MAX_PATCHES} matching terrain patches")
        norm = np.linalg.norm(normals, axis=1)
        if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
            raise ValueError("Each terrain patch needs a nonzero normal")
        areas = np.asarray(self.areas, dtype=np.float64)
        albedo = np.asarray(self.albedo, dtype=np.float64)
        if (areas.shape != (count,) or not np.isfinite(areas).all() or np.any(areas <= 0)
                or albedo.shape != (count,) or not np.isfinite(albedo).all()
                or np.any((albedo < 0) | (albedo > 1))):
            raise ValueError("Require positive areas and diffuse reflectance in [0, 1]")
        arrays = dict(centers=centers, normals=normals / norm[:, None], areas=areas, albedo=albedo,
                      mapped=_boolean(self.mapped, count, "mapped"),
                      transport_coverage=_boolean(self.transport_coverage, count, "transport_coverage"))
        for name, values in arrays.items():
            object.__setattr__(self, name, _readonly(values))


@dataclass(frozen=True)
class Atlas:
    """Known contributions can be nonzero even when the total is uncertified."""

    terrain: TerrainPatches
    direct_irradiance: np.ndarray
    secondary_irradiance: np.ndarray
    direct_radiance: np.ndarray
    secondary_radiance: np.ndarray
    direct_known: np.ndarray
    secondary_known: np.ndarray
    missing_pairs: np.ndarray
    contributing_pairs: np.ndarray
    excessive_form_factor: np.ndarray


@dataclass(frozen=True)
class Projection:
    """Raster, nearest terrain owner and equal-detector-area support summaries."""

    radiance: np.ndarray
    valid: np.ndarray
    owner: np.ndarray
    block_mean: np.ndarray
    block_known_pixels: np.ndarray
    block_projected_pixels: np.ndarray


def classify_segment(origin, end, first_hit, coverage_limit):
    """Classify a self-offset mutual ray using reference geometry callbacks.

    first_hit(origin, direction, limit) returns the first positive hit distance
    or None. coverage_limit(origin, direction, length) returns the distance
    to the first unsupported interval, or length if all of it is covered.
    """
    origin, end = _points([origin, end], "segment endpoints")
    vector = end - origin
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise ValueError("Nonzero visibility segment required")
    direction = vector / distance
    coverage = float(coverage_limit(origin, direction, distance))
    if not np.isfinite(coverage) or coverage < 0:
        raise ValueError("Finite nonnegative segment coverage required")
    limit = min(distance, coverage)
    if limit == 0:
        return Visibility.UNKNOWN
    hit = first_hit(origin, direction, limit)
    if hit is not None:
        hit = float(hit)
        if not np.isfinite(hit) or hit <= 0:
            raise ValueError("First hit must be a positive finite distance")
        if hit < distance and hit <= limit:
            return Visibility.BLOCKED
    return Visibility.CLEAR if coverage >= distance else Visibility.UNKNOWN


def _states(values, shape):
    values = np.asarray(values)
    if (values.shape != shape or values.dtype.kind not in "iu"
            or not np.isin(values, [int(state) for state in Visibility]).all()):
        raise ValueError("Explicit unknown/clear/blocked integer ray states required")
    return values


def build_atlas(terrain, sun_direction, solar_states, mutual_states, *, solar_irradiance=1.):
    """Approximate one Lambertian terrain bounce on the supplied finite map.

    mutual_states[i,j] describes the open segment from receiver i to emitter j.
    Generate solar and mutual states against reference geometry. The caller
    declares where this finite terrain represents the relevant contributors
    through transport_coverage. No camera image is used.
    """
    count = len(terrain.centers)
    solar = _states(solar_states, (count,))
    mutual = _states(mutual_states, (count, count))
    if not np.array_equal(mutual, mutual.T) or np.any(np.diag(mutual) != Visibility.BLOCKED):
        raise ValueError("Require symmetric mutual states and a blocked self diagonal")
    sun = np.asarray(sun_direction, dtype=np.float64)
    if (sun.shape != (3,) or not np.isfinite(sun).all() or np.linalg.norm(sun) <= 1e-12
            or not math.isfinite(solar_irradiance) or solar_irradiance < 0):
        raise ValueError("Require a finite Sun direction and nonnegative solar irradiance")
    sun_cosine = np.maximum(0., terrain.normals @ (sun / np.linalg.norm(sun)))
    incident = solar_irradiance * sun_cosine
    direct_known = terrain.mapped & ((solar != Visibility.UNKNOWN) | (incident == 0))
    direct = np.where(direct_known & (solar == Visibility.CLEAR), incident, 0.)
    emitting_radiance = direct * terrain.albedo / np.pi

    relative = terrain.centers[None, :, :] - terrain.centers[:, None, :]
    squared_distance = np.sum(relative * relative, axis=-1)
    receiver_dot = np.einsum("ijk,ik->ij", relative, terrain.normals)
    emitter_dot = -np.einsum("ijk,jk->ij", relative, terrain.normals)
    facing = (receiver_dot > 0) & (emitter_dot > 0)
    radius = np.sqrt(terrain.areas / np.pi)
    separated = squared_distance > (radius[:, None] + radius[None, :]) ** 2
    potential_emitter = (terrain.albedo > 0) & (incident > 0)
    unresolved = facing & potential_emitter[None, :] & (mutual != Visibility.BLOCKED)
    unresolved &= ~(separated & (mutual == Visibility.CLEAR) & direct_known[None, :])
    # A certified shadowed emitter contributes zero regardless of its mutual ray.
    unresolved &= ~((solar == Visibility.BLOCKED) & terrain.mapped)[None, :]
    if solar_irradiance > 0:
        # Provisional geometry cannot prove that an unmapped emitter contributes zero.
        unresolved |= (~terrain.mapped)[None, :] & (mutual != Visibility.BLOCKED)
    exchange = (facing & separated & (mutual == Visibility.CLEAR)
                & terrain.mapped[:, None] & terrain.mapped[None, :])
    factors = np.divide(receiver_dot * emitter_dot * terrain.areas[None, :],
                        np.pi * squared_distance**2, out=np.zeros_like(squared_distance),
                        where=exchange)
    secondary = np.pi * (factors @ emitting_radiance)
    contributing = exchange & (emitting_radiance[None, :] > 0)
    missing_pairs = np.count_nonzero(unresolved, axis=1)
    exceeds_hemisphere = (factors.sum(axis=1) > 1. + 1e-9) & (solar_irradiance > 0)
    secondary_known = terrain.mapped & ~exceeds_hemisphere & (missing_pairs == 0)
    secondary_known &= terrain.transport_coverage | (solar_irradiance == 0)
    diffuse = terrain.albedo / np.pi
    return Atlas(terrain, direct, secondary, diffuse * direct, diffuse * secondary,
                 direct_known, secondary_known, missing_pairs,
                 np.count_nonzero(contributing, axis=1), exceeds_hemisphere)


def _camera(pose, intrinsics):
    pose, intrinsics = (np.asarray(array, dtype=np.float64) for array in (pose, intrinsics))
    if (pose.shape != (4, 4) or intrinsics.shape != (3, 3)
            or not np.isfinite(pose).all() or not np.isfinite(intrinsics).all()):
        raise ValueError("Require finite camera-to-world pose and native intrinsics")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-8)
            or not np.array_equal(pose[3], [0., 0., 0., 1.])
            or not np.array_equal(intrinsics[2], [0., 0., 1.])
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or intrinsics[1, 0] != 0):
        raise ValueError("Require a proper camera-to-world pose and pinhole intrinsics")
    return pose, intrinsics


def _patch_pixels(center, axes, half_side, intrinsics, width, height, near):
    corners = center + half_side * np.array([axes[0] + axes[1], axes[0] - axes[1],
                                             -axes[0] + axes[1], -axes[0] - axes[1]])
    if np.all(corners[:, 2] <= near):
        return None
    if np.any(corners[:, 2] <= near):
        return np.s_[:, :]
    projected = (corners @ intrinsics.T)[:, :2] / corners[:, 2, None]
    limits = np.array([width - 1, height - 1])
    low, high = projected.min(axis=0), projected.max(axis=0)
    if np.any(low > limits) or np.any(high < 0):
        return None
    x0, y0 = np.maximum(0, np.floor(low)).astype(int)
    x1, y1 = np.minimum(limits, np.ceil(high)).astype(int)
    return np.s_[y0:y1+1, x0:x1+1]


def project_atlas(atlas, candidate_pose, intrinsics, *, shape=(512, 512), block=32, channel="secondary",
                  camera_covered=None, near_m=.001):
    """Rasterize reference patches under the candidate pose in native pixels.

    Opaque tangent squares have area equal to each coarse quadrature patch.
    Nearest positive-depth intersection wins. A tie within 1 nm keeps the
    smaller patch index. An unmapped, unlit or uncertified front patch still
    occludes. camera_covered can impose additional reference-only ray support.
    The ranker can use radiance and valid directly at native resolution.
    """
    terrain = atlas.terrain
    if (type(block) is not int or block <= 0 or len(shape) != 2
            or any(type(value) is not int or not 0 < value <= 512 or value % block for value in shape)
            or not math.isfinite(near_m) or near_m <= 0
            or channel not in ("direct", "secondary", "combined")):
        raise ValueError("Require valid block-aligned native dimensions, channel and near plane")
    pose, k = _camera(candidate_pose, intrinsics)
    rotation = pose[:3, :3]
    centers = (terrain.centers - pose[:3, 3]) @ rotation
    normals = terrain.normals @ rotation
    helper = np.eye(3)[np.argmin(np.abs(terrain.normals), axis=1)]
    tangent = np.cross(helper, terrain.normals)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    tangents = np.stack([tangent, np.cross(terrain.normals, tangent)], axis=1) @ rotation
    height, width = shape
    y, x = np.indices(shape)
    rays = np.stack([x, y, np.ones(shape)], axis=-1) @ np.linalg.inv(k).T
    depth = np.full(shape, np.inf)
    owner = np.full(shape, -1, dtype=np.int32)
    front = np.einsum("ij,ij->i", centers, normals) < 0
    for index, (center, normal, axes, area) in enumerate(zip(centers, normals, tangents, terrain.areas)):
        half_side = np.sqrt(area) / 2
        region = _patch_pixels(center, axes, half_side, k, width, height, near_m)
        if region is None:
            continue
        directions = rays[region]
        denominator = directions @ normal
        hit = np.divide(center @ normal, denominator, out=np.zeros_like(denominator),
                        where=np.abs(denominator) > 1e-12)
        offset = hit[..., None] * directions - center
        inside = (np.abs(offset @ axes.T) <= half_side + 1e-10).all(axis=-1)
        closer = inside & (hit > near_m) & (hit < depth[region] - 1e-9)
        depth[region][closer] = hit[closer]
        owner[region][closer] = index

    known = {"direct": atlas.direct_known, "secondary": atlas.secondary_known,
             "combined": atlas.direct_known & atlas.secondary_known}[channel]
    values = {"direct": atlas.direct_radiance, "secondary": atlas.secondary_radiance,
              "combined": atlas.direct_radiance + atlas.secondary_radiance}[channel]
    covered = (terrain.mapped if camera_covered is None
               else _boolean(camera_covered, len(centers), "camera_covered") & terrain.mapped)
    projected = owner >= 0
    indices = owner.clip(min=0)
    valid = projected & (known & front & covered)[indices]
    radiance = np.where(valid, values[indices], np.nan)
    blocks = (height // block, block, width // block, block)
    known_pixels = valid.reshape(blocks).sum(axis=(1, 3))
    projected_pixels = projected.reshape(blocks).sum(axis=(1, 3))
    totals = np.where(valid, radiance, 0.).reshape(blocks).sum(axis=(1, 3))
    means = np.divide(totals, known_pixels, out=np.full_like(totals, np.nan), where=known_pixels > 0)
    return Projection(radiance, valid, owner, means, known_pixels, projected_pixels)
