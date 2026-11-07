"""Bind a structured local-ENU reference mesh and its ray support to the v111 atlas."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from solar_map_v63 import grid_envelope
from solar_physics_v63 import TerrainEnvelope, classify_solar_ray, unit_vector
from solar_secondary_atlas_v111 import Atlas, TerrainPatches, Visibility, build_atlas, classify_segment


MAX_TILES_PER_AXIS = 16
MAX_RAY_DISTANCE_M = 50_000.


def mesh_array_sha256(vertices, faces):
    """Canonical hash for the exact node and face arrays passed to the ray backend."""
    digest = hashlib.sha256(b"solar-secondary-reference-enu-v112\0")
    for array, dtype in ((vertices, "<f8"), (faces, "<i8")):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FiniteScene:
    """An explicit assertion that the frozen triangles are the entire modeled scene."""

    mesh_sha256: str
    maximum_height_m: float


@dataclass(frozen=True)
class SunBinding:
    atlas: Atlas
    solar_states: np.ndarray
    reason_counts: dict[str, int]
    traced_rays: int


@dataclass(frozen=True)
class ReferenceBinding:
    terrain: TerrainPatches
    ray_normals: np.ndarray
    selected_faces: np.ndarray
    envelope: TerrainEnvelope
    mutual_states: np.ndarray
    mesh_sha256: str
    finite_scene: FiniteScene | None
    epsilon_m: float
    ray_cap_m: float
    mutual_rays: int
    first_hit: Callable[[np.ndarray, np.ndarray, float], float | None]

    def bind_sun(self, sun_direction, *, solar_irradiance=1.):
        """Construct one atlas for a hypothesized local-ENU Sun direction."""
        sun = unit_vector(sun_direction)
        if sun[2] <= 0:
            raise ValueError("The declared candidate Sun must be above the local horizon")
        envelope = self.envelope
        if self.finite_scene is not None:
            # Outside the exact finite render mesh, the declared scene is vacuum.
            # Make a per-Sun support rectangle that covers the capped ray segments.
            xy = self.terrain.centers[:, :2]
            limits = np.concatenate((xy, xy + self.ray_cap_m * sun[:2]))
            low = limits.min(axis=0) - 8 * self.epsilon_m
            high = limits.max(axis=0) + 8 * self.epsilon_m
            envelope = TerrainEnvelope(low[0], high[0], low[1], high[1],
                                       self.finite_scene.maximum_height_m, True)

        reasons = Counter()
        states = np.empty(len(self.terrain.centers), dtype=np.uint8)
        traced = 0

        def first(origin, direction, limit):
            nonlocal traced
            traced += 1
            return self.first_hit(origin, direction, limit)

        def coverage(_origin, _direction):
            return self.ray_cap_m

        for index, (point, surface, coarse) in enumerate(zip(
                self.terrain.centers, self.ray_normals, self.terrain.normals)):
            if np.dot(coarse, sun) <= 0:
                state, reason = Visibility.BLOCKED, "coarse_surface_faces_away"
            elif np.dot(surface, sun) <= 0:
                # The quadrature facet and aggregate disagree on illumination.
                state, reason = Visibility.UNKNOWN, "incidence_unresolved_at_patch_scale"
            else:
                ray = classify_solar_ray(point, surface, sun, envelope, first, self.epsilon_m, coverage)
                state, reason = ray["state"], ray["reason"]
            states[index] = state
            reasons[reason] += 1
        states.flags.writeable = False
        atlas = build_atlas(self.terrain, sun, states, self.mutual_states, solar_irradiance=solar_irradiance)
        return SunBinding(atlas, states, dict(reasons), traced)


def _coarse_patches(vertices, faces, grid_size, tiles):
    side = grid_size - 1
    # grid_faces stores the first triangle of every cell, followed by the second.
    cells = np.arange(len(faces)).reshape(2, side, side)
    cuts = np.arange(tiles + 1) * side // tiles
    centers, normals, areas, ray_normals, representative = [], [], [], [], []
    for y0, y1 in zip(cuts[:-1], cuts[1:]):
        for x0, x1 in zip(cuts[:-1], cuts[1:]):
            indices = cells[:, y0:y1, x0:x1].ravel()
            triangles = vertices[faces[indices]]
            a, b, c = triangles.transpose(1, 0, 2)
            crosses = np.cross(b - a, c - a)
            crosses *= np.sign(crosses[:, 2:3])
            double_areas = np.linalg.norm(crosses, axis=1)
            centroids = triangles.mean(axis=1)
            mean = (double_areas @ centroids) / double_areas.sum()
            nearest = np.argmin(np.sum((centroids[:, :2] - mean[:2]) ** 2, axis=1))
            centers.append(centroids[nearest])
            normals.append(unit_vector(crosses.sum(axis=0)))
            ray_normals.append(crosses[nearest] / double_areas[nearest])
            areas.append(.5 * double_areas.sum())
            representative.append(indices[nearest])
    return (np.asarray(centers), np.asarray(normals), np.asarray(areas),
            np.asarray(ray_normals), np.asarray(representative))


def _mutual_table(terrain, ray_normals, first_hit, envelope, finite, epsilon, cap):
    centers = terrain.centers
    points = centers + epsilon * ray_normals
    states = np.full((len(points), len(points)), Visibility.UNKNOWN, dtype=np.uint8)
    np.fill_diagonal(states, Visibility.BLOCKED)
    traced = 0

    def first(origin, direction, limit):
        nonlocal traced
        traced += 1
        return first_hit(origin, direction, limit)

    def coverage(origin, direction, length):
        supported = cap if finite else min(cap, envelope.supported_distance(origin, direction))
        return min(length, supported)

    for left in range(len(points) - 1):
        separation = centers[left + 1:] - centers[left]
        toward_other = separation @ terrain.normals[left]
        toward_left = -np.einsum("ij,ij->i", separation, terrain.normals[left + 1:])
        # A nonfacing exchange has zero weight. It receives no visibility label.
        for right in np.flatnonzero((toward_other > 0) & (toward_left > 0)) + left + 1:
            forward = classify_segment(points[left], points[right], first, coverage)
            if forward == Visibility.BLOCKED:
                state = Visibility.BLOCKED
            else:
                reverse = classify_segment(points[right], points[left], first, coverage)
                if reverse == Visibility.BLOCKED:
                    state = Visibility.BLOCKED
                elif forward == reverse == Visibility.CLEAR:
                    state = Visibility.CLEAR
                else:
                    state = Visibility.UNKNOWN
            states[left, right] = states[right, left] = state
    states.flags.writeable = False
    return states, traced


def prepare_reference_mesh(vertices, faces, first_hit, *, tiles_per_axis=MAX_TILES_PER_AXIS,
                           diffuse_albedo=.5, finite_scene=None, epsilon_m=.001,
                           max_ray_distance_m=MAX_RAY_DISTANCE_M):
    """Build coarse patches and cache candidate-independent mutual visibility.

    Coordinates must already be metres in the v59 scene's local ENU frame.
    first_hit must belong to these exact full-resolution triangles and return
    the positive distance to the nearest hit within the requested segment.
    """
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if (type(tiles_per_axis) is not int or not 1 <= tiles_per_axis <= MAX_TILES_PER_AXIS
            or not math.isfinite(diffuse_albedo) or not 0 <= diffuse_albedo <= 1
            or not math.isfinite(epsilon_m) or epsilon_m <= 0
            or not math.isfinite(max_ray_distance_m) or not 2 * epsilon_m <= max_ray_distance_m <= MAX_RAY_DISTANCE_M
            or not callable(first_hit) or faces.dtype.kind not in "iu"):
        raise ValueError("Invalid reference-ray callback or bounded patch construction settings")
    envelope = grid_envelope(vertices, faces)
    digest = mesh_array_sha256(vertices, faces)
    if finite_scene is not None:
        if (not isinstance(finite_scene, FiniteScene) or finite_scene.mesh_sha256 != digest
                or not math.isfinite(finite_scene.maximum_height_m)
                or finite_scene.maximum_height_m < float(vertices[:, 2].max())):
            raise ValueError("Finite-scene declaration must bind the exact full mesh and a valid ceiling")
    size = math.isqrt(len(vertices))
    centers, normals, areas, ray_normals, selected = _coarse_patches(
        vertices, faces, size, min(size - 1, tiles_per_axis))
    ray_normals.flags.writeable = selected.flags.writeable = False
    count = len(centers)
    terrain = TerrainPatches(centers, normals, areas, np.full(count, diffuse_albedo),
                             np.ones(count, bool), np.full(count, finite_scene is not None, bool))
    mutual, traced = _mutual_table(terrain, ray_normals, first_hit, envelope, finite_scene is not None,
                                  epsilon_m, max_ray_distance_m)
    return ReferenceBinding(terrain, ray_normals, selected, envelope, mutual, digest, finite_scene,
                            epsilon_m, max_ray_distance_m, traced, first_hit)


def prepare_blender_reference(vertices, faces, **kwargs):
    """Build the established Blender BVH directly from the same reference arrays."""
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromPolygons(np.asarray(vertices).tolist(), np.asarray(faces).tolist(),
                              all_triangles=True, epsilon=0.)

    def first_hit(origin, direction, limit):
        point, _, _, distance = bvh.ray_cast(tuple(origin), tuple(direction), float(limit))
        return None if point is None else float(distance)

    return prepare_reference_mesh(vertices, faces, first_hit, **kwargs)
