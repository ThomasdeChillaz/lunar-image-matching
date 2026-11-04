"""Procedural structured-mesh tests. No project assets, Blender or inference."""

import unittest

import numpy as np

from north_terrain_pilot_v32 import grid_faces
from solar_secondary_atlas_v111 import Visibility as V
from solar_secondary_mesh_v112 import FiniteScene, mesh_array_sha256, prepare_reference_mesh


def grid(axis, height):
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel(), height(x, y).ravel()))
    return points, grid_faces(len(axis))


class TriangleRays:
    """Test-only exact nearest intersections in tiny procedural scenes."""

    def __init__(self, vertices, faces):
        self.starts, ends, third = vertices[faces].transpose(1, 0, 2)
        self.a, self.b = ends - self.starts, third - self.starts
        self.calls = []

    def __call__(self, origin, direction, limit):
        self.calls.append(limit)
        p = np.cross(direction, self.b)
        det = np.einsum("ij,ij->i", self.a, p)
        inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        t = origin - self.starts
        u = inv * np.einsum("ij,ij->i", t, p)
        q = np.cross(t, self.a)
        v = inv * (q @ direction)
        distance = inv * np.einsum("ij,ij->i", self.b, q)
        hits = ((u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
                & (distance > 1e-9) & (distance <= limit))
        return float(distance[hits].min()) if hits.any() else None


class MeshAdapterTests(unittest.TestCase):
    def test_deterministic_patch_area_and_real_ray_origins(self):
        axis = np.array([-4., -2., -1., 0., 2., 4.])
        vertices, faces = grid(axis, lambda x, y: .2 * x - .1 * y)
        rays = TriangleRays(vertices, faces)
        first = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=3)
        second = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=3)
        self.assertEqual(len(first.terrain.centers), 9)
        self.assertEqual(len(np.unique(first.selected_faces)), 9)
        self.assertTrue(np.array_equal(first.selected_faces, second.selected_faces))
        self.assertTrue(np.array_equal(first.terrain.centers, vertices[faces[first.selected_faces]].mean(axis=1)))
        self.assertAlmostEqual(first.terrain.areas.sum(), 64. * np.sqrt(1.05))
        self.assertTrue(np.allclose(first.ray_normals, first.terrain.normals))
        self.assertEqual(first.mesh_sha256, mesh_array_sha256(vertices, faces.astype(np.int64)))

    def test_default_unknown_and_explicit_finite_clear(self):
        vertices, faces = grid(np.arange(-3., 4.), lambda x, _y: np.zeros_like(x))
        rays = TriangleRays(vertices, faces)
        opened = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=3)
        declaration = FiniteScene(opened.mesh_sha256, 0.)
        closed = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=3, finite_scene=declaration)
        for sun in ([0., 0., 1.], [1., 0., .002]):
            default = opened.bind_sun(sun)
            declared = closed.bind_sun(sun)
            self.assertTrue(np.all(default.solar_states == V.UNKNOWN))
            self.assertFalse(default.atlas.secondary_known.any())
            self.assertTrue(np.all(declared.solar_states == V.CLEAR))
            self.assertTrue(declared.atlas.secondary_known.all())
            self.assertFalse(np.any(declared.atlas.secondary_irradiance))
            self.assertLessEqual(declared.traced_rays, len(closed.terrain.centers))

    def test_observed_ridge_shadow_without_certifying_open_exterior(self):
        vertices, faces = grid(np.arange(-3., 4.), lambda x, _y: np.maximum(0., 2. - np.abs(x)))
        rays = TriangleRays(vertices, faces)
        opened = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=6)
        closed = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=6,
                                        finite_scene=FiniteScene(opened.mesh_sha256, 2.))
        test_sun = [.98, 0., .2]
        default, declared = opened.bind_sun(test_sun), closed.bind_sun(test_sun)
        x, y, _ = opened.terrain.centers.T
        west = (x < -2) & (abs(y) < 1)
        east = (x > 2) & (abs(y) < 1)
        self.assertTrue(west.any() and east.any())
        self.assertTrue(np.all(default.solar_states[west] == V.BLOCKED))
        self.assertTrue(np.all(default.solar_states[east] == V.UNKNOWN))
        self.assertTrue(np.all(declared.solar_states[west] == V.BLOCKED))
        self.assertTrue(np.all(declared.solar_states[east] == V.CLEAR))

    def test_mutual_coverage_is_symmetric_and_limited(self):
        vertices, faces = grid(np.arange(-3., 4.), lambda x, _y: np.abs(x) * .4)
        rays = TriangleRays(vertices, faces)
        binding = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=4)
        self.assertTrue(np.array_equal(binding.mutual_states, binding.mutual_states.T))
        self.assertTrue(np.all(binding.mutual_states.diagonal() == V.BLOCKED))
        self.assertTrue(np.any(binding.mutual_states == V.CLEAR))
        self.assertLessEqual(binding.mutual_rays, len(binding.terrain.centers) * (len(binding.terrain.centers) - 1))

        limited = TriangleRays(vertices, faces)
        short = prepare_reference_mesh(vertices, faces, limited, tiles_per_axis=4, max_ray_distance_m=.05)
        self.assertTrue(limited.calls)
        self.assertLessEqual(max(limited.calls), .05)
        self.assertFalse(np.any(short.mutual_states == V.CLEAR))

    def test_finite_valley_returns_certified_bounce_on_direct_dark_patches(self):
        vertices, faces = grid(np.arange(-3., 4.), lambda x, _y: np.abs(x) * .4)
        rays = TriangleRays(vertices, faces)
        declaration = FiniteScene(mesh_array_sha256(vertices, faces), float(vertices[:, 2].max()))
        reference = prepare_reference_mesh(vertices, faces, rays, tiles_per_axis=6, finite_scene=declaration)
        result = reference.bind_sun([.98, 0., .2])
        dark_bounce = (result.atlas.secondary_known & (result.atlas.direct_radiance == 0)
                       & (result.atlas.secondary_radiance > 0))
        self.assertTrue(dark_bounce.any())
        self.assertTrue(np.all(result.solar_states[dark_bounce] == V.BLOCKED))
        self.assertTrue(np.all(result.atlas.contributing_pairs[dark_bounce] > 0))

    def test_reject_bad_full_scene_identity_ceiling_and_unstructured_faces(self):
        vertices, faces = grid(np.arange(-2., 3.), lambda x, _y: abs(x))
        rays = TriangleRays(vertices, faces)
        digest = mesh_array_sha256(vertices, faces)
        for declaration in (FiniteScene("0" * 64, 2.), FiniteScene(digest, 1.)):
            with self.assertRaisesRegex(ValueError, "exact full mesh"):
                prepare_reference_mesh(vertices, faces, rays, finite_scene=declaration)
        with self.assertRaisesRegex(ValueError, "terrain"):
            prepare_reference_mesh(vertices, faces[::-1], rays)
        prepared = prepare_reference_mesh(vertices, faces, rays)
        with self.assertRaisesRegex(ValueError, "above"):
            prepared.bind_sun([1., 0., 0.])


if __name__ == "__main__":
    unittest.main()
