"""Analytic reference geometry only. No rendered images or project data."""

import math
import unittest

import numpy as np

from solar_secondary_atlas_v111 import (
    TerrainPatches,
    Visibility as V,
    build_atlas,
    classify_segment,
    project_atlas,
)


def terrain(centers, normals, *, areas=None, mapped=None, complete=None, albedo=None):
    count = len(centers)
    return TerrainPatches(
        centers, normals,
        np.ones(count) if areas is None else areas,
        np.full(count, .5) if albedo is None else albedo,
        np.ones(count, bool) if mapped is None else np.asarray(mapped, bool),
        np.ones(count, bool) if complete is None else np.asarray(complete, bool),
    )


def between(state):
    return [[V.BLOCKED, state], [state, V.BLOCKED]]


def pinhole():
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 4.
    k = np.array([[64., 0., 31.5], [0., 64., 31.5], [0., 0., 1.]])
    return pose, k


def ray_fixture_hit(origin, ray, limit):
    """The off-axis horizontal blocker occupies z=1, x=[-.9,-.6], y=[-.1,.1]."""
    if ray[2] == 0:
        return None
    distance = (1. - origin[2]) / ray[2]
    if 0 < distance <= limit:
        point = origin + distance * ray
        if -.9 <= point[0] <= -.6 and -.1 <= point[1] <= .1:
            return distance
    return None


def full_coverage(_origin, _direction, length):
    return length


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.receiver_reflector = terrain([[0., 0., 0.], [2., 0., 1.]],
                                          [[0., 0., 1.], [-1., 0., 0.]])
        self.sun = np.array([-.6, 0., .8])

    def test_direct_blocked_reflector_analytic_irradiance_and_projection(self):
        patch = self.receiver_reflector
        solar = []
        for point, normal in zip(patch.centers, patch.normals):
            start = point + .001 * (self.sun + normal)
            solar.append(classify_segment(start, start + 10. * self.sun, ray_fixture_hit, full_coverage))
        receiver, reflector = (patch.centers + .001 * patch.normals)
        mutual = classify_segment(receiver, reflector, ray_fixture_hit, full_coverage)
        self.assertEqual(solar, [V.BLOCKED, V.CLEAR])
        self.assertEqual(mutual, V.CLEAR)
        atlas = build_atlas(patch, self.sun, solar, between(mutual), solar_irradiance=2.)
        self.assertTrue(atlas.secondary_known.all())
        self.assertEqual(atlas.direct_irradiance[0], 0.)
        self.assertAlmostEqual(atlas.direct_irradiance[1], 1.2)
        # Reflector cosine at the receiver is 2/sqrt(5); receiver cosine is 1/sqrt(5).
        self.assertAlmostEqual(atlas.secondary_irradiance[0], .048 / math.pi)
        self.assertAlmostEqual(atlas.secondary_radiance[0], .024 / math.pi**2)
        self.assertEqual(atlas.contributing_pairs.tolist(), [1, 0])

        projection = project_atlas(atlas, *pinhole(), shape=(64, 64))
        self.assertEqual(projection.owner[31, 31], 0)
        self.assertTrue(projection.valid[31, 31])
        self.assertAlmostEqual(projection.radiance[31, 31], .024 / math.pi**2)
        self.assertGreater(int(projection.block_known_pixels.sum()), 0)
        self.assertEqual(projection.block_known_pixels.shape, (2, 2))
        self.assertTrue(np.isnan(projection.radiance[~projection.valid]).all())

    def test_zero_incident_light_is_known_zero(self):
        patch = terrain(self.receiver_reflector.centers, self.receiver_reflector.normals,
                        complete=[False, False])
        atlas = build_atlas(patch, self.sun, [V.UNKNOWN, V.UNKNOWN], between(V.UNKNOWN),
                            solar_irradiance=0.)
        self.assertTrue(atlas.direct_known.all() and atlas.secondary_known.all())
        self.assertFalse(np.any(atlas.secondary_irradiance))
        projection = project_atlas(atlas, *pinhole(), shape=(64, 64))
        self.assertTrue(np.any(projection.valid))
        self.assertTrue(np.all(projection.radiance[projection.valid] == 0))

    def test_unknown_solar_mutual_or_external_coverage_are_not_clear(self):
        patch = self.receiver_reflector
        source_unknown = build_atlas(patch, self.sun, [V.BLOCKED, V.UNKNOWN], between(V.CLEAR))
        mutual_unknown = build_atlas(patch, self.sun, [V.BLOCKED, V.CLEAR], between(V.UNKNOWN))
        external_unknown = build_atlas(terrain(patch.centers, patch.normals, complete=[False, True]),
                                       self.sun, [V.BLOCKED, V.CLEAR], between(V.CLEAR))
        for atlas in (source_unknown, mutual_unknown, external_unknown):
            self.assertFalse(atlas.secondary_known[0])
            projection = project_atlas(atlas, *pinhole(), shape=(64, 64))
            self.assertEqual(projection.owner[31, 31], 0)
            self.assertFalse(projection.valid[31, 31])
            self.assertTrue(np.isnan(projection.radiance[31, 31]))
        self.assertGreater(external_unknown.secondary_irradiance[0], 0.)
        blocked = build_atlas(patch, self.sun, [V.BLOCKED, V.UNKNOWN], between(V.BLOCKED))
        self.assertTrue(blocked.secondary_known[0])
        self.assertEqual(blocked.secondary_irradiance[0], 0.)

    def test_segment_loses_support_before_clearance(self):
        origin, end = [0., 0., 0.], [2., 0., 0.]
        def hole(_origin, _direction, _length):
            return .75
        no_hit = lambda *_: None
        self.assertEqual(classify_segment(origin, end, no_hit, hole), V.UNKNOWN)
        self.assertEqual(classify_segment(origin, end, lambda *_: .5, hole), V.BLOCKED)
        self.assertEqual(classify_segment(origin, end, lambda *_: 1.2, hole), V.UNKNOWN)

    def test_off_fov_creates_no_support(self):
        patch = terrain([[100., 0., 0.]], [[0., 0., 1.]])
        atlas = build_atlas(patch, [0., 0., 1.], [V.CLEAR], [[V.BLOCKED]])
        projection = project_atlas(atlas, *pinhole(), shape=(64, 64), channel="direct")
        self.assertFalse(np.any(projection.valid))
        self.assertTrue(np.all(projection.owner == -1))
        self.assertFalse(np.any(projection.block_projected_pixels))

    def test_unknown_foreground_occludes_valid_background(self):
        patch = terrain([[0., 0., 0.], [0., 0., 1.]], [[0., 0., 1.]] * 2,
                        areas=[16., 1.], mapped=[True, False])
        atlas = build_atlas(patch, [0., 0., 1.], [V.CLEAR, V.UNKNOWN], between(V.CLEAR))
        projection = project_atlas(atlas, *pinhole(), shape=(64, 64), channel="direct")
        self.assertEqual(projection.owner[31, 31], 1)
        self.assertFalse(projection.valid[31, 31])
        self.assertEqual(projection.owner[5, 5], 0)
        self.assertTrue(projection.valid[5, 5])
        self.assertGreater(projection.block_projected_pixels.sum(), projection.block_known_pixels.sum())

    def test_equal_depth_tie_and_additional_camera_coverage(self):
        patch = terrain([[0., 0., 0.], [0., 0., 0.]], [[0., 0., 1.]] * 2)
        atlas = build_atlas(patch, [0., 0., 1.], [V.CLEAR] * 2, between(V.BLOCKED))
        projection = project_atlas(atlas, *pinhole(), shape=(64, 64), channel="direct",
                                   camera_covered=np.array([False, True]))
        self.assertEqual(projection.owner[31, 31], 0)
        self.assertFalse(np.any(projection.valid))
        self.assertGreater(projection.block_projected_pixels.sum(), 0)

    def test_close_facing_patches_are_unresolved(self):
        source = self.receiver_reflector
        patch = terrain(source.centers, source.normals, areas=[1., 16.])
        atlas = build_atlas(patch, self.sun, [V.BLOCKED, V.CLEAR], between(V.CLEAR))
        self.assertEqual(atlas.missing_pairs[0], 1)
        self.assertFalse(atlas.secondary_known[0])
        self.assertEqual(atlas.secondary_irradiance[0], 0.)


if __name__ == "__main__":
    unittest.main()
