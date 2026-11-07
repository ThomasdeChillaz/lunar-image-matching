import unittest

import numpy as np

from north_camera_variation_v59 import rolled_pose
from north_render_plan_v31 import camera_to_world
from warp_candidates import (
    ARM_NAMES, ProjectionBank, ReferenceCloud, build_reference_cloud, common_cells, fitted_plane,
    rank_pose_bank, reference_pose_bank, splat_detector,
)


def cameras():
    pose = np.array(camera_to_world([0., 0., 100.], [0., 0., 0.]))
    k = np.array([[100., 0., 63.5], [0., 100., 63.5], [0., 0., 1.]])
    return pose, k


class WarpCandidateTests(unittest.TestCase):
    def test_reference_bank_is_deterministic_and_preserves_camera_convention(self):
        p = np.array(rolled_pose([20., -45., 1250.], [10., 5., 0.], 14.))
        bank = reference_pose_bank(p, count=16)
        np.testing.assert_array_equal(bank, reference_pose_bank(p, count=16))
        np.testing.assert_array_equal(bank[0], p)
        self.assertEqual(len(bank), 16)
        self.assertTrue(np.all(np.abs(bank[:, :3, 3] - p[:3, 3]) <= 500.))
        self.assertTrue(np.all(bank[:, 2, 2] <= -np.cos(np.deg2rad(52.))))
        np.testing.assert_allclose(np.linalg.det(bank[:, :3, :3]), 1.)

    def test_reference_depth_and_flat_control_share_source_rays(self):
        p, k = cameras()
        image = np.ones((32, 32))
        depth = np.full_like(image, 100.)
        depth[:, 16:] = 96.
        cloud = build_reference_cloud(image, depth, k, p, stride=8)
        np.testing.assert_array_equal(cloud.world[:, 2].reshape(4, 4), np.tile([0., 0., 4., 4.], (4, 1)))
        normal, offset = fitted_plane(cloud.world)
        np.testing.assert_allclose(cloud.flat_world @ normal, offset)
        self.assertGreater(np.ptp(cloud.flat_world[:, 2]), 1.)
        np.testing.assert_array_equal(cloud.radiance, 1.)
        np.testing.assert_allclose(cloud.world[:, :2] / (100. - cloud.world[:, 2:]),
                                   cloud.flat_world[:, :2] / (100. - cloud.flat_world[:, 2:]))

    def test_nearest_depth_including_unknown_radiance_owns_detector_bin(self):
        world = np.array([[2., 2., 2.], [1., 1., 1.], [3., 0., 1.], [3., 0., 1.],
                          [-.51, 2., 1.], [-.49, 2., 1.], [-1., -1., -1.]])
        values = np.array([5., np.nan, 7., 8., 9., 10., 11.])
        observed = splat_detector(world, values, np.eye(3), np.eye(4), (4, 4), 1)
        self.assertTrue(np.isnan(observed[1, 1]))
        self.assertEqual(observed[0, 3], 7.)
        self.assertEqual(observed[2, 0], 10.)
        self.assertEqual(np.isfinite(observed).sum(), 2)

    def test_known_low_light_shift_ranks_first_at_both_native_cell_scales(self):
        p, k = cameras()
        image = 1e-7 + 1e-6 * np.random.default_rng(20261104).random((8, 8)).repeat(16, 0).repeat(16, 1)
        cloud = build_reference_cloud(image, np.full_like(image, 100.), k, p)
        query = np.full_like(image, 1e-7)
        query[:, :96] = image[:, 32:]
        candidate_bank = np.repeat(p[None], 4, axis=0)
        candidate_bank[:, 0, 3] = [0., -16., 32., 32.]
        for size in (16, 32):
            kwargs = dict(cell_pixels=size, min_cells_per_quadrant=1)
            first = rank_pose_bank(cloud, query, k, candidate_bank, **kwargs)
            gained = rank_pose_bank(cloud, 7. * query + .01, k, candidate_bank, **kwargs)
            self.assertIsNone(first["reason"])
            self.assertEqual(first["orders"]["geometry_3d"][:2], [2, 3])
            self.assertEqual(first["orders"]["planar"][:2], [2, 3])
            self.assertEqual(gained["orders"]["geometry_3d"][:2], [2, 3])
            self.assertAlmostEqual(first["scores"]["geometry_3d"][2], 0., places=12)
            self.assertGreater(first["scores"]["gradient"][2], first["scores"]["geometry_3d"][2])

    def test_two_depth_layers_have_different_measured_image_displacement(self):
        p, k = cameras()
        image = 1e-7 + 1e-6 * np.random.default_rng(42).random((8, 8)).repeat(16, 0).repeat(16, 1)
        depth = np.full_like(image, 100.)
        depth[:64] = 80.
        cloud = build_reference_cloud(image, depth, k, p)
        query = np.full_like(image, 1e-7)
        # A camera shift of 32 m gives 100*32/80=40 px and 100*32/100=32 px.
        query[:64, :88] = image[:64, 40:]
        query[64:, :96] = image[64:, 32:]
        bank = np.repeat(p[None], 3, axis=0)
        bank[:, 0, 3] = [0., 24., 32.]
        for size in (16, 32):
            fixed = np.zeros((128//size, 128//size), dtype=bool)
            fixed[32//size:96//size, 0:64//size] = True
            # Four quadrants of full detector require the right side too.
            fixed[32//size:96//size, 64//size] = True
            result = rank_pose_bank(cloud, query, k, bank, cell_pixels=size, region=fixed,
                                    min_cells_per_quadrant=1)
            self.assertEqual(result["orders"]["geometry_3d"][0], 2)
            self.assertLess(result["scores"]["geometry_3d"][2], result["scores"]["planar"][2])

    def test_common_subbins_exclude_points_seen_by_only_one_arm(self):
        query = np.array([[1., 200.], [3., 400.]])
        real = np.array([[2., np.nan], [4., 500.]])
        planar = np.array([[7., 600.], [9., np.nan]])
        fixed = np.array([[True]])
        observed, actual, control, counts = common_cells(query, real, planar, 2, fixed, 2)
        for result, expected in zip((observed, actual, control, counts), (2., 3., 8., 2), strict=True):
            np.testing.assert_array_equal(result, [expected])
        self.assertIsNone(common_cells(query, real, planar, 2, fixed, 3))

    def test_projection_reuse_is_exact_and_cannot_change_geometry(self):
        p, k = cameras()
        image = np.random.default_rng(3).random((8, 8)).repeat(16, 0).repeat(16, 1)
        depth = np.full_like(image, 100.)
        first = build_reference_cloud(image, depth, k, p)
        second = build_reference_cloud(image + .1, depth, k, p)
        bank = np.repeat(p[None], 3, axis=0)
        bank[:, 0, 3] = [0., 16., 32.]
        cache = ProjectionBank(first, k, bank, image.shape, 4)
        rank_pose_bank(first, image, k, bank, cell_pixels=16, projection_bank=cache)
        cold = rank_pose_bank(second, image, k, bank, cell_pixels=32, min_cells_per_quadrant=1)
        warm = rank_pose_bank(second, image, k, bank, cell_pixels=32, min_cells_per_quadrant=1,
                              projection_bank=cache)
        self.assertEqual(cold, warm)
        depth[4, 4] = 91.
        wrong = build_reference_cloud(image, depth, k, p)
        with self.assertRaisesRegex(ValueError, "must bind"):
            rank_pose_bank(wrong, image, k, bank, cell_pixels=16, projection_bank=cache)

    def test_uniform_invalid_and_inadequate_candidates_fail_closed(self):
        p, k = cameras()
        y, x = np.indices((128, 128))
        image = (x + y).astype(float)
        cloud = build_reference_cloud(image, np.full_like(image, 100.), k, p)
        blank = rank_pose_bank(cloud, image * 0., k, [p], cell_pixels=16)
        self.assertEqual(blank["reason"], "uniform_observation_on_fixed_cells")
        image[64, 64] = np.nan
        invalid = rank_pose_bank(cloud, image, k, [p], cell_pixels=16)
        self.assertEqual(invalid["reason"], "invalid_observation_on_fixed_cells")
        image[64, 64] = 128.
        lacking_flat = ReferenceCloud(cloud.world, np.full_like(cloud.flat_world, np.nan), cloud.radiance)
        unsupported = rank_pose_bank(lacking_flat, image, k, [p], cell_pixels=16)
        self.assertEqual(unsupported["reason"], "insufficient_common_candidate_support")
        for result in (blank, invalid, unsupported):
            self.assertEqual(result["orders"], {name: [] for name in ARM_NAMES})


if __name__ == "__main__":
    unittest.main()
