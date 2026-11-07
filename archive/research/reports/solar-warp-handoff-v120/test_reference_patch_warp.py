"""Tiny generated raster, alignment, and unknown-coverage controls."""

import unittest

import numpy as np

from lunar_geometry_v5 import project_z, world_points_z
from north_render_plan_v31 import camera_to_world
from reference_patch_warp import ReferencePatchWarper, native_anchors_512


def fixture():
    shape = (40, 56)
    k = np.array([[56., 0., 27.5], [0., 56., 19.5], [0., 0., 1.]])
    ref = np.array(camera_to_world([0., 0., 100.], [0., 0., 0.]))
    anchors = np.array([[10, 10], [38, 10], [10, 29], [38, 29]])
    y, x = np.indices(shape)
    return shape, k, ref, anchors, y, x


class ReferenceWarpTests(unittest.TestCase):
    def test_fixed_native_anchor_order_and_identity_radiance(self):
        shape, k, ref, anchors, y, x = fixture()
        grid = native_anchors_512()
        np.testing.assert_array_equal(grid[0], [80, 80])
        np.testing.assert_array_equal(grid[-1], [431, 431])
        self.assertEqual(grid.shape, (16, 2))
        image = 1e-6 * (x + 2 * y)
        warp = ReferencePatchWarper(image, np.full(shape, 100.), k, ref, anchors, patch_pixels=8)
        views = warp.warp(k, ref, shape)
        a, b = views["geometry_3d"], views["planar"]
        np.testing.assert_array_equal(a.detector_valid, b.detector_valid)
        self.assertGreater(a.detector_valid.sum(), .95 * image.size)
        np.testing.assert_allclose(a.raster[a.detector_valid], image[a.detector_valid], atol=1e-11)
        np.testing.assert_allclose(a.raster, b.raster, atol=1e-11)
        np.testing.assert_array_equal(a.reference_xy, anchors)
        np.testing.assert_allclose(a.predicted_xy, anchors, atol=1e-11)
        np.testing.assert_array_equal(a.template_centres_xy, anchors)
        self.assertTrue(a.anchor_visible.all())
        np.testing.assert_allclose(a.templates[:, 4, 4], image[anchors[:, 1], anchors[:, 0]])

    def test_nonplanar_controls_keep_identical_true_anchor_and_canvas_support(self):
        shape, k, ref, anchors, y, x = fixture()
        image = np.sin(x / 3.) + np.cos(y / 4.) + 3.
        depth = 95. + 4. * np.sin(x / 8.) * np.cos(y / 10.)
        shifted = ref.copy()
        shifted[0, 3] += 14.
        warp = ReferencePatchWarper(image, depth, k, ref, anchors, patch_pixels=10)
        views = warp.warp(k, shifted, shape)
        a, b = views["geometry_3d"], views["planar"]
        truth = world_points_z(anchors, depth[anchors[:, 1], anchors[:, 0]], k, ref)
        projection, _, _ = project_z(truth, k, shifted)
        np.testing.assert_allclose(a.reference_world, truth, atol=1e-11)
        np.testing.assert_array_equal(a.reference_world, b.reference_world)
        np.testing.assert_allclose(a.predicted_xy, projection, atol=1e-11)
        np.testing.assert_array_equal(a.predicted_xy, b.predicted_xy)
        np.testing.assert_array_equal(a.valid, b.valid)
        np.testing.assert_array_equal(a.detector_valid, b.detector_valid)
        self.assertGreater(a.detector_valid.sum(), .65 * image.size)
        self.assertGreater(float(np.nanmax(np.abs(a.raster - b.raster))), .03)

    def test_discontinuity_and_disocclusion_do_not_invent_radiance(self):
        shape, k, ref, anchors, _, x = fixture()
        image = np.where(x < 28, 1., 5.)
        depth = np.where(x < 28, 75., 100.)
        shifted = ref.copy()
        shifted[0, 3] += 20.
        warp = ReferencePatchWarper(image, depth, k, ref, anchors, patch_pixels=8)
        output = warp.warp(k, shifted, shape)["geometry_3d"]
        self.assertTrue(np.isnan(output.raster[20, 13:17]).all())
        known = output.raster[output.detector_valid]
        self.assertGreater(len(known), 500)
        self.assertTrue(np.logical_or(np.isclose(known, 1.), np.isclose(known, 5.)).all())

    def test_missing_depth_radiance_and_out_of_view_are_unknown(self):
        shape, k, ref, anchors, y, x = fixture()
        image = (x + y).astype(float)
        image[25:34, 34:43] = np.nan
        valid = np.ones(shape, bool)
        valid[9:12, 9:12] = False
        warp = ReferencePatchWarper(image, np.full(shape, 100.), k, ref, anchors, patch_pixels=8,
                                    reference_valid=valid)
        views = warp.warp(k, ref, shape)
        for bank in views.values():
            self.assertTrue(np.isnan(bank.reference_world[0]).all())
            self.assertTrue(np.isnan(bank.predicted_xy[0]).all())
            self.assertFalse(bank.valid[0].any())
            self.assertFalse(bank.valid[3].any())
        shifted = ref.copy()
        shifted[0, 3] += 1_000.
        self.assertFalse(warp.warp(k, shifted, shape)["geometry_3d"].detector_valid.any())

    def test_occluded_anchor_keeps_true_geometry_but_cannot_emit_patch(self):
        shape, k, ref, _, _, x = fixture()
        anchors = np.array([[10, 10], [31, 10], [10, 29], [31, 29]])
        image = np.where(x < 28, 1., 5.)
        depth = np.where(x < 28, 50., 100.)
        shifted = ref.copy()
        shifted[0, 3] -= 20.
        warp = ReferencePatchWarper(image, depth, k, ref, anchors, patch_pixels=8)
        views = warp.warp(k, shifted, shape)
        for bank in views.values():
            self.assertTrue(np.isfinite(bank.reference_world[[1, 3]]).all())
            self.assertTrue(np.isfinite(bank.predicted_xy[[1, 3]]).all())
            self.assertFalse(bank.anchor_visible[[1, 3]].any())
            self.assertFalse(bank.valid[[1, 3]].any())


if __name__ == "__main__":
    unittest.main()
