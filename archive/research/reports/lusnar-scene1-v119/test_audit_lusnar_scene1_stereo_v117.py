"""Generated stereo and arithmetic controls. Never opens the archived imagery."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from audit_lusnar_scene1_stereo_v117 import (
    PALETTE_RGB, SemanticPaletteError, StereoPolicy, audit_tracks, semantic_terrain_mask, stereo_tracks,
)


def generated_geometry(*, ray, scale, reversed_rows=True, symmetric=False):
    h, w, f, baseline = 128, 128, 75., .31
    y, x = np.indices((h, w))
    axial = 2.3 + .015*x + .021*y
    if symmetric:
        axial = 2.3 + .015*x + .0003*(y-63.5)**2
    ray_norm = np.sqrt(1. + ((x-63.5)**2 + (y-63.5)**2) / f**2)
    raw = (axial * ray_norm if ray else axial) / scale
    yy, xx = np.mgrid[8:121:6, 8:121:6]
    xx, yy = xx.ravel(), yy.ravel()
    disparity = f * baseline / axial[yy, xx]
    ones = np.ones(len(xx))
    tracks = np.column_stack((xx, yy, xx - disparity, yy + .05, ones * .95, ones * .15, ones * .10))
    return tracks, raw[::-1].copy() if reversed_rows else raw, dict(focal_px=f, baseline_m=baseline)


class StereoControls(unittest.TestCase):
    def test_subpixel_signed_disparity_without_using_depth(self):
        rng = np.random.default_rng(54)
        left = rng.integers(0, 256, (88, 112), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (5, 5), .85)
        transform = np.array([[1., 0., -6.35], [0., 1., .2]])
        right = cv2.warpAffine(left, transform, left.shape[::-1], flags=cv2.INTER_LINEAR)
        valid = np.ones(left.shape, bool)
        right_valid = valid.copy()
        right_valid[:, -8:] = False
        p = replace(StereoPolicy(), grid=12, horizontal=14, min_ncc=.78)
        tracks = stereo_tracks(left, right, valid, right_valid, p)
        self.assertGreater(len(tracks), 20)
        self.assertAlmostEqual(float(np.median(tracks[:, 0] - tracks[:, 2])), 6.35, delta=.15)
        self.assertAlmostEqual(float(np.median(tracks[:, 1] - tracks[:, 3])), -.20, delta=.12)
        self.assertEqual(len(stereo_tracks(left*0, right*0, valid, right_valid, p)), 0)

    def test_identifies_ray_cm_and_bottom_first_storage(self):
        tracks, raw, geometry = generated_geometry(ray=True, scale=.01)
        result = audit_tracks(tracks, raw, StereoPolicy(), **geometry)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["selected_model"], {
            "row_order": "bottom_to_top_storage", "depth": "ray", "units": "centimetre", "scale_m": .01,
        })
        self.assertLess(result["models"][0]["best_centre"]["reprojection_px"]["p90"], .06)

    def test_identifies_axial_metres_and_top_first_storage(self):
        tracks, raw, geometry = generated_geometry(ray=False, scale=1., reversed_rows=False)
        result = audit_tracks(tracks, raw, StereoPolicy(), **geometry)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["selected_model"], {
            "row_order": "top_to_bottom_storage", "depth": "axial", "units": "metre", "scale_m": 1.,
        })

    def test_reports_ambiguity_and_rejects_bad_rectification(self):
        tracks, raw, geometry = generated_geometry(ray=True, scale=1., symmetric=True)
        result = audit_tracks(tracks, raw, StereoPolicy(), **geometry)
        self.assertEqual(result["reason"], "depth_orientation_units_or_ray_axis_ambiguous")
        self.assertIn("row_order", result["ambiguous_fields"])
        for target_axis, adjustment in ((3, 2.), (2, 20.)):
            broken = tracks.copy()
            broken[:, target_axis] += adjustment
            self.assertEqual(audit_tracks(broken, raw, StereoPolicy(), **geometry)["reason"],
                             "documented_rectified_baseline_not_supported")

    def test_enforces_common_terrain_and_depth_support(self):
        tracks, raw, geometry = generated_geometry(ray=False, scale=1.)
        mask = np.zeros(raw.shape, bool)
        self.assertEqual(audit_tracks(tracks, raw, StereoPolicy(), terrain_mask=mask, **geometry)["common_depth_tracks"], 0)
        raw[:64] = np.nan
        self.assertEqual(audit_tracks(tracks, raw, StereoPolicy(), **geometry)["common_depth_tracks"], 0)

    def test_publisher_mask_is_exact_and_unknown_colors_fail_closed(self):
        def bgr(code):
            return [code & 255, (code >> 8) & 255, code >> 16]
        colours = np.array([[bgr(PALETTE_RGB["regolith"]), bgr(PALETTE_RGB["sky"])]], np.uint8)
        terrain, counts = semantic_terrain_mask(colours)
        np.testing.assert_array_equal(terrain, [[True, False]])
        self.assertEqual(counts["sky"], 1)
        colours[0, 1, 0] ^= 1
        with self.assertRaises(SemanticPaletteError):
            semantic_terrain_mask(colours)


if __name__ == "__main__":
    unittest.main()
