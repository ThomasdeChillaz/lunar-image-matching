import unittest

import numpy as np

from handoff_patch import PATCH, search_patch


def texture(size, seed):
    values = np.random.default_rng(seed).random((size//8, size//8))
    return 1e-7 + 1e-6 * values.repeat(8, 0).repeat(8, 1)


def shifted(image, dx, dy):
    out = np.full_like(image, 1e-7)
    h, w = image.shape
    x = slice(max(dx, 0), w+min(dx, 0))
    y = slice(max(dy, 0), h+min(dy, 0))
    sx = slice(max(-dx, 0), w-min(max(dx, 0), w))
    sy = slice(max(-dy, 0), h-min(max(dy, 0), h))
    out[y, x] = image[sy, sx]
    return out


def put(image, xy, patch):
    x, y = np.asarray(xy) - PATCH//2
    image[y:y+PATCH, x:x+PATCH] = patch


class HandoffPatchTests(unittest.TestCase):
    def test_wide_faint_gain_offset_translation_outside_narrow_control(self):
        image = texture(320, 20261104)
        query = 7 * shifted(image, 73, -21) + 2e-5
        anchor = np.array([144.25, 144.75])
        wide = search_patch(image, query, anchor, anchor)
        narrow = search_patch(image, query, anchor, anchor, radius=16)
        self.assertIsNone(wide["reason"])
        np.testing.assert_allclose(wide["query_xy"], anchor + [73, -21])
        self.assertEqual(wide["round_trip_error_px"], 0.)
        self.assertEqual(wide["max_total_positions"], 2628)
        self.assertLessEqual(wide["evaluated_positions"], 2628)
        self.assertIsNone(narrow["query_xy"])
        self.assertEqual(narrow["max_total_positions"], 500)
        self.assertLessEqual(narrow["evaluated_positions"], 500)

    def test_narrow_control_has_same_nested_native_refinement(self):
        image = texture(160, 4)
        anchor = (80, 80)
        result = search_patch(image, shifted(image, 9, -3), anchor, anchor, radius=16)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["query_xy"], [89., 77.])
        self.assertEqual(result["forward"]["coarse"]["offset_xy"], [8, 0])
        self.assertEqual(result["forward"]["fine"]["offset_xy"], [9, -3])
        self.assertGreater(result["score"], .99)

    def test_outer_boundary_and_repeated_texture_fail_closed(self):
        image = texture(192, 14)
        anchor = (96, 96)
        at_boundary = search_patch(image, shifted(image, 16, 0), anchor, anchor, radius=16)
        self.assertEqual(at_boundary["reason"], "forward_boundary_peak")
        pattern = np.tile(texture(32, 15), (6, 6))
        ambiguous = search_patch(pattern, pattern, anchor, anchor)
        self.assertEqual(ambiguous["reason"], "forward_ambiguous_peak")
        for result in (at_boundary, ambiguous):
            self.assertIsNone(result["query_xy"])

    def test_reference_holes_query_holes_and_blank_reject(self):
        image = texture(160, 16)
        anchor = (80, 80)
        sparse = np.zeros_like(image, dtype=bool)
        for source, mask_a, mask_b, reason in (
            (image, sparse, None, "insufficient_fixed_reference_support"),
            (image, None, sparse, "forward_insufficient_target_support_or_texture"),
            (np.zeros_like(image), None, None, "forward_uniform_template"),
        ):
            with self.subTest(reason=reason):
                result = search_patch(source, image, anchor, anchor, radius=16,
                                      reference_valid=mask_a, query_valid=mask_b)
                self.assertEqual(result["reason"], reason)
                self.assertIsNone(result["query_xy"])

    def test_reverse_finds_decoy_and_blocks_nonreciprocal_peak(self):
        ref = texture(320, 1)
        query = texture(320, 2)
        pattern = texture(64, 3)
        perturbed = pattern + .6 * texture(64, 4)
        anchor = (128, 128)
        put(ref, anchor, perturbed)
        put(ref, (224, 128), pattern)
        put(query, (160, 128), pattern)
        result = search_patch(ref, query, anchor, anchor)
        self.assertEqual(result["forward"]["fine"]["offset_xy"], [32, 0])
        self.assertEqual(result["reverse"]["fine"]["offset_xy"], [64, 0])
        self.assertEqual(result["round_trip_error_px"], 96.)
        self.assertEqual(result["reason"], "nonreciprocal_peak")
        self.assertIsNone(result["query_xy"])


if __name__ == "__main__":
    unittest.main()
