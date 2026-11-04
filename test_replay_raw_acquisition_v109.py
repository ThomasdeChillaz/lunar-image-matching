"""Pure numeric replay tests. No scientific assets, subprocess rendering or data inputs."""

import struct
import unittest

import numpy as np

import replay_raw_acquisition_v109 as replay


class ReplayTests(unittest.TestCase):
    def test_file_header_rejects_half_float(self):
        def attribute(name, type_, value):
            return name + b"\0" + type_ + b"\0" + struct.pack("<I", len(value)) + value

        def header(pixel_type):
            fields = b"".join(name + b"\0" + struct.pack("<i4xii", pixel_type, 1, 1)
                              for name in (b"B", b"G", b"R"))
            return (b"\x76\x2f\x31\x01\x02\x00\x00\x00"
                    + attribute(b"channels", b"chlist", fields + b"\0")
                    + attribute(b"dataWindow", b"box2i", struct.pack("<iiii", 0, 0, 31, 31))
                    + b"\0")

        self.assertEqual(replay.read_exr_channels(header(2)), (32, 32))
        with self.assertRaisesRegex(ValueError, "float32"):
            replay.read_exr_channels(header(1))

    def test_repeatable_variation_vs_noise_and_support(self):
        first = np.zeros((64, 64, 3))
        first[:32] = 1.
        second = first + .25
        result = replay.seed_statistics(first, second, 32)
        self.assertAlmostEqual(result["across_seed_spatial_covariance"], .25)
        self.assertAlmostEqual(result["seed_difference_single_render_rms"], .25 / np.sqrt(2))
        self.assertAlmostEqual(result["aggregate_span"], 1.)
        masks = replay.seed_masks(first, second)
        self.assertEqual(masks["both_positive"]["count"], 64 * 32)
        self.assertEqual(masks["exactly_one_positive"]["count"], 64 * 32)
        self.assertNotEqual(masks["both_positive"]["packed_row_major_sha256"],
                            masks["exactly_one_positive"]["packed_row_major_sha256"])

    def test_repeat_fails_wrong_original_summary(self):
        first = np.ones((32, 32, 3))
        image = {key: first for key in ("combined", "direct", "indirect")}
        data = {"toy": {64: {0: image, 1: image}}}
        expected = {"toy": {"64": {
            name: {str(block): replay.seed_statistics(first, first, block) for block in (1, 32)}
            for name in image
        }}}
        replay.paired_seed_replay(data, expected)
        expected["toy"]["64"]["indirect"]["32"]["positive_support_fraction"] = .5
        with self.assertRaisesRegex(ValueError, "Independent"):
            replay.paired_seed_replay(data, expected)

    def test_direct_and_indirect_albedo_weighted(self):
        direct = np.full((2, 2, 3), .25)
        indirect = np.full_like(direct, .125)
        albedo = np.full_like(direct, .5)
        total = albedo * (direct + indirect)
        planes = dict(combined=total, albedo=albedo, direct=direct, indirect=indirect)
        reconstructed = replay.reconstructed(planes, {"recon_max_abs_error": 0.})
        self.assertTrue(np.array_equal(reconstructed["direct"], .5 * direct))
        self.assertTrue(np.array_equal(reconstructed["indirect"], .5 * indirect))


if __name__ == "__main__":
    unittest.main()
