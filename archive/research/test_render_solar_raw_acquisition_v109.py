"""Pure checks. No Blender invocation, optical data processing or compute reservation."""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import render_solar_raw_acquisition_v109 as raw


class RawAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = raw.read(raw.COHORT)
        cls.signal = raw.read(raw.SIGNAL)
        cls.rows = raw.select_rows(cls.cohort, cls.signal)

    def test_exact_case_roles_and_seeds(self):
        self.assertEqual(len(self.rows), 8)
        self.assertNotIn("u025/reference/light-1", [row["key"] for row in self.rows])
        timing = raw.schedule(self.rows, "timing")
        study = raw.schedule(self.rows, "study", extended=True)
        self.assertEqual((len(timing), len(study)), (4, 28))
        self.assertEqual(sum(row["scene"] == "legacy_white" for row in study), 8)
        black = [row["seed"] for row in timing + study if row["scene"] == "black_space"]
        self.assertEqual(len(black), len(set(black)))
        self.assertTrue(all(row["key"].startswith("u061/") for row in study if row["samples"] == 256))

    def test_wrong_role_join(self):
        self.signal["items"][0]["light_index"] = -1
        try:
            with self.assertRaisesRegex(ValueError, "join"):
                raw.select_rows(self.cohort, self.signal)
        finally:
            self.signal["items"][0]["light_index"] = 0

    def test_timing_admission_and_fixtures(self):
        rows = raw.schedule(self.rows, "timing")
        for row in rows:
            row["elapsed_seconds"] = {16: 4., 64: 13.}[row["samples"]]
        timing = dict(schema=1, phase="timing", complete=True, calibration_or_final_access=False,
                      cpu_threads=2, renderer_sha256=raw.BLENDER_SHA,
                      fixtures={"passed": True}, rows=rows)
        self.assertLess(raw.timing_basis(timing, raw.schedule(self.rows, "study", extended=True))
                        ["conservative_study_seconds"], 900)
        rows[0]["elapsed_seconds"] = 100
        with self.assertRaisesRegex(ValueError, "admit"):
            raw.timing_basis(timing, raw.schedule(self.rows, "study"))

        black, blocked, bright, half = (dict(combined_max=0., direct_max=0., indirect_mean=1e-3,
                                             combined_mean=2., direct_mean=.2) for _ in range(4))
        half["combined_mean"] = 1.
        values = dict(zip(raw.FIXTURE_NAMES, (black, blocked, bright, half)))
        raw.check_fixture(values)
        blocked["direct_max"] = 1.
        with self.assertRaisesRegex(ValueError, "fixture"):
            raw.check_fixture(values)

    def test_reconstruction_and_independent_seed_statistics(self):
        albedo = np.full((2, 2, 3), .5)
        direct, indirect = np.full_like(albedo, .2), np.full_like(albedo, .1)
        planes = dict(albedo=albedo, direct=direct, indirect=indirect, combined=(direct + indirect) * albedo)
        self.assertEqual(raw.check_components(planes), 0.)
        planes["combined"] = direct + indirect
        with self.assertRaisesRegex(ValueError, "differs"):
            raw.check_components(planes)

        arr = np.zeros((raw.SIZE, raw.SIZE, 3))
        arr[:raw.SIZE // 2] = 1
        offset = .125
        stats = raw.compare_seed_planes(arr, arr + offset)
        self.assertAlmostEqual(stats["32"]["seed_difference_single_render_rms"], offset / np.sqrt(2))
        self.assertAlmostEqual(stats["32"]["across_seed_spatial_covariance"], .25)

    def test_raw_exr_header_checks_actual_storage(self):
        def attr(name, kind, data):
            return name + b"\0" + kind + b"\0" + struct.pack("<I", len(data)) + data

        def header(pixel_type, size):
            channels = b"".join(name + b"\0" + struct.pack("<iB3xii", pixel_type, 0, 1, 1)
                                for name in (b"B", b"G", b"R")) + b"\0"
            box = struct.pack("<4i", 0, 0, size - 1, size - 1)
            return (b"\x76\x2f\x31\x01\x02\x00\x00\x00" + attr(b"channels", b"chlist", channels)
                    + attr(b"dataWindow", b"box2i", box) + b"\0")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_header.exr"
            path.write_bytes(header(2, 512))
            raw.check_exr_header(path, 512)
            with self.assertRaisesRegex(ValueError, "window"):
                raw.check_exr_header(path, 64)
            path.write_bytes(header(1, 512))
            with self.assertRaisesRegex(ValueError, "float32"):
                raw.check_exr_header(path, 512)

    def test_metadata_freeze_is_reproducible(self):
        protocol, _, _ = raw.build("timing")
        self.assertEqual(protocol, raw.build("timing")[0])
        self.assertEqual(protocol["split"], "model_selection")
        self.assertEqual(len(protocol["camera_mesh_identity"]), 2)
        self.assertFalse(protocol["calibration_or_final_access"])


if __name__ == "__main__":
    unittest.main()
