"""Metadata-only admission controls and generated independent score arithmetic."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import control_parallax_v126 as control
import score_parallax_v126 as scorer


def receipt(seconds):
    arms = {name: {"passed": True, "patch_searches": 12} for name in control.ARMS}
    return dict(schema="solar_parallax_procedural_v126", passed=True, procedural_only=True,
                archived_lunar_images_opened=False, native_shape=[512, 512], row_wall_seconds=seconds,
                rows=[dict(passed=True, multidepth=bool(i), arms=arms, row_wall_seconds=value)
                      for i, value in enumerate(seconds)])


class PrepTests(unittest.TestCase):
    def test_scope_is_frozen_before_lunar_measurements(self):
        self.assertEqual(control.forecast_scope(receipt([5., 5.]))[0], 16)
        self.assertEqual(control.forecast_scope(receipt([10., 10.]))[0], 8)
        for invalid in (receipt([15., 15.]), receipt([True, 1.]), receipt([0., 1.])):
            with self.assertRaises(ValueError):
                control.forecast_scope(invalid)
        for count in (8, 16):
            pairing = control.pairing_grid(count)
            self.assertEqual(len(pairing), count)
            self.assertEqual(pairing[:4], [["exposure-0", a, b] for a in (0, 1) for b in (0, 1)])
        self.assertEqual({row[0] for row in control.pairing_grid(8)}, {"exposure-0", "exposure-1"})

    def test_timing_preparation_pins_failed_prior_without_reading_lunar_arrays(self):
        actual = control.legacy.sha

        def metadata_only(path):
            p = Path(path)
            self.assertNotIn(p.suffix, (".npz", ".exr", ".png"))
            self.assertNotEqual(p, control.PREDICTIONS)
            return actual(path)

        with patch.object(control.legacy, "sha", side_effect=metadata_only):
            plan = control.build("timing")
        self.assertTrue(plan["prepared_only"])
        self.assertFalse(plan["v125_gate_passed"])
        self.assertEqual(plan["pairings"], [])
        self.assertEqual(plan["seeds"], {"detector_scale": 32, "ranking": "planar", "max_ordered_hypotheses": 8, "warps": 1})
        self.assertFalse(plan["follow_on"]["automatic"])

    def test_sealed_bank_uses_only_plane_rank_and_no_oracle(self):
        bank = np.repeat(np.eye(4)[None], 9, axis=0)
        bank[:, 0, 3] = np.arange(9)
        row = dict(tag="exposure-0", reference_seed=0, query_seed=0, cell_pixels=32, candidate_count=9,
                   orders={"planar": [8, 7, 6, 5, 4, 3, 2, 1, 0], "geometry_3d": list(range(9))})
        raw = dict(banks={"exposure-0": bank.tolist()}, rows=[row], calibration_or_final_access=False)
        order, chosen = control.seed_bank(raw, "exposure-0", 0, 0)
        self.assertEqual(order, [8, 7, 6, 5, 4, 3, 2, 1])
        np.testing.assert_array_equal(chosen, bank[order])

    def test_independent_support_and_raw_pose_scoring(self):
        xy = scorer.ANCHORS
        selected = [0, 5, 18, 23, 30, 35]
        points = np.column_stack([10. * xy, np.full(len(xy), 10.)])
        peak = {"fine": dict(reason=None, offset_xy=[0, 0], score=.9, margin=.2)}

        def observation(i):
            return dict(anchor_index=i, reference_xy=xy[i], reference_world=points[i], query_xy=xy[i],
                        seed_projection_xy=xy[i], patch_measurement=dict(
                            query_xy=xy[i], reason=None, radius_pixels=16, forward=peak, reverse=peak,
                            round_trip_error_px=0., support_pixels=2500, evaluated_positions=100, max_total_positions=500,
                        ))

        arm = dict(selected_anchors=selected, patch_searches=6, measured_patch_count=6, failures=[], inliers=6,
                   inlier_indices=list(range(6)), raw_pose=np.eye(4).tolist(), reason=None,
                   measurement_seconds=.1, pose_seconds=.2,
                   measurements=[observation(i) for i in selected])
        truth = xy, np.ones(len(xy), dtype=bool), points, np.eye(4), np.eye(3), 10.
        score = scorer.score_arm(arm, truth)
        self.assertTrue(score["correct"] and score["strict"] and score["qualifying_pose"])
        self.assertEqual(score["correct_native_correspondences"]["2"], 6)
        with self.assertRaises(ValueError):
            scorer.score_arm(dict(arm, inliers=5), truth)
        with self.assertRaises(ValueError):
            scorer.score_arm(dict(arm, selected_anchors=list(reversed(selected))), truth)
        broken = {**arm["measurements"][0], "seed_projection_xy": xy[0] + [1., 0.]}
        with self.assertRaises(ValueError):
            scorer.score_arm(dict(arm, measurements=[broken, *arm["measurements"][1:]]), truth)

    def test_two_faint_wins_and_no_paired_losses(self):
        miss, hit = dict(correct=False, qualifying_pose=False), dict(correct=True, qualifying_pose=True)
        rows = [dict(pairing=pair, arms=dict(parallax=hit if i < 2 else miss, uniform=miss))
                for i, pair in enumerate(control.pairing_grid(8))]
        self.assertTrue(scorer.feasibility(rows)["passed"])
        rows[-1]["arms"]["uniform"] = hit
        self.assertFalse(scorer.feasibility(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
