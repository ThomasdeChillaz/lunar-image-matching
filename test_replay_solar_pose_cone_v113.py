"""Procedural scoring and content-boundary checks for the v113 metric replay."""

import math
import unittest
from unittest.mock import patch

import numpy as np

import replay_solar_pose_cone_v113 as replay


def camera(theta=0., x=0.):
    angle = math.radians(theta)
    c, s = math.cos(angle), math.sin(angle)
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, :3] = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]]) @ pose[:3, :3]
    pose[0, 3] = x
    return pose


def scores(candidates, final=None, positive=True, truth=None):
    return replay.measure_slot(candidates, final, None, None, positive, camera() if truth is None else truth, 100.)


class ConeReplayTests(unittest.TestCase):
    def test_camera_axes_and_cone_boundary(self):
        for degrees in (0., 21., 51.999, 52., 52.001, 90., 180.):
            with self.subTest(degrees=degrees):
                result = replay.feasibility(camera(degrees))
                self.assertAlmostEqual(result["off_nadir_deg"], degrees, places=8)
                self.assertEqual(result["within_simulator_cone"], degrees <= 52.)

    def test_correctness_and_physical_recall_are_separate(self):
        # The historical error tolerance can label a pose outside the prior
        # correct when the camera truth itself is close to the boundary.
        truth = camera(51.5)
        outside = scores([camera(52.5)], truth=truth)
        self.assertTrue(outside["oracle_correct"])
        self.assertFalse(outside["physical_oracle_correct"])
        self.assertEqual(outside["cone_feasible_count"], 0)
        raw_miss = scores([camera(x=2.), camera()], final=None)
        self.assertTrue(raw_miss["physical_oracle_correct"])
        self.assertFalse(raw_miss["raw_top1"]["correct"])
        self.assertFalse(raw_miss["compact_final"]["pose_returned"])

    def test_recall_gate_requires_both_positive_seeds_and_no_negative_increase(self):
        baseline = [scores([camera(100.)]), scores([camera(110.)]),
                    scores([camera(120.)], positive=False), scores([camera(130.)], positive=False)]
        cone = [scores([camera()]), scores([camera()]),
                scores([camera(30.)], positive=False), scores([], positive=False)]
        rows = [dict(positive=i < 2, v112=a, cone_v113=b) for i, (a, b) in enumerate(zip(baseline, cone))]
        result = replay.summarize(rows)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["arms"]["cone_v113"]["physical_candidate_oracle_recall"], 2)
        self.assertEqual(result["arms"]["cone_v113"]["compact_final"]["positive_false_negatives"], 2)
        self.assertEqual(result["arms"]["cone_v113"]["raw_top1"]["raw_nonoverlap_outputs"], 1)
        self.assertEqual(result["paired_positive_changes"]["raw_top1"], {"gains": 2, "losses": 0})
        rows[1]["cone_v113"] = scores([])
        self.assertFalse(replay.summarize(rows)["gate"]["both_positive_banks_contain_correct_pose"])
        rows[2]["cone_v113"] = scores([], final=camera(30.), positive=False)
        self.assertFalse(replay.summarize(rows)["gate"]["compact_final_nonoverlap_outputs_did_not_increase"])

    def test_pose_stage_diagnostics_recount_unconstrained_later_poses(self):
        diagnostic = dict(schema="solar_blind_cone_diagnostics_v113", cone_degrees=52.,
                          initial_v73=dict.fromkeys(replay.INITIAL_COUNTERS, 0),
                          native_v84={stage: dict.fromkeys(replay.NATIVE_COUNTERS, 0)
                                      for stage in ("native_ap3p", "native_lm")},
                          pre_final_beam_poses=2, pre_final_off_nadir_poses=1,
                          ordinary_final_off_nadir=True, ordinary_final_iterations_reported=2000)
        replay.verify_diagnostics(diagnostic, [camera(10.), camera(100.)], camera(70.))
        diagnostic["pre_final_off_nadir_poses"] = 0
        with self.assertRaisesRegex(ValueError, "outcomes differ"):
            replay.verify_diagnostics(diagnostic, [camera(10.), camera(100.)], camera(70.))

    def test_no_protected_truth_when_sealed_cone_or_opaque_hash_fails(self):
        with patch.object(replay, "verify_cone", side_effect=ValueError("broken cone seal")):
            with patch.object(replay, "old_and_truth") as truth:
                with self.assertRaisesRegex(ValueError, "broken cone seal"):
                    replay.replay("sha")
        truth.assert_not_called()
        with patch.object(replay, "verify_cone", return_value=({}, [])), \
                patch.object(replay, "verify_dispatch", return_value={
                    "opaque_input_sha256": {}, "metadata_sha256": {}}), \
                patch.object(replay, "verify_opaque_lineage", side_effect=ValueError("changed exposure")), \
                patch.object(replay, "old_and_truth") as truth:
            with self.assertRaisesRegex(ValueError, "changed exposure"):
                replay.replay("sha")
        truth.assert_not_called()


if __name__ == "__main__":
    unittest.main()
