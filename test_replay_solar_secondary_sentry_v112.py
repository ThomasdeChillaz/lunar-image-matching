"""Pure coverage/scoring checks and the protected replay ordering invariant."""

import unittest
from unittest.mock import patch

import numpy as np

import replay_solar_secondary_sentry_v112 as replay


def translated(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


class SentryReplayTests(unittest.TestCase):
    def test_threshold_native_mask_intersection_and_quadrants(self):
        masks = np.zeros((2, 512, 512), dtype=bool)
        for y, x in ((0, 0), (0, 32), (0, 256), (0, 288), (256, 0), (256, 32), (256, 256)):
            masks[:, y:y + 16, x:x + 32] = True
        masks[0, 256:272, 288:320] = True
        expected = dict(common_pixels=7 * 512, retained_pixels=6 * 512, retained_blocks=6,
                        quadrant_blocks=[2, 2, 2, 1], active_quadrants=[0, 1, 2])
        self.assertEqual(replay.support_statistics(masks), expected)
        masks[0, 0, 0] = False
        reduced = replay.support_statistics(masks)
        self.assertEqual(reduced["active_quadrants"], [1, 2])
        self.assertEqual(reduced["retained_pixels"], 4 * 512)

    def test_coverage_fallback_requires_original_order(self):
        fallback = dict(order=[0, 1], reason="insufficient_shared_quadrant_support",
                        scores=None, quadrant_losses=None)
        self.assertEqual(replay.ordered_mode(fallback, [0, 1], True), [0, 1])
        with self.assertRaisesRegex(ValueError, "cannot change"):
            replay.ordered_mode(dict(fallback, order=[1, 0]), [0, 1], True)
        with self.assertRaisesRegex(ValueError, "coverage failure"):
            replay.ordered_mode(dict(fallback, reason=None), [0, 1], True)

    def test_no_truth_open_on_seal_verification_failure(self):
        with patch.object(replay, "verify_blind", side_effect=ValueError("changed blind seal")):
            with patch.object(replay, "open_postseal_truth") as truth:
                with self.assertRaisesRegex(ValueError, "changed blind seal"):
                    replay.replay("rank", "complete", "watchdog")
        truth.assert_not_called()

    def test_pose_threshold_negative_never_compares_enu_and_empty_stays_failure(self):
        self.assertTrue(replay.pose_score(translated(1.), True, np.eye(4), 100.)["correct"])
        self.assertFalse(replay.pose_score(translated(1.0001), True, np.eye(4), 100.)["correct"])
        negative = replay.pose_score(translated(1.), False, None, 100.)
        self.assertTrue(negative["nonoverlap_output"])
        self.assertIsNone(negative["translation_error_m"])
        self.assertFalse(replay.pose_score(None, True, np.eye(4), 100.)["correct"])

    def test_summary_keeps_both_replicates_and_distinguishes_final_ap3p(self):
        assignments, banks = [], []
        for index, slot in enumerate(replay.SLOTS):
            assignments.append(dict(id=slot, case_id=replay.POSITIVE if index < 2 else replay.NEGATIVE,
                                    replicate=index % 2))
            poses = [translated(2.), translated(0.)] if index in (0, 2) else []
            if poses:
                orders = {method: [0, 1] for method in replay.METHODS if method != "compact_final"}
                orders["combined"] = [1, 0]
            else:
                orders = {method: [] for method in replay.METHODS if method != "compact_final"}
            banks.append(dict(id=slot, poses=poses, orders=orders,
                              final=translated(0.) if index == 1 else None,
                              coverage_pass=bool(poses), reason=None, distinct_poses=len(poses), support=None))
        summary = replay.summarize(replay.score(banks, assignments, np.eye(4), 100.))
        self.assertEqual(summary["positive_case_ids"], 1)
        self.assertEqual(summary["positive_renderer_replicates"], 2)
        self.assertEqual(summary["candidate_oracle_correct"], 1)
        methods = summary["methods"]
        self.assertEqual(methods["original_beam"]["correct_top1"], 0)
        self.assertEqual(methods["combined"]["correct_top1"], 1)
        self.assertEqual(methods["combined"]["gains_vs_original_beam"], 1)
        self.assertEqual(methods["combined"]["nonoverlap_outputs"], 1)
        self.assertEqual(methods["compact_final"]["correct_top1"], 1)
        self.assertEqual(methods["compact_final"]["nonoverlap_outputs"], 0)


if __name__ == "__main__":
    unittest.main()
