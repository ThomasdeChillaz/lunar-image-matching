"""Procedural controls only. No lunar images, checkpoints or case labels."""

import math
from types import FunctionType, SimpleNamespace
import unittest

import cv2
import numpy as np

import solar_pose_proposals_v73 as original
from solar_pose_cone_v113 import _specialized_program, propose_with_simulator_cone, within_simulator_cone


def rvec(off_nadir_deg):
    return np.array([[math.pi-math.radians(off_nadir_deg)], [0.], [0.]])


class ConeTests(unittest.TestCase):
    def test_downward_and_boundary_rotation(self):
        self.assertTrue(within_simulator_cone(rvec(0.)))
        self.assertTrue(within_simulator_cone(rvec(52.)))
        self.assertFalse(within_simulator_cone(rvec(52.001)))
        self.assertFalse(within_simulator_cone(rvec(90.)))
        self.assertFalse(within_simulator_cone(np.zeros(3)))
        self.assertFalse(within_simulator_cone([np.nan, 0., 0.]))

    def test_root_filter_and_lm_guard_preserve_samples_and_objective(self):
        world = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                          [1, 1, 0], [2, 0, 0], [1, 2, 0], [0, 3, 0]], float)
        pixels = world[:, :2]+100.
        good_root = rvec(35.)
        wrong_root = rvec(110.)
        bad_refit = rvec(80.)
        accepted_refit = rvec(25.)

        def run(cone, refit):
            samples = []
            scores = []
            fit_inputs = []

            def roots(points, image, *_args, **_kwargs):
                samples.append((points.copy(), image.copy()))
                return 2, [wrong_root.copy(), good_root.copy()], [np.array([1., 0., 0.]), np.array([3., 0., 0.])]

            def lm(_points, _image, _k, _dist, rv, tv):
                fit_inputs.append((rv.copy(), tv.copy()))
                return refit.copy(), np.zeros(3)

            def errors(_world, query, _k, rv, tv):
                scores.append((float(rv[0, 0]), float(tv[0])))
                return np.full(len(query), float(tv[0]))

            bridge = SimpleNamespace(error=cv2.error, Rodrigues=cv2.Rodrigues, SOLVEPNP_AP3P=cv2.SOLVEPNP_AP3P,
                                     solvePnPGeneric=roots, solvePnPRefineLM=lm)
            scope = dict(vars(original), cv2=bridge, errors=errors, _within_simulator_cone=within_simulator_cone)
            if cone:
                exec(_specialized_program(original.__file__), scope)
                function = scope["propose"]
            else:
                function = FunctionType(original.propose.__code__, scope, "propose", original.propose.__defaults__)
            return function(world, pixels, np.eye(3), iterations=4), samples, scores, fit_inputs

        (pose, report), samples, scores, fit_inputs = run(True, bad_refit)
        (_, before), original_samples, _, _ = run(False, bad_refit)
        for constrained, unmodified in zip(samples, original_samples, strict=True):
            np.testing.assert_array_equal(constrained[0], unmodified[0])
            np.testing.assert_array_equal(constrained[1], unmodified[1])
        self.assertEqual(report["minimal_sets_executed"], before["minimal_sets_executed"])
        self.assertEqual(report["roots_evaluated"], before["roots_evaluated"])
        self.assertEqual(report["roots_rejected_off_nadir"], 4)
        self.assertEqual(report["lm_refits_rejected_off_nadir"], 1)
        self.assertEqual(len(fit_inputs), 1)
        self.assertTrue(all(value == float(good_root[0, 0]) for value, _ in scores))
        np.testing.assert_allclose(pose[:3, :3], cv2.Rodrigues(good_root)[0].T, atol=1e-12)
        self.assertEqual(report["final_reprojection_threshold_native_px"], 3.)
        self.assertFalse(report["accepted"])

        (pose, report), _, _, fit_inputs = run(True, accepted_refit)
        self.assertEqual(report["lm_refits_rejected_off_nadir"], 0)
        self.assertEqual(len(fit_inputs), 2)
        np.testing.assert_allclose(pose[:3, :3], cv2.Rodrigues(accepted_refit)[0].T, atol=1e-12)

    def test_actual_opencv_on_constructed_3d_points(self):
        rng = np.random.default_rng(7)
        world = rng.uniform(-6., 6., (16, 3))
        expected_center = np.array([0., 0., 80.])
        rotation = cv2.Rodrigues(rvec(0.))[0]
        k = np.array([[400., 0., 255.5], [0., 400., 255.5], [0., 0., 1.]])
        detector = (world-expected_center) @ rotation.T @ k.T
        pixels = detector[:, :2]/detector[:, 2, None]
        guess, report = propose_with_simulator_cone(world, pixels, k, iterations=16)
        self.assertEqual(report["minimal_sets_executed"], 16)
        self.assertEqual(report["strict3_support_diagnostic"], len(world))
        self.assertFalse(report["accepted"])
        np.testing.assert_allclose(guess[:3, 3], expected_center, atol=1e-6)
        np.testing.assert_allclose(guess[:3, :3], rotation.T, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
