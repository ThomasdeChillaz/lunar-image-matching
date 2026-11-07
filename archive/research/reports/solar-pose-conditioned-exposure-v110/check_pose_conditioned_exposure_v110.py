"""Run fixed, tiny procedural exposure-planning controls, with no acquired images."""
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

from pose_conditioned_exposure_v110 import (
    Budget, HEIGHT_VARIANTS_M, POLICIES, Sensor, clipped_poisson_moments, plan_exposures,
)


def photometry():
    charge_mean, variance, saturation = clipped_poisson_moments(
        np.array([0., 2., 10_000.]), 3, 1.5,
    )
    manual = [math.exp(-2)*2**k/math.factorial(k) for k in range(3)]
    manual.append(1 - math.fsum(manual))
    first = math.fsum(k*p for k, p in enumerate(manual))
    second = math.fsum(k*k*p for k, p in enumerate(manual))
    np.testing.assert_allclose(charge_mean, [0, first, 3], atol=1e-12, rtol=0)
    np.testing.assert_allclose(variance, [2.25, second-first**2+2.25, 2.25], atol=1e-12, rtol=0)
    np.testing.assert_allclose(saturation, [0, manual[-1], 1], atol=1e-12, rtol=0)

    # A plausible well size exercises the closed form without a sum over
    # 100,000 outcomes. Far from the cap, ordinary Poisson moments apply.
    charge_mean, variance, saturation = clipped_poisson_moments(
        np.array([20., 50_000., 100_000., 200_000.]), 100_000, 3.,
    )
    np.testing.assert_allclose(charge_mean[[0, 1, 3]], [20., 50_000., 100_000.], rtol=1e-12)
    np.testing.assert_allclose(variance[[0, 1, 3]], [29., 50_009., 9.], rtol=1e-12)
    np.testing.assert_allclose(saturation[[0, 1, 3]], [0., 0., 1.], atol=1e-12)
    assert 99_500 < charge_mean[2] < 100_000 and 0.49 < saturation[2] < .51


def complementary_atlas():
    """Deliberately constructed case where bright and dark ambiguities differ."""
    base = np.array([[30., 1., 11., 8., 4.],
                     [70., 1.1, 11., 8., 4.],
                     [32., 4., 11., 8., 4.]])
    perturbation = np.array([1., .99, 1.01, .98, 1.02, .96, 1.04])
    pixels = np.array([1., .98, 1.01, 1.02])
    irradiance = base[:, :, None, None] * perturbation[None, None, :, None] * pixels
    return irradiance, np.ones(irradiance.shape, bool)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    photometry()
    sensor = Sensor(1., .05, 1.5, 80, 1.)
    budget = Budget(9, 3, (1, 2, 3, 4, 5))
    weights = np.array([.35, .30, .35])
    points = [(40, 40), (170, 40), (300, 40), (430, 40), (40, 170)]
    gradient = [2., 1., 20., 18., 15.]
    levels, coverage = complementary_atlas()
    observations = {
        name: plan_exposures(levels, coverage, weights, points, gradient, sensor, budget, name)
        for name in POLICIES
    }
    for row in observations.values():
        actions = row["records"]
        assert len({rec["action"] for rec in actions}) == row["frames"] == budget.frames
        assert sum(rec["duration_ticks"] for rec in actions) == budget.total_ticks == row["integration_ticks"]
        assert row["pixels_per_frame"] == levels.shape[-1]
    joint = observations["joint"]
    fixed = observations["fixed_hypothesis"]
    assert joint["expected_separation"] > fixed["expected_separation"]
    ticks = {rec["action"]: rec["duration_ticks"] for rec in joint["records"]}
    assert ticks[0] < budget.fixed_ticks < ticks[1]
    assert observations["fixed_gradient"]["expected_separation"] == 0

    # One known-identical pair and two pairs with no shared atlas pixels.
    uninformative = np.ones(levels.shape)
    unobserved = np.ones(levels.shape, bool)
    unobserved[2, :, -1] = False
    uninformative[2, :, -1] = np.nan
    without_information = plan_exposures(
        uninformative, unobserved, weights, points, gradient, sensor, budget, "joint",
    )
    assert without_information["expected_separation"] == 0
    assert [row["status"] for row in without_information["pair_diagnostics"]] == [
        "unresolved_on_complete_atlas", "unresolved_with_missing_coverage", "unresolved_with_missing_coverage",
    ]
    assert [row["actions_with_shared_pixels"] for row in without_information["pair_diagnostics"]] == [5, 0, 0]
    assert [r["duration_ticks"] for r in without_information["records"]] == [3, 3, 3]

    # The nominal predictions differ, but all possible integration times
    # preserve the overlap of their declared irradiance intervals.
    overlap = np.empty((2, 5, 7, 4))
    overlap[0] = np.array([1., 1., 2., 3., 4., 5., 5.])[None, :, None]
    overlap[1] = np.array([7., 3., 4., 5., 5., 6., 7.])[None, :, None]
    wide_intervals = plan_exposures(
        overlap, np.ones_like(overlap, bool), [.5, .5], points, gradient, sensor, budget, "joint",
    )
    assert wide_intervals["pair_diagnostics"][0]["status"] == "unresolved_on_complete_atlas"
    assert wide_intervals["expected_separation"] == 0

    # Different radiances are observationally identical at the finite well.
    clipped = np.full((2, 5, 7, 4), 10_000.)
    clipped[1] *= 2
    saturated = plan_exposures(
        clipped, np.ones_like(clipped, bool), [.5, .5], points, gradient, sensor, budget, "joint",
    )
    assert saturated["pair_diagnostics"][0]["status"] == "unresolved_on_complete_atlas"
    assert saturated["expected_separation"] == 0
    assert all(row["max_supported_saturation_probability"] == 1 for row in saturated["records"])

    source = Path(__file__).with_name("pose_conditioned_exposure_v110.py")
    result = dict(
        schema="solar_pose_conditioned_exposure_fixture_v110",
        height_variants_m=list(HEIGHT_VARIANTS_M),
        analytic_only=True, archived_inputs_opened=0, render_calls=0, matcher_calls=0,
        optimizer_updates=0, calibration_or_final_access=False, source_sha256=digest(source),
        fixture_sha256=digest(__file__), photometry_controls_passed=True,
        comparison=observations, indistinguishable=without_information,
        overlapping_variants=wide_intervals, saturated=saturated,
    )
    with Path(out).open("x") as file:
        json.dump(result, file, allow_nan=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
