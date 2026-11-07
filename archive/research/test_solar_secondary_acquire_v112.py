"""Focused metadata and gate checks without project terrain or Blender."""

from copy import deepcopy
import unittest

from solar_physics_v63 import lighting_schedule
from solar_secondary_acquire_v112 import ENDPOINTS, frozen_sentry, gate_decision


def fixed_shape_roster():
    endpoints, jobs = [], []
    for name in ENDPOINTS:
        scope = {"split": "model_selection", "component": name}
        site = dict(**scope, centre_latitude_deg=86.)
        endpoints.append(dict(endpoint=name, split="model_selection", site=site, camera_pair=scope,
                              lighting=[lighting_schedule(86.)[i] for i in (8, 23)]))
        for role, light in (("reference", 8), ("query", 8), ("query", 23)):
            for replicate in (0, 1):
                jobs.append(dict(view_id=f"v112/{name}/{role}/light-{light:02d}", endpoint=name, role=role,
                                 light=light, replicate=replicate, seed=len(jobs) + 1, samples_per_pixel=64,
                                 source="black_space_float32_combined"))
    sentry = [jobs[i] for i in (0, 1, 4, 5, 10, 11)]
    return dict(
        schema="solar_secondary_ranking_admission_reconciled_v112", calibration_or_final_access=False,
        endpoints=endpoints, render_jobs=jobs, cases=[{}] * 16,
        compute_design=dict(cpu_threads_per_job=2, stage_caps_seconds=dict(geometry_and_sentry_renders=500)),
        early_coverage_stop={"acquire_first_view_seed_pairs": [
            {key: row[key] for key in ("view_id", "replicate", "seed")} for row in sentry
        ]},
    )


def oracle_rows():
    return [dict(eligible=True, oracle_outcomes=[dict(eligible=True, correct=True) for _ in range(2)])
            for _ in ENDPOINTS]


class AcquisitionTests(unittest.TestCase):
    def test_exact_sentry_includes_matched_negative_and_distinct_seeds(self):
        roster = fixed_shape_roster()
        endpoints, jobs = frozen_sentry(roster)
        self.assertEqual(tuple(endpoints), ENDPOINTS)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len({row["seed"] for row in jobs}), 6)
        self.assertEqual([row["endpoint"] for row in jobs].count(ENDPOINTS[1]), 2)
        self.assertTrue(all(row["light"] == 23 for row in jobs if row["role"] == "query"))
        altered = deepcopy(roster)
        altered["endpoints"][0]["lighting"][0]["sun_enu"][0] += .1
        with self.assertRaisesRegex(ValueError, "Sun schedule"):
            frozen_sentry(altered)

    def test_sentry_rejects_repeated_exposure_and_heldout_endpoint(self):
        roster = fixed_shape_roster()
        roster["early_coverage_stop"]["acquire_first_view_seed_pairs"][5] = \
            roster["early_coverage_stop"]["acquire_first_view_seed_pairs"][0]
        with self.assertRaisesRegex(ValueError, "six fixed"):
            frozen_sentry(roster)
        roster = fixed_shape_roster()
        roster["endpoints"][1]["site"]["split"] = "final_test"
        with self.assertRaisesRegex(ValueError, "outside model selection"):
            frozen_sentry(roster)

    def test_one_failed_oracle_stays_in_denominator(self):
        rows = oracle_rows()
        self.assertTrue(gate_decision(rows, True)["passed"])
        rows[0]["oracle_outcomes"][1]["correct"] = False
        decision = gate_decision(rows, True)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["eligible_directions"], 8)
        self.assertEqual(decision["correct_oracle_poses"], 7)
        self.assertEqual(decision["oracle_success"], .875)

    def test_geometry_ineligible_or_nonoverlap_invalid_stops_sentry(self):
        rows = oracle_rows()
        self.assertFalse(gate_decision(rows, False)["passed"])
        rows[0]["eligible"] = False
        for outcome in rows[0]["oracle_outcomes"]:
            outcome.update(eligible=False, correct=False)
        decision = gate_decision(rows, True)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["eligible_directions"], 6)
        self.assertEqual(decision["oracle_success"], 1.)
        rows[1]["oracle_outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "denominator"):
            gate_decision(rows, True)


if __name__ == "__main__":
    unittest.main()
