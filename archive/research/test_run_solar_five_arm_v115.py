"""Metadata and control-flow tests for the frozen five-arm worker."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_solar_five_arm_v115 as runner


def saved_metadata():
    modern = runner.read("reports/solar-modern-arm-v110/pilot-protocol.json")
    direct = runner.read("reports/solar-direct-arm-v113/pilot-protocol.json")
    manifest = runner.read(modern["manifest"])
    cohort = runner.read(runner.COHORT[0])
    settings = dict(modern, profiles=modern["profiles"] + direct["profiles"],
                    purpose=runner.PURPOSE, stage_mode=runner.STAGE_MODE, comparison_case_count=96,
                    comparison_case_ids=cohort["subsets"]["96"]["case_ids"], output=str(runner.OUTPUT),
                    max_calendar_seconds=runner.CAP)
    return settings, manifest, cohort


class FiveArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.manifest, cls.cohort = saved_metadata()

    def test_select_exact_ordered_96_without_changing_authority(self):
        rows = runner.strict_cases(self.protocol, self.manifest, self.manifest["cases"], self.cohort)
        self.assertEqual([row["case_id"] for row in rows], self.cohort["subsets"]["96"]["case_ids"])
        self.assertEqual(sum(row["positive"] for row in rows), 48)
        self.assertEqual(len(self.protocol["case_ids"]), 384)

        reordered = dict(self.protocol, comparison_case_ids=self.protocol["comparison_case_ids"][::-1])
        with self.assertRaisesRegex(ValueError, "Exact ordered"):
            runner.strict_cases(reordered, self.manifest, self.manifest["cases"], self.cohort)
        cached = dict(self.protocol, case_cache_seals={"historical": "unacceptable"})
        with self.assertRaisesRegex(ValueError, "fresh"):
            runner.strict_cases(cached, self.manifest, self.manifest["cases"], self.cohort)

    def test_changed_scoring_truth_does_not_change_inference_or_direct_metadata(self):
        first = self.manifest["cases"][0]
        changed = deepcopy(first)
        for key in ("scoring_geometry", "scoring_labels", "secondary_known_query_sun_world",
                    "query_unit_index", "positive", "query_component", "frame_metadata"):
            changed[key] = object()
        self.assertEqual(runner.inference_inputs(first), runner.inference_inputs(changed))
        proof, _ = runner.reference_binding(changed, self.protocol)
        self.assertEqual(proof, runner.reference_binding(first, self.protocol)[0])
        self.assertNotIn(first["scoring_geometry"], proof["input_sha256"])
        changed["inference_inputs"]["truth"] = "illegal"
        with self.assertRaisesRegex(ValueError, "projection"):
            runner.inference_inputs(changed)

    def test_scoring_bytes_are_not_opened_in_the_inference_hash_step(self):
        case = self.manifest["cases"][0]
        with patch.object(runner, "checked_file") as verify:
            runner.check_case_bytes(case, self.protocol["input_sha256"], scoring=False)
            first = {call.args[0] for call in verify.call_args_list}
            self.assertTrue({case["query"], case["reference_geometry"]} <= first)
            self.assertTrue(first.isdisjoint(case["scoring_inputs"].values()))
            verify.reset_mock()
            runner.check_case_bytes(case, self.protocol["input_sha256"], scoring=True)
            self.assertEqual({call.args[0] for call in verify.call_args_list},
                             set(case["scoring_inputs"].values()))

    def test_evaluator_opens_scoring_hashes_once_after_inference(self):
        case = self.manifest["cases"][0]
        order = []
        fake_input = SimpleNamespace(known_query_sun=None, solar_lookup=None)
        backend = SimpleNamespace(profile={"backend": "sift"}, infer=lambda _inp: order.append("infer"))

        class Loader:
            last_setup = {}

            def get(self, observed, track):
                self.observed = observed
                order.append("load_input")
                return fake_input

        loader = Loader()

        def evaluate(adapter, inp, scoring_case, scoring_loader):
            adapter.infer(inp)
            scoring_loader(scoring_case, [])
            scoring_loader(scoring_case, [])
            return {"inference_seconds": 1.}, {}

        def checked(_case, _pins, *, scoring):
            order.append("hash_truth" if scoring else "hash_inference")

        with patch.object(runner, "evaluate_case", side_effect=evaluate), \
                patch.object(runner, "check_case_bytes", side_effect=checked), \
                patch.object(runner, "capture_case_report", return_value=({}, {})), \
                patch.object(runner, "load_aligned_scoring", side_effect=lambda *_: order.append("read_truth")):
            runner.evaluate_one(backend, loader, case, self.protocol)
        self.assertEqual(order, ["hash_inference", "load_input", "infer", "hash_truth", "read_truth", "read_truth"])
        self.assertEqual(set(loader.observed), set(runner.INPUT_KEYS))

    def test_outer_clock_includes_checkout_and_never_resets(self):
        start = datetime(2026, 11, 5, 2, tzinfo=timezone.utc)
        published = start + timedelta(seconds=runner.CAP + 300)
        protocol = dict(max_calendar_seconds=runner.CAP, absolute_deadline_utc=published.isoformat())
        outer = start + timedelta(seconds=runner.CAP)
        inner_start = start + timedelta(seconds=100)
        context = dict(outer_started_utc=start.isoformat(), outer_absolute_deadline_utc=outer.isoformat(),
                       cap_seconds=runner.CAP)
        inner, _, effective = runner.deadline_bounds(protocol, context, inner_start)
        self.assertEqual(inner, inner_start + timedelta(seconds=runner.CAP))
        self.assertEqual(effective, outer)
        invalid = dict(context, outer_absolute_deadline_utc=(outer + timedelta(seconds=1)).isoformat())
        with self.assertRaisesRegex(ValueError, "outer clock"):
            runner.deadline_bounds(protocol, invalid, inner_start)
        with self.assertRaisesRegex(ValueError, "outer clock"):
            runner.deadline_bounds(protocol, context, outer)


if __name__ == "__main__":
    unittest.main()
