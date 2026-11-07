"""Metadata-only tests. No model import or scientific assets are opened."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import prepare_v121 as prep
import run_v121 as runner


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = prep.build()

    def test_exact_frozen_paired_roster(self):
        p = self.protocol
        self.assertEqual(p["pilot_case_ids"], list(prep.EXPECTED))
        self.assertEqual(len(p["case_ids"]), 384)
        positives = {c["case_id"]: c for c in p["pilot_case_metadata"] if c["positive"]}
        negatives = [c for c in p["pilot_case_metadata"] if not c["positive"]]
        self.assertEqual(len(positives), len(negatives))
        self.assertEqual(len({c["component"] for c in positives.values()}), 4)
        self.assertEqual(sum(c["light_a"] != c["light_b"] for c in positives.values()), 2)
        self.assertEqual(set(positives), {c["matched_positive_case"] for c in negatives})
        self.assertTrue(p["prepared_only"])
        self.assertFalse(p["execution_authorized"])

    def test_builder_only_opens_frozen_metadata_and_sources(self):
        actual = Path.read_bytes
        seen = []

        def metadata_only(path):
            self.assertNotIn(path.suffix, (".npz", ".png", ".exr", ".pt", ".pth", ".safetensors", ".zip"))
            self.assertNotIn("runs/solar-five-arm-v115/comparison/", str(path))
            self.assertNotIn("solar-compute-reservations-v67.json", str(path))
            seen.append(path)
            return actual(path)

        with patch.object(Path, "read_bytes", metadata_only):
            self.assertEqual(prep.build(), self.protocol)
        self.assertGreater(len(seen), 100)

    def test_terminal_status_is_bound_to_original_attempt(self):
        p = self.protocol["prerequisite_v115"]
        record = dict(run_id=p["run_id"], status="completed", conclusion="success",
                      html_url=f"https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/{p['run_id']}",
                      ticket_commit=p["ticket_commit"], ticket_sha256=p["ticket_sha256"],
                      protocol_sha256=p["sha256"], source_commit=p["source_commit"], nonce=p["nonce"],
                      run_started_at="2026-11-05T00:31:03Z", job={"conclusion": "success"}, artifacts={"outer": {}})
        watchdog = dict(html_url=record["html_url"], workflow_source_commit=record["ticket_commit"],
                        ticket_sha256=record["ticket_sha256"], nonce=record["nonce"],
                        workflow_started_utc=record["run_started_at"], workflow_run_id=p["run_id"],
                        run_attempt=1, source_publication_commit=p["source_commit"],
                        started_utc=record["run_started_at"], calendar_seconds=15_000)
        dispatch = dict(created_utc="2026-11-05T05:20:00Z")
        runner.check_official_terminal(record, watchdog, self.protocol, dispatch)
        for altered in (dict(status="in_progress"), dict(conclusion="failure"),
                        dict(ticket_commit="b" * 40), dict(artifacts={})):
            with self.assertRaises(ValueError):
                runner.check_official_terminal({**record, **altered}, watchdog, self.protocol, dispatch)
        with self.assertRaises(ValueError):
            runner.check_official_terminal(record, {**watchdog, "run_attempt": 2}, self.protocol, dispatch)

    def test_official_verifier_receipt_cannot_omit_outer_or_independent_replay(self):
        prior = self.protocol["prerequisite_v115"]
        outer = {"digest": "sha256:" + "b" * 64, "size_in_bytes": 192}
        evidence = {k: {"path": k, "sha256": "a" * 64}
                    for k in ("nested_cell_inventory", "replay", "comparison", "archive", "outer_watchdog")}
        records = {role: evidence["archive" if role == "scientific_archive" else role]
                   for role in ("nested_cell_inventory", "replay", "comparison", "scientific_archive", "outer_watchdog")}
        audit = dict(schema="solar_five_arm_official_artifact_audit_v115", run_id=prior["run_id"],
                     conclusion="success", complete_archive_verified=True, independently_replayed=True,
                     new_inference=False, ticket_sha256=prior["ticket_sha256"], protocol_sha256=prior["sha256"],
                     archived_downloads={"outer": outer}, records=records)
        official = {"artifacts": {"outer": outer}}
        runner.check_artifact_audit(audit, official, evidence, self.protocol)
        for altered in (dict(independently_replayed=False), dict(archived_downloads={}), dict(records={})):
            with self.assertRaises(ValueError):
                runner.check_artifact_audit({**audit, **altered}, official, evidence, self.protocol)

    def test_preexisting_canonical_charge_only(self):
        old = dict(job_id=prep.V115_SHA + "/evaluation", category="baseline", seconds=18_000,
                   refund_on_failure=False, reserved_utc="2026-11-05T00:30:05Z")
        charge = dict(job_id="f" * 64 + prep.JOB_SUFFIX, category="baseline", seconds=prep.CAP,
                      refund_on_failure=False, reserved_utc="2026-11-05T05:20:15Z")
        dispatch = dict(created_utc="2026-11-05T05:20:00Z", latest_start_utc="2026-11-05T05:24:00Z")
        started = datetime(2026, 11, 5, 5, 20, 20, tzinfo=timezone.utc)
        b = self.protocol["budget"]
        contract = dict(compute_ledger=b["ledger"], compute_ledger_sha256=b["ledger_sha256"],
                        reservation_journal=b["canonical_journal"])
        with (
            patch.object(runner, "checked", return_value=json.dumps(contract).encode()),
            patch("solar_learning_pipeline_v73.remaining", return_value={"baseline": 4895}),
            patch("solar_learning_pipeline_v73.journal_state") as journal,
        ):
            journal.return_value = ({}, {"reservations": [old, charge]}, {})
            self.assertEqual(runner.budget_charge(self.protocol, "f" * 64, dispatch, started, old), (charge, 4895))
            for rows in ([old], [charge], [old, {**charge, "seconds": prep.CAP+1}],
                         [old, {**charge, "refund_on_failure": True}]):
                journal.return_value = ({}, {"reservations": rows}, {})
                with self.assertRaises(ValueError):
                    runner.budget_charge(self.protocol, "f" * 64, dispatch, started, old)

    def test_missing_authorization_stops_before_terminal_or_model(self):
        with (
            patch.object(runner, "load_protocol", return_value=(self.protocol, "0" * 64)),
            patch.object(runner, "read_dispatch", side_effect=FileNotFoundError),
            patch.object(runner, "terminal_audit") as terminal,
            patch.object(runner.subprocess, "Popen") as child,
        ):
            with self.assertRaises(FileNotFoundError):
                runner.run("0" * 40, Path("."), Path("."), Path("."))
            terminal.assert_not_called()
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
