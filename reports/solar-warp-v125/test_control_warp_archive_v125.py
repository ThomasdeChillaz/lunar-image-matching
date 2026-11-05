"""Metadata and admission controls without timing, arrays, or reservations."""

from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import control_warp_archive_v125 as control


class PreparationTests(unittest.TestCase):
    def test_original_roster_controls_and_failure_are_frozen(self):
        checks = []
        actual = control.legacy.sha

        def metadata_only(path):
            item = Path(path)
            self.assertIn(item.suffix, (".json", ".py"))
            checks.append(item)
            return actual(item)

        with patch.object(control.legacy, "sha", side_effect=metadata_only):
            timing = control.immutable("timing")
            archive = control.immutable("archive")
        self.assertEqual(timing["schema"], "solar_warp_archival_gate_v117")
        self.assertEqual(timing["controller_revision"], "v125")
        self.assertEqual((timing["cap_seconds"], archive["cap_seconds"]), (60, 180))
        self.assertEqual((archive["primary_cell_pixels"], archive["sensitivity_cell_pixels"]), (32, 16))
        self.assertEqual(archive["expected_prediction_rows"], 32)
        self.assertEqual(archive["cpu_threads"], 2)
        self.assertTrue(archive["query_pose_read_after_prediction_seal"])
        self.assertFalse(archive["calibration_or_final_access"])
        self.assertGreater(timing["original_v117_timing"]["projected_archive_seconds"], 180)
        self.assertFalse(timing["original_v117_timing"]["archive_started"])
        self.assertIn(control.OLD_FAILURE, checks)

    def test_prediction_arguments_cannot_access_scoring(self):
        p = control.immutable("archive")
        root = (control.REPO / control.WORK / "archive").resolve()
        args = dict(repo=str(control.REPO), inputs=p["inputs"]["path"], inputs_sha256=p["inputs"]["sha256"],
                    output=str(control.WORK / "archive/predictions.json"), labels=None,
                    labels_sha256=None, predictions=None, predictions_sha256=None)
        with patch.object(control.legacy, "read") as read:
            self.assertEqual(control.archived_arguments("predict", p, root, {"arguments": args}), args)
            with self.assertRaises(ValueError):
                control.archived_arguments("predict", p, root, {"arguments": {**args, "labels": p["labels"]["path"]}})
            read.assert_not_called()

    def test_scoring_requires_prediction_seal_before_label_arguments(self):
        p = control.immutable("archive")
        root = (control.REPO / control.WORK / "archive").resolve()
        wanted = "1" * 64
        args = dict(repo=str(control.REPO), inputs=p["inputs"]["path"], inputs_sha256=p["inputs"]["sha256"],
                    output=str(control.WORK / "archive/scores.json"), labels=p["labels"]["path"],
                    labels_sha256=p["labels"]["sha256"], predictions=str(control.WORK / "archive/predictions.json"),
                    predictions_sha256=wanted)
        seal = dict(rows=32, sha256=wanted, evidence_bytes=1024)
        with patch.object(control.legacy, "read", return_value=seal) as read:
            with patch.object(control.legacy, "sha", return_value=wanted) as sha:
                self.assertEqual(control.archived_arguments("score", p, root, {"arguments": args}), args)
            read.assert_called_once_with(root / "PREDICTIONS_SEALED.json")
            sha.assert_called_once_with(root / "predictions.json")
        with patch.object(control.legacy, "read", return_value={**seal, "rows": 31}):
            with patch.object(control.legacy, "sha", return_value=wanted), self.assertRaises(ValueError):
                control.archived_arguments("score", p, root, {"arguments": args})

    def test_archive_cannot_spend_without_timing_or_original_charges(self):
        p = control.immutable("archive")
        now = datetime(2026, 11, 5, tzinfo=timezone.utc)
        prior = {"run_id": 41941904269}
        timing = {"charge": {"job_id": "v125-timing"}}
        old = p["original_v117_timing"]["charge"]
        dispatch = dict(p, prepared_only=False, predecessor=prior, timing=timing, created_utc=now.isoformat(),
                        latest_start_utc=(now + control.legacy.LEASE).isoformat())
        with (
            patch.object(control.legacy, "read", return_value=dispatch),
            patch.object(control, "immutable", return_value=p),
            patch.object(control.legacy, "publication"),
            patch.object(control, "predecessor", return_value=prior),
            patch.object(control, "timing_proof", return_value=timing) as timing_proof,
            patch.object(control.legacy, "now", return_value=now),
            patch.object(control, "journal_state") as journal,
            patch.object(control.legacy, "remaining", return_value={"baseline": 180}),
            patch.object(control.legacy, "load_contract"),
            patch.object(control.legacy, "reserve") as reserve,
        ):
            journal.return_value = (None, {"reservations": [old, timing["charge"]]}, None)
            self.assertEqual(control.admission("archive", None, "0" * 40), dispatch)
            journal.return_value = (None, {"reservations": [old]}, None)
            with self.assertRaises(ValueError):
                control.admission("archive", None, "0" * 40)
            timing_proof.side_effect = FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                control.run("archive", None, "0" * 40)
            reserve.assert_not_called()

    def test_either_clock_can_end_execution(self):
        now = datetime(2026, 11, 5, tzinfo=timezone.utc)
        with patch.object(control.legacy, "now", return_value=now), patch.object(control.time, "monotonic", return_value=2.):
            control.ensure_time(now + timedelta(seconds=1), 3.)
            for wall, monotonic in ((now, 3.), (now + timedelta(seconds=1), 2.)):
                with self.assertRaises(TimeoutError):
                    control.ensure_time(wall, monotonic)

    def test_legacy_archive_arguments_and_capture_schema_compose(self):
        p = dict(control.immutable("archive"), prepared_only=False)
        source = control.REPO / "reports/solar-warp-handoff-v120/capture_audit.py"
        spec = spec_from_file_location("legacy_capture_v120_test", source)
        capture = module_from_spec(spec)
        spec.loader.exec_module(capture)
        original_directory = Path.cwd()
        with TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            try:
                os.chdir(repo)
                root = control.WORK / "archive"
                root.mkdir(parents=True)
                stages = []

                def issue(mode, arguments):
                    verified = control.archived_arguments(mode, p, root.resolve(), {"arguments": arguments})
                    self.assertEqual(verified, arguments)
                    return root / f"{mode}-ticket.json"

                def step(command, mode):
                    stages.append(mode)
                    if mode == "predict":
                        rows = []
                        for i in range(4):
                            tag = f"exposure-{i}"
                            for r in (0, 1):
                                for q in (0, 1):
                                    for scale in (32, 16):
                                        name = f"cells-{tag}-r{r}-q{q}-s{scale}.npz"
                                        path = root / name
                                        path.write_bytes(b"x")
                                        rows.append(dict(tag=tag, reference_seed=r, query_seed=q, cell_pixels=scale,
                                                         cell_evidence=dict(path=name, sha256=control.legacy.sha(path))))
                        control.legacy.put(root / "predictions.json", dict(rows=rows, evidence_bytes=len(rows)))
                    elif mode == "score":
                        control.legacy.put(root / "scores.json", {})
                    else:
                        self.assertTrue(command[1].endswith("verify_warp_archive_v117.py"))
                        control.legacy.put(root / "independent-replay.json", dict(
                            passed=True, replayed_rows=32, gate={}, prediction_sha256=control.legacy.sha(root / "predictions.json"),
                            scores_sha256=control.legacy.sha(root / "scores.json"),
                        ))

                with patch.object(control, "REPO", repo):
                    extra = control.legacy.archive(p, root, issue, step)
                completion = dict(extra, phase="archive", complete=True, calibration_or_final_access=False,
                                  steps=stages, source_commit="a" * 40, schema="solar_warp_projection_completion_v125")
                capture.validate_dispatch(completion, p)
                self.assertEqual(stages, ["predict", "score", "independent-replay"])
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
