from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lunar_matching.artifacts import audit_archive
from lunar_matching.cli import main
from lunar_matching.evidence import load_evidence, format_table
from lunar_matching.paths import repository_root, research_root


def test_measured_evidence_keeps_failed_cases_and_all_controls(capsys):
    assert main(["status", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["positive_pairs"] == 4 and result["positive_terrain_components"] == 3
    assert len(result["methods"]) == 7
    assert {row["correct"] for row in result["methods"].values()} == {1}
    assert "raw poses" in result["evidence"]


def test_saved_metric_tampering_is_rejected(tmp_path):
    result = load_evidence()
    changed = deepcopy(result)
    changed["methods"]["beam_active128"]["correct"] = 4
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="disagree"):
        load_evidence(path)
    changed = deepcopy(result)
    changed["case_results"] = changed["case_results"][:-1]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="four positives"):
        load_evidence(path)
    assert "No superiority claim" in format_table(result)


def test_every_original_public_file_remains_byte_identical():
    receipt = audit_archive(repository_root())
    assert receipt["passed"] and receipt["preserved_files"] == 2288


def test_package_import_and_help_work_outside_repository_without_loading_legacy(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(repository_root() / "src"))
    code = "import lunar_matching.cli,sys; assert 'torch' not in sys.modules; assert 'beam_matcher_v143' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    result = subprocess.run([sys.executable, "-m", "lunar_matching", "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert "status" in result.stdout and "localize" in result.stdout


def test_cli_rejects_existing_output_before_any_worker(tmp_path, capsys):
    output = tmp_path / "already.json"
    output.write_text("original")
    assert main(["smoke", "--output", str(output)]) == 2
    assert output.read_text() == "original"
    assert "already exists" in capsys.readouterr().err


def test_legacy_compatibility_resolves_inputs_and_caller_output(monkeypatch, tmp_path):
    from lunar_matching import compat
    captured = {}

    def run(args, cwd):
        captured.update(args=args, cwd=cwd)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(compat.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    assert compat.legacy_main("match_terrain.py", ["--protocol", "reports/solar-beam-v143/protocol.json",
                                                "--output", "result.json"]) == 0
    assert captured["cwd"] == research_root()
    assert captured["args"][-1] == str(tmp_path / "result.json")
    assert str(research_root() / "reports/solar-beam-v143/protocol.json") in captured["args"]


def test_generated_native_control_runs_without_import_pollution():
    from lunar_matching.backend import run_backend
    before = set(sys.modules)
    result = run_backend("smoke")
    assert result["passed"] and result["generated_only"] and not result["lunar_images_opened"]
    assert result["deliberately_injected_control_hypotheses"]
    assert all(row["attempted_native_fields"] == 64 for row in result["rows"].values())
    assert not ({"policy", "beam_matcher_v143", "likelihood"} & (set(sys.modules) - before))


def test_saved_geometry_replay_uses_the_frozen_workspace():
    from lunar_matching.backend import run_backend
    if not (research_root() / "runs/solar-beam-v143/predictions.json").is_file():
        pytest.skip("Download the public saved-output capsules to replay geometry.")
    result = run_backend("replay")
    assert result["passed"] and result["method_case_cells"] == 56
    assert result["historical_control_identities"] == 24
    assert result["private_authority_used"] is False
