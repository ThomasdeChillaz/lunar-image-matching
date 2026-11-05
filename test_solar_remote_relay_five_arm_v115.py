"""Procedural signed v115 relay guards. No runner, matcher, or real budget writes."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

import solar_remote_relay_five_arm_v115 as relay


def _dates():
    start = datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)
    return start, {
        "original_latest_start_utc": (start + timedelta(minutes=30)).isoformat(),
        "original_absolute_deadline_utc": (start + timedelta(minutes=30, seconds=relay.CAP)).isoformat(),
        "max_calendar_seconds": relay.CAP,
    }


def test_workflow_bootstrap_consumes_the_original_pipeline_cap():
    start, ticket = _dates()
    boot_done = start + timedelta(seconds=70)
    outer, deadline, setup_deadline = relay._outer_bounds(ticket, start.isoformat(), now=boot_done)
    assert outer == start
    assert (deadline - boot_done).total_seconds() == 17_930
    assert setup_deadline == start + timedelta(minutes=30)
    with pytest.raises(TimeoutError, match="original pipeline"):
        relay._outer_bounds(ticket, start.isoformat(), now=start + timedelta(seconds=18_000))


def test_pipeline_deadline_never_exceeds_the_frozen_absolute():
    start, ticket = _dates()
    ticket["original_absolute_deadline_utc"] = (start + timedelta(seconds=1234)).isoformat()
    _, deadline, setup = relay._outer_bounds(ticket, start.isoformat(), now=start + timedelta(seconds=5))
    assert deadline == setup == start + timedelta(seconds=1234)
    with pytest.raises(TimeoutError, match="invalid"):
        relay._outer_bounds(ticket, (start + timedelta(seconds=6)).isoformat(), now=start)


def test_exact_scientific_wire_scope_rejects_prospective_and_scope_drift():
    from solar_direct_transport_v113 import ASSET_URL

    all_ids = [f"case-{i:03d}" for i in range(384)]
    protocol = {
        "prepared_only": False, "output": relay.OUTPUT, "stage_mode": "matched_modern_direct_arm_v113",
        "purpose": "matched_development_comparison", "split": "model_selection", "cpu_threads": 2,
        "tracks": ["estimated_sun"], "accepted_results": False, "calibration_or_final_access": False,
        "max_calendar_seconds": relay.CAP, "reservation_journal": relay.CANONICAL_JOURNAL,
        "case_ids": all_ids, "comparison_case_count": 96, "comparison_case_ids": all_ids[:96],
        "profiles": [{"name": name} for name in (*relay.ARM_NAMES, relay.DIRECT_NAME)],
        "direct_supplement": {
            "part": {"url": ASSET_URL, "bytes": 123, "sha256": "a" * 64},
            "manifest_sha256": "b" * 64,
        },
    }
    relay._scope_wire(protocol)
    for replacement in (
        {"prepared_only": True}, {"accepted_results": True}, {"calibration_or_final_access": True},
        {"tracks": ["estimated_sun", "known_sun"]}, {"cpu_threads": 4},
        {"comparison_case_count": 48}, {"case_cache_seals": {"prior": True}},
        {"profiles": protocol["profiles"][:-1]}, {"max_calendar_seconds": relay.CAP + 1},
    ):
        with pytest.raises(ValueError, match="source-frozen"):
            relay._scope_wire(dict(protocol, **replacement))


def test_all_original_hydrated_author_checkpoints_required():
    protocol = json.loads(Path("reports/solar-direct-arm-v113/pilot-protocol.json").read_bytes())
    hydration = json.loads(Path("reports/solar_arm_v110_remote_watchdog.json").read_bytes())
    relay._validate_hydrated_weights(hydration, protocol)
    missing = deepcopy(hydration)
    missing["official_checkpoint_receipts"].pop()
    with pytest.raises(ValueError, match="four locked"):
        relay._validate_hydrated_weights(missing, protocol)
    replaced = deepcopy(hydration)
    replaced["official_checkpoint_receipts"][0]["source"] = "https://example.org/model.pth"
    with pytest.raises(ValueError, match="four locked"):
        relay._validate_hydrated_weights(replaced, protocol)


def _workflow_fixture():
    nonce = "1" * 64
    commit = "2" * 40
    repo = relay.REPOSITORY
    ticket = {
        "expected_branch": relay.BRANCH_PREFIX + nonce, "expected_run_title": relay.RUN_PREFIX + nonce,
        "source_publication_commit": "3" * 40, "nonce": nonce,
    }
    ref = "refs/heads/" + ticket["expected_branch"]
    env = {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": repo, "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ATTEMPT": "1", "RUNNER_OS": "Linux", "RUNNER_ARCH": "ARM64",
        "RUNNER_ENVIRONMENT": "github-hosted", "GITHUB_REF": ref, "GITHUB_RUN_ID": "100",
        "GITHUB_SHA": commit, "GITHUB_WORKFLOW_SHA": commit,
        "GITHUB_WORKFLOW_REF": repo + "/" + relay.WORKFLOW + "@" + ref,
    }
    prefix = "repos/" + repo
    current = {
        "id": 100, "event": "workflow_dispatch", "run_attempt": 1, "head_sha": commit,
        "head_branch": ticket["expected_branch"], "path": relay.WORKFLOW,
        "display_title": ticket["expected_run_title"], "repository": {"full_name": repo},
        "head_repository": {"full_name": repo}, "run_started_at": "2026-11-05T00:00:00Z",
    }
    records = {
        prefix + "/actions/runs/100": current,
        prefix + relay._history_path(1): {"workflow_runs": [current]},
    }
    return ticket, commit, env, prefix, records


def test_only_fresh_exact_arm_workflow_and_one_public_invocation_are_accepted():
    ticket, commit, env, prefix, records = _workflow_fixture()
    assert relay.workflow_binding(ticket, commit, env, records.__getitem__)["workflow_run_id"] == 100
    for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("RUNNER_ARCH", "X64"), ("RUNNER_ENVIRONMENT", "self-hosted")):
        with pytest.raises(ValueError, match="first|attempt"):
            relay.workflow_binding(ticket, commit, dict(env, **{key: value}), records.__getitem__)
    for run in relay.PREDECESSOR_RUNS:
        with pytest.raises(ValueError, match="bindings differ"):
            relay.workflow_binding(ticket, commit, dict(env, GITHUB_RUN_ID=str(run)), records.__getitem__)
    history = records[prefix + relay._history_path(1)]["workflow_runs"]
    history.append(dict(history[0], id=101))
    with pytest.raises(ValueError, match="Duplicate"):
        relay.workflow_binding(ticket, commit, env, records.__getitem__)


def test_readonly_actions_tokens_do_not_enter_scientific_or_author_processes(monkeypatch):
    for key in relay.TOKEN_ENV:
        monkeypatch.setenv(key, "synthetic-token")
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    scientific = relay._scientific_env()
    assert scientific["GITHUB_RUN_ID"] == "100" and scientific["OMP_NUM_THREADS"] == "2"
    assert set(relay.TOKEN_ENV).isdisjoint(scientific)
    with relay._without_action_token():
        import os
        assert set(relay.TOKEN_ENV).isdisjoint(os.environ)
    assert all(os.environ.get(key) == "synthetic-token" for key in relay.TOKEN_ENV)


def test_workflow_keeps_exact_qualified_wheels_and_separate_partial_artifact():
    workflow = Path(relay.WORKFLOW).read_text()
    assert "on:\n  workflow_dispatch:" in workflow
    assert "permissions:\n  contents: read\n  actions: read" in workflow
    assert "timeout-minutes: 300" in workflow
    assert "persist-credentials: false" in workflow
    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "python-version: '3.12.15'" in workflow
    assert "pip install --no-deps --require-hashes -r requirements_arm_cp312_v113.txt" in workflow
    assert workflow.count("GH_TOKEN:") == 2
    assert workflow.index(" verify-ticket ") < workflow.index("pip install ")
    assert "-partial" in workflow and "scientific_results.zip" in workflow


def test_stale_canonical_history_does_not_write_an_issuing_marker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    journal = Path(relay.CANONICAL_JOURNAL)
    journal.parent.mkdir()
    initial = {
        "ledger_sha256": "a" * 64,
        "reservations": [{"job_id": "existing/fixed"}],
    }
    journal.write_text(json.dumps(initial))
    protocol = {"baseline_admission": {"initial_reservations": []}, "compute_ledger_sha256": "a" * 64}
    with pytest.raises(ValueError, match="no v115 charge"):
        relay._reserve_frozen_history(protocol, "b" * 64, "c" * 40)
    assert json.loads(journal.read_text()) == initial
    assert not Path(relay.AUTHORIZATION).exists()
    assert not Path(relay.AUTHORIZATION + ".ISSUING.json").exists()
