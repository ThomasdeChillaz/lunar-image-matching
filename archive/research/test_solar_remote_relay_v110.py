"""Procedural v110 admission, signature, privacy and archive transport controls."""

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import solar_remote_relay_v110 as relay
from solar_capsule_overlay_v110 import ADDED_SOURCES


def frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime.now(timezone.utc)
    ledger = Path("reports/ledger-fixture.json")
    relay.legacy.write_new(ledger, {"accounting_complete": True, "remaining_baseline_wall_seconds": 35000})
    ledger_sha = relay.legacy.sha(ledger)
    path = Path(relay.CANONICAL_JOURNAL)
    old_entry = {
        "job_id": relay.BASE["protocol_sha256"] + "/evaluation",
        "seconds": 1800, "category": "baseline", "reserved_utc": (now - timedelta(minutes=10)).isoformat(),
        "refund_on_failure": False,
    }
    relay.legacy.write_new(path, {"ledger_sha256": ledger_sha, "reservations": [old_entry]})
    protocol = {
        "created_utc": (now - timedelta(minutes=1)).isoformat(),
        "latest_start_utc": (now + timedelta(minutes=59)).isoformat(),
        "absolute_deadline_utc": (now + timedelta(minutes=89)).isoformat(),
        "publication_dispatch_lease_seconds": 3600,
        "max_calendar_seconds": 1800, "case_ids": ["fixture-positive"], "tracks": ["estimated_sun"],
        "compute_ledger": str(ledger), "compute_ledger_sha256": ledger_sha,
        "baseline_admission": {"initial_reservations": [old_entry], "remaining_at_preparation_seconds": 33200},
        "overlay_transport_v110": relay.BASE, "output": relay.OUTPUT,
        "reservation_journal": relay.CANONICAL_JOURNAL,
        "source_sha256": {
            **{name: "b" * 64 for name in ADDED_SOURCES},
            relay.PROBE: relay.PROBE_SHA256, relay.FAILURE: relay.FAILURE_SHA256,
        },
    }
    relay.legacy.write_new(relay.PROTOCOL, protocol)
    return protocol, path, old_entry


def build_ticket(protocol, raw):
    auth = relay._review_inline_bytes(raw)
    nonce = auth["nonce"]
    return {
        "schema_version": 110, "repository": relay.REPOSITORY, "workflow": relay.WORKFLOW,
        "workflow_name": relay.WORKFLOW_NAME, "transport": relay.TRANSPORT, "nonce": nonce,
        "authorized_attempt": 1, "source_publication_commit": auth["source_publication_commit"],
        "protocol_path": relay.PROTOCOL, "protocol_sha256": auth["protocol_sha256"],
        "authorization": {
            "path": relay.AUTHORIZATION, "sha256": relay._digest(raw), "inline_base64": base64.b64encode(raw).decode(),
        },
        "base": relay.BASE, "assembled_manifest_sha256": "a" * 64,
        "original_latest_start_utc": protocol["latest_start_utc"],
        "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
        "max_calendar_seconds": 1800, "required_runner": "ubuntu-24.04-arm", "cpu_threads": 2,
        "expected_run_title": relay.RUN_PREFIX + nonce, "expected_branch": relay.BRANCH_PREFIX + nonce,
        "refunds": False, "automatic_retry": False, "primary_only": True,
    }


def test_synthetic_canonical_charge_is_atomic_and_distinct(tmp_path, monkeypatch):
    protocol, path, old_entry = frozen_fixture(tmp_path, monkeypatch)
    source = "a" * 40
    digest = relay.legacy.sha(relay.PROTOCOL)
    result = relay._reserve_frozen_history(protocol, digest, source)
    raw = Path(relay.AUTHORIZATION).read_bytes()
    ticket = build_ticket(protocol, raw)
    relay.legacy.write_new(relay.TICKET, ticket)
    actual = relay.validate_ticket(relay.TICKET, relay.legacy.sha(relay.TICKET), result["nonce"])
    assert actual == ticket
    journal = relay.legacy.read(path)
    assert journal["reservations"][0] == old_entry
    new_entry, = journal["reservations"][1:]
    assert new_entry["job_id"] == digest + "/evaluation"
    assert new_entry["seconds"] == 1800 and new_entry["refund_on_failure"] is False
    assert result["nonce"] != relay.legacy.charge_nonce(relay.BASE["protocol_sha256"], old_entry)
    assert Path(relay.AUTHORIZATION + ".ISSUING.json").is_file()
    assert not Path(relay.BASE["authorization_path"]).exists()
    with pytest.raises(ValueError, match="journal changed"):
        relay._reserve_frozen_history(protocol, digest, source)
    assert relay.legacy.read(path) == journal


def test_stale_canonical_freeze_creates_no_charge_or_sentinel(tmp_path, monkeypatch):
    protocol, path, old_entry = frozen_fixture(tmp_path, monkeypatch)
    journal = relay.legacy.read(path)
    journal["reservations"].append(dict(old_entry, job_id="already-running/fixture"))
    path.write_bytes(relay._bytes(journal))
    with pytest.raises(ValueError, match="no v110 charge"):
        relay._reserve_frozen_history(protocol, relay.legacy.sha(relay.PROTOCOL), "a" * 40)
    assert relay.legacy.read(path) == journal
    assert not Path(relay.AUTHORIZATION).exists()
    assert not Path(relay.AUTHORIZATION + ".ISSUING.json").exists()


@pytest.mark.parametrize("secret", ("/Users/example/private/path", "ghp_" + "A" * 40))
def test_inline_metadata_scans_the_decoded_journal(tmp_path, monkeypatch, secret):
    protocol, path, old_entry = frozen_fixture(tmp_path, monkeypatch)
    relay._reserve_frozen_history(protocol, relay.legacy.sha(relay.PROTOCOL), "a" * 40)
    auth = relay.legacy.read(relay.AUTHORIZATION)
    journal = relay.legacy.read(path)
    journal["reservations"][0]["job_id"] = secret
    raw = relay._bytes(journal)
    auth.update(
        canonical_journal_bytes_base64=base64.b64encode(raw).decode(), canonical_journal_sha256=relay._digest(raw),
    )
    with pytest.raises(ValueError, match="private path or credential"):
        relay._review_inline_bytes(relay._bytes(auth))


@pytest.mark.parametrize("field, bad", (
    ("workflow", ".github/workflows/solar-remote-cpu-v98.yml"),
    ("automatic_retry", True),
    ("primary_only", False),
    ("expected_branch", "solar-v98-ticket-invalid"),
    ("base", {}),
))
def test_ticket_rejects_reuse_or_scope_drift(tmp_path, monkeypatch, field, bad):
    protocol, _, _ = frozen_fixture(tmp_path, monkeypatch)
    relay._reserve_frozen_history(protocol, relay.legacy.sha(relay.PROTOCOL), "a" * 40)
    ticket = build_ticket(protocol, Path(relay.AUTHORIZATION).read_bytes())
    ticket[field] = bad
    relay.legacy.write_new(relay.TICKET, ticket)
    with pytest.raises(ValueError, match="one-use ticket"):
        relay.validate_ticket(relay.TICKET)


def public_records():
    commit, nonce = "a" * 40, "b" * 64
    part = relay._science_part()
    ticket = {"expected_branch": relay.BRANCH_PREFIX + nonce}
    records = {
        "": {"full_name": relay.REPOSITORY, "private": False, "visibility": "public", "default_branch": "main"},
        "/actions/workflows/" + Path(relay.WORKFLOW).name: {"state": "active", "path": relay.WORKFLOW},
        "/git/ref/heads/" + ticket["expected_branch"]: {
            "ref": "refs/heads/" + ticket["expected_branch"], "object": {"type": "commit", "sha": commit},
        },
        "/git/commits/" + commit: {"sha": commit, "verification": {"verified": True, "reason": "valid"}},
        "/releases/tags/solar-arm-transport-v98": {
            "tag_name": "solar-arm-transport-v98", "draft": False, "published_at": "fixture",
            "assets": [{
                "id": 123, "name": part["url"].rsplit("/", 1)[-1], "browser_download_url": part["url"],
                "state": "uploaded", "size": part["bytes"], "digest": "sha256:" + part["sha256"],
            }],
        },
    }
    return commit, ticket, records


def test_public_signed_commit_and_reused_exact_release_are_required():
    commit, ticket, records = public_records()
    assert relay.public_dispatch_evidence(ticket, commit, api=records.__getitem__)["release_asset_id"] == 123
    signature = deepcopy(records["/git/commits/" + commit])
    records["/git/commits/" + commit]["verification"]["verified"] = False
    with pytest.raises(ValueError, match="verify"):
        relay.public_dispatch_evidence(ticket, commit, api=records.__getitem__)
    records["/git/commits/" + commit] = signature
    records["/releases/tags/solar-arm-transport-v98"]["assets"][0]["size"] -= 1
    with pytest.raises(ValueError, match="immutable science asset"):
        relay.public_dispatch_evidence(ticket, commit, api=records.__getitem__)


def test_worker_rejects_old_probe_and_duplicate_workflow_runs():
    commit, ticket, _ = public_records()
    nonce = ticket["expected_branch"].removeprefix(relay.BRANCH_PREFIX)
    ticket.update(
        source_publication_commit="c" * 40, nonce=nonce, expected_run_title=relay.RUN_PREFIX + nonce,
    )
    ref = "refs/heads/" + ticket["expected_branch"]
    env = {
        "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": relay.REPOSITORY, "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_RUN_ATTEMPT": "1", "RUNNER_OS": "Linux", "RUNNER_ARCH": "ARM64",
        "RUNNER_ENVIRONMENT": "github-hosted", "GITHUB_REF": ref, "GITHUB_RUN_ID": "100",
        "GITHUB_SHA": commit, "GITHUB_WORKFLOW_SHA": commit,
        "GITHUB_WORKFLOW_REF": relay.REPOSITORY + "/" + relay.WORKFLOW + "@" + ref,
    }
    prefix = "repos/" + relay.REPOSITORY
    history = prefix + relay._history_path(1)
    current = {
        "id": 100, "event": "workflow_dispatch", "run_attempt": 1, "head_sha": commit,
        "head_branch": ticket["expected_branch"], "path": relay.WORKFLOW,
        "display_title": ticket["expected_run_title"],
        "repository": {"full_name": relay.REPOSITORY}, "head_repository": {"full_name": relay.REPOSITORY},
    }
    records = {prefix + "/actions/runs/100": current, history: {"workflow_runs": [current]}}
    assert relay.workflow_binding(ticket, commit, env, records.__getitem__)["workflow_run_id"] == 100
    records[history]["workflow_runs"].append(dict(current, id=101))
    with pytest.raises(ValueError, match="Duplicate"):
        relay.workflow_binding(ticket, commit, env, records.__getitem__)
    for old in (relay.FAILED_RUN, relay.PROBE_RUN):
        with pytest.raises(ValueError, match="bindings differ"):
            relay.workflow_binding(ticket, commit, dict(env, GITHUB_RUN_ID=str(old)), records.__getitem__)


def test_legacy_downloader_receives_no_old_authority(monkeypatch, tmp_path):
    lease = datetime.now(timezone.utc).isoformat()
    calls = []
    monkeypatch.setattr(relay.legacy, "fetch_capsule", lambda binding, destination, active_deadline: calls.append(
        (binding, destination, active_deadline)
    ))
    target = tmp_path / "science.zip"
    relay._fetch_science_data({"original_latest_start_utc": lease, "authorization": {"private": True}}, target, lease)
    data, destination, deadline = calls[0]
    assert set(data) == {"capsule", "original_latest_start_utc"}
    assert data["capsule"]["parts"] == [relay._science_part()]
    assert data["original_latest_start_utc"] == lease and destination == target and deadline == lease
