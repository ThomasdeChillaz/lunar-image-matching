"""Independent synthetic overlay controls. No remote calls or reservations."""

import base64
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import solar_capsule_overlay_v110 as overlay
from solar_remote_relay_v98 import charge_nonce


def write(root, path, value):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else overlay._json_bytes(value)
    file.write_bytes(data)
    return overlay._sha(data)


def snapshot(ledger, history):
    data = overlay._json_bytes({"ledger_sha256": ledger, "reservations": history})
    return {
        "canonical_journal_bytes_base64": base64.b64encode(data).decode(),
        "canonical_journal_sha256": overlay._sha(data),
    }


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT, text=True).strip()


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    base = tmp_path / "v98"
    payload = base / "payload"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    metadata = tmp_path / "metadata"
    old_protocol_path = overlay.BASE["protocol_path"]
    old_auth_path = overlay.BASE["authorization_path"]
    ledger = "a" * 64
    case_file = "runs/science/scene.json"
    old_source = "old_frozen_source.py"
    old_source_data = b"# Old original source\n"
    case_data = b'{"synthetic":"fixture"}\n'
    p = {
        "input_sha256": {case_file: write(payload, case_file, case_data)},
        "source_sha256": {old_source: write(payload, old_source, old_source_data)},
        "output": "runs/solar-modern-arm-v98/pilot",
        "created_utc": "2026-11-04T21:00:00+00:00",
        "latest_start_utc": "2026-11-04T22:00:00+00:00",
        "absolute_deadline_utc": "2026-11-04T22:30:00+00:00",
        "publication_dispatch_lease_seconds": 3600,
        "max_calendar_seconds": 1800,
        "baseline_admission": {
            "initial_reservations": [], "remaining_at_preparation_seconds": 36000,
            "maximum_full_calendar_seconds": 18000, "category": "baseline",
            "refund_on_failure": False,
        },
        "case_ids": ["test1", "test2"], "tracks": ["estimated_sun"],
        "reservation_journal": "runs/budget.json", "compute_ledger_sha256": ledger,
        "calibration_or_final_access": False, "cpu_threads": 2, "split": "model_selection",
        "profiles": [{"name": "test", "backend": "solar"}],
    }
    old_proto_sha = write(payload, old_protocol_path, p)
    entry = {
        "job_id": old_proto_sha + "/evaluation", "seconds": 1800, "category": "baseline",
        "reserved_utc": "2026-11-04T21:05:00+00:00", "refund_on_failure": False,
    }
    old_auth = {
        "nonce": charge_nonce(old_proto_sha, entry), "created_utc": "2026-11-04T21:05:00+00:00",
        **snapshot(ledger, [entry]),
    }
    old_auth_sha = write(payload, old_auth_path, old_auth)
    old_manifest = {
        "schema_version": 75, "capsule_kind": "model_selection_primary_full_sun", "complete": True,
        "protocol_path": old_protocol_path, "protocol_sha256": old_proto_sha,
        "pre_reservation_authorization": {"path": old_auth_path, "sha256": old_auth_sha},
        "files": {**p["input_sha256"], **p["source_sha256"], old_protocol_path: old_proto_sha, old_auth_path: old_auth_sha},
        "max_member_bytes": 2_000_000, "max_total_bytes": 2_000_000, "calendar_seconds": 5.0,
    }
    old_manifest_sha = write(base, "manifest.json", old_manifest)
    base_pin = dict(
        overlay.BASE, protocol_sha256=old_proto_sha, authorization_sha256=old_auth_sha,
        manifest_sha256=old_manifest_sha,
    )
    monkeypatch.setattr(overlay, "BASE", base_pin)
    write(checkout, old_source, old_source_data)
    git(checkout, "init", "-q")
    git(checkout, "config", "user.name", "Local Fixture")
    git(checkout, "config", "user.email", "fixture@example.com")

    new = deepcopy(p)
    new.update(
        output=overlay.OUTPUT, created_utc="2026-11-04T22:40:00+00:00",
        latest_start_utc="2026-11-04T23:40:00+00:00",
        absolute_deadline_utc="2026-11-05T00:10:00+00:00",
        overlay_transport_v110=base_pin,
    )
    new["baseline_admission"].update(initial_reservations=[entry], remaining_at_preparation_seconds=34200)
    new["source_sha256"].update({
        source: write(checkout, source, b"# New fixture source\n") for source in overlay.ADDED_SOURCES
    })
    return base, checkout, metadata, new, entry, case_file, old_source


def fresh_binding(pilot, *, alter_history=None, mutate_auth=None):
    base, checkout, metadata, protocol, old_entry, _, _ = pilot
    digest = write(checkout, overlay.PROTOCOL, protocol)
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", "Synthetic local fixture")
    commit = git(checkout, "rev-parse", "HEAD")
    entry = dict(old_entry, job_id=digest + "/evaluation", reserved_utc="2026-11-04T22:45:00+00:00")
    history = [old_entry, entry] if alter_history is None else alter_history(old_entry, entry)
    nonce = charge_nonce(digest, entry)
    auth = {
        "schema_version": 75, "repository": overlay.REPOSITORY,
        "protocol_path": overlay.PROTOCOL, "protocol_sha256": digest, "source_publication_commit": commit,
        "canonical_journal": protocol["reservation_journal"], "authorized_attempt": 1, "refunds": False,
        "automatic_retry": False, "nonce": nonce, "original_latest_start_utc": protocol["latest_start_utc"],
        "original_absolute_deadline_utc": protocol["absolute_deadline_utc"],
        "max_calendar_seconds": protocol["max_calendar_seconds"], "case_ids": protocol["case_ids"],
        "tracks": ["estimated_sun"], "reservation": {"reservation": entry},
        "created_utc": "2026-11-04T22:45:00+00:00", **snapshot(protocol["compute_ledger_sha256"], history),
    }
    if mutate_auth:
        mutate_auth(auth)
    return {
        "base": overlay.BASE, "protocol_path": overlay.PROTOCOL, "protocol_sha256": digest,
        "authorization_path": overlay.AUTHORIZATION,
        "authorization_sha256": write(metadata, overlay.AUTHORIZATION, auth),
        "source_publication_commit": commit, "nonce": nonce,
    }


def test_overlay_reuses_frozen_payload_and_excludes_old_authorization(pilot, tmp_path):
    base, checkout, metadata, _, _, case_file, old_source = pilot
    binding = fresh_binding(pilot)
    manifest, _ = overlay.plan_overlay(base, checkout, metadata, binding)
    binding["assembled_manifest_sha256"] = overlay._sha(manifest)
    destination = tmp_path / "scratch"
    result = overlay.assemble_overlay(base, checkout, metadata, destination, binding)
    assert result["manifest_sha256"] == binding["assembled_manifest_sha256"]
    new_payload = destination / "payload"
    assert not (new_payload / overlay.BASE["authorization_path"]).exists()
    assert not (new_payload / overlay.BASE["protocol_path"]).exists()
    assert (new_payload / overlay.AUTHORIZATION).exists()
    assert (new_payload / case_file).samefile(base / "payload" / case_file)
    assert (new_payload / old_source).read_bytes() == (checkout / old_source).read_bytes()
    assert (base / "payload" / overlay.BASE["authorization_path"]).exists()
    with pytest.raises(FileExistsError):
        overlay.assemble_overlay(base, checkout, metadata, destination, binding)


@pytest.mark.parametrize("field,value", [
    ("case_ids", ["new-case"]),
    ("profiles", [{"name": "new", "backend": "solar"}]),
    ("input_sha256", {}),
    ("split", "final"),
    ("calibration_or_final_access", True),
    ("tracks", ["estimated_sun", "known_sun"]),
    ("cpu_threads", 3),
    ("max_calendar_seconds", 1801),
])
def test_overlay_cannot_change_scientific_or_budget_contract(pilot, field, value):
    base, checkout, metadata, protocol, *_ = pilot
    protocol[field] = value
    binding = fresh_binding(pilot)
    with pytest.raises(ValueError, match="scientific case"):
        overlay.plan_overlay(base, checkout, metadata, binding)


def test_inherited_sources_and_old_reservation_are_immutable(pilot):
    base, checkout, metadata, protocol, _, _, old_source = pilot
    protocol["source_sha256"][old_source] = "f" * 64
    binding = fresh_binding(pilot)
    with pytest.raises(ValueError, match="inherited source"):
        overlay.plan_overlay(base, checkout, metadata, binding)


@pytest.mark.parametrize("bad_history", ("omit", "rewrite", "duplicate"))
def test_fresh_authorization_must_keep_exact_original_journal_history(pilot, bad_history):
    base, checkout, metadata, *_ = pilot

    def alter(old, new):
        if bad_history == "omit":
            return [new]
        if bad_history == "rewrite":
            return [dict(old, seconds=1799), new]
        return [old, new, new]

    binding = fresh_binding(pilot, alter_history=alter)
    with pytest.raises(ValueError, match="extend"):
        overlay.plan_overlay(base, checkout, metadata, binding)


def test_old_auth_digest_and_new_ticket_manifest_digest_are_both_enforced(pilot, tmp_path):
    base, checkout, metadata, *_ = pilot
    binding = fresh_binding(pilot)
    output = tmp_path / "scratch"
    binding["assembled_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="signed ticket"):
        overlay.assemble_overlay(base, checkout, metadata, output, binding)
    assert not output.exists()
    original = base / "payload" / overlay.BASE["authorization_path"]
    original.write_text(original.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        overlay.plan_overlay(base, checkout, metadata, binding)


def test_fresh_auth_cannot_enable_automatic_retry(pilot):
    base, checkout, metadata, *_ = pilot
    binding = fresh_binding(pilot, mutate_auth=lambda auth: auth.update(automatic_retry=True))
    with pytest.raises(ValueError, match="authorization"):
        overlay.plan_overlay(base, checkout, metadata, binding)


def test_baseline_headroom_must_include_the_original_failed_charge(pilot):
    base, checkout, metadata, protocol, *_ = pilot
    protocol["baseline_admission"]["remaining_at_preparation_seconds"] = 36000
    binding = fresh_binding(pilot)
    with pytest.raises(ValueError, match="headroom"):
        overlay.plan_overlay(base, checkout, metadata, binding)
