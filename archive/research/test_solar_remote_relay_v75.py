"""Procedural metadata/array/stub checks only; no scientific evidence is emitted."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from datetime import datetime, timedelta, timezone
import zipfile
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import solar_remote_relay_v75 as relay
ORIGINAL_PRIMARY_SCOPE = relay.primary_scope


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def authorized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime.now(timezone.utc)
    dump("reports/ledger.json", dict(accounting_complete=True, remaining_baseline_wall_seconds=100, remaining_training_wall_seconds=100))
    p = dict(max_calendar_seconds=10, compute_ledger="reports/ledger.json", compute_ledger_sha256=relay.sha("reports/ledger.json"),
        reservation_journal=relay.CANONICAL_JOURNAL, latest_start_utc=(now + timedelta(minutes=5)).isoformat(),
        absolute_deadline_utc=(now + timedelta(minutes=6)).isoformat(), case_ids=["procedural-only-case"], tracks=["estimated_sun"], purpose="cost_pilot", output="runs/procedural-default")
    dump("reports/protocol.json", p)
    monkeypatch.setattr(relay, "primary_scope", lambda *a, **k: 10)
    monkeypatch.setattr(relay, "published", lambda *a, **k: None)
    auth = relay.pre_reserve("reports/protocol.json", "a" * 40, "reports/authorization.json")
    return dict(protocol=p, auth=auth, authorization="reports/authorization.json", protocol_path="reports/protocol.json")


@pytest.fixture
def ticket(authorized):
    auth = authorized["auth"]
    manifest = dict(schema_version=75, capsule_kind="model_selection_primary_full_sun", protocol_path=auth["protocol_path"],
        protocol_sha256=auth["protocol_sha256"], publication_commit=auth["source_publication_commit"],
        pre_reservation_authorization=dict(path=authorized["authorization"], sha256=relay.sha(authorized["authorization"])),
        files={authorized["authorization"]: relay.sha(authorized["authorization"])})
    dump("capsule/manifest.json", manifest)
    Path("capsule.zip").write_bytes(b"procedural zip-byte hash fixture only")
    parts = [dict(url="https://github.com/ThomasdeChillaz/lunar-image-matching/releases/download/procedural/part.zip",
        sha256=relay.sha("capsule.zip"), bytes=Path("capsule.zip").stat().st_size)]
    result = relay.freeze_ticket(authorized["authorization"], "capsule/manifest.json", "capsule.zip", parts, "reports/ticket.json")
    return dict(authorized, ticket=result, ticket_path="reports/ticket.json", parts=parts)


def test_precharge_is_one_exact_positive_nonrefunded_canonical_entry(authorized):
    state = relay.read(relay.CANONICAL_JOURNAL)
    entry = state["reservations"][0]
    assert entry == authorized["auth"]["reservation"]["reservation"]
    assert entry["job_id"] == relay.sha(authorized["protocol_path"]) + "/evaluation"
    assert entry["seconds"] == 10 and entry["refund_on_failure"] is False
    assert relay.utc(entry["reserved_utc"]).tzinfo is not None
    with pytest.raises(FileExistsError):
        relay.pre_reserve(authorized["protocol_path"], "a" * 40, authorized["authorization"])
    with pytest.raises(ValueError, match="already charged"):
        relay.pre_reserve(authorized["protocol_path"], "a" * 40, "reports/another-auth.json")
    assert len(relay.read(relay.CANONICAL_JOURNAL)["reservations"]) == 1


def test_exact_snapshot_replayed_after_later_append_without_replacing_local(authorized):
    current = relay.read(relay.CANONICAL_JOURNAL)
    current["reservations"].append(dict(job_id="later", category="baseline", seconds=1, reserved_utc="2026-01-01T00:00:00+00:00", refund_on_failure=False))
    dump(relay.CANONICAL_JOURNAL, current); before = Path(relay.CANONICAL_JOURNAL).read_bytes()
    relay.validate_authorization(authorized["auth"], authorized["protocol_path"], True)
    assert Path(relay.CANONICAL_JOURNAL).read_bytes() == before
    current["reservations"][0]["seconds"] = 9; dump(relay.CANONICAL_JOURNAL, current)
    with pytest.raises(ValueError, match="append-only"):
        relay.validate_authorization(authorized["auth"], authorized["protocol_path"], True)


@pytest.mark.parametrize("field,value", [("authorized_attempt", 2), ("refunds", True), ("automatic_retry", True), ("canonical_journal", "runs/new.json"), ("protocol_sha256", "0" * 64)])
def test_authorization_cannot_change_charge_identity_or_add_retry(authorized, field, value):
    changed = copy.deepcopy(authorized["auth"]); changed[field] = value
    with pytest.raises(ValueError): relay.validate_authorization(changed, authorized["protocol_path"])


def test_ticket_binds_identical_snapshot_and_archive_parts(ticket):
    t = relay.validate_ticket(ticket["ticket_path"], relay.sha(ticket["ticket_path"]), ticket["auth"]["nonce"])
    assert t["authorization"]["sha256"] == relay.sha(ticket["authorization"])
    changed = copy.deepcopy(ticket["parts"]); changed[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="part hashes"):
        relay.freeze_ticket(ticket["authorization"], "capsule/manifest.json", "capsule.zip", changed, "reports/other-ticket.json")
    with pytest.raises(FileExistsError):
        relay.freeze_ticket(ticket["authorization"], "capsule/manifest.json", "capsule.zip", ticket["parts"], "reports/second-ticket.json")


@pytest.mark.parametrize("path", ["../x", "/tmp/x", "x/../y", "x//y", "$(touch x)", "`id`", "x;id", "x\ny", "x\\y"])
def test_no_shell_paths_or_relocation(path):
    with pytest.raises(ValueError): relay.relative(path)


def fake_context(ticket):
    env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=relay.REPOSITORY, GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_RUN_ATTEMPT="1",
        RUNNER_OS="Linux", RUNNER_ARCH="ARM64", GITHUB_RUN_ID="123", GITHUB_SHA="b" * 40, GITHUB_WORKFLOW_SHA="b"*40, GITHUB_REF="refs/tags/" + ticket["expected_tag"])
    run = dict(id=123, event="workflow_dispatch", run_attempt=1, head_sha="b" * 40, path=relay.WORKFLOW,
        display_title=ticket["expected_run_title"], repository=dict(full_name=relay.REPOSITORY), head_repository=dict(full_name=relay.REPOSITORY),
        html_url="https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/123", run_started_at="2026-01-01T00:00:00Z")
    return env, run


def test_actual_workflow_source_and_run_identity_required(ticket):
    t = ticket["ticket"]; env, run = fake_context(t)
    def api(path): return run if "/actions/runs/" in path else dict(workflow_runs=[run])
    result = relay.workflow_binding(t, env, api)
    assert result["workflow_run_id"] == 123
    for key, value in [("GITHUB_RUN_ATTEMPT", "2"), ("RUNNER_ARCH", "X64"), ("GITHUB_REPOSITORY", "attacker/repo"), ("GITHUB_REF", "refs/heads/main"), ("GITHUB_WORKFLOW_SHA", "c"*40)]:
        with pytest.raises(ValueError): relay.workflow_binding(t, dict(env, **{key: value}), api)
    wrong = dict(run, head_sha="c" * 40)
    with pytest.raises(ValueError, match="bindings"):
        relay.workflow_binding(t, env, lambda path: wrong if "/actions/runs/" in path else dict(workflow_runs=[wrong]))


def test_duplicate_dispatch_and_rerun_rejected_even_on_new_vm(ticket):
    t = ticket["ticket"]; env, run = fake_context(t)
    duplicate = dict(run, id=124)
    with pytest.raises(ValueError, match="Duplicate"):
        relay.workflow_binding(t, env, lambda path: run if "/actions/runs/" in path else dict(workflow_runs=[run, duplicate]))
    with pytest.raises(ValueError):
        relay.workflow_binding(t, env, lambda path: dict(run, run_attempt=2) if "/actions/runs/" in path else dict(workflow_runs=[run]))


def test_dispatch_marker_survives_uncertain_network_and_prevents_second_attempt(ticket, monkeypatch):
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: Path(ticket["ticket_path"]).read_bytes())
    monkeypatch.setattr(relay, "_gh_json", lambda *a: dict(object=dict(type="commit", sha="b" * 40)))
    calls = []
    def fail(args, **kwargs):
        calls.append(args); raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(relay.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError): relay.dispatch_once(ticket["ticket_path"], "b" * 40)
    assert Path(ticket["ticket_path"] + ".DISPATCHED.json").is_file()
    assert len(relay.read(relay.CANONICAL_JOURNAL)["reservations"]) == 1
    with pytest.raises(FileExistsError): relay.dispatch_once(ticket["ticket_path"], "b" * 40)
    assert len(calls) == 1 and relay.REPOSITORY in calls[0]


def test_remote_restore_exact_bytes_never_creates_a_new_charge(ticket, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    original = Path(relay.CANONICAL_JOURNAL).read_bytes(); Path(relay.CANONICAL_JOURNAL).unlink()
    result = relay.restore_precharged_journal(ticket["ticket"])
    assert Path(relay.CANONICAL_JOURNAL).read_bytes() == original
    assert result == ticket["auth"]["reservation"]
    dump(relay.CANONICAL_JOURNAL, dict(ledger_sha256="0" * 64, reservations=[]))
    before = Path(relay.CANONICAL_JOURNAL).read_bytes()
    with pytest.raises(ValueError, match="Never replace"): relay.restore_precharged_journal(ticket["ticket"])
    assert Path(relay.CANONICAL_JOURNAL).read_bytes() == before


def test_remote_consume_has_no_second_reserve_and_requires_parent_context(ticket, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123"); monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    dump(ticket["ticket_path"] + ".WORKER_CONTEXT.json", dict(ticket_sha256=relay.sha(ticket["ticket_path"]), nonce=ticket["ticket"]["nonce"], workflow_run_id=123, run_attempt=1, parent_pid=os.getppid()))
    before = Path(relay.CANONICAL_JOURNAL).read_bytes()
    assert relay.consume_pre_reservation(ticket["ticket_path"], ticket["protocol_path"]) == ticket["auth"]["reservation"]
    assert Path(relay.CANONICAL_JOURNAL).read_bytes() == before
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    with pytest.raises(ValueError): relay.consume_pre_reservation(ticket["ticket_path"], ticket["protocol_path"])


def test_inner_worker_context_and_exact_original_charge_required(ticket, monkeypatch):
    p = dict(ticket["protocol"], output="runs/procedural-evaluation")
    dump(ticket["protocol_path"], p)
    # Protocol bytes are frozen before authorization in production. This test
    # uses the unchanged fixture protocol through a stub read solely to model
    # the parent's output path without manufacturing scientific evidence.
    original_read = relay.read
    monkeypatch.setattr(relay, "read", lambda path: p if str(path) == ticket["protocol_path"] else original_read(path))
    monkeypatch.setenv("GITHUB_RUN_ID", "123"); monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(relay, "validate_authorization", lambda *a, **k: b"fixture")
    dump(ticket["ticket_path"] + ".WORKER_CONTEXT.json", dict(ticket_sha256=relay.sha(ticket["ticket_path"]), nonce=ticket["ticket"]["nonce"], workflow_run_id=123, run_attempt=1, parent_pid=777))
    dump(Path(p["output"]) / "watchdog_admission.json", dict(controller_pid=os.getppid(), protocol_sha256=relay.sha(ticket["protocol_path"])))
    dump(Path(p["output"]) / "reservation.json", ticket["auth"]["reservation"])
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: "777\n")
    assert relay.validate_worker_ticket(ticket["ticket_path"], ticket["protocol_path"]) == ticket["auth"]["reservation"]
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: "778\n")
    with pytest.raises(ValueError, match="descend"): relay.validate_worker_ticket(ticket["ticket_path"], ticket["protocol_path"])


def test_local_precharge_scope_does_not_apply_current_intel_pilot_software(authorized, monkeypatch):
    import solar_evaluation_protocol_v74 as builder
    import run_solar_evaluation_v70 as runner
    monkeypatch.setattr(runner, "software_identity", lambda: (_ for _ in ()).throw(AssertionError("Local Intel is not target ARM")))
    observed = []
    monkeypatch.setattr(builder, "validate_runtime", lambda *a, **k: observed.append("metadata-only"))
    p = dict(authorized["protocol"], builder_schema_version=74, schema_version=70, prepared_only=False, split="model_selection",
        calibration_or_final_access=False, cpu_threads=2, purpose="modern_comparison", remote_cpu_stratum=relay.RUNTIME,
        profiles=[dict(name="sift", backend="sift")], output="runs/procedural/full", manifest="reports/manifest.json", geometry_audit="reports/audit.json",
        learning_contract="reports/learning.json", input_sha256={}, source_sha256={name:"0"*64 for name in(relay.SELF,relay.WORKFLOW,"solar_capsule_v75.py","solar_portable_evidence_v75.py")})
    assert ORIGINAL_PRIMARY_SCOPE(authorized["protocol_path"], p, check_assets=False) == 10
    assert observed == ["metadata-only"]


def test_browser_preparation_needs_no_gh_and_consumes_same_dispatch_marker(ticket, monkeypatch):
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: Path(ticket["ticket_path"]).read_bytes())
    monkeypatch.setattr(relay, "_gh_json", lambda *a: (_ for _ in ()).throw(AssertionError("No gh/network allowed")))
    result = relay.prepare_browser_dispatch(ticket["ticket_path"], "b"*40)
    assert result["mode"] == "prepared_browser_dispatch_once" and result["inputs"]["ticket_nonce"] == ticket["ticket"]["nonce"]
    with pytest.raises(FileExistsError): relay.prepare_browser_dispatch(ticket["ticket_path"], "b"*40)


def test_lease_applies_to_actual_worker_start_and_never_cross_host_monotonic(ticket):
    t = ticket["ticket"]
    with pytest.raises(TimeoutError): relay._alive_lease(t, relay.utc(t["original_latest_start_utc"]) + timedelta(microseconds=1))
    # Historical authorization remains valid after the launch lease; it is not launch authority.
    relay.validate_authorization(ticket["auth"], ticket["protocol_path"], True)
    assert "start_monotonic" not in ticket["auth"]


def physical_fixture(tmp_path, mask):
    name = str(tmp_path / "reference.npz")
    parameters = np.array([[0., 0.], [1.25, 0.]])
    suns = np.array([[0., 0., 1.], [0.01, 0., np.sqrt(1-.01**2)]])
    arrays = dict(solar_materialized_mask=np.asarray(mask, bool), sun_parameters=parameters, coarse_sun_mask=np.array([True, False]),
        sun_vectors=suns, solar_states=np.zeros((256, 2), np.uint8), solar_variants=np.zeros((256, 2, 7), np.uint8))
    np.savez(name, **arrays)
    sidecar = dict(complete=True, precompute="full", uncomputed_solar_columns=0, computed_solar_columns=2, output_sha256=relay.sha(name), query_truth_used=False,
        sun_catalogue=[dict(hour_angle_deg=h, subsolar_latitude_deg=d, sun_enu=s.tolist()) for (h,d), s in zip(parameters, suns)])
    dump(name + ".json", sidecar)
    return name, arrays, sidecar, {name: dict(receipt_path=name + ".json", sha256=relay.sha(name + ".json"))}


def test_full_computed_unknown_is_valid_but_uncomputed_is_rejected(tmp_path):
    name, arrays, sidecar, receipts = physical_fixture(tmp_path, [True, True])
    relay.validate_primary_archives([dict(reference_geometry=name)], receipts)  # UNKNOWN is retained, never invented clear.
    arrays["solar_materialized_mask"][1] = False; np.savez(name, **arrays)
    sidecar["output_sha256"] = relay.sha(name); dump(name + ".json", sidecar); receipts[name]["sha256"] = relay.sha(name + ".json")
    with pytest.raises(ValueError, match="Uncomputed"): relay.validate_primary_archives([dict(reference_geometry=name)], receipts)


def test_missing_fine_column_or_query_specific_vector_rejected(tmp_path):
    name, arrays, sidecar, receipts = physical_fixture(tmp_path, [True, True])
    arrays["sun_parameters"][1, 0] = 1.1; np.savez(name, **arrays); sidecar["output_sha256"] = relay.sha(name)
    dump(name + ".json", sidecar); receipts[name]["sha256"] = relay.sha(name + ".json")
    with pytest.raises(ValueError, match="generic Sun columns"): relay.validate_primary_archives([dict(reference_geometry=name)], receipts)


def test_zip_traversal_duplicate_and_symlink_rejected(tmp_path):
    for index, name in enumerate(("../escape", "payload/../../escape")):
        archive = tmp_path / f"bad{index}.zip"
        with zipfile.ZipFile(archive, "w") as z: z.writestr(name, b"x")
        with pytest.raises(ValueError): relay.unpack_capsule(archive, tmp_path / f"out{index}")
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("payload/link"); info.external_attr = 0o120777 << 16; z.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symlinks"): relay.unpack_capsule(archive, tmp_path / "linkout")


def test_parent_terminates_nested_scientific_sessions(monkeypatch):
    class Child:
        pid = 100
        def wait(self, timeout=None): return 0
    killed = []
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: "100 1\n101 100\n102 101\n999 1\n")
    monkeypatch.setattr(relay.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    relay.terminate_descendants(Child())
    assert {pid for pid, sig in killed} == {100, 101, 102}
    assert (102, signal.SIGTERM) in killed and (102, signal.SIGKILL) in killed


def test_actual_science_and_partial_failure_outputs_preserved_without_inputs(ticket):
    output = Path("runs/procedural-science"); output.mkdir(parents=True)
    dump(output / "execution.json", dict(procedural=True))
    dump(output / "FAILED.json", dict(procedural=True))
    (output / "case.npz").write_bytes(b"procedural-only array placeholder")
    (output / "worker.log").write_text("procedural worker failure\n")
    remote = Path("runs/remote-test"); remote.mkdir(); (remote / "capsule.zip").write_bytes(b"do not return model/input archive")
    result = relay.preserve_results(remote, dict(output=str(output)), ticket["ticket_path"], ticket["protocol_path"])
    with zipfile.ZipFile(result["path"]) as archive:
        names = archive.namelist()
        assert str(output / "case.npz") in names and str(output / "FAILED.json") in names and str(output / "worker.log") in names
        assert not any(name.endswith("capsule.zip") for name in names)
    assert result["includes_failed_partial_outputs"] is True


def test_bootstrap_failure_preserves_failed_charged_ticket_receipt(ticket, monkeypatch):
    monkeypatch.setattr(relay, "workflow_binding", lambda *a: dict(workflow_source_commit="b"*40, workflow_run_id=123, run_attempt=1))
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **k: Path(ticket["ticket_path"]).read_bytes())
    monkeypatch.setattr(relay, "fetch_capsule", lambda *a, **k: (_ for _ in ()).throw(ValueError("procedural broken capsule")))
    Path(relay.WORKFLOW).parent.mkdir(parents=True, exist_ok=True); Path(relay.WORKFLOW).write_text("procedural source placeholder")
    before = Path(relay.CANONICAL_JOURNAL).read_bytes()
    with pytest.raises(ValueError, match="broken capsule"):
        relay.execute_remote(ticket["ticket_path"], relay.sha(ticket["ticket_path"]), ticket["ticket"]["nonce"])
    record = relay.read(Path("runs/solar-remote-v75") / ticket["ticket"]["nonce"] / "remote_watchdog.json")
    assert record["complete"] is False and record["failure"]["phase"] == "setup"
    assert record["original_reservation"] == ticket["auth"]["reservation"]
    assert Path(relay.CANONICAL_JOURNAL).read_bytes() == before and record["reservation_refunded"] is False


def test_workflow_keeps_readonly_official_cpu_profile_and_quoted_inputs():
    text = (Path(__file__).parent / ".github/workflows/solar-remote-cpu-v75.yml").read_text()
    assert "runs-on: ubuntu-24.04-arm" in text and "contents: read" in text and "actions: read" in text
    assert "torch==2.8.0 torchvision==0.23.0" in text and "timeout-minutes: 360" in text
    assert '"$TICKET_PATH"' in text and "cancel-in-progress: false" in text and "persist-credentials: false" in text


def test_optional_runner_loads_full_physics_without_linux_blender(monkeypatch):
    path = Path(__file__).with_name("run_solar_evaluation_v70.py")
    spec = importlib.util.spec_from_file_location("remote_candidate_runner_for_test", path)
    candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
    geometry = SimpleNamespace(solar_materialized_mask=np.array([True, True]))
    monkeypatch.setitem(sys.modules, "solar_frontend_v63", SimpleNamespace(ReferenceGeometry=SimpleNamespace(load=lambda path: geometry)))
    calls = []
    def unexpected(*args): calls.append(args); raise AssertionError("Remote primary cannot construct Blender lookup")
    monkeypatch.setitem(sys.modules, "solar_lookup_v70", SimpleNamespace(ReferenceSunLookup=SimpleNamespace(from_protocol=unexpected)))
    class FakeImage:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def convert(self, mode): return np.zeros((512, 512), np.uint8)
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda path: FakeImage())
    case = dict(reference="procedural-reference.png", query="procedural-query.png", reference_geometry="procedural-only.npz",
        solar_lookup_protocol="opaque-original-protocol.json", query_intrinsics=np.eye(3).tolist(), reference_sun_world=[0.,0.,1.])
    loader = candidate.InputLoader("a"*40, "/original/Mac/Blender", fully_materialized_primary=True)
    result = loader.get(case, "estimated_sun")
    assert result.geometry is geometry and result.solar_lookup is None and not calls
    with pytest.raises(ValueError, match="primary"): loader.get(case, "known_sun")
    geometry.solar_materialized_mask[1] = False
    loader = candidate.InputLoader(fully_materialized_primary=True)
    with pytest.raises(ValueError, match="all physical columns"): loader.get(case, "estimated_sun")


def test_optional_worker_rejects_bad_ticket_before_any_torch_or_model_import(monkeypatch):
    path = Path(__file__).with_name("run_solar_evaluation_v70.py")
    spec = importlib.util.spec_from_file_location("remote_worker_guard_for_test", path)
    candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
    monkeypatch.setattr(relay, "validate_worker_ticket", lambda *a: (_ for _ in ()).throw(ValueError("bad parent context")))
    with pytest.raises(ValueError, match="bad parent context"): candidate.worker("procedural.json", "a"*40, "procedural-ticket.json")
