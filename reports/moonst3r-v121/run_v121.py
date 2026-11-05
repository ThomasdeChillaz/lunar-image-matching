"""Guarded local MOONSt3R pilot and separate model-free saved-array replay."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from zipfile import ZipFile

from prepare_v121 import (
    CAP, COHORT, COHORT_SHA, DIR, EXPECTED, OUTPUT, PROTOCOL, ROOT, V115, V115_SHA,
    build, checked, protocol_bytes, repo_file, sha,
)


DISPATCH = DIR + "/dispatch.json"
PARENT_OUTPUT = "runs/solar-five-arm-v115/comparison"
NAMES = {
    "official_run": "github_v115_metadata.json",
    "artifact_audit": "artifact_audit_v115.json",
    "summary": PARENT_OUTPUT + "/summary.json",
    "seal": PARENT_OUTPUT + "/COMPLETE.json",
    "replay": "independent_replay_v115.json",
    "comparison": "matched_comparison_v115.json",
    "nested_cell_inventory": "nested_cell_inventory_v115.json",
}
ENV = dict(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
           VECLIB_MAXIMUM_THREADS="2", NUMEXPR_MAX_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
TRACK = "estimated_sun"


def utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("Timezone-aware frozen clock required")
    return stamp.astimezone(timezone.utc)


def now():
    return datetime.now(timezone.utc)


def file_sha(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_new(path, value):
    with path.open("xb") as file:
        file.write(protocol_bytes(value))


def load_protocol():
    path = repo_file(ROOT, PROTOCOL)
    raw = path.read_bytes()
    prepared = build()
    if raw != protocol_bytes(prepared):
        raise ValueError("Prospective protocol differs from the published case-selection rule or source pins")
    return prepared, sha(raw)


def read_dispatch(protocol_sha, protocol):
    path = repo_file(ROOT, DISPATCH)
    raw = path.read_bytes()
    result = json.loads(raw)
    prior = protocol["prerequisite_v115"]
    remote = "runs/solar-remote-five-arm-v115/" + prior["nonce"]
    names = {**NAMES, "outer_watchdog": remote + "/remote_watchdog.json", "archive": remote + "/scientific_results.zip"}
    if (set(result) != {"schema", "execution_authorized", "protocol_sha256", "max_calendar_seconds",
                        "created_utc", "latest_start_utc", "absolute_deadline_utc", "v115_evidence",
                        "v115_official_zip_sha256", "artifact_verifier_source_sha256"}
            or result["schema"] != "moonst3r_v121_local_dispatch"
            or result["execution_authorized"] is not True or result["protocol_sha256"] != protocol_sha
            or result["max_calendar_seconds"] != CAP or set(result["v115_evidence"]) != set(names)
            or result["artifact_verifier_source_sha256"] != prior["artifact_verifier_source_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", result["v115_official_zip_sha256"])):
        raise ValueError("Published v121 authorization and all original v115 terminal evidence are required")
    for role, value in result["v115_evidence"].items():
        if (set(value) != {"path", "sha256"} or value["path"] != names[role]
                or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
            raise ValueError("Each v115 terminal file requires a path and a full digest")
    created, latest, absolute = (utc(result[k]) for k in ("created_utc", "latest_start_utc", "absolute_deadline_utc"))
    if not created < latest <= created + timedelta(minutes=10) or absolute != latest + timedelta(seconds=CAP):
        raise ValueError("Invalid immutable dispatch lease or cap")
    return result, sha(raw)


def publication(protocol, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Full public source commit required")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    for path in (PROTOCOL, DISPATCH, *protocol["source_sha256"]):
        local = repo_file(ROOT, path).read_bytes()
        if subprocess.check_output(["git", "show", f"{commit}:{path}"], timeout=5) != local:
            raise ValueError(f"Local bytes differ from public main for {path}")


def check_official_terminal(record, watchdog, protocol, dispatch):
    prior = protocol["prerequisite_v115"]
    started = utc(record["run_started_at"])
    ended = utc(watchdog["started_utc"]) + timedelta(seconds=watchdog["calendar_seconds"])
    if (record["run_id"] != prior["run_id"]
            or record["status"] != "completed" or record["conclusion"] != "success"
            or record["job"]["conclusion"] != "success" or set(record["artifacts"]) != {"outer"}
            or record["ticket_commit"] != prior["ticket_commit"] or record["ticket_sha256"] != prior["ticket_sha256"]
            or record["protocol_sha256"] != prior["sha256"] or record["source_commit"] != prior["source_commit"]
            or record["nonce"] != prior["nonce"]
            or record["html_url"] != f"https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/{prior['run_id']}"
            or record["html_url"] != watchdog["html_url"] or record["ticket_commit"] != watchdog["workflow_source_commit"]
            or record["ticket_sha256"] != watchdog["ticket_sha256"] or record["nonce"] != watchdog["nonce"]
            or started != utc(watchdog["workflow_started_utc"])
            or watchdog["workflow_run_id"] != prior["run_id"] or watchdog["run_attempt"] != prior["run_attempt"]
            or watchdog["source_publication_commit"] != prior["source_commit"]
            or not started < ended <= utc(dispatch["created_utc"])):
        raise ValueError("Original official v115 attempt must have terminated successfully before dispatch")


def check_artifact_audit(audit, record, evidence, protocol):
    prior = protocol["prerequisite_v115"]
    record_roles = {"nested_cell_inventory": "nested_cell_inventory", "replay": "replay", "comparison": "comparison",
                    "scientific_archive": "archive", "outer_watchdog": "outer_watchdog"}
    if (audit["schema"] != "solar_five_arm_official_artifact_audit_v115"
            or audit["run_id"] != prior["run_id"] or audit["conclusion"] != "success"
            or audit["complete_archive_verified"] is not True or audit["independently_replayed"] is not True
            or audit["new_inference"] is not False or audit["ticket_sha256"] != prior["ticket_sha256"]
            or audit["protocol_sha256"] != prior["sha256"] or audit["archived_downloads"] != record["artifacts"]
            or audit["records"] != {role: evidence[source] for role, source in record_roles.items()}):
        raise ValueError("Require the exact official archive verifier receipt with independent replay")


def recorded_v115_charge(dispatch, audit_root):
    entry = dispatch["v115_evidence"]["outer_watchdog"]
    data = json.loads(checked(audit_root.resolve(strict=True), entry["path"], entry["sha256"]))
    return data["original_reservation"]["reservation"]


def terminal_audit(protocol, dispatch, audit_root, official_zip):
    """Validate the original audit and archive without opening saved inference arrays."""
    from summarize_solar_five_arm_v115 import (
        _archive_controls, _one_member, _worker, cohort_cases, validate_evidence, validate_rows,
    )

    audit_root = audit_root.resolve(strict=True)
    if not audit_root.is_dir() or audit_root.is_relative_to(ROOT):
        raise ValueError("Use the official verifier's preserved evidence directory outside the scientific checkout")
    evidence = dispatch["v115_evidence"]
    data = {role: json.loads(checked(audit_root, entry["path"], entry["sha256"]))
            for role, entry in evidence.items() if role != "archive"}
    watchdog, summary = data["outer_watchdog"], data["summary"]
    official = data["official_run"]
    check_official_terminal(official, watchdog, protocol, dispatch)
    check_artifact_audit(data["artifact_audit"], official, evidence, protocol)
    rest = official["artifacts"]["outer"]
    expected_name = f"solar-five-arm-v115-{official['nonce']}-{official['run_id']}-attempt-1"
    expected_url = f"https://api.github.com/repos/ThomasdeChillaz/lunar-image-matching/actions/artifacts/{rest['id']}/zip"
    outer_sha = dispatch["v115_official_zip_sha256"]
    official_zip = official_zip.resolve(strict=True)
    if (rest["name"] != expected_name or rest["archive_download_url"] != expected_url or rest["expired"] is not False
            or rest["digest"] != f"sha256:{outer_sha}" or official_zip.stat().st_size != rest["size_in_bytes"]
            or file_sha(official_zip) != outer_sha):
        raise ValueError("Original Actions ZIP differs from exact GitHub REST SHA-256 and size")
    with ZipFile(official_zip) as original_outer:
        for role in ("outer_watchdog", "archive"):
            with _one_member(original_outer, evidence[role]["path"]) as stream:
                if file_digest(stream, "sha256").hexdigest() != evidence[role]["sha256"]:
                    raise ValueError(f"Official Actions ZIP contains a different {role}")
    inventory, comparison = data["nested_cell_inventory"], data["comparison"]["evidence"]
    if (inventory["protocol_sha256"] != V115_SHA or inventory["expected_cells"] != 480
            or inventory["sealed_cells"] != 480 or inventory["invalid_markers"]
            or any(comparison[f"{key}_sha256"] != wanted for key, wanted in {
                "protocol": V115_SHA, **{k: evidence[k]["sha256"] for k in ("summary", "seal", "replay", "archive")},
                "watchdog": evidence["outer_watchdog"]["sha256"],
            }.items()) or comparison["workflow_run_id"] != official["run_id"]
            or comparison["source_commit"] != official["source_commit"]):
        raise ValueError("Official sealed cell inventory or independent comparison binding differs")
    original = json.loads(checked(ROOT, V115, V115_SHA))
    cohort = json.loads(checked(ROOT, COHORT, COHORT_SHA))
    manifest = json.loads(checked(ROOT, protocol["manifest"], protocol["input_sha256"][protocol["manifest"]]))
    cases, _ = cohort_cases(cohort, manifest)
    validate_evidence(original, V115_SHA, summary, evidence["summary"]["sha256"], data["seal"], data["replay"],
                      watchdog, cohort, manifest)
    _worker(summary)
    validate_rows(summary, data["replay"], cases)
    archive = repo_file(audit_root, evidence["archive"]["path"])
    if evidence["archive"]["sha256"] != watchdog["result_artifact"]["sha256"] or file_sha(archive) != evidence["archive"]["sha256"]:
        raise ValueError("Original v115 raw archive differs from the outer watchdog")
    with ZipFile(archive) as zipped:
        for role, filename in (("summary", "summary.json"), ("seal", "COMPLETE.json")):
            with _one_member(zipped, f"{original['output']}/{filename}") as stream:
                if file_digest(stream, "sha256").hexdigest() != evidence[role]["sha256"]:
                    raise ValueError(f"Archived {role} differs from the audited local copy")
        _archive_controls(zipped, original, V115_SHA)
    return watchdog["original_reservation"]["reservation"]


def budget_charge(protocol, protocol_sha, dispatch, started, parent_charge):
    from solar_learning_pipeline_v73 import journal_state, remaining

    pins = protocol["budget"]
    contract = json.loads(checked(ROOT, pins["learning_contract"], pins["learning_contract_sha256"]))
    if (contract["compute_ledger"], contract["compute_ledger_sha256"], contract["reservation_journal"]) != (
        pins["ledger"], pins["ledger_sha256"], pins["canonical_journal"],
    ):
        raise ValueError("Original canonical budget contract differs")
    _, journal, _ = journal_state(pins["ledger"], pins["ledger_sha256"], pins["canonical_journal"])
    rows = journal["reservations"]
    if parent_charge not in rows or parent_charge["job_id"] != V115_SHA + "/evaluation":
        raise ValueError("Original v115 reservation must remain charged")
    charge, = [row for row in rows if row["job_id"] == protocol_sha + pins["job_suffix"]]
    if (charge["category"] != "baseline" or charge["seconds"] != CAP or charge["refund_on_failure"] is not False
            or not utc(dispatch["created_utc"]) <= utc(charge["reserved_utc"]) <= started <= utc(dispatch["latest_start_utc"])):
        raise ValueError("Original fixed no-refund local baseline charge or lease differs")
    return charge, remaining(contract)["baseline"]


def runtime_check(protocol):
    expected = protocol["runtime"]
    if platform.machine() != expected["machine"] or platform.python_version() != expected["python"]:
        raise ValueError("Use the preserved local x86_64 Python runtime")
    for name, version in expected["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Locked runtime differs for {name}")
    if any(os.environ.get(key) != value for key, value in ENV.items()):
        raise ValueError("Use the exact two-thread worker environment")


def require_root():
    if Path.cwd() != ROOT:
        raise ValueError("Run from the lunar-image-matching repository root")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def worker(commit, controller_pid, audit_root, official_zip):
    protocol, protocol_sha = load_protocol()
    dispatch, dispatch_sha = read_dispatch(protocol_sha, protocol)
    root = repo_file(ROOT, OUTPUT)
    receipt = read_json(root / "execution.json")
    if (controller_pid != os.getppid() or receipt["controller_pid"] != controller_pid
            or receipt["publication_commit"] != commit or receipt["protocol_sha256"] != protocol_sha
            or receipt["dispatch_sha256"] != dispatch_sha or receipt["v115_terminal_audited"] is not True):
        raise ValueError("Worker requires the original admitted local controller and exact execution receipt")
    started = utc(receipt["started_utc"])
    publication(protocol, commit)
    prior = recorded_v115_charge(dispatch, audit_root)
    charge, _ = budget_charge(protocol, protocol_sha, dispatch, started, prior)
    if (charge != receipt["reservation"] or prior != receipt["original_v115_charge"]
            or prior != terminal_audit(protocol, dispatch, audit_root, official_zip)):
        raise ValueError("Worker charge differs from the controller receipt")

    def check():
        if time.monotonic() - receipt["start_monotonic"] >= CAP or now() >= utc(receipt["hard_deadline_utc"]):
            raise TimeoutError("Original local two-thread calendar cap exhausted")

    check()
    runtime_check(protocol)
    import cv2
    import numpy as np
    import torch
    from native_baseline_pose_v7 import controls
    from native_pretrained_adapters_v73 import replay_native_control
    from run_solar_evaluation_v110 import InferenceResultCapture, InputLoader, software_identity, worker_peak_rss_snapshot
    from run_solar_five_arm_v115 import evaluate_one
    from solar_evaluation_v70 import Backend, digest

    torch.set_num_threads(2)
    torch.set_num_interop_threads(protocol["runtime"]["interop_threads"])
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    profile, = protocol["profiles"]
    manifest = json.loads(checked(ROOT, protocol["manifest"], protocol["input_sha256"][protocol["manifest"]]))
    by_id = {case["case_id"]: case for case in manifest["cases"]}
    peak_before = worker_peak_rss_snapshot()
    tick = time.monotonic()
    backend = InferenceResultCapture(Backend(profile))
    model_load = time.monotonic() - tick
    check()
    write_new(root / "provenance.json", backend.provenance)
    tick = time.monotonic()
    control = controls(backend.model)
    write_new(root / "fresh_control.json", control)
    replay_native_control(control)
    control_time = time.monotonic() - tick
    check()
    rows = []
    loader = InputLoader(commit, fully_materialized_primary=True)
    try:
        for case_id in EXPECTED:
            check()
            tick = time.monotonic()
            case = by_id[case_id]
            row, arrays = evaluate_one(backend, loader, case, protocol)
            key = digest(dict(case=case, profile=profile, track=TRACK, sources=protocol["source_sha256"]))
            array_path = root / f"{key}.npz"
            with array_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row.update({name: case[name] for name in ("case_id", "component", "query_component", "positive",
                                                      "unit_index", "query_unit_index", "light_a", "light_b")})
            row.update(profile=profile["name"], track=TRACK, saved_arrays=str(array_path.relative_to(ROOT)),
                       saved_arrays_sha256=file_sha(array_path), reused_case_cache=False,
                       case_calendar_seconds=time.monotonic() - tick)
            row_path = root / f"{key}.json"
            write_new(row_path, row)
            write_new(root / f"{key}.COMPLETE.json", dict(
                complete=True, identity=key, row_sha256=file_sha(row_path), arrays_sha256=row["saved_arrays_sha256"],
            ))
            rows.append(row)
            check()
    finally:
        loader.close()
    write_new(root / "summary.json", dict(
        complete=True, protocol_sha256=protocol_sha, dispatch_sha256=dispatch_sha, purpose=protocol["purpose"],
        manifest_sha256=protocol["input_sha256"][protocol["manifest"]], accepted_results=False,
        calibration_or_final_access=False, fresh_inference_cases=len(rows), cached_cases=0, rows=rows,
        fresh_control_sha256=file_sha(root / "fresh_control.json"), model_load_seconds=model_load,
        fresh_control_seconds=control_time, software=software_identity(),
        process_lifetime_peak_before=peak_before, process_lifetime_peak_after=worker_peak_rss_snapshot(),
        comparison_scope="Same eight v115 development cases. Local hardware latency only. Accuracy pilot without ranking.",
    ))


def run(commit, dependencies, audit_root, official_zip):
    start_tick, started = time.monotonic(), now()
    protocol, protocol_sha = load_protocol()
    dispatch, dispatch_sha = read_dispatch(protocol_sha, protocol)
    publication(protocol, commit)
    if not utc(dispatch["created_utc"]) <= started <= utc(dispatch["latest_start_utc"]):
        raise ValueError("Run is outside the original published dispatch lease")
    original_v115_charge = recorded_v115_charge(dispatch, audit_root)
    charge, remaining = budget_charge(protocol, protocol_sha, dispatch, started, original_v115_charge)
    if original_v115_charge != terminal_audit(protocol, dispatch, audit_root, official_zip):
        raise ValueError("Exact original v115 reservation differs after terminal audit")
    deadline = min(started + timedelta(seconds=CAP), utc(dispatch["absolute_deadline_utc"]))
    if now() >= deadline or time.monotonic() - start_tick >= CAP:
        raise TimeoutError("Admission consumed the original allowance")
    deps = dependencies.resolve(strict=True)
    if not deps.is_dir():
        raise ValueError("Supply the existing locked local dependency directory")
    root = repo_file(ROOT, OUTPUT)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "execution.json", dict(
        controller_pid=os.getpid(), publication_commit=commit, protocol_sha256=protocol_sha, dispatch_sha256=dispatch_sha,
        started_utc=started.isoformat(), start_monotonic=start_tick, hard_deadline_utc=deadline.isoformat(),
        v115_terminal_audited=True, original_v115_charge=original_v115_charge, reservation=charge,
        remaining_canonical_baseline_seconds=remaining, cpu_threads=2,
    ))
    child, failure, interrupts = None, None, []
    previous = {signum: signal.signal(signum, lambda sig, _: interrupts.append(sig)) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (root / "worker.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, "worker", "--publication-commit", commit, "--controller-pid", str(os.getpid()),
                 "--audit-root", str(audit_root.resolve()), "--official-zip", str(official_zip.resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, **ENV, "PYTHONPATH": str(deps)},
            )
            while child.poll() is None:
                if interrupts or now() >= deadline or time.monotonic() - start_tick >= CAP:
                    raise TimeoutError("Original external calendar allowance exhausted or interrupted")
                time.sleep(.1)
        if child.returncode or interrupts or now() >= deadline or time.monotonic() - start_tick >= CAP:
            raise RuntimeError("Local worker failed or finished outside its original cap")
        summary = read_json(root / "summary.json")
        if (summary["complete"] is not True or summary["protocol_sha256"] != protocol_sha
                or summary["dispatch_sha256"] != dispatch_sha or tuple(row["case_id"] for row in summary["rows"]) != EXPECTED):
            raise ValueError("Completed worker inventory does not match the prospective cohort")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        elapsed = time.monotonic() - start_tick
        if failure is None and (elapsed >= CAP - 1 or now() >= deadline - timedelta(seconds=1)):
            failure = {"type": "TimeoutError", "message": "Original cap exhausted while sealing"}
        write_new(root / "watchdog.json", dict(
            complete=failure is None, failure=failure, elapsed_seconds=elapsed, reservation=charge,
            child_returncode=None if child is None else child.returncode,
            protocol_sha256=protocol_sha, dispatch_sha256=dispatch_sha,
        ))
        if failure is not None:
            write_new(root / "FAILED.json", failure)
    if failure is not None:
        raise TimeoutError(failure["message"])
    write_new(root / "COMPLETE.json", dict(
        complete=True, external_watchdog_passed=True, summary_sha256=file_sha(root / "summary.json"),
        watchdog_sha256=file_sha(root / "watchdog.json"), protocol_sha256=protocol_sha, dispatch_sha256=dispatch_sha,
    ))


def replay(commit, output):
    """Recompute the saved raw metrics with the independent v98 verifier."""
    protocol, protocol_sha = load_protocol()
    dispatch, dispatch_sha = read_dispatch(protocol_sha, protocol)
    publication(protocol, commit)
    root = repo_file(ROOT, OUTPUT)
    execution, watchdog, seal = (read_json(root / filename) for filename in ("execution.json", "watchdog.json", "COMPLETE.json"))
    if ((root / "FAILED.json").exists() or seal["complete"] is not True or watchdog["complete"] is not True
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or watchdog["reservation"] != execution["reservation"] or execution["publication_commit"] != commit
            or execution["v115_terminal_audited"] is not True or not 0 < watchdog["elapsed_seconds"] < CAP
            or seal["watchdog_sha256"] != file_sha(root / "watchdog.json")
            or any(record["dispatch_sha256"] != dispatch_sha or record["protocol_sha256"] != protocol_sha
                   for record in (execution, watchdog, seal))):
        raise ValueError("Original local watchdog, publication and completed comparison seal required")
    budget_charge(protocol, protocol_sha, dispatch, utc(execution["started_utc"]), execution["original_v115_charge"])
    summary = read_json(root / "summary.json")
    if (tuple(row["case_id"] for row in summary["rows"]) != EXPECTED
            or summary["fresh_control_sha256"] != file_sha(root / "fresh_control.json")):
        raise ValueError("Fresh control or row inventory differs")
    for row in summary["rows"]:
        path = repo_file(ROOT, row["saved_arrays"])
        if path.parent != root:
            raise ValueError("Saved predictions escape the sealed local run")
        key = path.stem
        row_path = root / f"{key}.json"
        row_seal = read_json(root / f"{key}.COMPLETE.json")
        if (row_seal["complete"] is not True or row_seal["identity"] != key
                or read_json(row_path) != row or row_seal["row_sha256"] != file_sha(row_path)
                or row_seal["arrays_sha256"] != row["saved_arrays_sha256"]):
            raise ValueError("Individual prediction row seal differs")
    from verify_solar_evaluation_v98 import replay as verify

    return verify(root, PROTOCOL, output)


if __name__ == "__main__":
    require_root()
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    for action, key, kind in (("run", "dependencies", Path), ("worker", "controller-pid", int), ("replay", "output", Path)):
        command = sub.add_parser(action)
        command.add_argument("--publication-commit", required=True)
        command.add_argument("--" + key, type=kind, required=True)
        if action != "replay":
            command.add_argument("--audit-root", type=Path, required=True)
            command.add_argument("--official-zip", type=Path, required=True)
    args = cli.parse_args()
    if args.action == "run":
        run(args.publication_commit, args.dependencies, args.audit_root, args.official_zip)
    elif args.action == "worker":
        worker(args.publication_commit, args.controller_pid, args.audit_root, args.official_zip)
    else:
        replay(args.publication_commit, args.output)
