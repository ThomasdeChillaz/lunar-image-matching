"""Source-published paired development pilot over the frozen v86 case bank."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
PILOT = "reports/solar-distributional100-pilot-v89.json"
REPLAY = "reports/solar-distributional100-replay-v89.json"
LEDGER = "reports/solar_compute_admission_v67.json"
JOURNAL = "runs/solar-compute-reservations-v67.json"
JOB_SUFFIX = "/matched_development_evaluation"
TRACKS = ["estimated_sun"]
CONTROL = "frozen_step0_A1B0C1_quantization_aware"
PROFILE_NAMES = [CONTROL, "coarse_response_step0", "coarse_response_step100"]
COARSE_SOURCE = "solar_distributional_evaluation_v90.py"
RUNNER_SOURCE = "run_solar_distributional_evaluation_v90.py"


def utc_now():
    return datetime.now(timezone.utc)


def utc(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() != timedelta(0):
        raise ValueError("Explicit UTC clock required")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fixed_v86_cases(basis_path=BASIS, full=False):
    from solar_evaluation_v70 import validate_manifest

    basis = read(basis_path)
    if basis["stage_mode"] != "preselection_compact_controls" or basis["tracks"] != TRACKS:
        raise ValueError("Original frozen primary v86 development pilot required")
    manifest = read(basis["manifest"])
    audit = read(basis["geometry_audit"])
    for path in (basis["manifest"], basis["geometry_audit"]):
        if sha(path) != basis["input_sha256"][path]:
            raise ValueError("Original selection manifest/audit changed")
    if manifest["geometry_audit_sha256"] != sha(basis["geometry_audit"]):
        raise ValueError("Manifest geometry lineage differs")
    cases = validate_manifest(manifest, audit)
    if [case["case_id"] for case in cases] != basis["case_ids"] or len(cases) != 384:
        raise ValueError("Original complete 384-case bank required")
    index = {case["case_id"]: case for case in cases}
    selected = [index[key] for key in basis["pilot_case_ids"]]
    if len({case["case_id"] for case in selected}) != 8 or sum(case["positive"] for case in selected) != 4:
        raise ValueError("Exact v86 eight-case paired pilot required")
    return basis, manifest, cases if full else selected


def check_publication(protocol_path, source_hashes, public_metadata_hashes, commit, check):
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Full hexadecimal public source commit required")
    required = {**source_hashes, **public_metadata_hashes, str(protocol_path): sha(protocol_path)}
    for path, expected in required.items():
        check()
        local = Path(path).read_bytes()
        if hashlib.sha256(local).hexdigest() != expected:
            raise ValueError("Frozen file changed: " + path)
        public = subprocess.check_output(["git", "show", commit + ":" + path], timeout=5)
        if public != local:
            raise ValueError("File not published byte-for-byte: " + path)
        check()


def validate_dispatch(p, started, cap_ceiling):
    cap = p["max_calendar_seconds"]
    if type(cap) is not int or not 5 < cap <= cap_ceiling:
        raise ValueError("Explicit fixed calendar cap within original ceiling required")
    created = utc(p["created_utc"])
    latest = utc(p["latest_start_utc"])
    absolute = utc(p["absolute_deadline_utc"])
    lease = p["publication_dispatch_lease_seconds"]
    if type(lease) is not int or not 0 < lease <= 1800 or latest != created + timedelta(seconds=lease):
        raise ValueError("Invalid frozen dispatch lease")
    if absolute != latest + timedelta(seconds=cap) or not created <= started <= latest:
        raise ValueError("Dispatch outside the original published lease")
    return min(started + timedelta(seconds=cap), absolute)


def check_common_protocol(p):
    output = Path(p["output"])
    if output.is_absolute() or not output.is_relative_to("runs") or ".." in output.parts:
        raise ValueError("Use a new repository-relative runs/ directory")
    if (p.get("prepared_only") is not False or p["cpu_threads"] != 2 or p["tracks"] != TRACKS
            or p["split"] != "model_selection" or p["calibration_or_final_access"]
            or p["compute_ledger"] != LEDGER or p["reservation_journal"] != JOURNAL):
        raise ValueError("Admitted two-thread development-only primary protocol required")


def evaluate_profiles(p, cases, commit, check, backend_factory):
    import numpy as np
    from run_solar_evaluation_v70 import (
        InferenceResultCapture,
        InputLoader,
        capture_case_report,
        worker_peak_rss_snapshot,
    )
    from solar_evaluation_v70 import digest, evaluate_case

    root = Path(p["output"])
    rows = []
    profile_memory = []
    loader = InputLoader(commit, p.get("blender_executable"))
    try:
        for profile in p["profiles"]:
            check()
            memory = worker_peak_rss_snapshot()
            tick = time.perf_counter()
            backend = InferenceResultCapture(backend_factory(profile))
            load_seconds = time.perf_counter() - tick
            write(root / (profile["name"] + ".provenance.json"), backend.provenance)
            for case_index, case in enumerate(cases):
                check()
                tick = time.perf_counter()
                for path, expected in case["input_sha256"].items():
                    if sha(path) != expected:
                        raise ValueError("Frozen case input differs: " + path)
                inp = loader.get(case, TRACKS[0])
                if inp.known_query_sun is not None:
                    raise ValueError("Primary inference supplied query Sun truth")
                row, arrays = evaluate_case(backend, inp, case)
                row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(backend, arrays)
                key = digest(dict(case=case, profile=profile, track=TRACKS[0], sources=p["source_sha256"]))
                array_path = root / (key + ".npz")
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row.update({name: case[name] for name in (
                    "case_id", "component", "query_component", "positive", "unit_index", "query_unit_index",
                    "light_a", "light_b",
                )})
                row.update(
                    profile=profile["name"],
                    track=TRACKS[0],
                    reference_setup=loader.last_setup.copy(),
                    complete_localization_cold_seconds=row["inference_seconds"] + sum(loader.last_setup.values()),
                    case_calendar_seconds=time.perf_counter() - tick,
                    profile_load_seconds=load_seconds if case_index == 0 else 0.0,
                    saved_arrays=str(array_path),
                    saved_arrays_sha256=sha(array_path),
                )
                row_path = root / (key + ".json")
                write(row_path, row)
                write(root / (key + ".COMPLETE.json"), dict(
                    complete=True, identity=key, row_path=str(row_path), row_sha256=sha(row_path),
                    arrays_path=str(array_path), arrays_sha256=row["saved_arrays_sha256"],
                ))
                rows.append(row)
                print(profile["name"], case["case_id"], row["reason"], flush=True)
            profile_memory.append(dict(profile=profile["name"], before=memory, after=worker_peak_rss_snapshot()))
            del backend
    finally:
        loader.close()
    return rows, profile_memory


def guarded_worker(path, commit, controller_pid, admission, backend_factory):
    p = read(path)
    root = Path(p["output"])
    execution = read(root / "execution.json")
    if os.getppid() != controller_pid or execution["controller_pid"] != controller_pid:
        raise ValueError("Worker requires the original live controller")
    if execution["protocol_sha256"] != sha(path) or execution["publication_commit"] != commit:
        raise ValueError("Worker source provenance differs")
    receipt = read(root / "reservation.json")["reservation"]
    journal = read(JOURNAL)
    if journal["ledger_sha256"] != p["compute_ledger_sha256"] or receipt not in journal["reservations"]:
        raise ValueError("Exact worker reservation absent from original journal")
    if (receipt["job_id"] != sha(path) + JOB_SUFFIX or receipt["category"] != "baseline"
            or receipt["seconds"] != p["max_calendar_seconds"] or receipt["refund_on_failure"]):
        raise ValueError("Worker budget identity differs")
    deadline = utc(execution["work_deadline_utc"])

    def check():
        if utc_now() >= deadline or time.monotonic() - execution["start_monotonic"] >= p["max_calendar_seconds"] - 5:
            raise TimeoutError("Original worker allowance exhausted")

    check()
    p, cases, hard_deadline = admission(path, commit, utc(execution["started_utc"]))
    if hard_deadline != utc(execution["hard_deadline_utc"]):
        raise ValueError("Worker cannot reset original deadline")
    with (root / "WORKER_STARTED").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    import cv2
    import torch
    from run_solar_evaluation_v70 import software_identity
    from solar_evaluation_v70 import aggregate_rows

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    rows, profile_memory = evaluate_profiles(p, cases, commit, check, backend_factory)
    check()
    write(root / "summary.json", dict(
        complete=True, protocol_sha256=sha(path), publication_commit=commit, software=software_identity(),
        profile_memory=profile_memory, rows=rows, table=aggregate_rows(rows), selection_performed=False,
        calibration_or_final_access=False, purpose=p["purpose"], profiles=p["profiles"],
        cost_scope="Complete fresh case wall time including inference, common-pose replay, scoring and NPZ write.",
    ))


def guarded_run(path, commit, python, admission, worker_script):
    from solar_compute_budget_v67 import reserve

    started = utc_now()
    monotonic = time.monotonic()
    p, cases, deadline = admission(path, commit, started)
    check_common_protocol(p)
    root = Path(p["output"])
    work_deadline = deadline - timedelta(seconds=5)
    cap = p["max_calendar_seconds"]
    if utc_now() >= work_deadline or time.monotonic() - monotonic >= cap - 5:
        raise TimeoutError("Preflight consumed original allowance")
    root.mkdir(parents=True, exist_ok=False)
    previous = {}
    interrupts = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda signum, frame: interrupts.append(signum))
    worker = None
    charge = None
    error = None
    try:
        charge = reserve(LEDGER, p["compute_ledger_sha256"], JOURNAL, sha(path) + JOB_SUFFIX, "baseline", cap)
        write(root / "reservation.json", charge)
        write(root / "execution.json", dict(
            controller_pid=os.getpid(), publication_commit=commit, protocol_sha256=sha(path),
            started_utc=started.isoformat(), start_monotonic=monotonic,
            hard_deadline_utc=deadline.isoformat(), work_deadline_utc=work_deadline.isoformat(),
        ))
        if interrupts or utc_now() >= work_deadline or time.monotonic() - monotonic >= cap - 5:
            raise TimeoutError("Reservation and startup consumed the original allowance")
        with (root / "worker.log").open("x") as log:
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            worker = subprocess.Popen(
                [python, worker_script, "worker", "--protocol", path, "--publication-commit", commit,
                 "--controller-pid", str(os.getpid())], stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=env,
            )
            while worker.poll() is None:
                if interrupts or utc_now() >= work_deadline or time.monotonic() - monotonic >= cap - 5:
                    raise TimeoutError("Original evaluation allowance exhausted or interrupted")
                time.sleep(0.1)
        if worker.returncode or interrupts or utc_now() >= work_deadline:
            raise RuntimeError("Evaluation worker failed, was interrupted, or finished late")
        summary = read(root / "summary.json")
        actual = [(row["profile"], row["case_id"], row["track"]) for row in summary["rows"]]
        expected = [(profile["name"], case["case_id"], TRACKS[0]) for profile in p["profiles"] for case in cases]
        if not summary["complete"] or summary["protocol_sha256"] != sha(path) or actual != expected:
            raise ValueError("Original worker case inventory incomplete")
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if worker is not None and worker.poll() is None:
            for sig, wait_seconds in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
                try:
                    os.killpg(worker.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    worker.wait(timeout=max(0.0, min(wait_seconds, (deadline - utc_now()).total_seconds())))
                    break
                except subprocess.TimeoutExpired:
                    pass
        if utc_now() >= deadline or time.monotonic() - monotonic >= cap:
            error = dict(type="TimeoutError", message="Original hard deadline exhausted")
        for sig, previous_handler in previous.items():
            signal.signal(sig, previous_handler)
        write(root / "watchdog.json", dict(
            complete=error is None, protocol_sha256=sha(path), publication_commit=commit,
            started_utc=started.isoformat(), ended_utc=utc_now().isoformat(),
            calendar_seconds=time.monotonic() - monotonic, cap_seconds=cap,
            hard_deadline_utc=deadline.isoformat(), work_deadline_utc=work_deadline.isoformat(),
            compute_reservation=charge, child_returncode=None if worker is None else worker.returncode,
            failure=error,
        ))
    if error is not None:
        raise RuntimeError(error)
    write(root / "COMPLETE.json", dict(
        external_watchdog_passed=True, protocol_sha256=sha(path),
        summary_sha256=sha(root / "summary.json"), watchdog_sha256=sha(root / "watchdog.json"),
    ))


def _checkpoint_schedule(schedule, proof):
    pilot, replay = proof["pilot"], proof["replay"]
    inputs = {PILOT, REPLAY, pilot["bank_prior"]}
    for protocol in (pilot, replay):
        inputs.update(str(Path(protocol["output"]) / name) for name in ("COMPLETE.json", "worker_summary.json", "watchdog.json"))
    if schedule == "pilot100":
        science = Path(pilot["output"]) / "science"
        summary = read(science / "summary.json")
        steps = (0, 100)
    elif schedule == "study2000":
        from solar_distributional_study_runner_v90 import verify_output

        path = "reports/solar-distributional2000-study-v90.json"
        protocol = read(path)
        root = Path(protocol["output"])
        science = root / "science"
        summary = read(science / "summary.json")
        seal = read(root / "COMPLETE.json")
        watchdog = read(root / "watchdog.json")
        worker = read(root / "worker_summary.json")
        if (not seal["external_watchdog_passed"] or not watchdog["complete"] or not worker["complete"]
                or any(record["protocol_sha256"] != sha(path) for record in (seal, watchdog, worker))
                or seal["watchdog_sha256"] != sha(root / "watchdog.json")
                or seal["worker_summary_sha256"] != sha(root / "worker_summary.json")
                or worker["science_summary_sha256"] != verify_output(protocol, root)
                or protocol["pilot_replay_evidence"] != proof["receipts"]):
            raise ValueError("Completed source-published primary 2,000-step study required")
        inputs.update({path, *(str(root / name) for name in ("COMPLETE.json", "watchdog.json", "worker_summary.json"))})
        steps = (0, 500, 1000, 1500, 2000)
    else:
        raise ValueError("Use the exact pilot100 or study2000 checkpoint schedule")
    inputs.add(str(science / "summary.json"))
    return science, summary, steps, inputs


def binding(settings):
    from solar_distributional_study_v90 import require_replayed_pilot

    mode = settings["mode"]
    if mode not in {"pilot", "full"}:
        raise ValueError("Explicit eight-case pilot or complete development bank required")
    proof = require_replayed_pilot(PILOT, REPLAY)
    basis, _, cases = fixed_v86_cases(full=mode == "full")
    cap = settings["max_calendar_seconds"]
    ceiling = basis["max_calendar_seconds"] if mode == "pilot" else 12000
    if type(cap) is not int or not 5 < cap <= ceiling:
        raise ValueError("Requested pilot/full cap exceeds the original allowance")
    control = next(profile for profile in basis["profiles"] if profile["name"] == CONTROL)
    config = {key: control["configuration"][key] for key in (
        "proposal_angles", "pose_initialization", "solar_mode", "active", "policy", "budget",
    )}
    if (config["proposal_angles"] != [0.0, 90.0, 180.0, 270.0] or config["budget"] != 128
            or config["pose_initialization"] != "quantization_aware" or config["solar_mode"] != "shared"
            or not config["active"] or config["policy"] != "active" or control["configuration"]["response"]):
        raise ValueError("Original matched frozen v84 control changed")
    pilot = proof["pilot"]
    science, summary, steps, public_inputs = _checkpoint_schedule(settings["checkpoint_schedule"], proof)
    profiles = [control]
    local_inputs = {control["checkpoint"]: control["configuration"]["checkpoint_sha256"]}
    for step in steps:
        checkpoint = str(science / f"step_{step}.pt")
        checksum = summary["artifacts"][f"step_{step}.pt"]
        profiles.append(dict(
            name=f"coarse_response_step{step}", backend="distributional_coarse128", checkpoint=checkpoint,
            checkpoint_sha256=checksum, expected_step=step, prior_path=pilot["bank_prior"],
            prior_sha256=pilot["bank_prior_sha256"], teacher_directory=pilot["teacher_directory"],
            frozen_descriptor_sha256=pilot["expected_frozen_descriptor_sha256"], configuration=config,
        ))
        local_inputs[checkpoint] = checksum
    local_inputs.update({str(Path(pilot["teacher_directory"]) / name): checksum for name, checksum in pilot["teacher_sha256"].items()})
    local_inputs.update({path: basis["input_sha256"][path] for path in (basis["manifest"], basis["geometry_audit"])})
    # The public evaluation protocol binds the exact hashes of every local
    # completed result, including receipts that are not in the public tree.
    public_anchors = {BASIS, PILOT, REPLAY, pilot["bank_prior"]}
    for key in ("protocol", "summary", "watchdog_receipt"):
        path = control["training_evidence"][key]
        local_inputs[path] = control["training_evidence"][key + "_sha256"]
    local_inputs.update({path: sha(path) for path in public_inputs - public_anchors})
    sources = {
        **basis["source_sha256"], **pilot["source_sha256"],
        COARSE_SOURCE: sha(COARSE_SOURCE), RUNNER_SOURCE: sha(RUNNER_SOURCE),
        "solar_distributional_study_v90.py": sha("solar_distributional_study_v90.py"),
    }
    if settings["checkpoint_schedule"] == "study2000":
        name = "solar_distributional_study_runner_v90.py"
        sources[name] = sha(name)
    cost = None
    measured_path = settings.get("cost_pilot_protocol")
    if mode == "full":
        from run_solar_evaluation_v70 import measured_cost

        measured = read(measured_path)
        measured_root = Path(measured["output"])
        receipt = read(measured_root / "COMPLETE.json")
        watchdog = read(measured_root / "watchdog.json")
        measured_summary = read(measured_root / "summary.json")
        _, _, pilot_cases = fixed_v86_cases()
        expected_rows = [(profile["name"], case["case_id"], TRACKS[0]) for profile in profiles for case in pilot_cases]
        actual_rows = [(row["profile"], row["case_id"], row["track"]) for row in measured_summary["rows"]]
        if (not receipt["external_watchdog_passed"] or not watchdog["complete"]
                or not measured_summary["complete"] or measured["mode"] != "pilot"
                or measured["source_sha256"] != sources or measured_summary["profiles"] != profiles
                or receipt["protocol_sha256"] != sha(measured_path)
                or measured_summary["protocol_sha256"] != sha(measured_path)
                or receipt["summary_sha256"] != sha(measured_root / "summary.json")
                or receipt["watchdog_sha256"] != sha(measured_root / "watchdog.json")
                or actual_rows != expected_rows):
            raise ValueError("Original measured fresh matched-profile cost pilot required")
        cost = measured_cost(measured_summary["rows"], len(cases), TRACKS, profiles)
        if cost["conservative_total_seconds"] > cap:
            raise ValueError("Original full admission refused by measured cost: " + str(cost["conservative_total_seconds"]))
        public_anchors.add(measured_path)
        for name in ("COMPLETE.json", "summary.json", "watchdog.json"):
            path = str(measured_root / name)
            local_inputs[path] = sha(path)
    elif measured_path is not None:
        raise ValueError("Pilot must use fresh inference, no historical case cache")
    return dict(
        mode=mode, checkpoint_schedule=settings["checkpoint_schedule"], output=settings["output"],
        purpose="coarse_response_cost_pilot" if mode == "pilot" else "coarse_response_complete_development_comparison",
        max_calendar_seconds=cap, profiles=profiles, split="model_selection",
        case_ids=[case["case_id"] for case in cases], cpu_threads=2, tracks=TRACKS,
        calibration_or_final_access=False, checkpoint_selection=False,
        learning_contract=basis["learning_contract"], compute_ledger=basis["compute_ledger"],
        compute_ledger_sha256=basis["compute_ledger_sha256"], reservation_journal=basis["reservation_journal"],
        blender_executable=basis["blender_executable"], source_sha256=sources,
        metadata_sha256={path: sha(path) for path in sorted(public_anchors)},
        local_input_sha256=local_inputs, cost_pilot_protocol=measured_path, full_cost_estimate=cost,
    ), cases, ceiling


def prepare(settings_path, output):
    fields, _, _ = binding(read(settings_path))
    write(output, dict(fields, prepared_only=True, created_utc=None, latest_start_utc=None,
                       absolute_deadline_utc=None, publication_dispatch_lease_seconds=1800))


def admission(path, commit, started):
    from solar_learning_pipeline_v73 import journal_state, load_contract, remaining

    p = read(path)
    check_common_protocol(p)
    fields, cases, ceiling = binding(p)
    deadline = validate_dispatch(p, started, ceiling)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original evaluation preflight allowance exhausted")

    if any(p[key] != value for key, value in fields.items()):
        raise ValueError("Published profile, checkpoint, case or source identity differs")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for name, expected in p["local_input_sha256"].items():
        check()
        if sha(name) != expected:
            raise ValueError("Frozen local input changed: " + name)
    contract = load_contract(p["learning_contract"])
    _, journal, _ = journal_state(LEDGER, p["compute_ledger_sha256"], JOURNAL)
    already_charged = any(
        row["job_id"] == sha(path) + JOB_SUFFIX and row["seconds"] == p["max_calendar_seconds"]
        and row["category"] == "baseline" for row in journal["reservations"]
    )
    if not already_charged and p["max_calendar_seconds"] > remaining(contract)["baseline"]:
        raise ValueError("Original cumulative baseline allowance exceeded")
    return p, cases, deadline


def backend_factory(profile):
    if profile["backend"] == "distributional_coarse128":
        from solar_distributional_evaluation_v90 import CoarseResponseBackend

        return CoarseResponseBackend(profile)
    from solar_evaluation_v70 import Backend

    return Backend(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--settings", required=True)
    prepare_command.add_argument("--output", required=True)
    for command in ("run", "worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--publication-commit", required=True)
        if command == "run":
            sub.add_argument("--python", default=sys.executable)
        else:
            sub.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.settings, args.output)
    elif args.command == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, RUNNER_SOURCE)
    else:
        guarded_worker(args.protocol, args.publication_commit, args.controller_pid, admission, backend_factory)
