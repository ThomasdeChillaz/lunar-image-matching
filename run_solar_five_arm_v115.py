"""Ticket-only same-worker five-arm comparison over the immutable 96-case subset."""

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

from plan_solar_matched_arm_v113 import ARM_NAMES, COHORT, DIRECT_NAME, TRACK
from run_solar_evaluation_v110 import (
    InputLoader, InferenceResultCapture, REQUIRED_SOURCES as V110_SOURCES, TIMING_SCOPE,
    capture_case_report, comparison_identity, software_identity, worker_memory_scope,
    worker_peak_rss_snapshot, write_new,
)
from solar_aligned_scoring_v91 import load_aligned_scoring
from solar_evaluation_v70 import (
    Backend, aggregate_rows, digest, evaluate_case, paired_component_bootstrap, read, sha,
    validate_profile_training, validate_profiles,
)
from solar_portable_evidence_v75 import safe
from verify_solar_matched_arm_v113 import exact_case_ids


SOURCE = "run_solar_five_arm_v115.py"
PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
OUTPUT = Path("runs/solar-five-arm-v115/comparison")
CAP = 18_000
EXPECTED_ROWS = 480
PURPOSE = "matched_development_comparison"
STAGE_MODE = "matched_modern_direct_arm_v113"
PROFILE_NAMES = (*ARM_NAMES, DIRECT_NAME)
PROFILE_BACKENDS = ("solar", "sift", "modern", "modern", "direct_render")
INPUT_KEYS = ("reference", "query", "reference_geometry", "query_intrinsics",
              "reference_sun_world", "solar_lookup_protocol")
REFERENCE_KEYS = (
    "unit_index", "component", "reference_geometry", "reference_geometry_sha256",
    "solar_lookup_protocol", "solar_lookup_protocol_sha256", "source_reference_geometry_path",
    "source_reference_geometry_sha256", "scene_mesh_path", "scene_mesh_sha256", "source_audit_sha256",
)
REQUIRED_SOURCES = V110_SOURCES | {
    SOURCE, "qualify_five_arm_v115.py", "solar_remote_relay_five_arm_v115.py",
    "solar_aligned_scoring_v91.py", "solar_direct_arm_bridge_v103.py", "solar_direct_controls_v113.py",
    "direct_render_adapter_v73.py", "prepare_solar_modern_arm_v98.py",
    "verify_solar_matched_arm_v113.py", "plan_solar_matched_arm_v113.py", "solar_portable_evidence_v75.py",
}


def utc(value):
    moment = datetime.fromisoformat(value)
    if moment.utcoffset() is None:
        raise ValueError("Require the original UTC timestamps")
    return moment.astimezone(timezone.utc)


def checked_file(name, expected):
    path = Path(safe(name))
    if path.resolve() != Path.cwd() / path or sha(path) != expected:
        raise ValueError("Frozen file changed: " + str(name))
    return path


def strict_cases(protocol, manifest, cases, cohort):
    if (protocol.get("purpose") != PURPOSE or protocol.get("stage_mode") != STAGE_MODE
            or protocol.get("comparison_case_count") != 96 or protocol.get("output") != str(OUTPUT)
            or protocol.get("schema_version") != 70 or protocol.get("builder_schema_version") != 74
            or protocol.get("max_calendar_seconds") != CAP or protocol.get("cpu_threads") != 2
            or protocol.get("split") != "model_selection" or protocol.get("tracks") != [TRACK]
            or protocol.get("calibration_or_final_access") is not False or protocol.get("accepted_results") is not False
            or protocol.get("automatic_model_promotion") is not False or protocol.get("case_cache_seals")
            or tuple((p["name"], p["backend"]) for p in protocol["profiles"])
            != tuple(zip(PROFILE_NAMES, PROFILE_BACKENDS))):
        raise ValueError("Only the fresh 96-case raw primary five-arm scope is supported")
    all_ids = [row["case_id"] for row in cases]
    if len(all_ids) != 384 or all_ids != protocol["case_ids"] or manifest["cases"] != cases:
        raise ValueError("Require the original full ordered 384-case authority")
    ids = exact_case_ids(protocol, cohort)
    by_case = {row["case_id"]: row for row in cases}
    selected = [by_case[identity] for identity in ids]
    selected_ids = set(ids)
    if (len(selected_ids) != 96 or sum(row["positive"] for row in selected) != 48
            or any(row["split"] != "model_selection" for row in selected)
            or any(not row["positive"] and row["matched_positive_case"] not in selected_ids for row in selected)):
        raise ValueError("The frozen subset must contain 48 positives and their 48 matched nonoverlaps")
    return selected


def admission(protocol_path, commit, *, check_assets):
    from solar_evaluation_protocol_v74 import qualified_cases
    from solar_remote_relay_five_arm_v115 import primary_scope

    if str(protocol_path) != PROTOCOL or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Require the exact five-arm protocol and source publication")
    p = read(protocol_path)
    if p.get("prepared_only") is not False or not REQUIRED_SOURCES <= set(p["source_sha256"]):
        raise ValueError("Prepared settings or incomplete execution closure cannot run")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    for path in (protocol_path, *p["source_sha256"]):
        checked = Path(safe(path))
        if (checked.resolve() != Path.cwd() / checked
                or subprocess.check_output(["git", "show", f"{commit}:{path}"], timeout=5) != checked.read_bytes()):
            raise ValueError("Execution bytes differ from the published commit: " + path)
        if path != protocol_path and sha(checked) != p["source_sha256"][path]:
            raise ValueError("Source pin differs: " + path)
    primary_scope(protocol_path, p, check_assets=check_assets)
    validate_profiles(p["profiles"], "modern_comparison")
    for profile in p["profiles"]:
        validate_profile_training(profile, "modern_comparison")
    cohort_path, expected = COHORT
    if p["input_sha256"].get(cohort_path) != expected:
        raise ValueError("Published subset authority must be frozen")
    cohort = read(checked_file(cohort_path, expected))
    manifest, _, cases = qualified_cases(p["manifest"], p["geometry_audit"])
    selected = strict_cases(p, manifest, cases, cohort)
    return p, manifest, selected


def inference_inputs(case):
    """Only this fixed input projection reaches the shared image/geometry loader."""
    inputs = case["inference_inputs"]
    expected = {key: case[key] for key in INPUT_KEYS}
    if inputs != expected:
        raise ValueError("Primary input manifest is not the reference-only projection")
    return expected


def reference_binding(case, p):
    """Direct identity binding receives only the declared reference-side proof."""
    meta = {key: case["reference_asset_metadata"][key] for key in REFERENCE_KEYS}
    names = (
        case["reference_geometry"], case["solar_lookup_protocol"], meta["source_reference_geometry_path"],
        meta["scene_mesh_path"], p["geometry_audit"],
    )
    proof = {key: case[key] for key in ("split", "unit_index", "component", "reference_geometry",
                                      "solar_lookup_protocol")}
    proof["reference_asset_metadata"] = meta
    proof["input_sha256"] = {name: case["input_sha256"][name] for name in names}
    pins = {name: p["input_sha256"][name] for name in names}
    return proof, pins


def check_case_bytes(case, expected, *, scoring):
    score_paths = set(case["scoring_inputs"].values())
    if score_paths != ({case["scoring_labels"], case["scoring_geometry"]} if case["positive"] else set()):
        raise ValueError("Frozen scoring access declaration differs")
    for path, expected_hash in case["input_sha256"].items():
        if (path in score_paths) == scoring:
            if expected.get(path) != expected_hash:
                raise ValueError("Case asset differs from the published input map")
            checked_file(path, expected_hash)
    if not score_paths <= set(case["input_sha256"]):
        raise ValueError("Every scoring-only input must have its original byte hash")


def evaluate_one(backend, loader, case, protocol):
    check_case_bytes(case, protocol["input_sha256"], scoring=False)
    inp = loader.get(inference_inputs(case), TRACK)
    if inp.known_query_sun is not None or inp.solar_lookup is not None:
        raise ValueError("Primary materialized comparison cannot consume query truth or online Blender lookup")
    runner = backend
    if backend.profile["backend"] == "direct_render":
        from solar_direct_arm_bridge_v103 import bind_case

        tick = time.monotonic()
        proof, pins = reference_binding(case, protocol)
        runner = InferenceResultCapture(bind_case(backend._backend, proof, pins))
        loader.last_setup["direct_reference_identity_setup_seconds"] = time.monotonic() - tick

    verified_scoring = False

    def load_scoring(scoring_case, points):
        nonlocal verified_scoring
        if not verified_scoring:
            check_case_bytes(scoring_case, protocol["input_sha256"], scoring=True)
            verified_scoring = True
        return load_aligned_scoring(scoring_case, points)

    # This common evaluator calls infer and the shared AP3P estimator before
    # invoking the scoring loader. All five arms also share label alignment.
    row, arrays = evaluate_case(runner, inp, case, scoring_loader=load_scoring)
    row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(runner, arrays)
    row["reference_setup"] = loader.last_setup.copy()
    row["complete_localization_cold_seconds"] = row["inference_seconds"] + sum(loader.last_setup.values())
    row["timing_scope"] = TIMING_SCOPE["complete_localization_cold_seconds"]
    return row, arrays


def run_controls(profile, backend, root, protocol_sha, check):
    from prepare_solar_modern_arm_v98 import fresh_pretrained_control

    tick = time.monotonic()
    if profile["backend"] == "modern":
        result = fresh_pretrained_control(profile, backend._backend.model)
        write_new(root / (digest(profile) + ".actual_ARM_control.json"), result)
    elif profile["backend"] == "direct_render":
        from solar_direct_controls_v113 import controls

        result = controls(backend._backend.model, protocol_sha, check, qualification=True, measure=time.monotonic)
        write_new(root / "actual_ARM_Direct_control.json", result)
        if result["passed"] is not True:
            raise ValueError("Fresh same-worker Direct controls failed")
    else:
        return 0.
    check()
    return time.monotonic() - tick


def reference_costs(protocol, manifest, cases):
    results = []
    for geometry in sorted({case["reference_geometry"] for case in cases}):
        receipt = manifest["reference_precompute_receipts"][geometry]
        internal = read(receipt["receipt_path"])["calendar_seconds"]
        outer = protocol["reference_preparation_costs"]["outer_calendar_seconds_by_archive"][geometry]
        count = sum(case["reference_geometry"] == geometry for case in cases)
        results.append(dict(
            reference_geometry=geometry, **receipt, measured_calendar_seconds=outer,
            internal_sidecar_calendar_seconds=internal, startup_inclusive_outer_calendar_seconds=outer,
            queries_per_profile_per_track=count, fixed_study_amortized_seconds_per_query=outer/count,
            note="Offline preparation stays separate from online and model-load time.",
        ))
    return results


def deadline_bounds(protocol, context, inner_start):
    cap = protocol["max_calendar_seconds"]
    published = utc(protocol["absolute_deadline_utc"])
    outer_start = utc(context["outer_started_utc"])
    outer = utc(context["outer_absolute_deadline_utc"])
    if (cap != CAP or context["cap_seconds"] != cap or not outer_start <= inner_start < outer
            or outer > min(outer_start + timedelta(seconds=cap), published)):
        raise ValueError("Authenticated outer clock exceeds its original allowance")
    inner = min(inner_start + timedelta(seconds=cap), published)
    return inner, outer, min(inner, outer)


def worker(protocol_path, commit, ticket):
    from prepare_solar_modern_arm_v98 import check_worker_runtime
    from solar_remote_relay_five_arm_v115 import validate_worker_ticket, worker_context

    if ticket is None:
        raise ValueError("The five-arm worker requires its original one-use ticket")
    context = worker_context(ticket, protocol_path)
    authorized = validate_worker_ticket(ticket, protocol_path)
    import cv2
    import torch

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    check_worker_runtime()
    p, manifest, cases = admission(protocol_path, commit, check_assets=False)
    root = Path(p["output"])
    protocol_sha = sha(protocol_path)
    watchdog = read(root / "watchdog_admission.json")
    charge_path = root / "reservation.json"
    charge = read(charge_path)
    started = time.monotonic()
    start = watchdog["start_monotonic"]
    inner, outer, effective = deadline_bounds(p, context, utc(watchdog["started_utc"]))
    if (watchdog["protocol_sha256"] != protocol_sha or watchdog["reservation_sha256"] != sha(charge_path)
            or charge["reservation"] != authorized["reservation"]
            or charge["reservation"]["job_id"] != protocol_sha + "/evaluation"
            or charge["reservation"]["category"] != "baseline" or charge["reservation"]["seconds"] != CAP
            or charge["reservation"]["refund_on_failure"] is not False
            or watchdog["controller_pid"] != os.getppid()
            or watchdog["publication_commit"] != commit
            or utc(watchdog["absolute_deadline_utc"]) != inner or utc(watchdog["outer_absolute_deadline_utc"]) != outer
            or utc(watchdog["effective_deadline_utc"]) != effective or not 0 <= started - start < CAP):
        raise ValueError("Worker requires the original controller, charge and effective deadline")

    def check():
        if time.monotonic() - start >= CAP or datetime.now(timezone.utc) >= effective:
            raise TimeoutError("Original end-to-end five-arm allowance exhausted")

    rows, profile_memory = [], []
    current = None
    scope = worker_memory_scope(p["profiles"])
    loader = InputLoader(commit, fully_materialized_primary=True)
    try:
        check()
        write_new(root / "execution.json", dict(
            protocol_sha256=protocol_sha, publication_commit=commit, software=software_identity(),
            memory_measurement=scope, worker_peak_rss=worker_peak_rss_snapshot(), timing_scope=TIMING_SCOPE,
            purpose=PURPOSE, raw_uncalibrated=True, final_reserve_access=False, profile_order=list(PROFILE_NAMES),
            initialization_only_cost_controls=[], outer_started_utc=context["outer_started_utc"],
            effective_deadline_utc=effective.isoformat(),
        ))
        for index, profile in enumerate(p["profiles"]):
            check()
            current = dict(profile=profile["name"], profile_index=index, worker_pid=os.getpid(),
                           before=worker_peak_rss_snapshot(), after=None, completed=False)
            profile_memory.append(current)
            before_load = time.monotonic()
            backend = InferenceResultCapture(Backend(profile))
            model_load = time.monotonic() - before_load
            control_time = run_controls(profile, backend, root, protocol_sha, check)
            write_new(root / (digest(profile) + ".provenance.json"), backend.provenance)
            for case_index, case in enumerate(cases):
                check()
                before = time.monotonic()
                row, arrays = evaluate_one(backend, loader, case, p)
                row.update(
                    case_id=case["case_id"], component=case["component"], query_component=case["query_component"],
                    positive=case["positive"], profile=profile["name"], track=TRACK,
                    initialization_only_cost_control=False, unit_index=case["unit_index"],
                    query_unit_index=case["query_unit_index"], light_a=case["light_a"], light_b=case["light_b"],
                    reused_case_cache=False, profile_load_seconds=model_load,
                    fresh_control_calendar_seconds=control_time if case_index == 0 else 0.,
                    raw_inference_execution_scope="Fresh single original backend inference.",
                    memory_measurement_scope="Current worker lifetime peaks are recorded before and after each profile.",
                )
                # Include the serialized prediction array in the measured full row cost.
                key = digest(dict(case=case, profile=profile, track=TRACK, sources=p["source_sha256"]))
                array_path = root / (key + ".npz")
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row.update(saved_arrays=str(array_path), saved_arrays_sha256=sha(array_path))
                elapsed = time.monotonic() - before
                row.update(case_calendar_seconds=elapsed, case_total_calendar_seconds=elapsed +
                           (model_load + control_time if case_index == 0 else 0.))
                row_path = root / (key + ".json")
                write_new(row_path, row)
                write_new(root / (key + ".COMPLETE.json"), dict(
                    complete=True, identity=key, row_path=str(row_path), row_sha256=sha(row_path),
                    arrays_path=str(array_path), arrays_sha256=row["saved_arrays_sha256"],
                    source_protocol_sha256=protocol_sha,
                ))
                rows.append(row)
                print(profile["name"], case["case_id"], row["reason"], flush=True)
            del backend
            current.update(after=worker_peak_rss_snapshot(), completed=True,
                           fresh_inference_cases=len(cases), cached_cases=0)
        expected = [(profile["name"], case["case_id"], TRACK) for profile in p["profiles"] for case in cases]
        actual = [(row["profile"], row["case_id"], row["track"]) for row in rows]
        if len(actual) != EXPECTED_ROWS or actual != expected:
            raise ValueError("The complete ordered 480-cell matched denominator is required")
        check()
        summary = dict(
            complete=True, purpose=PURPOSE, rows=rows, software=software_identity(),
            comparison_identity=comparison_identity(p, manifest), timing_scope=TIMING_SCOPE,
            memory_measurement=dict(scope, profile_snapshots=profile_memory), protocol_sha256=protocol_sha,
            manifest_sha256=sha(p["manifest"]), source_publication_commit=commit,
            calendar_seconds=time.monotonic() - start, profiles=p["profiles"],
            case_ids=[case["case_id"] for case in cases], fresh_inference_cases=len(rows), cached_cases=0,
            accepted_results=False, calibration_or_final_access=False, automatic_model_promotion=False,
            scientific_model_evidence=True, offline_reference_preparation=reference_costs(p, manifest, cases),
            raw_results=aggregate_rows(rows), worker_peak_rss=worker_peak_rss_snapshot(),
            paired_terrain_intervals={control: paired_component_bootstrap(rows, DIRECT_NAME, control, TRACK)
                                      for control in ARM_NAMES},
        )
        check()
        write_new(root / "summary.json", summary)
    except BaseException as error:
        if current is not None and current["after"] is None:
            current.update(after=worker_peak_rss_snapshot(), completed=False,
                           fresh_inference_cases=sum(row["profile"] == current["profile"] for row in rows),
                           cached_cases=0)
        write_new(root / "WORKER_FAILED.json", dict(
            error_type=type(error).__name__, error=str(error), completed_rows=len(rows),
            calendar_seconds=time.monotonic() - start, timing_scope=TIMING_SCOPE,
            memory_measurement=dict(scope, profile_snapshots=profile_memory),
        ))
        raise
    finally:
        loader.close()


def run(protocol_path, commit, python, ticket):
    from solar_remote_relay_five_arm_v115 import consume_pre_reservation, controller_context

    if ticket is None:
        raise ValueError("Only the original precharged five-arm ticket may start")
    context = controller_context(ticket, protocol_path)
    p, _, _ = admission(protocol_path, commit, check_assets=True)
    root = Path(p["output"])
    if root.exists():
        raise FileExistsError("Preserve the previous complete or partial five-arm result")
    charge = consume_pre_reservation(ticket, protocol_path)
    start = time.monotonic()
    now = datetime.now(timezone.utc)
    inner, outer, effective = deadline_bounds(p, context, now)
    root.mkdir(parents=True)
    write_new(root / "reservation.json", charge)
    protocol_sha = sha(protocol_path)
    write_new(root / "watchdog_admission.json", dict(
        protocol_sha256=protocol_sha, publication_commit=commit, reservation_sha256=sha(root / "reservation.json"),
        start_monotonic=start, started_utc=now.isoformat(), absolute_deadline_utc=inner.isoformat(),
        original_protocol_absolute_deadline_utc=p["absolute_deadline_utc"], controller_pid=os.getpid(),
        outer_started_utc=context["outer_started_utc"], outer_absolute_deadline_utc=outer.isoformat(),
        effective_deadline_utc=effective.isoformat(),
    ))
    child, interrupted = None, []

    def check():
        if interrupted:
            raise InterruptedError("Comparison interrupted, preserve every completed cell")
        if time.monotonic() - start >= CAP or datetime.now(timezone.utc) >= effective:
            raise TimeoutError("External watchdog exhausted the original end-to-end allowance")

    def pending(signum, _):
        interrupted.append(signum)

    old = {sig: signal.signal(sig, pending) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        check()
        command = [python, str(Path(__file__)), "--protocol", protocol_path, "--publication-commit", commit,
                   "--pre-reserved-ticket", ticket, "--worker"]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
        with (root / "worker.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            while child.poll() is None:
                check()
                time.sleep(.1)
            check()
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        summary_path = root / "summary.json"
        summary = read(summary_path)
        if (summary["complete"] is not True or summary["protocol_sha256"] != protocol_sha
                or len(summary["rows"]) != EXPECTED_ROWS or summary["fresh_inference_cases"] != EXPECTED_ROWS
                or summary["cached_cases"] != 0):
            raise ValueError("Worker did not complete the fresh 480-cell comparison")
        checksum = sha(summary_path)
        check()
        write_new(root / "COMPLETE.json", dict(
            complete=True, summary_sha256=checksum, protocol_sha256=protocol_sha,
            external_watchdog_passed=True, observed_calendar_seconds=time.monotonic() - start,
            original_allowance_seconds=CAP, effective_deadline_utc=effective.isoformat(),
        ))
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
        write_new(root / "FAILED.json", dict(error_type=type(error).__name__, error=str(error),
                                          calendar_seconds=time.monotonic() - start,
                                          original_allowance_seconds=CAP, automatic_restart=False))
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--pre-reserved-ticket", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.protocol, args.publication_commit, args.pre_reserved_ticket)
    else:
        run(args.protocol, args.publication_commit, args.python, args.pre_reserved_ticket)


if __name__ == "__main__":
    main()
