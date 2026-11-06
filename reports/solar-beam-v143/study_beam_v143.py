"""Prepare/execute the complete persistent-beam study under one explicit cap.

Preparation never issues approval. All prior fixed charges remain intact. A new
1800-second allowance must record an actual user resource decision before run.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPORT = Path("reports/solar-beam-v143")
RUN = Path("runs/solar-beam-v143")
AUTH = Path("runs/solar-beam-v143-allowance")
LEDGER, JOURNAL = AUTH/"ledger.json", AUTH/"reservations.json"
CAP = 1800
METHODS = ["direct", "legacy_no_extra", "legacy_active32", "single_uniform128",
           "beam_uniform128", "beam_geometry128", "beam_active128"]
spec = spec_from_file_location("preserved_blind_study_v140_for_v143", ROOT/"reports/solar-blind-v140/continuation_v140.py")
parent = module_from_spec(spec); spec.loader.exec_module(parent)
read, sha, check, write = parent.read, parent.sha, parent.check, parent.write


def prior_state():
    protocol = read(parent.HOME/"protocol.json")
    parent.guard(protocol)
    completed = read(parent.WORK/"COMPLETE.json")
    check(completed["complete"] and not (parent.WORK/"FAILED.json").exists(), "Preserve completed v140 comparison")
    journal = read(parent.JOURNAL)
    check(journal["ledger_sha256"] == sha(parent.LEDGER) and len(journal["reservations"]) == 1
          and journal["reservations"][0]["seconds"] == 900
          and journal["reservations"][0]["refund_on_failure"] is False, "Preserve full900-second prior charge")
    return protocol


def build():
    previous = prior_state()
    sources = dict(previous["sources"])
    for p in REPORT.iterdir():
        if p.suffix in (".py", ".md"):
            sources[str(p)] = sha(p)
    inputs = dict(previous["inputs"])
    for name in ("predictions.json", "PREDICTIONS_SEALED.json", "scores.json", "replay.json"):
        p = parent.WORK/name; inputs[str(p)] = sha(p)
    cases = read(previous["inference_manifest"])
    inference = {previous["inference_manifest"]: inputs[previous["inference_manifest"]]}
    for case in cases:
        check(set(case) == {"id", "reference_geometry", "catalogue", "query_image", "query_intrinsics"}, "Inference-only case schema required")
        for record in [case["reference_geometry"], case["query_image"], *[c["image"] for c in case["catalogue"]]]:
            check(inputs[record["path"]] == record["sha256"], "Inference record differs from frozen data")
            inference[record["path"]] = record["sha256"]
    check(previous["scoring_manifest"] not in inference, "Query truth cannot enter inference")
    return dict(schema="prepared_persistent_beam_study_v143", prepared_only=True,
        preparation_is_not_execution_authority=True, requested_cap_seconds=CAP,
        sources=sources, inputs=inputs, inference_inputs=inference,
        parent_journals={str(parent.JOURNAL): sha(parent.JOURNAL), str(parent.old.JOURNAL): sha(parent.old.JOURNAL),
                         "runs/solar-compute-reservations-v67.json": sha("runs/solar-compute-reservations-v67.json")},
        prior_extra_fixed_charges_seconds=3300,
        inference_manifest=previous["inference_manifest"], scoring_manifest=previous["scoring_manifest"],
        historical_predictions=str(parent.WORK/"predictions.json"), case_ids=previous["case_ids"],
        methods=METHODS, comparison_cells=56, native_pixels=512, field_cap_per_new_control=128,
        maximum_lights_retained=4, maximum_camera_light_branches=8,
        field_budget_rule="Common floor(min(128,shared_reference_candidate_capacity)/qualified_branches)*qualified_branches across all new controls; every branch attempt charged, failed support included",
        single_branch_baseline="Same new initializer, common field cap and final pose estimator; only best branch kept",
        frozen_baselines="Direct, legacy no-extra and legacy active32 must repeat saved native endpoints/pose/estimator",
        primary_comparison="beam_active128 versus strongest non-beam baseline, including single_uniform128",
        policy_comparison="beam_active128 versus beam_uniform128 and beam_geometry128 at the same beam and field budget",
        timing="Two complete24-light/2048-camera seven-arm generated controls;1.5*8*worst pair+45s must fit original remaining deadline",
        prior_result_cases_are_exposed_development=True, original_targets_unchanged=True,
        cpu_threads=2, opencv_threads=1, paid_compute=False, training=False,
        query_truth_inputs=False, known_query_sun=False, accepted_pose_study=False,
        calibration_or_final_access=False, full_roster_or_no_lunar_inference=True,
        model_promotion=False, outreach=False)


def guard(protocol):
    prior_state()
    parent.old.pins(protocol)
    for name, expected in protocol["parent_journals"].items():
        check(sha(name) == expected, "Prior accounting changed")
    check(all(p.resolve() == (ROOT/p).absolute() for p in (AUTH, LEDGER, JOURNAL)), "No authority path aliases")


def published(commit, protocol):
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"], check=True)
    for name in (*protocol["sources"], str(REPORT/"protocol.json")):
        check(subprocess.check_output(["git", "show", commit+":"+name]) == Path(name).read_bytes(), "Source differs from public freeze: "+name)


def validate_approval(ledger, protocol):
    check(ledger.get("schema") == "explicit_user_authorized_beam_comparison_v143" and ledger.get("user_approved") is True
          and ledger.get("accounting_complete") is True, "Actual new user resource approval required")
    check(ledger.get("remaining_baseline_wall_seconds") == CAP and ledger.get("remaining_training_wall_seconds") == 0
          and ledger.get("paid_compute") is False and ledger.get("threads_per_scientific_job") == 2
          and ledger.get("calibration_or_final_access") is False, "Fixed cap and resource scope cannot expand")
    check(ledger.get("protocol_sha256") == sha(REPORT/"protocol.json")
          and ledger.get("parent_journals") == protocol["parent_journals"], "Bind exact protocol and prior charges")
    check(isinstance(ledger.get("approval_text"), str) and ledger["approval_text"].strip()
          and isinstance(ledger.get("user_approval_source"), str) and ledger["user_approval_source"].strip(),
          "Actual decision text and provenance required; task preparation is not approval")


def deadline(reservation):
    check(reservation["category"] == "baseline" and reservation["seconds"] == CAP
          and reservation["refund_on_failure"] is False, "One original1800-second fixed reservation required")
    start = datetime.fromisoformat(reservation["reserved_utc"])
    check(start.tzinfo is not None, "Timezone-aware original deadline required")
    return start+timedelta(seconds=CAP)


def forecast_fits(timing, remaining):
    rows = timing.get("samples", [])
    check(len(rows) == 2 and {r["layered"] for r in rows} == {False, True}
          and all(r["all_seven_pose_controls_pass"] and r["lights"] == 24 and r["cameras"] == 2048
                  and r["maximum_branches"] == 8 and r["requested_native_field_cap"] == 128
                  and 0 < r["common_effective_native_fields"] <= 128 and r["elapsed_seconds"] > 0 for r in rows),
          "Complete full-scope cost controls required")
    forecast = 1.5*8*max(r["elapsed_seconds"] for r in rows)+45.
    check(abs(forecast-timing["comparison_forecast_seconds"]) < 1e-8, "Forecast arithmetic changed")
    return forecast <= remaining


def sealed(protocol):
    raw, seal = read(RUN/"predictions.json"), read(RUN/"PREDICTIONS_SEALED.json")
    check(raw["query_truth_used"] is False and raw["accepted_results"] is False and seal["rows"] == len(raw["rows"]) == 8
          and [r["id"] for r in raw["rows"]] == protocol["case_ids"] and seal["sha256"] == sha(RUN/"predictions.json"),
          "All eight truth-free predictions must be sealed before query scoring")


def checked_input(protocol, operation, name):
    if operation == "timing":
        allowed = {}
    elif operation == "predict":
        allowed = protocol["inference_inputs"]
    elif operation in ("score", "replay"):
        allowed = protocol["inputs"]
    else:
        raise ValueError("Unknown input phase")
    check(name in allowed and sha(name) == allowed[name], "Changed or phase-forbidden input: "+name)
    return Path(name)


def worker(operation, ticket_path):
    import resource
    protocol, ticket = read(REPORT/"protocol.json"), read(ticket_path)
    guard(protocol); validate_approval(read(LEDGER), protocol)
    check(ticket["parent_pid"] == os.getppid() and ticket["protocol_sha256"] == sha(REPORT/"protocol.json")
          and ticket["approval_sha256"] == sha(LEDGER), "Original admitted controller and source required")
    journal = read(JOURNAL)
    check(journal["ledger_sha256"] == sha(LEDGER) and journal["reservations"] == [ticket["reservation"]], "Preserve original single charge")
    stop = deadline(ticket["reservation"])
    check(ticket["deadline"] == stop.isoformat() and datetime.now(timezone.utc) < stop, "Original deadline expired")
    lease, target = os.fstat(ticket["lease_fd"]), (parent.old.PRIVATE/"scientific.lock").stat()
    check((lease.st_dev, lease.st_ino) == (target.st_dev, target.st_ino), "Inherited serial scientific lease required")
    signal.setitimer(signal.ITIMER_REAL, max(.001, (stop-datetime.now(timezone.utc)).total_seconds()))
    published(ticket["commit"], protocol)
    sys.path.insert(0, str(ROOT/REPORT))
    import engine_beam_v143 as engine
    engine.frozen.cv2.setNumThreads(1)
    checked = lambda name: checked_input(protocol, operation, name)
    put = lambda name, value: write(RUN/name, value)
    get = lambda name: read(RUN/name)
    tick = time.perf_counter()
    if operation == "timing": engine.timing(put)
    elif operation == "predict": engine.predict(protocol, put, checked)
    elif operation == "score": sealed(protocol); engine.score(protocol, put, checked, get)
    elif operation == "replay": sealed(protocol); engine.replay(protocol, put, checked, get)
    else: raise ValueError("Unknown operation")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    put(operation+"-runtime.json", dict(elapsed_seconds=time.perf_counter()-tick,
        whole_worker_peak_rss_bytes=int(peak if sys.platform == "darwin" else peak*1024),
        memory_scope="Whole worker, including shared caches and imports; not per-method memory"))


def execute(commit, approval_sha256):
    from solar_compute_budget_v67 import reserve
    protocol = read(REPORT/"protocol.json")
    guard(protocol); published(commit, protocol)
    check(sha(LEDGER) == approval_sha256, "Exact new private user approval required")
    validate_approval(read(LEDGER), protocol)
    check(not RUN.exists() and not RUN.is_symlink() and not JOURNAL.exists(), "Preserve prior attempts and charges")
    with (parent.old.PRIVATE/"scientific.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        reservation = reserve(LEDGER, approval_sha256, JOURNAL, sha(REPORT/"protocol.json")+"/beam-v143", "baseline", CAP)["reservation"]
        stop = deadline(reservation); RUN.mkdir()
        ticket = dict(commit=commit, parent_pid=os.getpid(), lease_fd=lease.fileno(),
            protocol_sha256=sha(REPORT/"protocol.json"), approval_sha256=approval_sha256,
            reservation=reservation, deadline=stop.isoformat())
        write(RUN/"execution.json", ticket); tick = time.perf_counter()
        try:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       VECLIB_MAXIMUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONPATH=str(ROOT))
            for operation in ("timing", "predict", "score", "replay"):
                remaining = (stop-datetime.now(timezone.utc)).total_seconds()
                check(remaining > 0, "Original shared deadline expired")
                if operation == "predict":
                    check(forecast_fits(read(RUN/"timing.json"), remaining), "Complete comparison cannot fit; no smaller substitute or free retry")
                with (RUN/(operation+".log")).open("x") as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", operation,
                        "--ticket", str(RUN/"execution.json")], env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    try: status = child.wait(timeout=remaining)
                    except BaseException:
                        os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
                check(status == 0 and datetime.now(timezone.utc) < stop, "Worker failed or original deadline expired")
                if operation == "predict": write(RUN/"PREDICTIONS_SEALED.json", dict(rows=8, sha256=sha(RUN/"predictions.json")))
            guard(protocol)
            write(RUN/"COMPLETE.json", dict(ticket, complete=True, elapsed_seconds=time.perf_counter()-tick))
        except BaseException as error:
            write(RUN/"FAILED.json", dict(ticket, complete=False, error=type(error).__name__, message=str(error)))
            raise


if __name__ == "__main__":
    check(Path.cwd() == ROOT, "Run from original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run"); run.add_argument("--commit", required=True); run.add_argument("--approval-sha256", required=True)
    job = sub.add_parser("worker"); job.add_argument("operation", choices=("timing", "predict", "score", "replay")); job.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze": write(REPORT/"protocol.json", build())
    elif args.action == "run": execute(args.commit, args.approval_sha256)
    else: worker(args.operation, args.ticket)
