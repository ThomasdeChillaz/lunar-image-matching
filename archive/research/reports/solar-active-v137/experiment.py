"""Source-published six-arm study under the explicitly approved40-minute supplement."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    sys.path.insert(0, str(path))
spec = spec_from_file_location("immutable_joint_v135_for_supplement", HERE.parent/"solar-joint-v135/experiment.py")
previous = module_from_spec(spec); spec.loader.exec_module(previous)
base = previous.base
from admission import CAPS, TOTAL, before_deadline, charge, deadline, serial_lease, state, validate

HOME = Path("reports/solar-active-v137")
WORK = Path("runs/solar-active-v137")
PRIVATE = Path("runs/solar-active-v137-allowance")
LEDGER = PRIVATE/"ledger.json"
JOURNAL = PRIVATE/"reservations.json"
CACHE = Path("runs/solar-local-v134/comparison/predictions.json")
JOINT_CACHE = Path("runs/solar-joint-v135/comparison/predictions.json")


def original_guard():
    summary = base.read(HOME/"allowance.json")
    base.check(base.sha(base.legacy.LEDGER) == summary["original_ledger_sha256"]
               and base.sha(base.legacy.JOURNAL) == summary["original_journal_sha256"],
               "Original canonical accounting changed; no supplement alias/refund allowed")
    base.check(base.remaining(base.read(base.legacy.CONTRACT)) == summary["original_remaining_seconds"],
               "Original remaining allowance changed")


def build():
    import platform
    import cv2
    import numpy as np
    original_guard()
    summary = base.read(HOME/"allowance.json")
    base.check(base.sha(LEDGER) == summary["private_ledger_sha256"], "Private approval ledger changed")
    validate(base.read(LEDGER), summary)
    prepared = base.read("reports/solar-active-v136/preparation.json")
    for name, digest in prepared["source_sha256"].items():
        base.check(base.sha(name) == digest, "Prepared algorithm changed: "+name)
    historical = base.read("reports/solar-local-v134/protocol.json")
    p = {name:deepcopy(historical[name]) for name in ("input_sha256", "cases", "inputs", "labels", "libraries")}
    p["sources"] = deepcopy(prepared["source_sha256"])
    for name in ("experiment.py", "engine.py", "admission.py", "test_experiment_v137.py", "METHOD.md", "allowance.json"):
        p["sources"][str(HOME/name)] = base.sha(HOME/name)
    for name in (str(CACHE), str(JOINT_CACHE), "reports/solar-active-v136/preparation.json",
                 "reports/solar-local-v134/protocol.json", "reports/solar-joint-v135/protocol.json",
                 "runs/solar-local-v134/comparison/COMPLETE.json", "runs/solar-local-v134/comparison/PREDICTIONS_SEALED.json",
                 "runs/solar-joint-v135/comparison/COMPLETE.json", "runs/solar-joint-v135/comparison/PREDICTIONS_SEALED.json"):
        p["input_sha256"][name] = base.sha(name)
    base.check(len(p["cases"]) == 32 and sum(c["positive"] for c in p["cases"]) == 16
               and base.sha("reports/solar-local-v134/protocol.json") == prepared["case_roster_sha256"],
               "Preserve the full frozen32-case roster")
    p.update(schema="approved_sequential_acquisition_study_v137", caps=CAPS.copy(),
        total_supplement_seconds=TOTAL, separately_reserved_recovery_seconds=480,
        allowance_sha256=base.sha(HOME/"allowance.json"), original_accounting_unchanged=True,
        cached_predictions=str(CACHE), joint_pilot_predictions=str(JOINT_CACHE),
        fresh_methods=["direct", "joint_no_extra", "uniform", "appearance", "geometry", "active"],
        cached_methods=20, total_method_case_cells=832,
        fresh_initializer=True, cached_initializer_used_as_input=False,
        reference_caching="Both reference geometry projections and Direct reference descriptors; no query-feature or query-camera cache. Cold misses and preparation costs recorded.",
        exact_controls="Fresh camera hypotheses and Direct endpoints/pose/estimator must repeat v134. The eight available no-extra joint-pilot rows must repeat v135.",
        patch_budget=dict(base=36,extra=32,policies=["uniform", "appearance", "geometry", "active"]),
        timing_rule="Full32-extra-search flat/layered generated controls, all six arms and fresh initialization. Forecast1.5*32*worst measured complete pair+45s. Estimate, not worst-case guarantee; hard original deadlines enforce caps.",
        intensity_control="24 generated gamma variants exercise Direct library extraction in timing only; these are not sunlight simulation.",
        primary_method="active", primary_pose="Unchanged six-match/three-pixel AP3P on measured native image-grid endpoints, not the soft camera proposal.",
        control_order="Four-policy Latin rotation over the four noise combinations; Direct first/last alternates. Same order for matched positives/nonoverlaps.",
        preliminary_success_rule="At least two more correct positives than every completed cached/no-extra/Direct control, no more wrong positive/nonoverlap outputs than Direct, and strictly more correct positives than each equal-budget acquisition control.",
        original_requirements_unchanged=True, calibrated_acceptance=False,
        scope="Consumed known-light development;16 positives/16 nonoverlaps, four acquisition conditions on three terrain components. Not final, transfer or new geography.",
        calibration_or_final_access=False, parameters_added=0, native_pixels=512, cpu_threads=2)
    p["runtime"] = dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                        platform=platform.platform(), opencv_threads=1)
    for name, digest in p["input_sha256"].items():
        base.check(base.sha(name) == digest, "Frozen input changed: "+name)
    return p


def published(commit, protocol):
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"], check=True)
    for name in (*protocol["sources"], str(HOME/"protocol.json")):
        base.check(subprocess.check_output(["git", "show", commit+":"+name]) == Path(name).read_bytes(),
                   "Published source differs: "+name)


def complete_predictions(protocol, run):
    seal = base.read(run/"PREDICTIONS_SEALED.json")
    raw = base.read(run/"predictions.json")
    base.check(seal["sha256"] == base.sha(run/"predictions.json") and seal["rows"] == 32
               and raw["protocol_sha256"] == base.sha(HOME/"protocol.json")
               and raw["query_truth_used"] is False and [r["case"] for r in raw["rows"]] == protocol["cases"],
               "Scoring requires all32 sealed query-truth-free predictions")
    return raw


def worker(operation, ticket_path):
    import resource
    started = time.perf_counter()
    ticket = base.read(ticket_path); protocol = base.read(HOME/"protocol.json")
    stage = "timing" if operation == "timing" else "comparison"
    base.check(ticket["parent_pid"] == os.getppid()
               and ticket["protocol_sha256"] == base.sha(HOME/"protocol.json")
               and ticket["stage"] == stage, "Worker requires its original admitted controller")
    published(ticket["commit"], protocol)
    journal, _ = state(LEDGER, JOURNAL, base.read(HOME/"allowance.json"), original_guard)
    reservation = ticket["charge"]
    base.check(reservation in journal["reservations"] and reservation["seconds"] == CAPS[stage]
               and reservation["job_id"] == base.sha(HOME/"protocol.json")+"/active-v137-"+stage
               and ticket["deadline"] == deadline(reservation).isoformat() and before_deadline(reservation),
               "Original supplementary charge/deadline required")
    # A child also enforces the original deadline if its parent disappears.
    signal.setitimer(signal.ITIMER_REAL, max(.001, (deadline(reservation)-datetime.now(timezone.utc)).total_seconds()))
    output = ticket_path.parent
    if operation in ("timing", "predict"):
        from engine import timing, predict
        (timing(protocol, output/"timing.json", base) if operation == "timing"
         else predict(protocol, output/"predictions.json", base))
    else:
        raw = complete_predictions(protocol, output)
        scores = [dict(case=row["case"], methods=base.score_result(row["result"], row["case"],
                   protocol["inputs"], protocol["labels"])) for row in raw["rows"]]
        if operation == "score":
            base.put(output/"scores.json", scores)
        else:
            base.check(scores == base.read(output/"scores.json"), "Independent geometry-metric replay differs")
            base.put(output/"replay.json", dict(passed=True, pairs=32, method_case_cells=832,
                predictions_sha256=base.sha(output/"predictions.json")))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    base.put(output/(operation+"-runtime.json"), dict(elapsed_seconds=time.perf_counter()-started,
        peak_rss_bytes=int(rss if sys.platform == "darwin" else 1024*rss),
        memory_scope="Whole worker and shared reference caches; not a per-model ranking."))


def execute_locked(stage, commit, lease_fd):
    protocol = base.read(HOME/"protocol.json"); published(commit, protocol)
    base.check(protocol == build(), "Prospective source, data, controls or scope changed")
    if stage == "comparison":
        completed, measured = (base.read(WORK/"timing"/name) for name in ("COMPLETE.json", "timing.json"))
        base.check(completed["complete"] and measured["admitted"]
                   and measured["full_extra_search_budget"] == 32
                   and completed["protocol_sha256"] == base.sha(HOME/"protocol.json"),
                   "The full-budget generated timing gate must pass first")
    base.check(LEDGER.resolve() == (ROOT/LEDGER).absolute()
               and JOURNAL.resolve() == (ROOT/JOURNAL).absolute(), "No supplementary ledger/journal aliases")
    output = WORK/stage; output.mkdir(parents=True, exist_ok=False)
    reservation = charge(LEDGER, JOURNAL, base.read(HOME/"allowance.json"),
                         base.sha(HOME/"protocol.json"), stage, original_guard)
    started = time.monotonic(); end = deadline(reservation)
    ticket = dict(parent_pid=os.getpid(), protocol_sha256=base.sha(HOME/"protocol.json"),
                  commit=commit, stage=stage, charge=reservation, deadline=end.isoformat())
    execution = output/"execution.json"; base.put(execution, ticket)
    try:
        for operation in (("timing",) if stage == "timing" else ("predict", "score", "replay")):
            with (output/(operation+".log")).open("x") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", operation,
                    "--ticket", str(execution)], stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease_fd,),
                    env={**os.environ, **base.ENV, "PYTHONPATH": str(ROOT)})
                try:
                    seconds = (end-datetime.now(timezone.utc)).total_seconds()
                    base.check(seconds > 0, "Original supplement deadline expired")
                    status = child.wait(timeout=seconds)
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
            base.check(status == 0 and before_deadline(reservation), "Worker failed or original deadline elapsed")
            if operation == "predict":
                base.put(output/"PREDICTIONS_SEALED.json", dict(rows=32, sha256=base.sha(output/"predictions.json")))
        original_guard()
        base.put(output/"COMPLETE.json", dict(ticket, complete=True, elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        base.put(output/"FAILED.json", dict(ticket, complete=False, error=type(error).__name__, message=str(error)))
        raise


def execute(stage, commit):
    with serial_lease(PRIVATE/"scientific.lock") as lease_fd:
        execute_locked(stage, commit, lease_fd)


if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    launch = sub.add_parser("run"); launch.add_argument("stage", choices=tuple(CAPS)); launch.add_argument("--commit", required=True)
    child = sub.add_parser("worker"); child.add_argument("operation", choices=("timing", "predict", "score", "replay")); child.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze": base.put(HOME/"protocol.json", build())
    elif args.action == "run": execute(args.stage, args.commit)
    else: worker(args.operation, args.ticket)
