"""Repeat the identical comparison with JSON-safe rejected-match diagnostics."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = spec_from_file_location("preserved_v133_experiment", HERE.parent/"solar-local-v133/experiment.py")
old = module_from_spec(spec); spec.loader.exec_module(old)
base = old.base
HOME = Path("reports/solar-local-v134")
WORK = Path("runs/solar-local-v134")
CAP = 600


def timing_proof():
    failure = base.read("runs/solar-local-v133/comparison/FAILED.json")
    _, journal, _ = base.journal_state(base.legacy.LEDGER, base.legacy.LEDGER_SHA, base.legacy.JOURNAL)
    base.check(not failure["complete"] and failure["charge"] in journal["reservations"]
               and failure["charge"]["seconds"] == 900 and not failure["charge"]["refund_on_failure"],
               "Preserve original failed full-study reservation")
    log = (HOME/"preserved-v133-runtime.log").read_text()
    match = re.search(r"^\s*(\d+\.\d+) real\s+", log, flags=re.MULTILINE)
    base.check(match is not None, "Original actual full-inference runtime required")
    elapsed = float(match.group(1))
    trace = (HOME/"preserved-v133-predict.log").read_text()
    base.check("ORIGINAL_PREDICT(protocol, temporary)" in trace
               and 'put(output, dict(protocol_sha256' in trace
               and "Out of range float values are not JSON compliant: inf" in trace,
               "Failure must occur at the final prediction dump, after the complete inference loop")
    base.check(not (Path("runs/solar-local-v133/comparison")/"scores.json").exists(),
               "Failed attempt must remain unscored")
    projected = 1.5*elapsed+30.
    base.check(projected <= CAP, "Measured identical-inference repeat exceeds prospective cap")
    return dict(original_wall_seconds=elapsed, projected_seconds=projected, cap_seconds=CAP,
                admitted=True, all_inference_completed_but_outputs_invalid=True,
                prior900_second_charge_preserved=True, new_matching_or_selection=False)


def build():
    p = deepcopy(base.read("reports/solar-local-v133/protocol.json"))
    p.update(schema="finite_rejected_diagnostics_native_v134", caps=dict(comparison=CAP),
             numerical_matching_and_scoring_unchanged=True,
             diagnostic_fix="Only positive-infinite margins on already rejected measurements become null with an unavailable-comparison flag. Accepted endpoints, thresholds, camera solutions, all20 methods and all32 image pairs are unchanged.",
             admission=timing_proof())
    for name in ("experiment.py", "finite_diagnostics.py", "test_finite_diagnostics_v134.py", "replay_saved.py", "METHOD.md"):
        p["sources"][str(HOME/name)] = base.sha(HOME/name)
    for name in ("reports/solar-local-v133/protocol.json", "runs/solar-local-v133/comparison/FAILED.json",
                 "runs/solar-local-v133/comparison/execution.json",
                 "runs/solar-local-v133/comparison/predict.log",
                 "runs/solar-local-v133/comparison/predictions-before-library-timing.json",
                 "reports/solar-local-v134/preserved-v133-runtime.log",
                 "reports/solar-local-v134/preserved-v133-predict.log",
                 "reports/solar-local-v134/preserved-v133-failure.json"):
        p["input_sha256"][name] = base.sha(name)
    return p


def execute(commit):
    p = base.read(HOME/"protocol.json")
    base.publish_check(commit, p)
    base.check(p == build(), "Published repeat scope, failure or source changed")
    base.check(base.remaining(base.read(base.legacy.CONTRACT))["baseline"] >= CAP, "Original cumulative allowance exhausted")
    output = WORK/"comparison"
    output.mkdir(parents=True, exist_ok=False)
    charge = base.reserve(base.legacy.LEDGER, base.legacy.LEDGER_SHA, base.legacy.JOURNAL,
                          base.sha(HOME/"protocol.json")+"/v127-comparison", "baseline", CAP)["reservation"]
    started = time.monotonic()
    deadline = datetime.fromisoformat(charge["reserved_utc"])+timedelta(seconds=CAP)
    allowance = min(CAP, (deadline-base.now()).total_seconds())
    execution = dict(parent_pid=os.getpid(), protocol_sha256=base.sha(HOME/"protocol.json"),
                     commit=commit, charge=charge, deadline=deadline.isoformat())
    ticket = output/"execution.json"
    base.put(ticket, execution)
    try:
        for operation in ("predict", "score", "replay"):
            with (output/(operation+".log")).open("x") as log:
                child = subprocess.Popen([sys.executable, __file__, "worker", operation, "--ticket", str(ticket)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, **base.ENV, "PYTHONPATH": str(ROOT)})
                try:
                    status = child.wait(timeout=max(.01, allowance-(time.monotonic()-started)))
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
            base.check(status == 0 and time.monotonic()-started < allowance and base.now() < deadline,
                       "Worker failed or original repeat cap elapsed")
            if operation == "predict":
                base.put(output/"PREDICTIONS_SEALED.json", dict(sha256=base.sha(output/"predictions.json"), rows=len(p["cases"])))
        base.put(output/"COMPLETE.json", dict(execution, complete=True, elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        base.put(output/"FAILED.json", dict(execution, complete=False, error=type(error).__name__, message=str(error)))
        raise


base.HOME, base.WORK, base.CAP = HOME, WORK, dict(comparison=CAP)

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run"); run.add_argument("--commit", required=True)
    worker = sub.add_parser("worker"); worker.add_argument("stage", choices=("predict", "score", "replay")); worker.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME/"protocol.json", build())
    elif args.action == "run":
        execute(args.commit)
    else:
        if args.stage == "predict":
            from reference_cache import install
            install()
            from finite_diagnostics import install as safe_diagnostics
            safe_diagnostics()
        base.worker(args.stage, args.ticket)
