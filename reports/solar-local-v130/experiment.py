"""Prospective 900-second comparison admitted from the complete v129 cost evidence."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/solar-local-v129"
sys.path.insert(0, str(OLD))
spec = spec_from_file_location("frozen_cached_v129", OLD / "experiment.py")
cached = module_from_spec(spec); spec.loader.exec_module(cached)
base = cached.base
HOME = Path("reports/solar-local-v130")
WORK = Path("runs/solar-local-v130")
CAP = 900


def timing_proof():
    path = Path("runs/solar-local-v129/timing")
    timing = base.read(path / "timing.json")
    failure = base.read(path / "FAILED.json")
    execution = base.read(path / "execution.json")
    _, journal, _ = base.journal_state(base.legacy.LEDGER, base.legacy.LEDGER_SHA, base.legacy.JOURNAL)
    base.check(failure["complete"] is False and failure["error"] == "ValueError"
               and failure["charge"] == execution["charge"] and execution["charge"] in journal["reservations"]
               and execution["charge"]["seconds"] == 60 and not execution["charge"]["refund_on_failure"]
               and execution["protocol_sha256"] == base.sha(OLD / "protocol.json"), "Original failed timing and charge required")
    samples = timing["samples"]
    base.check(timing["procedural_only"] and not timing["archived_images_opened"]
               and not timing["admitted"] and len(samples) == 2
               and [r["multidepth"] for r in samples] == [False, True]
               and all(r["ordered_camera_cache_parity"] and len(r["methods"]) == 5
                       and all(r["methods"].values()) for r in samples), "Both complete native controls must pass")
    forecast = 1.5*(32*max(r["warm_pair_seconds"] for r in samples)
                   +4*max(r["cold_reference_overhead_seconds"] for r in samples)
                   +timing["conservative_104_reference_feature_seconds"]+40.)
    base.check(forecast == timing["projected_seconds"] and 600 < forecast <= CAP,
               "Use the unchanged complete timing formula; no weakened safety factor")
    return dict(path=str(path / "timing.json"), sha256=base.sha(path / "timing.json"),
                forecast_seconds=forecast, previous_limit_seconds=600, new_limit_seconds=CAP,
                previous_refusal_preserved=True, cache_parity_and_nonnull_pose_checks_passed=True)


def build():
    p = deepcopy(base.read(OLD / "protocol.json"))
    p["schema"] = "measured_admission_native_measurement_repair_v130"
    p["caps"]["comparison"] = CAP
    for file in (HOME / "experiment.py", HOME / "test_admission.py", HOME / "replay_saved.py",
                 Path("reports/solar-local-v129/replay_saved.py")):
        p["sources"][str(file)] = base.sha(file)
    for file in (Path("reports/solar-local-v129/protocol.json"), *Path("runs/solar-local-v129/timing").glob("*"),
                 *Path("reports/solar-local-v130/preflight").glob("*")):
        if file.is_file():
            p["input_sha256"][str(file)] = base.sha(file)
    p["admission"] = timing_proof()
    p["matching_scoring_and_roster_unchanged_from_v129"] = True
    base.check(all(not Path(name).is_absolute() for name in (*p["sources"], *p["input_sha256"])),
               "All frozen dependencies must have portable repository-relative paths")
    p["preflight_manifest_correction"] = "Repository-relative dependency paths; earlier publication-check refusal preserved before any scientific inference or reservation."
    p["per_job_cap_amendment"] = "New prospective reservation from the unchanged original cumulative allowance. Prior 60/600-second attempts and their refusals remain untouched."
    return p


def execute(commit):
    p = base.read(HOME / "protocol.json")
    base.publish_check(commit, p)
    base.check(p == build(), "Published evidence or prospective scope changed")
    base.check(base.remaining(base.read(base.legacy.CONTRACT))["baseline"] >= CAP,
               "Cannot exceed the unchanged original cumulative allowance")
    output = WORK / "comparison"
    output.mkdir(parents=True, exist_ok=False)
    charge = base.reserve(base.legacy.LEDGER, base.legacy.LEDGER_SHA, base.legacy.JOURNAL,
                          base.sha(HOME / "protocol.json")+"/v127-comparison", "baseline", CAP)["reservation"]
    started = time.monotonic()
    deadline = datetime.fromisoformat(charge["reserved_utc"])+timedelta(seconds=CAP)
    allowance = min(CAP, (deadline-base.now()).total_seconds())
    execution = dict(parent_pid=os.getpid(), protocol_sha256=base.sha(HOME / "protocol.json"),
                     commit=commit, charge=charge, deadline=deadline.isoformat())
    ticket = output / "execution.json"
    base.put(ticket, execution)
    try:
        for operation in ("predict", "score", "replay"):
            with (output / (operation+".log")).open("x") as log:
                child = subprocess.Popen([sys.executable, __file__, "worker", operation, "--ticket", str(ticket)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, **base.ENV, "PYTHONPATH":str(ROOT)})
                try:
                    status = child.wait(timeout=max(.01, allowance-(time.monotonic()-started)))
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
            base.check(status == 0 and time.monotonic()-started < allowance and base.now() < deadline,
                       "Worker failed or original prospective deadline elapsed")
            if operation == "predict":
                base.put(output / "PREDICTIONS_SEALED.json", dict(sha256=base.sha(output / "predictions.json"), rows=len(p["cases"])))
        base.put(output / "COMPLETE.json", dict(execution, complete=True, elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        base.put(output / "FAILED.json", dict(execution, complete=False, error=type(error).__name__, message=str(error)))
        raise


base.HOME, base.WORK = HOME, WORK
base.CAP["comparison"] = CAP

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run"); run.add_argument("--commit", required=True)
    worker = sub.add_parser("worker"); worker.add_argument("stage", choices=("predict", "score", "replay")); worker.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME / "protocol.json", build())
    elif args.action == "run":
        execute(args.commit)
    else:
        if args.stage == "predict":
            from reference_cache import install
            install()
        base.worker(args.stage, args.ticket)
