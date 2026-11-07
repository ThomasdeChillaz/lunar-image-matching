"""Instrument the unchanged v127 pilot before admitting archived inference."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/solar-local-v127"
sys.path.insert(0, str(OLD))
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("frozen_measurement_v127", OLD / "experiment.py")
base = module_from_spec(spec)
spec.loader.exec_module(base)
HOME = Path("reports/solar-local-v128")
WORK = Path("runs/solar-local-v128")
ORIGINAL_TIMING = base.timing


def build():
    protocol = deepcopy(base.read(OLD / "protocol.json"))
    protocol["schema"] = "instrumented_native_measurement_repair_v128"
    protocol["sources"][str(HOME / "experiment.py")] = base.sha(__file__)
    failure = "runs/solar-local-v127/timing/FAILED.json"
    protocol["input_sha256"][failure] = base.sha(failure)
    protocol["v127_timing_failed_preserved"] = failure
    protocol["algorithm_changes_from_v127"] = False
    protocol["instrumentation"] = "Timestamp unchanged initialization, patch arms and Direct. Preserve the two-case pilot and 60/600-second caps."
    return protocol


def timing(output):
    import pipeline

    def logged(name, function):
        def call(*args, **kwargs):
            start = time.perf_counter()
            print(name + " started", flush=True)
            result = function(*args, **kwargs)
            print(name + " completed in " + str(time.perf_counter()-start), flush=True)
            return result
        return call

    pipeline.initialize = logged("coarse_initialization", pipeline.initialize)
    pipeline.patch_measure = logged("patch_arm", pipeline.patch_measure)
    pipeline.direct = logged("Direct", pipeline.direct)
    ORIGINAL_TIMING(output)


base.HOME, base.WORK = HOME, WORK
base.build, base.timing = build, timing
base.__file__ = str(Path(__file__).resolve())

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run")
    run.add_argument("stage", choices=tuple(base.CAP))
    run.add_argument("--commit", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("stage", choices=("timing", "predict", "score", "replay"))
    worker.add_argument("--ticket", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME / "protocol.json", build())
    elif args.action == "run":
        base.execute(args.stage, args.commit)
    else:
        base.worker(args.stage, args.ticket)
