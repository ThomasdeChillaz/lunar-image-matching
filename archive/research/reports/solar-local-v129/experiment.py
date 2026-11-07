"""Same native comparison with exact reference-only reuse and measured admission."""

import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/solar-local-v127"
sys.path.insert(0, str(OLD))
spec = spec_from_file_location("frozen_measurement_v127", OLD / "experiment.py")
base = module_from_spec(spec); spec.loader.exec_module(base)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOME = Path("reports/solar-local-v129")
WORK = Path("runs/solar-local-v129")


def build():
    protocol = deepcopy(base.read(OLD / "protocol.json"))
    protocol["schema"] = "reference_cached_native_measurement_repair_v129"
    for name in ("reference_cache.py", "experiment.py", "test_reference_cache.py"):
        file = HOME / name
        protocol["sources"][str(file)] = base.sha(file)
    for name in ("reports/solar-local-v127/protocol.json", "runs/solar-local-v127/timing/FAILED.json",
                 "runs/solar-local-v128/timing/FAILED.json", "runs/solar-local-v128/timing/timing.log"):
        protocol["input_sha256"][name] = base.sha(name)
    protocol["reference_only_cache"] = "Exact geometry-keyed visibility and image/settings-keyed reference SIFT features. Current query features, ranking and matching always rerun."
    protocol["timing_forecast"] = "1.5 * (32 * maximum measured warm per-pair work + 4 * maximum cold reference overhead + 104 * maximum reference SIFT extraction + 40 seconds IO/scoring). All 24 SIFT references and two matched-light replicates per four references, and every case retained."
    protocol["previous_failed_caps_preserved"] = True
    return protocol


def timing(output):
    import cv2
    import numpy as np
    from pipeline import run
    from reference_cache import CachedReferenceSift, ReferenceProjectionCache, install
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    install()
    samples = []
    for variant in (False, True):
        cache = ReferenceProjectionCache()
        args, valid, _ = generated(variant)
        reference, depth, k, _, query, qk, _ = args
        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
        start = time.perf_counter()
        first, _ = cache.initialize(reference, query, depth, k, pose, qk)
        cold = time.perf_counter()-start
        print(dict(multidepth=variant, cold_initialization_seconds=cold), flush=True)
        start = time.perf_counter()
        second, _ = cache.initialize(reference, query, depth, k, pose, qk)
        warm = time.perf_counter()-start
        base.check(np.array_equal(first, second), "Reference cache changed the ordered camera hypotheses")
        encoded = np.rint(255*np.clip(reference/reference.max(), 0, 1)).astype(np.uint8)
        start = time.perf_counter()
        result = run(reference, query, depth, k, pose, qk, second, [encoded]*24,
                     reference_valid=np.ones_like(valid), initialization_seconds=warm)
        measurement = time.perf_counter()-start
        row = dict(multidepth=variant, cold_initialization_seconds=cold, warm_initialization_seconds=warm,
                   measurement_seconds=measurement, warm_pair_seconds=warm+measurement,
                   cold_reference_overhead_seconds=max(0., cold-warm),
                   ordered_camera_cache_parity=True,
                   methods={name: item["pose"] is not None for name, item in result["methods"].items()})
        samples.append(row)
        print(row, flush=True)
    reference_feature_preparation = 104*max(CachedReferenceSift.store.miss_seconds, default=0.)
    projected = 1.5*(32*max(r["warm_pair_seconds"] for r in samples)
                      + 4*max(r["cold_reference_overhead_seconds"] for r in samples)
                      + reference_feature_preparation+40.)
    base.put(output, dict(procedural_only=True, archived_images_opened=False, samples=samples,
                          projected_seconds=projected, admitted=projected <= base.CAP["comparison"],
                          conservative_104_reference_feature_seconds=reference_feature_preparation,
                          scope="Reference-resident workload with cold geometry preparation counted for all four references. SIFT feature preparation actually incurred is retained in measured work."))
    base.check(projected <= base.CAP["comparison"], "Complete comparison exceeds unchanged 600-second cap")


base.HOME, base.WORK = HOME, WORK
base.build, base.timing = build, timing
base.__file__ = str(Path(__file__).resolve())

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run"); run.add_argument("stage", choices=tuple(base.CAP)); run.add_argument("--commit", required=True)
    worker = sub.add_parser("worker"); worker.add_argument("stage", choices=("timing", "predict", "score", "replay")); worker.add_argument("--ticket", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME / "protocol.json", build())
    elif args.action == "run":
        base.execute(args.stage, args.commit)
    else:
        if args.stage == "predict":
            from reference_cache import install
            install()
        base.worker(args.stage, args.ticket)
