"""Bounded full-radiance, subpixel and 3D-template development comparison."""

import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = spec_from_file_location("frozen_v131_experiment", HERE.parent/"solar-local-v131/experiment.py")
old = module_from_spec(spec); spec.loader.exec_module(old)
base = old.base
HOME = Path("reports/solar-local-v132")
WORK = Path("runs/solar-local-v132")
CAP = dict(timing=60, comparison=900)


def build():
    p = deepcopy(base.read("reports/solar-local-v131/protocol.json"))
    p.update(schema="native_radiance_subpixel_terrain_v132", caps=CAP.copy(),
             additional_parameters=0,
             technique="Full linear native radiance, two-pixel Gaussian integration, common-support masked NCC, independent reciprocal measurement, local quadratic subpixel peaks and full reference-3D raster templates.",
             new_methods=["native_plane_integer", "native_plane_subpixel", "native_terrain_subpixel"],
             matched_new_patch_budget=36,
             ablation_scope="Plane integer vs plane subpixel isolates peak fitting; plane vs terrain subpixel isolates geometric warp. Change from v131 also includes smoothing/radiance and 36 rather than12 anchors; do not attribute that entire gain to one factor.",
             admission="New prospective timing and 900-second comparison reservations within the original cumulative envelope; all earlier charges and results remain unchanged.",
             timing_rule="1.5 times completed v131 elapsed, adjusted upward for current generated optimizer speed, plus 1.5*32 times worst new generated three-arm measurement and30s. No archived inference during timing.")
    for name in ("experiment.py", "native_radiance.py", "test_native_radiance_v132.py", "replay_saved.py", "METHOD.md"):
        p["sources"][str(HOME/name)] = base.sha(HOME/name)
    for name in ("reports/solar-local-v131/protocol.json", "reports/solar-local-v131/results.json",
                 "runs/solar-local-v131/comparison/predictions.json",
                 "runs/solar-local-v131/comparison/PREDICTIONS_SEALED.json",
                 "runs/solar-local-v131/comparison/COMPLETE.json", "runs/solar-local-v131/timing/timing.json"):
        p["input_sha256"][name] = base.sha(name)
    base.check(base.read("runs/solar-local-v131/comparison/COMPLETE.json")["complete"], "Preserve completed shared-camera study")
    return p


def timing(output):
    import cv2
    import numpy as np
    from native_radiance import control, measure
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    rows = []
    historical = base.read("runs/solar-local-v131/timing/timing.json")["samples"]
    for index, variant in enumerate((False, True)):
        args, valid, truth = generated(variant)
        ref, depth, k, old_pose, query, qk, _ = args
        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
        expected = pose@np.linalg.inv(old_pose)@truth
        seeds = control.pipeline.coarse.reference_pose_bank(pose, count=8)
        tick = time.perf_counter()
        hypotheses = control.refine(ref, query, depth, k, pose, qk, seeds,
                                    reference_valid=np.ones_like(valid))
        optimizer = time.perf_counter()-tick
        base.check(hypotheses and np.linalg.norm(hypotheses[0]["pose"][:3, 3]-expected[:3, 3]) < 2.,
                   "Generated image-only camera gate failed")
        tick = time.perf_counter()
        matched = measure(ref, query, depth, k, pose, qk, hypotheses[0]["pose"][None],
                          reference_valid=np.ones_like(valid))
        measured = time.perf_counter()-tick
        for name, row in matched.items():
            base.check(row["pose"] is not None and np.linalg.norm(np.asarray(row["pose"])[:3, 3]-expected[:3, 3]) < 2.,
                       "Generated native measurement and common-PnP gate failed: "+name)
        rows.append(dict(multidepth=variant, optimizer_seconds=optimizer, measurement_seconds=measured,
                         speed_ratio=optimizer/historical[index]["optimizer_seconds"], passed=True))
        print(rows[-1], flush=True)
    prior = base.read("runs/solar-local-v131/comparison/COMPLETE.json")["elapsed_seconds"]
    projected = 1.5*prior*max(1., *(r["speed_ratio"] for r in rows)) + 1.5*32*max(r["measurement_seconds"] for r in rows)+30.
    base.put(output, dict(procedural_only=True, archived_images_opened=False, samples=rows,
                          projected_seconds=projected, admitted=projected <= CAP["comparison"]))
    base.check(projected <= CAP["comparison"], "Measured ten-arm study exceeds frozen cap")


base.HOME, base.WORK, base.CAP = HOME, WORK, CAP
base.build, base.timing, base.__file__ = build, timing, str(Path(__file__).resolve())

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run"); run.add_argument("stage", choices=tuple(CAP)); run.add_argument("--commit", required=True)
    worker = sub.add_parser("worker"); worker.add_argument("stage", choices=("timing", "predict", "score", "replay")); worker.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME/"protocol.json", build())
    elif args.action == "run":
        base.execute(args.stage, args.commit)
    else:
        if args.stage == "predict":
            from reference_cache import install
            install()
            import native_radiance
            native_radiance.control.pipeline.run = native_radiance.run
        base.worker(args.stage, args.ticket)
