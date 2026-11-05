"""Retain the refused v132 pilot and compare the same AP3P in a reference frame."""

import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = spec_from_file_location("preserved_v132_experiment", HERE.parent/"solar-local-v132/experiment.py")
old = module_from_spec(spec); spec.loader.exec_module(old)
base = old.base
HOME = Path("reports/solar-local-v133")
WORK = Path("runs/solar-local-v133")
CAP = dict(timing=60, comparison=900)
ORIGINAL_PREDICT = base.predict


def predict(protocol, output):
    # The inherited controller charges library decoding to the original Direct
    # name. Give its rigid-frame control that identical cost before sealing.
    temporary = output.with_name("predictions-before-library-timing.json")
    ORIGINAL_PREDICT(protocol, temporary)
    data = base.read(temporary)
    for row in data["rows"]:
        row["result"]["methods"]["framed_direct"]["inference_seconds"] += row["library_input_seconds"]
    base.put(output, data)


def build():
    p = deepcopy(base.read("reports/solar-local-v132/protocol.json"))
    p.update(schema="reference_frame_native_radiance_v133", caps=CAP.copy(),
             estimator_change="Same frozen AP3P/RANSAC, 2000 iterations, 3-native-pixel threshold and refit. All ten methods additionally solve in the reference-camera coordinate frame; original world-frame results also retained.",
             primary_method="framed_native_terrain_subpixel",
             primary_baseline="framed_direct",
             library_timing="Original and framed Direct both include the same complete reference-library decoding cost.",
             admission="Separate original-envelope 60/900-second reservations. Preserve v132 generated failure and its full60-second charge, without changing any prior deadline or gate.",
             generated_gate="Every new framed arm must recover both generated motions within2m using estimated image-only initialization; the original world-frame failures remain explicit controls.",
             success_rule="At least two more correct positives than framed Direct AND the strongest of the seven preexisting controls in either coordinate frame, with no increase in wrong positives or raw nonoverlap outputs compared with framed Direct.")
    for name in ("experiment.py", "reference_frame.py", "test_reference_frame_v133.py", "replay_saved.py", "METHOD.md"):
        p["sources"][str(HOME/name)] = base.sha(HOME/name)
    for name in ("reports/solar-local-v132/protocol.json", "runs/solar-local-v132/timing/FAILED.json",
                 "runs/solar-local-v132/timing/execution.json", "runs/solar-local-v132/timing/timing.log"):
        p["input_sha256"][name] = base.sha(name)
    failure = base.read("runs/solar-local-v132/timing/FAILED.json")
    _, journal, _ = base.journal_state(base.legacy.LEDGER, base.legacy.LEDGER_SHA, base.legacy.JOURNAL)
    base.check(not failure["complete"] and failure["charge"] in journal["reservations"]
               and failure["charge"]["seconds"] == 60 and not failure["charge"]["refund_on_failure"],
               "Original refused generated pilot and charge must remain")
    return p


def timing(output):
    import cv2
    import numpy as np
    from reference_frame import native, reestimate, solve
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    rows = []
    historical = base.read("runs/solar-local-v131/timing/timing.json")["samples"]
    for index, variant in enumerate((False, True)):
        args, valid, truth = generated(variant)
        ref, depth, k, old_pose, query, qk, _ = args
        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
        expected = pose@np.linalg.inv(old_pose)@truth
        seeds = native.control.pipeline.coarse.reference_pose_bank(pose, count=8)
        tick = time.perf_counter()
        hypotheses = native.control.refine(ref, query, depth, k, pose, qk, seeds,
                                           reference_valid=np.ones_like(valid))
        optimizer = time.perf_counter()-tick
        base.check(hypotheses and np.linalg.norm(hypotheses[0]["pose"][:3, 3]-expected[:3, 3]) < 2.,
                   "Generated camera optimization failed")
        tick = time.perf_counter()
        matched = native.measure(ref, query, depth, k, pose, qk, hypotheses[0]["pose"][None],
                                 reference_valid=np.ones_like(valid))
        # One rigid-coordinate re-estimation is measured for every matcher,
        # conservatively using the largest of the available native match sets.
        checks = {}
        refit_seconds = []
        for name, row in matched.items():
            start = time.perf_counter()
            fixed = reestimate(row, depth, k, pose, qk)
            refit_seconds.append(time.perf_counter()-start)
            error = None if fixed["pose"] is None else float(np.linalg.norm(np.asarray(fixed["pose"])[:3, 3]-expected[:3, 3]))
            checks[name] = dict(world_frame_pose_returned=row["pose"] is not None,
                                reference_frame_pose_returned=fixed["pose"] is not None,
                                reference_frame_translation_error_m=error,
                                measured_correspondences=len(row["reference_xy"]))
            base.check(error is not None and error < 2., "Reference-frame generated native/PnP gate failed: "+name)
        measured = time.perf_counter()-tick+10*max(refit_seconds)
        rows.append(dict(multidepth=variant, optimizer_seconds=optimizer, measurement_seconds=measured,
                         speed_ratio=optimizer/historical[index]["optimizer_seconds"], passed=True,
                         measurements=checks))
        print(rows[-1], flush=True)
    # Direct can return up to2048 SIFT endpoints, far more than36 anchors.
    # Include actual large-set solver costs and adverse outlier mixtures.
    generator = np.random.default_rng(20261133)
    points = generator.uniform(64., 448., (2048, 2))
    world, _ = native.control.pipeline.map_points_z(points, depth, k, pose)
    targets, _, _ = native.control.project_z(world, qk, expected)
    solver_samples = []
    for fraction in (0., .5, 1.):
        observed = targets.copy()
        count = int(len(points)*fraction)
        observed[:count] = generator.uniform(0., 511., (count, 2))
        tick = time.perf_counter()
        solve(points, observed, depth, k, pose, qk)
        solver_samples.append(dict(matches=len(points), outlier_fraction=fraction,
                                   seconds=time.perf_counter()-tick))
    prior = base.read("runs/solar-local-v131/comparison/COMPLETE.json")["elapsed_seconds"]
    additional = max(r["measurement_seconds"] for r in rows)+10*max(r["seconds"] for r in solver_samples)
    projected = 1.5*prior*max(1., *(r["speed_ratio"] for r in rows))+1.5*32*additional+30.
    base.put(output, dict(procedural_only=True, archived_images_opened=False, samples=rows,
                          projected_seconds=projected, admitted=projected <= CAP["comparison"],
                          original_v132_failure_preserved=True, large_match_set_solver=solver_samples))
    base.check(projected <= CAP["comparison"], "Measured twenty-arm study exceeds unchanged900-second cap")


base.HOME, base.WORK, base.CAP = HOME, WORK, CAP
base.build, base.timing, base.predict, base.__file__ = build, timing, predict, str(Path(__file__).resolve())

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
            import reference_frame
            reference_frame.native.control.pipeline.run = reference_frame.run
        base.worker(args.stage, args.ticket)
