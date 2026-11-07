"""Source-frozen, original-budget pilot of shared-camera correlation potentials."""

import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = spec_from_file_location("frozen_native_v134_for_joint_v135", HERE.parent/"solar-local-v134/experiment.py")
parent = module_from_spec(spec); spec.loader.exec_module(parent)
base = parent.base
HOME = Path("reports/solar-joint-v135")
WORK = Path("runs/solar-joint-v135")
CAP = dict(timing=15, comparison=80)
CACHE = Path("runs/solar-local-v134/comparison/predictions.json")


def build():
    original = base.read("reports/solar-local-v134/protocol.json")
    complete = base.read("runs/solar-local-v134/comparison/COMPLETE.json")
    seal = base.read("runs/solar-local-v134/comparison/PREDICTIONS_SEALED.json")
    base.check(complete["complete"] and complete["protocol_sha256"] == base.sha("reports/solar-local-v134/protocol.json")
               and seal["sha256"] == base.sha(CACHE), "Completed sealed initializer/control source required")
    sources = deepcopy(original["sources"])
    for name in ("experiment.py", "likelihood.py", "test_likelihood_v135.py", "replay_saved.py", "METHOD.md"):
        sources[str(HOME/name)] = base.sha(HOME/name)
    pins = deepcopy(original["input_sha256"])
    for name in ("reports/solar-local-v134/protocol.json", str(CACHE),
                 "runs/solar-local-v134/comparison/COMPLETE.json",
                 "runs/solar-local-v134/comparison/PREDICTIONS_SEALED.json",
                 "reports/solar-local-v134/results.json"):
        pins[name] = base.sha(name)
    cases = [deepcopy(c) for c in original["cases"] if c["reference_seed"] == c["query_seed"] == 0]
    base.check(len(cases) == 8 and sum(c["positive"] for c in cases) == 4,
               "All four conditions and matched nonoverlaps required")
    return dict(schema="shared_camera_correlation_potentials_v135", sources=sources,
                input_sha256=pins, cases=cases, inputs=original["inputs"], labels=original["labels"],
                caps=CAP.copy(), cpu_threads=2, native_pixels=512, parameters_added=0,
                cache=str(CACHE), cached_initializer="First frozen v134 photometric camera, if available; exact case/image bindings. No query depth, true pose or Sun.",
                primary_method="joint", baseline="cached_framed_direct",
                hard_control="Image-only hard peaks on exactly the same native correlation fields; must repeat frozen v134 native3D endpoints and AP3P exactly.",
                scope="Consumed known-light development; four camera conditions/three terrain components; one predetermined noise pairing per condition. Not new geography or a complete-runtime comparison.",
                calibration_or_final_access=False, accepted_results=False,
                cached_baselines="All20 frozen v134 method outputs copied exactly and scored on these same eight cases; no fresh baseline inference or timing claim.",
                success_rule="At least one additional correct positive versus the strongest cached completed control and fresh hard-peak control, with no more wrong positive/nonoverlap outputs than Direct. This is a preliminary four-positive signal; all original full-study targets remain unchanged.",
                timing_rule="Two generated image/pose controls. Forecast8 cases at1.5 times the worst measured shared fields+hard+joint stage, extrapolating camera optimizer calls to their fixed maximum, plus12 seconds for guards/IO/scoring.",
                original_requirements_unchanged=True)


def timing(output):
    import cv2
    import numpy as np
    from likelihood import MAX_STEPS, run
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    rows = []
    for variant in (False, True):
        args, valid, truth = generated(variant)
        ref, depth, k, pose, query, qk, _ = args
        tick = time.perf_counter()
        result = run(ref, query, depth, k, pose, qk, pose, reference_valid=np.ones_like(valid))
        for name in ("hard", "joint"):
            camera = result["methods"][name]["pose"]
            base.check(camera is not None and np.linalg.norm(np.asarray(camera)[:3, 3]-truth[:3, 3]) < 2.,
                       "Generated image-to-AP3P control failed: "+name)
        values = result["timing"]
        calls = result["fit"]["calls"]
        max_calls = 1+12+MAX_STEPS*(8+12)
        worst_fit = values["camera_fit"]*max_calls/max(calls, 1)
        bounded = values["fields"]+values["hard_measure"]+values["joint_measure"]+worst_fit
        rows.append(dict(multidepth=variant, passed=True, elapsed_seconds=time.perf_counter()-tick,
                         timing=values, optimizer_calls=calls, maximum_optimizer_calls=max_calls,
                         conservative_pair_seconds=bounded))
        print(rows[-1], flush=True)
    projected = 1.5*8*max(r["conservative_pair_seconds"] for r in rows)+12.
    base.put(output, dict(procedural_only=True, archived_images_opened=False, samples=rows,
                          projected_seconds=projected, admitted=projected <= CAP["comparison"]))
    base.check(projected <= CAP["comparison"], "Joint pilot exceeds remaining original comparison allowance")


def predict(protocol, output):
    import cv2
    import numpy as np
    from replay_raw_acquisition_v109 import decode, luminance
    from likelihood import run
    cv2.setNumThreads(1)
    cache = base.read(CACHE)
    base.check(cache["query_truth_used"] is False and base.sha(CACHE) == protocol["input_sha256"][str(CACHE)],
               "Query-truth-free frozen cache changed")
    frozen = {base_key(r["case"]): r for r in cache["rows"]}
    def checked(name):
        base.check(base.sha(name) == protocol["input_sha256"][name], "Input changed: "+name)
        return Path(name)
    rows = []
    for number, case in enumerate(protocol["cases"]):
        started = time.perf_counter()
        cached = frozen[base_key(case)]
        ref = protocol["inputs"][case["reference_index"]]
        qry = protocol["inputs"][case["query_index"]]
        reference = luminance(decode(checked(ref["reference_radiance"][0]["path"]).read_bytes(), exr_size=512))
        query = luminance(decode(checked(qry["query_radiance"][0]["path"]).read_bytes(), exr_size=512))
        with np.load(checked(ref["reference_geometry"]["path"]), allow_pickle=False) as data:
            depth, k, pose, valid = [data[n].copy() for n in ("axial_depth", "intrinsics", "cam2world", "valid")]
        query_k = np.asarray(qry["query_intrinsics"])
        initialization = cached["result"]["photometric_initialization"]
        initial = None if not initialization else np.asarray(initialization[0]["pose"])
        input_seconds = time.perf_counter()-started
        result = run(reference, query, depth, k, pose, query_k, initial, reference_valid=valid)
        old = cached["result"]["methods"]["framed_native_terrain_subpixel"]
        for name in ("reference_xy", "query_xy", "pose", "estimator"):
            base.check(result["methods"]["hard"][name] == old[name], "Fresh hard-peak control changed: "+name)
        result["hard_control_exact"] = True
        fields = result.pop("fields")
        field_path = output.parent/f"fields-{number:02}.npz"
        if fields is None or not fields["records"]:
            arrays = dict(scores=np.empty((0, 49, 49)), support=np.empty((0, 49, 49), bool),
                          centers=np.empty((0, 2)), local_world=np.empty((0, 3)), anchor_indices=np.empty(0, int))
        else:
            arrays = dict(scores=fields["scores"], support=fields["support"],
                          centers=np.asarray([r["center"] for r in fields["records"]]),
                          local_world=np.asarray([r["local_world"] for r in fields["records"]]),
                          anchor_indices=np.asarray([r["anchor_index"] for r in fields["records"]]))
        with field_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        result["native_correlation_fields"] = dict(path=str(field_path), sha256=base.sha(field_path), missing_support="Boolean mask; stored zero is not evidence.")
        for name in ("hard", "joint", "joint_camera_proposal"):
            result["methods"][name]["conditional_inference_seconds"] += input_seconds
        for name, row in cached["result"]["methods"].items():
            result["methods"]["cached_"+name] = deepcopy(row)
        result["methods"]["cached_camera_proposal"] = dict(reference_xy=[], query_xy=[],
            pose=None if initial is None else initial.tolist(), proposal_only=True)
        rows.append(dict(case=case, result=result, input_seconds=input_seconds,
                         elapsed_seconds=time.perf_counter()-started,
                         cached_initializer_sha256=base.sha(CACHE)))
        print(dict(case=number, elapsed_seconds=rows[-1]["elapsed_seconds"],
                   fields=len(arrays["scores"]), control_exact=True), flush=True)
    base.put(output, dict(protocol_sha256=base.sha(HOME/"protocol.json"),
                          query_truth_used=False, rows=rows))


def base_key(case):
    return (case["reference_index"], case["query_index"], case["reference_seed"], case["query_seed"], case["positive"])


base.HOME, base.WORK, base.CAP = HOME, WORK, CAP
base.build, base.timing, base.predict, base.__file__ = build, timing, predict, str(Path(__file__).resolve())

if __name__ == "__main__":
    base.check(Path.cwd() == ROOT, "Run from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    launch = sub.add_parser("run"); launch.add_argument("stage", choices=tuple(CAP)); launch.add_argument("--commit", required=True)
    worker = sub.add_parser("worker"); worker.add_argument("stage", choices=("timing", "predict", "score", "replay")); worker.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        base.put(HOME/"protocol.json", build())
    elif args.action == "run":
        base.execute(args.stage, args.commit)
    else:
        base.worker(args.stage, args.ticket)
