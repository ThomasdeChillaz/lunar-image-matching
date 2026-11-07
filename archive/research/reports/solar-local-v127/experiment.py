"""Bounded native measurement repair and matched Direct comparison."""

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import file_digest
from itertools import permutations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HOME = Path("reports/solar-local-v127")
WORK = Path("runs/solar-local-v127")
V126 = Path("reports/solar-parallax-v126")
V117 = Path("reports/solar-warp-v117")
for p in (ROOT, ROOT / V126, ROOT / V117):
    sys.path.insert(0, str(p))

import control_parallax_v126 as parent
import control_warp_archive_v117 as legacy
from solar_compute_budget_v67 import reserve
from solar_learning_pipeline_v73 import remaining, journal_state

CAP = {"timing": 60, "comparison": 600}
ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
           VECLIB_MAXIMUM_THREADS="2", NUMEXPR_MAX_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def now():
    return datetime.now(timezone.utc)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def roster():
    inputs = read(V117 / "inference-inputs.json")["exposures"]
    labels = read(V117 / "scoring-inputs.json")["exposures"]
    main = read("reports/solar-five-arm-v115/comparison-protocol.json")
    profile = next(p for p in main["profiles"] if p["backend"] == "direct_render")
    libraries = read(profile["render_evidence"]["manifest"])["libraries"]
    manifest = read(main["manifest"])
    bounds = {}
    for case in manifest["cases"]:
        if case["positive"]:
            continue
        cert = case["nonoverlap_certificate"]
        for endpoint in ("reference", "query"):
            component = case["component"] if endpoint == "reference" else case["query_component"]
            found = cert[endpoint + "_bounds"]
            check(component not in bounds or bounds[component] == found, "Component coverage certificate changed")
            bounds[component] = found
    positives = {c["case_id"]: c for c in manifest["cases"] if c["positive"]}
    components = [positives[label["case_id"]]["component"] for label in labels]
    from solar_evaluation_v70 import bounds_disjoint

    mapping = next(order for order in permutations(range(4)) if all(
        order[i] != i and components[i] != components[order[i]]
        and bounds_disjoint(bounds[components[i]], bounds[components[order[i]]]) for i in range(4)))
    rows, selected_libraries = [], {}
    for index, (item, label) in enumerate(zip(inputs, labels, strict=True)):
        unit = positives[label["case_id"]]["unit_index"]
        library, = [p for p in libraries if p["unit_index"] == unit]
        check(library["reference_geometry_sha256"] == item["reference_geometry"]["sha256"],
              "Direct library must use the exact same reference camera")
        selected_libraries[str(index)] = library["entries"]
        for r in (0, 1):
            for q in (0, 1):
                for positive in (True, False):
                    target = index if positive else mapping[index]
                    rows.append(dict(case_id=label["case_id"] if positive else
                                     label["case_id"] + "-to-" + labels[target]["case_id"],
                                     reference_index=index, query_index=target, reference_seed=r, query_seed=q,
                                     positive=positive, component=components[index], query_component=components[target],
                                     nonoverlap=None if positive else dict(
                                         frame=manifest["footprint_frame"], full_image_footprints=True,
                                         reference_bounds=bounds[components[index]], query_bounds=bounds[components[target]]),
                                     ))
    return inputs, labels, rows, main, profile, selected_libraries


def build():
    previous = read("runs/solar-parallax-v126/archive/COMPLETE.json")
    previous_replay = read("runs/solar-parallax-v126/archive/independent-replay.json")
    check(previous["complete"] and previous_replay["passed"] and previous_replay["rows"] == 16,
          "The original completed 0/16 experiment and replay must exist")
    inputs, labels, cases, main, profile, libraries = roster()
    sources = parent.build("timing")["source_sha256"]
    for file in ("masked_patch.py", "pipeline.py", "experiment.py", "test_masked_patch.py",
                 "test_pipeline.py", "test_experiment.py"):
        name = str(HOME / file)
        sources[name] = sha(name)
    for name in ("direct_render_adapter_v73.py", "historical_compact_adapters_v73.py",
                 "reports/solar-warp-v125/fast_projection.py"):
        sources[name] = sha(name)
    pins = {str(V117 / "inference-inputs.json"): sha(V117 / "inference-inputs.json"),
            str(V117 / "scoring-inputs.json"): sha(V117 / "scoring-inputs.json"),
            "runs/solar-parallax-v126/archive/COMPLETE.json": sha("runs/solar-parallax-v126/archive/COMPLETE.json"),
            "runs/solar-parallax-v126/archive/independent-replay.json": sha("runs/solar-parallax-v126/archive/independent-replay.json")}
    pins["reports/solar-five-arm-v115/comparison-protocol.json"] = sha("reports/solar-five-arm-v115/comparison-protocol.json")
    for item in inputs:
        for record in (item["reference_geometry"], *item["reference_radiance"], *item["query_radiance"]):
            pins[record["path"]] = record["sha256"]
    for label in labels:
        for key in ("query_geometry", "joint_scoring_geometry"):
            pins[label[key]["path"]] = label[key]["sha256"]
    for entries in libraries.values():
        for entry in entries:
            pins[entry["image"]] = entry["image_sha256"]
    pins[profile["render_evidence"]["manifest"]] = profile["render_evidence"]["manifest_sha256"]
    pins[main["manifest"]] = main["input_sha256"][main["manifest"]]
    for name, digest in pins.items():
        check(sha(name) == digest, "Input bytes changed: " + name)
    return dict(schema="native_measurement_repair_v127", sources=sources, input_sha256=pins, cases=cases,
                inputs=inputs, labels=labels, libraries=libraries, caps=CAP, cpu_threads=2, native_pixels=512,
                calibration_or_final_access=False, accepted_results=False,
                baseline="Published Direct SIFT settings over all24 native reference renders plus the matched-light reference.",
                scope="Exposed known-light development. All16 positives and16 certified nonoverlap pairs.",
                success_rule="At least two more correct positive poses than Direct with no increase in wrong positive or nonoverlap poses.",
                original_requirements_unchanged=True, previous_charge=previous["charge"])


def publish_check(commit, protocol):
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"], check=True)
    for name in (*protocol["sources"], str(HOME / "protocol.json")):
        check(subprocess.check_output(["git", "show", commit + ":" + name]) == Path(name).read_bytes(),
              "Published source differs: " + name)


def score_result(result, case, inputs, labels):
    import numpy as np
    from lunar_geometry_v5 import correspondences_z, median_reference_range_z
    scored = {}
    if case["positive"]:
        record = labels[case["reference_index"]]["joint_scoring_geometry"]
        check(sha(record["path"]) == record["sha256"], "Scoring source changed")
        with np.load(record["path"], allow_pickle=False) as raw:
            fields = [raw[k].copy() for k in ("reference_depth", "query_depth", "reference_intrinsics",
                      "reference_cam2world", "query_intrinsics", "query_cam2world")]
        distance = median_reference_range_z(fields[0], fields[2])
    for name, row in result["methods"].items():
        p, q = (np.asarray(row[k], float).reshape(-1, 2) for k in ("reference_xy", "query_xy"))
        check(p.shape == q.shape and np.isfinite(p).all() and np.isfinite(q).all(), "Invalid saved matches")
        has_pose = row["pose"] is not None
        correct = strict = False
        error = None
        counts = {}
        if case["positive"]:
            _, target, _, visible = correspondences_z(*fields, points=p)
            residual = np.linalg.norm(q[visible] - target, axis=1)
            counts = {str(t): dict(correct=int((residual <= t).sum()), visible=int(visible.sum()), returned=len(p))
                      for t in (1, 2, 4, 8, 12)}
            if has_pose:
                pose = np.asarray(row["pose"])
                relative = pose[:3, :3].T @ fields[-1][:3, :3]
                angle = float(np.degrees(np.arctan2(np.linalg.norm([
                    relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                    relative[1, 0]-relative[0, 1]]), np.trace(relative)-1)))
                shift = float(np.linalg.norm(pose[:3, 3]-fields[-1][:3, 3]))
                correct, strict = angle <= 2. and shift <= .01*distance, angle <= .5 and shift <= .001*distance
                error = dict(rotation_deg=angle, translation_m=shift, reference_range_m=distance)
        scored[name] = dict(correct=correct, strict=strict, pose_returned=has_pose,
                            wrong_positive=case["positive"] and has_pose and not correct,
                            raw_nonoverlap=not case["positive"] and has_pose, correspondence=counts, pose_error=error)
    return scored


def predict(protocol, output):
    import cv2
    import numpy as np
    from PIL import Image
    from replay_raw_acquisition_v109 import decode, luminance
    from pipeline import initialize, run
    cv2.setNumThreads(1)
    def checked_asset(name):
        check(sha(name) == protocol["input_sha256"][name], "Input changed during inference: " + name)
        return Path(name)
    rows = []
    for case in protocol["cases"]:
        started = time.perf_counter()
        item = protocol["inputs"][case["reference_index"]]
        target = protocol["inputs"][case["query_index"]]
        reference = luminance(decode(checked_asset(item["reference_radiance"][case["reference_seed"]]["path"]).read_bytes(), exr_size=512))
        query = luminance(decode(checked_asset(target["query_radiance"][case["query_seed"]]["path"]).read_bytes(), exr_size=512))
        with np.load(checked_asset(item["reference_geometry"]["path"]), allow_pickle=False) as raw:
            depth, k, pose, valid = [raw[key].copy() for key in ("axial_depth", "intrinsics", "cam2world", "valid")]
        query_k = np.asarray(target["query_intrinsics"])
        common_input_seconds = time.perf_counter()-started
        tick = time.perf_counter()
        seeds, initialization = initialize(reference, query, depth, k, pose, query_k)
        init_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        library = []
        for entry in protocol["libraries"][str(case["reference_index"])]:
            with Image.open(checked_asset(entry["image"])) as image:
                library.append(np.asarray(image.convert("L")).copy())
        library_input_seconds = time.perf_counter()-tick
        result = run(reference, query, depth, k, pose, query_k, seeds, library,
                     reference_valid=valid, initialization_seconds=init_seconds)
        for name, method in result["methods"].items():
            method["inference_seconds"] += common_input_seconds + (library_input_seconds if name == "direct" else 0.)
        rows.append(dict(case=case, result=result, initialization=initialization,
                         common_input_seconds=common_input_seconds, library_input_seconds=library_input_seconds,
                         elapsed_seconds=time.perf_counter()-started))
    put(output, dict(protocol_sha256=sha(HOME / "protocol.json"), query_truth_used=False, rows=rows))


def timing(output):
    import cv2
    from pipeline import initialize, run
    from smoke_v126 import generated
    import numpy as np
    cv2.setNumThreads(1)
    samples = []
    for variant in (False, True):
        args, valid, truth = generated(variant)
        reference, depth, k, pose, query, qk, _ = args
        pose = np.diag([1., -1., -1., 1.])
        pose[2, 3] = 2000.
        started = time.perf_counter()
        seeds, _ = initialize(reference, query, depth, k, pose, qk)
        init_seconds = time.perf_counter()-started
        encoded = np.rint(255*np.clip(reference/reference.max(), 0, 1)).astype(np.uint8)
        result = run(reference, query, depth, k, pose, qk, seeds, [encoded]*24,
                     reference_valid=np.ones_like(valid), initialization_seconds=init_seconds)
        samples.append(dict(elapsed_seconds=time.perf_counter()-started,
                            methods={name: row["pose"] is not None for name, row in result["methods"].items()}))
    projected = 1.5*(max(row["elapsed_seconds"] for row in samples)*32+40.)
    put(output, dict(procedural_only=True, archived_images_opened=False, samples=samples,
                     projected_seconds=projected, admitted=projected <= CAP["comparison"]))
    check(projected <= CAP["comparison"], "Full matched comparison exceeds its frozen cap")


def worker(stage, path):
    receipt = read(path)
    protocol = read(HOME / "protocol.json")
    check(receipt["parent_pid"] == os.getppid() and receipt["protocol_sha256"] == sha(HOME / "protocol.json"),
          "Worker must use its admitted controller")
    publish_check(receipt["commit"], protocol)
    _, journal, _ = journal_state(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL)
    phase = "timing" if stage == "timing" else "comparison"
    charge = receipt["charge"]
    check(charge in journal["reservations"] and charge["category"] == "baseline"
          and charge["seconds"] == CAP[phase] and charge["refund_on_failure"] is False
          and charge["job_id"] == receipt["protocol_sha256"] + "/v127-" + phase,
          "Missing original fixed charge")
    check(now() < datetime.fromisoformat(receipt["deadline"]), "Original deadline elapsed")
    if stage == "timing":
        timing(path.parent / "timing.json")
    elif stage == "predict":
        predict(protocol, path.parent / "predictions.json")
    else:
        predictions = path.parent / "predictions.json"
        seal = read(path.parent / "PREDICTIONS_SEALED.json")
        check(sha(predictions) == seal["sha256"], "Scoring requires the sealed predictions")
        raw = read(predictions)
        check([row["case"] for row in raw["rows"]] == protocol["cases"]
              and raw["query_truth_used"] is False
              and raw["protocol_sha256"] == sha(HOME / "protocol.json"), "Incomplete raw grid")
        scores = [dict(case=row["case"], methods=score_result(row["result"], row["case"],
                       protocol["inputs"], protocol["labels"])) for row in raw["rows"]]
        if stage == "score":
            put(path.parent / "scores.json", scores)
        else:
            check(scores == read(path.parent / "scores.json"), "Independent source-label replay disagrees")
            put(path.parent / "replay.json", dict(passed=True, rows=len(scores),
                                                  predictions_sha256=sha(predictions)))


def execute(stage, commit):
    protocol = read(HOME / "protocol.json")
    publish_check(commit, protocol)
    check(protocol == build(), "Prospectively frozen data, parameters, or sources changed")
    if stage == "comparison":
        old = read(WORK / "timing/COMPLETE.json")
        measured = read(WORK / "timing/timing.json")
        check(old["complete"] and measured["admitted"] and old["protocol_sha256"] == sha(HOME / "protocol.json"),
              "Timing must pass before the full comparison")
    check(remaining(read(legacy.CONTRACT))["baseline"] >= CAP[stage], "Original cumulative allowance exhausted")
    output = WORK / stage
    output.mkdir(parents=True, exist_ok=False)
    charge = reserve(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL,
                     sha(HOME / "protocol.json") + "/v127-" + stage, "baseline", CAP[stage])["reservation"]
    started = time.monotonic()
    deadline = datetime.fromisoformat(charge["reserved_utc"]) + timedelta(seconds=CAP[stage])
    allowance = min(CAP[stage], (deadline-now()).total_seconds())
    execution = dict(parent_pid=os.getpid(), protocol_sha256=sha(HOME / "protocol.json"),
                     commit=commit, charge=charge, deadline=deadline.isoformat())
    ticket = output / "execution.json"
    put(ticket, execution)
    try:
        for operation in (("timing",) if stage == "timing" else ("predict", "score", "replay")):
            with (output / (operation + ".log")).open("x") as log:
                child = subprocess.Popen([sys.executable, __file__, "worker", operation, "--ticket", str(ticket)],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, **ENV, "PYTHONPATH": str(ROOT)})
                try:
                    status = child.wait(timeout=max(.01, allowance-(time.monotonic()-started)))
                except BaseException:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise
            check(status == 0 and time.monotonic()-started < allowance and now() < deadline,
                  "Worker failed or original cap elapsed")
            if operation == "predict":
                put(output / "PREDICTIONS_SEALED.json", dict(sha256=sha(output / "predictions.json"), rows=len(protocol["cases"])))
        put(output / "COMPLETE.json", dict(execution, complete=True, elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        put(output / "FAILED.json", dict(execution, complete=False, error=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    check(Path.cwd() == ROOT, "Run from the original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("run")
    run.add_argument("stage", choices=tuple(CAP))
    run.add_argument("--commit", required=True)
    w = sub.add_parser("worker")
    w.add_argument("stage", choices=("timing", "predict", "score", "replay"))
    w.add_argument("--ticket", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        put(HOME / "protocol.json", build())
    elif args.action == "run":
        execute(args.stage, args.commit)
    else:
        worker(args.stage, args.ticket)
