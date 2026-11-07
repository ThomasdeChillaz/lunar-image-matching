"""Fresh six-arm native development execution; query truth stays with the scorer."""

from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"solar-active-v136"))
sys.path.insert(0, str(HERE.parent/"solar-local-v129"))
from matcher import POLICIES, joint, run as active_run
from reference_cache import CachedReferenceSift, FeatureStore, ReferenceProjectionCache

pipeline = joint.native.control.pipeline
photometric = joint.native.control
FRESH_NAMES = ("direct", "joint_no_extra", *POLICIES)


def setup():
    """Identical reference-only caching opportunities; query work stays fresh."""
    cv2.setNumThreads(1)
    CachedReferenceSift.store = FeatureStore()
    pipeline.SiftLibraryMatcher = CachedReferenceSift
    return ReferenceProjectionCache()


def initializer(reference, query, depth, k, pose, query_k, valid, cache):
    tick = perf_counter()
    seeds, coarse = cache.initialize(reference, query, depth, k, pose, query_k)
    hypotheses = photometric.refine(reference, query, depth, k, pose, query_k, seeds,
                                    reference_valid=valid)
    evidence = [dict(h, pose=h["pose"].tolist()) for h in hypotheses]
    return (None if not hypotheses else hypotheses[0]["pose"], evidence,
            dict(seconds=perf_counter()-tick, coarse=coarse,
                 query_truth_used=False, reference_projection_cache=True))


def no_extra(reference, query, depth, k, pose, query_k, initial, valid):
    tick = perf_counter()
    fields = joint.prepare_fields(reference, query, depth, k, pose, query_k, initial,
                                  reference_valid=valid)
    camera, fit = joint.optimize_camera(fields, pose, query_k)
    row = joint.measure(fields, depth, k, pose, query_k, camera=camera)
    return dict(row, conditional_stage_seconds=perf_counter()-tick, camera_fit=fit)


def direct(reference, query, depth, k, pose, query_k, valid, library):
    tick = perf_counter()
    store = CachedReferenceSift.store
    before = (store.hits, store.misses, len(store.miss_seconds))
    a, b, encoding = pipeline.common_images(reference, query, valid)
    row = pipeline.direct(a, b, library, depth, k, pose, query_k)
    row = joint.frame.reestimate(dict(row, inference_seconds=perf_counter()-tick),
                                depth, k, pose, query_k)
    return dict(row, encoding=encoding, reference_feature_cache=dict(
        hits=store.hits-before[0], misses=store.misses-before[1],
        preparation_seconds=sum(store.miss_seconds[before[2]:]), query_features_cached=False))


def exact_core(first, second, description):
    for key in ("reference_xy", "query_xy", "pose", "estimator"):
        if first[key] != second[key]:
            raise ValueError(description+" changed: "+key)


def pair(reference, query, depth, k, pose, query_k, valid, initial, library, rotation):
    """Fixed solver for every arm; counterbalance policy order across noise pairs."""
    policy_order = POLICIES[rotation:]+POLICIES[:rotation]
    tick = perf_counter()
    reference_only_up = pose[:3, :3].T@np.array([0., 0., 1.])
    # All declared renderer cameras use the retained component-local ENU frame.
    # This converts its up direction into the reference-camera point frame.
    before = None
    if rotation % 2 == 0:
        before = direct(reference, query, depth, k, pose, query_k, valid, library)
    baseline = no_extra(reference, query, depth, k, pose, query_k, initial, valid)
    active_tick = perf_counter()
    acquisition = active_run(reference, query, depth, k, pose, query_k, initial,
        reference_valid=valid, height_direction_reference=reference_only_up,
        policies=policy_order, budget=32)
    acquisition_seconds = perf_counter()-active_tick
    exact_core(baseline, acquisition["methods"]["joint_no_extra"], "Separate/internal no-extra control")
    direct_row = before if before is not None else direct(reference, query, depth, k, pose, query_k, valid, library)
    methods = dict(direct=direct_row, joint_no_extra=baseline)
    for name in POLICIES:
        row = acquisition["methods"][name]
        # Empty/unsupported inputs preserve failure in the denominator.
        seconds = row.get("conditional_stage_seconds", acquisition_seconds)
        methods[name] = dict(row, conditional_stage_seconds=seconds)
    return dict(methods=methods, executed_policy_order=list(policy_order),
        direct_order="first" if before is not None else "last",
        measured_execution_seconds=perf_counter()-tick,
        height_direction_reference=reference_only_up.tolist(),
        known_matching_illumination=True, calibrated_acceptance=False,
        independent_landmark_validation=False)


def timing(protocol, output, base):
    from smoke_v126 import generated
    cache = setup(); rows = []
    for variant in (False, True):
        args, valid, truth = generated(variant)
        reference, depth, k, old_pose, query, query_k, _ = args
        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
        expected = pose@np.linalg.inv(old_pose)@truth
        # Distinct generated intensity patterns exercise reference extraction;
        # these are not sunlight simulations or additional terrain scenes.
        scaled = reference/reference.max()
        library = [scaled**gamma for gamma in np.linspace(.7, 1.3, 24)]
        tick = perf_counter()
        initial, hypotheses, initialization = initializer(reference, query, depth, k, pose, query_k,
                                                           np.ones_like(valid), cache)
        base.check(initial is not None, "Generated camera initializer missing")
        result = pair(reference, query, depth, k, pose, query_k, np.ones_like(valid), initial,
                      library, 2 if variant else 0)
        for name, row in result["methods"].items():
            camera = row["pose"]
            base.check(camera is not None and np.linalg.norm(np.array(camera)[:3, 3]-expected[:3, 3]) < 2.,
                       "Full-budget generated pose control failed: "+name)
            if name in POLICIES:
                base.check(row["acquisition"]["attempted_extra_patches"] == 32,
                           "Full32-search pilot cannot be replaced by a smaller smoke")
        elapsed = perf_counter()-tick
        rows.append(dict(multidepth=variant, passed=True, elapsed_seconds=elapsed,
                         initializer=initialization, methods={name:dict(
                             pose_returned=row["pose"] is not None,
                             extra_searches=row.get("acquisition", {}).get("attempted_extra_patches", 0))
                             for name, row in result["methods"].items()}))
        print(dict(multidepth=variant, elapsed_seconds=elapsed, all_six_pass=True, extra_budget=32), flush=True)
    projected = 1.5*32*max(row["elapsed_seconds"] for row in rows)+45.
    base.put(output, dict(procedural_only=True, lunar_images_opened=False, samples=rows,
        full_extra_search_budget=32, fresh_initializer=True, complete_six_arm_pipeline=True,
        projected_seconds=projected, cap_seconds=1800, admitted=projected <= 1800,
        forecast_is_estimate_not_worst_case_guarantee=True,
        watchdog_preserves_original_deadline=True))
    base.check(projected <= 1800, "Complete six-arm forecast exceeds its frozen1800-second cap")


def predict(protocol, output, base):
    from PIL import Image
    from replay_raw_acquisition_v109 import decode, luminance
    cache = setup()
    previous = base.read(protocol["cached_predictions"])
    lookup = {key(row["case"]): row for row in previous["rows"]}
    joint_lookup = {key(row["case"]): row for row in base.read(protocol["joint_pilot_predictions"])["rows"]}
    base.check(previous["query_truth_used"] is False, "Historical predictions must be query-truth free")
    def checked(name):
        base.check(base.sha(name) == protocol["input_sha256"][name], "Input changed: "+name)
        return Path(name)
    rows = []
    for index, case in enumerate(protocol["cases"]):
        case_start = perf_counter()
        tick = perf_counter()
        ref, qry = (protocol["inputs"][case[n]] for n in ("reference_index", "query_index"))
        reference = luminance(decode(checked(ref["reference_radiance"][case["reference_seed"]]["path"]).read_bytes(), exr_size=512))
        query = luminance(decode(checked(qry["query_radiance"][case["query_seed"]]["path"]).read_bytes(), exr_size=512))
        with np.load(checked(ref["reference_geometry"]["path"]), allow_pickle=False) as raw:
            depth, k, pose, valid = [raw[name].copy() for name in ("axial_depth", "intrinsics", "cam2world", "valid")]
        query_k = np.asarray(qry["query_intrinsics"])
        input_seconds = perf_counter()-tick
        initial, hypotheses, initialization = initializer(reference, query, depth, k, pose, query_k, valid, cache)
        historical = lookup[key(case)]
        base.check(hypotheses == historical["result"]["photometric_initialization"],
                   "Fresh photometric initializer differs from frozen control")
        tick = perf_counter(); library = []
        for entry in protocol["libraries"][str(case["reference_index"])]:
            with Image.open(checked(entry["image"])) as image:
                library.append(np.asarray(image.convert("L")).copy())
        library_seconds = perf_counter()-tick
        rotation = (2*case["reference_seed"]+case["query_seed"]) % 4
        result = pair(reference, query, depth, k, pose, query_k, valid, initial, library, rotation)
        exact_core(result["methods"]["direct"], historical["result"]["methods"]["framed_direct"], "Fresh Direct")
        if key(case) in joint_lookup:
            exact_core(result["methods"]["joint_no_extra"], joint_lookup[key(case)]["result"]["methods"]["joint"],
                       "Fresh no-extra versus completed joint pilot")
        result["fresh_initializer_exact"] = result["fresh_direct_exact"] = True
        for name, method in result["methods"].items():
            if name == "direct":
                method["online_seconds"] = method["inference_seconds"]+input_seconds+library_seconds
            else:
                method["online_seconds"] = method["conditional_stage_seconds"]+input_seconds+initialization["seconds"]
                method["initializer_computation_included"] = True
        for name, row in historical["result"]["methods"].items():
            result["methods"]["cached_"+name] = deepcopy(row)
        record = dict(case=case, result=result, fresh_initializer=hypotheses,
                      initialization=initialization, input_seconds=input_seconds,
                      library_input_seconds=library_seconds, case_execution_seconds=perf_counter()-case_start)
        base.put(output.parent/f"prediction-{index:02}.json", record)
        rows.append(record)
        print(dict(case=index, positive=case["positive"], initializer_exact=True, direct_exact=True,
                   fresh_methods=6, extra_budget=32), flush=True)
    base.put(output, dict(protocol_sha256=base.sha("reports/solar-active-v137/protocol.json"),
                          query_truth_used=False, rows=rows))


def key(case):
    return tuple(case[name] for name in ("reference_index", "query_index", "reference_seed", "query_seed", "positive"))
