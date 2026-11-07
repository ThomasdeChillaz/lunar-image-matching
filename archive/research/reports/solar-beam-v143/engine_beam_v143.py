"""Prospective seven-arm comparison; called only by an admitted controller.

The predict phase opens inference inputs only. Query truth and historical saved
outputs are available only to scoring/replay after the full prediction seal.
"""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/"solar-blind-v139"))
import engine_v139 as frozen
from beam_matcher_v143 import POLICIES, active, run as run_beam
from localize_beam import validate_case, REFERENCE_KEYS, REQUIRED_REFERENCE_KEYS
from catalogue_v138 import Catalogue

METHODS = ("direct", "legacy_no_extra", "legacy_active32", "single_uniform128",
           "beam_uniform128", "beam_geometry128", "beam_active128")


def models(catalogue, query, depth, k, pose, qk, valid, cache, rotation, *, control_poses=None):
    active.joint.native.direction = active.joint.safe_direction
    start = perf_counter()
    direct = frozen.direct(catalogue, query, depth, k, pose, qk, valid) if rotation % 2 == 0 else None
    order = POLICIES[rotation:] + POLICIES[:rotation]
    raw = run_beam(catalogue, query, depth, k, pose, qk, reference_valid=valid,
        height_direction_reference=pose[:3, :3].T @ np.array([0., 0., 1.]),
        policies=order, field_budget=128, poses=control_poses, cache=cache, research_output=True)["raw_research"]
    tick = perf_counter()
    hypotheses, legacy_initialization = frozen.initialize(catalogue, query, depth, k, pose, qk,
        reference_valid=valid, cache=cache, poses=control_poses)
    legacy_init_seconds = perf_counter()-tick
    if hypotheses:
        best = hypotheses[0]
        kwargs = dict(reference_valid=valid, height_direction_reference=pose[:3, :3].T @ np.array([0., 0., 1.]),
                      policies=("active",))
        tick = perf_counter()
        no_extra = active.run(catalogue.images[best["light_slot"]], query, depth, k, pose, qk,
            best["pose"], budget=0, **kwargs)["methods"]["joint_no_extra"]
        no_extra["online_seconds"] = legacy_init_seconds+perf_counter()-tick
        old_active = active.run(catalogue.images[best["light_slot"]], query, depth, k, pose, qk,
            best["pose"], budget=32, **kwargs)["methods"]["active"]
        old_active["online_seconds"] = legacy_init_seconds+old_active["conditional_stage_seconds"]
        sun = best["catalogue_sun_world"]
    else:
        null = active.joint.frame.solve([], [], depth, k, pose, qk)
        no_extra, old_active = (dict(null, online_seconds=legacy_init_seconds) for _ in range(2))
        sun = None
    for value in (no_extra, old_active):
        value["estimated_catalogue_sun_proposal"] = sun
    if direct is None:
        direct = frozen.direct(catalogue, query, depth, k, pose, qk, valid)
    direct["online_seconds"] = direct["conditional_stage_seconds"]
    methods = dict(direct=direct, legacy_no_extra=no_extra, legacy_active32=old_active,
        **{p+"128": raw["methods"][p] for p in POLICIES})
    return dict(methods={n: methods[n] for n in METHODS}, initialization=raw["initialization"],
        legacy_initialization=legacy_initialization, common_effective_native_fields=raw["common_effective_native_fields"],
        shared_reference_candidate_capacity=raw["shared_reference_candidate_capacity"],
        new_policy_order=list(order), whole_seven_arm_seconds=perf_counter()-start,
        shared_beam_preparation_seconds=raw["shared_preparation_seconds"], legacy_init_seconds=legacy_init_seconds,
        parameters_added=0, query_truth_used=False, accepted_pose_study=False,
        scope="Four reference catalogue lights / up to eight persistent cameras;128total native fields per new control")


def timing(write):
    samples = []
    for i, layered in enumerate((False, True)):
        args = frozen.generated(layered)
        tick = perf_counter(); result = models(*args[:-1], frozen.setup(), i)
        for name, row in result["methods"].items():
            if row["pose"] is None or np.linalg.norm(np.asarray(row["pose"])[:3, 3]-args[-1][:3, 3]) >= 2.:
                raise ValueError("Full seven-arm generated pose gate failed: " + name)
        for p in POLICIES:
            acquisition = result["methods"][p+"128"]["acquisition"]
            if acquisition["attempted_native_fields"] != result["common_effective_native_fields"]:
                raise ValueError("Full common effective native-field control required")
        sample = dict(layered=layered, elapsed_seconds=perf_counter()-tick, all_seven_pose_controls_pass=True,
            lights=24, cameras=result["initialization"]["coarse"]["pose_count"], maximum_branches=8,
            requested_native_field_cap=128, common_effective_native_fields=result["common_effective_native_fields"])
        samples.append(sample); print(sample, flush=True)
    write("timing.json", dict(samples=samples, comparison_forecast_seconds=1.5*8*max(s["elapsed_seconds"] for s in samples)+45.,
        generated_only=True, physical_sunlight_simulation=False, lunar_images_opened=False))


def predict(protocol, write, checked):
    rows, cache = [], frozen.setup()
    for i, case in enumerate(json.loads(checked(protocol["inference_manifest"]).read_bytes())):
        validate_case(case); tick = perf_counter()
        catalogue = Catalogue(np.array([frozen.decode_clean16(checked(c["image"]["path"])) for c in case["catalogue"]]),
            np.array([c["sun_world"] for c in case["catalogue"]]), tuple(c["light_id"] for c in case["catalogue"]))
        query = frozen.decode_clean16(checked(case["query_image"]["path"]))
        with np.load(checked(case["reference_geometry"]["path"]), allow_pickle=False) as z:
            if not REQUIRED_REFERENCE_KEYS <= set(z.files) <= REFERENCE_KEYS:
                raise ValueError("Reference archive contains unexpected fields")
            depth, k, pose, valid = (z[n].copy() for n in ("axial_depth", "intrinsics", "cam2world", "valid"))
        io_seconds = perf_counter()-tick
        result = models(catalogue, query, depth, k, pose, np.asarray(case["query_intrinsics"]), valid, cache, i % 4)
        for row in result["methods"].values():
            row["online_seconds"] += io_seconds
        rows.append(dict(id=case["id"], result=result, input_decode_seconds=io_seconds))
        print(dict(completed=i+1, total=8, id=case["id"]), flush=True)
    write("predictions.json", dict(rows=rows, query_truth_used=False, accepted_results=False))


def score(protocol, write, checked, read):
    frozen.score(protocol, write, checked, read)


def replay(protocol, write, checked, read):
    from replay_beam_v143 import verify
    result = verify(protocol, checked, read)
    write("replay.json", result)
