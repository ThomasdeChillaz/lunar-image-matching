"""Source-gated, two-thread development evaluation under an external watchdog.

Modes: a predeclared cost pilot, all five pose-selected checkpoints, the exact
eight A/B/C ablations plus pooled control, and additional/modern controls.
No automatic final inference, confidence fitting, training or model promotion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
from datetime import datetime, timezone, timedelta
import numpy as np
from solar_evaluation_v70 import (Backend, InferenceInput, TRACKS, aggregate_rows, checkpoint_ranking,
    digest, evaluate_case, paired_component_bootstrap, read, sha, validate_manifest, validate_profiles, validate_profile_training)

REQUIRED_SOURCES = {
    "solar_evaluation_v70.py", "run_solar_evaluation_v70.py", "verify_solar_evaluation_v70.py", "localize_terrain.py",
    "solar_frontend_v63.py", "solar_assignment_v63.py", "solar_native_v63.py",
    "solar_response_v63.py", "solar_physics_v63.py", "solar_active_v63.py",
    "solar_compute_budget_v67.py", "native_baseline_pose_v7.py",
    "localization_robust_pose_v4.py", "localization_pose_refit_v3.py",
    "lunar_geometry_v2.py", "lunar_geometry_v5.py", "compact_lunar.py",
    "contrastive_terrain.py", "rotation_bank_pose_v4.py", "native_crop_alignment_v57.py",
    "solar_rotation_v71.py", "solar_sun_search_v68.py", "solar_lookup_v70.py",
    "solar_lookup_worker_v70.py", "solar_map_v63.py", "north_terrain_pilot_v32.py",
    "north_render_plan_v31.py", "dataset.py", "matchers.py", "geometry.py",
    "localization_diagnostics_v3.py", "lunargeo_dataset.py", "remote_archive.py",
    "train_balanced_localization_v3.py", "localization_refiner_v3.py",
    "prepare_localization_training_v3.py",
    "solar_pose_proposals_v73.py",
    "solar_inference_cache_v84.py", "solar_response_cache_v84.py", "solar_acquisition_cache_v84.py",
    "solar_evaluation_protocol_v74.py", "build_solar_evaluation_protocol_v74.py",
    "solar_learning_pipeline_v73.py", "verify_solar_development_v68.py",
    'north_camera_contract_v59.py',
    'north_camera_variation_v59.py',
    'north_camera_visibility_v59.py',
    'north_development_plan_v35.py',
    'renderer_depth_v34.py',
    'solar_completion_v68.py',
    'solar_data_v65.py',
    'verify_north_camera_qualified_v64.py',
    'verify_north_development_v35.py',
}


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


MEMORY_SCOPE = ("Current worker process LIFETIME high-water RSS up to this snapshot, including runtime, models, data and allocator/cache history. "
    "Excludes child processes; not current RSS, isolated model/case RAM, or newly allocated memory. "
    "Deleting a backend does not reset this high-water mark; shared-profile values cannot rank model memory.")
TIMING_SCOPE = dict(
    inference_seconds="Model-resident Backend.infer, including its pose/online physics; independent pose replay and scoring are separate.",
    complete_localization_cold_seconds="Model-resident inference plus input/reference setup ACTUALLY incurred by the reused loader; not a forced cold setup per case. Model startup, offline preparation and cached history are separate.",
    profile_load_seconds="Profile construction/validation measured once and shared across its cases/tracks; do not add to every row.",
    cached_rows="Pinned historical inference/setup/startup timings; current cache-copy time is not fresh model latency.",
    offline_reference_preparation="Measured startup-inclusive offline preparation and its fixed-study amortization are reported separately.",
    raw_inference_reporting="Single-call observer bookkeeping is inside inference_seconds; raw trace JSON/NPZ packing is outside inference_seconds and inside case_calendar_seconds.")


def worker_peak_rss_snapshot():
    """Cheap stdlib-only self-process snapshot; telemetry errors never alter inference."""
    import math
    result = dict(available=False, worker_pid=os.getpid(), platform=sys.platform,
        metric="process_lifetime_peak_rss", peak_rss_bytes=None, scope=MEMORY_SCOPE,
        child_processes_included=False, newly_allocated_memory=False)
    if sys.platform not in ("darwin", "linux"):
        return dict(result, unavailable_reason="unsupported_platform", error=None)
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
            raise ValueError("Finite nonnegative ru_maxrss required")
        scale = 1 if sys.platform == "darwin" else 1024
        return dict(result, available=True, peak_rss_bytes=int(raw*scale), native_maxrss=raw,
            native_maxrss_unit="bytes" if sys.platform == "darwin" else "KiB", unavailable_reason=None, error=None)
    except Exception as error:
        return dict(result, unavailable_reason="resource_snapshot_error",
            error=dict(type=type(error).__name__, message=str(error)))


def worker_memory_scope(profiles):
    return dict(worker_pid=os.getpid(), profile_order=[profile["name"] for profile in profiles],
        job_scope="standalone_profile_worker_process" if len(profiles) == 1 else "shared_multi_profile_worker_process",
        scope=MEMORY_SCOPE, child_processes_included=False, per_case_memory_measured=False,
        isolated_model_memory_measured=False, per_model_memory_ranking_supported=False)


RAW_ARRAY_PREFIX = "_raw_inference_"
RAW_MAX_ARRAY_BYTES = 16*1024*1024
RAW_MAX_JSON_BYTES = 512*1024
RAW_MAX_NODES = 100000
_NO_INFERENCE_RESULT = object()


class InferenceResultCapture:
    """One delegate call, unchanged result; owned/deleted as the profile backend."""
    def __init__(self, backend):
        self._backend = backend; self._result = _NO_INFERENCE_RESULT; self._calls = 0
    def __getattr__(self, name):
        return getattr(self._backend, name)
    def infer(self, inp):
        self._result = _NO_INFERENCE_RESULT; self._calls += 1
        result = self._backend.infer(inp)
        self._result = result
        return result
    def release(self):
        result, calls = self._result, self._calls
        self._result = _NO_INFERENCE_RESULT; self._calls = 0
        return result, calls


def pack_raw_inference(result, arrays):
    """Exact numeric arrays go to existing NPZ; JSON contains typed pointers.

    No model/provenance/tensor objects or reprs are retained. Unsupported values
    are explicitly unavailable; bounds fail instead of silently truncating.
    Array insertion is transactional, with no object arrays or pickle.
    """
    import math
    if any(name.startswith(RAW_ARRAY_PREFIX) for name in arrays):
        raise ValueError("Reserved raw-inference NPZ namespace collision")
    packed = {}; unsupported = []; nonfinite = []; nodes = 0; total_bytes = 0
    def array_value(value, kind="ndarray", container=None):
        nonlocal total_bytes
        if value.dtype.kind not in "biufc" or value.dtype.hasobject:
            return None
        total_bytes += value.nbytes
        if total_bytes > RAW_MAX_ARRAY_BYTES:
            raise ValueError("Raw inference trace exceeds predeclared numeric-byte bound")
        name = RAW_ARRAY_PREFIX + str(len(packed)).zfill(5)
        packed[name] = value.copy()
        count = int((~np.isfinite(value)).sum()) if value.dtype.kind in "fc" else 0
        out = {"$raw":kind, "npz_key":name, "dtype":value.dtype.str,
            "shape":list(value.shape), "nonfinite_count":count}
        if container is not None: out["container"] = container
        return out
    def encode(value, path="$", depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > RAW_MAX_NODES or depth > 32:
            raise ValueError("Raw inference trace exceeds predeclared structure bound")
        if isinstance(value, np.ndarray):
            encoded = array_value(value)
            if encoded is not None: return encoded
        elif isinstance(value, np.generic):
            encoded = array_value(np.asarray(value), "numpy_scalar")
            if encoded is not None: return encoded
        elif value is None or type(value) in (str, bool, int):
            return value
        elif type(value) is float:
            if math.isfinite(value): return value
            label = "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
            nonfinite.append(path)
            return {"$raw":"nonfinite_float", "value":label}
        elif type(value) in (list, tuple):
            if value and all(isinstance(v, np.ndarray) and v.dtype.kind in "biufc" and not v.dtype.hasobject
                    and v.shape == value[0].shape and v.dtype == value[0].dtype for v in value):
                # In particular, eight256-anchor assignment arrays become ONE
                # typed NPZ array per update, preserving element dtype exactly.
                if total_bytes + sum(v.nbytes for v in value) > RAW_MAX_ARRAY_BYTES:
                    raise ValueError("Raw inference trace exceeds predeclared numeric-byte bound")
                stacked = np.stack(value, dtype=value[0].dtype, casting="no")
                return array_value(stacked, "ndarray_sequence", "tuple" if type(value) is tuple else "list")
            items = [encode(v, path+"["+str(i)+"]", depth+1) for i,v in enumerate(value)]
            return items if type(value) is list else {"$raw":"tuple", "items":items}
        elif type(value) is dict and all(type(k) is str for k in value):
            items = {key:encode(v,path+"."+key,depth+1) for key,v in value.items()}
            return items if "$raw" not in value else {"$raw":"escaped_mapping", "items":items}
        unsupported.append(dict(path=path, python_type=type(value).__module__+"."+type(value).__qualname__))
        return {"$raw":"unsupported", "python_type":unsupported[-1]["python_type"], "value_preserved":False}
    encoded = encode(result)
    size = len(json.dumps(encoded, separators=(",",":"), allow_nan=False).encode("utf-8"))
    if size > RAW_MAX_JSON_BYTES:
        raise ValueError("Raw inference trace exceeds predeclared JSON-byte bound")
    arrays.update(packed)
    return encoded, dict(format="raw-inference-npz-v82", encoding_complete=not unsupported,
        unsupported_values=unsupported, nonfinite_scalar_paths=nonfinite,
        serialized_array_count=len(packed), serialized_array_bytes=total_bytes,
        compact_json_bytes=size, structure_nodes=nodes,
        effective_C_policy="unrecorded_in_frozen_solver", C_fallback_reason="unrecorded_in_frozen_solver")


def capture_case_report(backend, arrays):
    tick = time.perf_counter(); result, calls = backend.release()
    if calls != 1: raise ValueError("Case reporter requires exactly one original backend inference call")
    if result is _NO_INFERENCE_RESULT:
        encoded = None; info = dict(format="raw-inference-npz-v82", encoding_complete=False,
            capture_status="backend_did_not_return", effective_C_policy="unrecorded_in_frozen_solver",
            C_fallback_reason="unrecorded_in_frozen_solver")
    else:
        encoded, info = pack_raw_inference(result, arrays); info["capture_status"] = "returned_result"
    info.update(backend_infer_calls=calls, serialization_seconds=time.perf_counter()-tick,
        scope="Observer bookkeeping is inside inference_seconds; JSON/NPZ packing is outside it and inside case_calendar_seconds. Serialized byte counts are not model-memory measurements.")
    return encoded, info


def software_identity():
    import torch, cv2, platform
    return dict(torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(), cpu_threads=2)


def admission(protocol_path, commit):
    p = read(protocol_path)
    if p.get("prepared_only", False): raise ValueError("Prepared template cannot start scientific inference")
    if p.get("builder_schema_version") != 74:
        raise ValueError("New scientific evaluation requires an artifact-driven schema74 freeze")
    from solar_evaluation_protocol_v74 import validate_runtime
    validate_runtime(p, protocol_path)
    if p["schema_version"] != 70 or p["split"] != "model_selection" or p["calibration_or_final_access"] or p["cpu_threads"] != 2:
        raise ValueError("Frozen development-only two-thread protocol required")
    if p["tracks"] not in (["estimated_sun"], list(TRACKS)):
        raise ValueError("Primary estimated-Sun track required; secondary explicitly separated")
    if type(p["max_calendar_seconds"]) is not int or p["max_calendar_seconds"] <= 0:
        raise ValueError("Original finite wall allowance required")
    validate_profiles(p["profiles"], p["purpose"])
    for profile in p["profiles"]: validate_profile_training(profile, p["purpose"])
    if p["purpose"] == "checkpoint_selection" and p["tracks"] != ["estimated_sun"]:
        raise ValueError("Checkpoint selection uses primary unfiltered poses only")
    if not REQUIRED_SOURCES <= set(p["source_sha256"]):
        raise ValueError("Complete executed inference/geometry/watchdog source pins required")
    if any(profile["backend"] == "modern" for profile in p["profiles"]):
        modern_sources = {"modern_baselines_v2.py", "dataset.py", "geometry.py", "matchers.py"}
        if not modern_sources <= set(p["source_sha256"]):
            raise ValueError("Modern adapter and coordinate/weight-lock source pins required")
    for profile in p["profiles"]:
        if profile["backend"] == "native_pretrained":
            from native_pretrained_adapters_v73 import validate_native_profile, required_native_sources, validate_external_comparison
            gate = validate_native_profile(profile)
            validate_external_comparison(profile, p)
            actual_protocol = read(profile["pretrained_evidence"]["protocol"])
            if not required_native_sources(profile, actual_protocol) <= set(p["source_sha256"]):
                raise ValueError("Complete native pretrained adapter/control/source/weight-lock pins required")
            evidence = profile["pretrained_evidence"]
            for field in ("protocol", "summary", "controls", "provenance"):
                if p["input_sha256"].get(evidence[field]) != evidence[field+"_sha256"]:
                    raise ValueError("Actual pretrained gate evidence must be bound to the published evaluation protocol")
    for profile in p["profiles"]:
        if profile["backend"] == "historical_compact":
            from historical_compact_adapters_v73 import validate_historical_profile, historical_profile_inputs
            validate_historical_profile(profile)
            required_inputs = historical_profile_inputs(profile)
        elif profile["backend"] == "direct_render":
            from direct_render_adapter_v73 import validate_render_profile, render_profile_inputs
            validate_render_profile(profile)
            required_inputs = render_profile_inputs(profile)
        else:
            continue
        for name, expected in profile["source_sha256"].items():
            if p["source_sha256"].get(name) != expected:
                raise ValueError("Historical/render source must be bound to comparison freeze: " + name)
        for name, expected in required_inputs.items():
            if p["input_sha256"].get(name) != expected:
                raise ValueError("Actual historical/render evidence must be pinned: " + name)
    for path in [protocol_path, *p["source_sha256"]]:
        if subprocess.check_output(["git", "show", commit+":"+path]) != Path(path).read_bytes():
            raise ValueError("Executed protocol/source not identical to its source publication: "+path)
    for path, h in p["source_sha256"].items():
        if sha(path) != h: raise ValueError("Source pin mismatch: "+path)
    for path, h in p["input_sha256"].items():
        if sha(path) != h: raise ValueError("Input pin mismatch: "+path)
    for path, h in p.get("case_cache_seals", {}).items():
        if sha(path) != h: raise ValueError("Pinned completed case-cache seal changed")
    if p["manifest"] not in p["input_sha256"] or p["geometry_audit"] not in p["input_sha256"]:
        raise ValueError("Manifest and complete geometry audit must be pinned")
    m = read(p["manifest"]); audit = read(p["geometry_audit"])
    if m["geometry_audit_sha256"] != sha(p["geometry_audit"]):
        raise ValueError("Manifest was not derived from this full geometry audit")
    cases = validate_manifest(m, audit)
    for geometry in {c["reference_geometry"] for c in cases}:
        r = m["reference_precompute_receipts"][geometry]
        if sha(r["receipt_path"]) != r["sha256"]:
            raise ValueError("Reference-map preparation timing/provenance receipt changed")
        receipt = read(r["receipt_path"])
        if not receipt["complete"] or not np.isfinite(receipt["calendar_seconds"]) or receipt["calendar_seconds"] < 0:
            raise ValueError("Measured complete reference-map preparation receipt required")
    if p["case_ids"] != [c["case_id"] for c in cases]:
        raise ValueError("Full ordered eligible positive/negative case list must be frozen")
    for c in cases:
        required = {c[n] for n in ("reference", "query", "reference_geometry")}
        if c.get("solar_lookup_protocol"):
            required.add(c["solar_lookup_protocol"])
            if "solar_lookup_v70.py" not in p["source_sha256"]:
                raise ValueError("Online reference-only solar lookup implementation must be source-pinned")
        if c["positive"]: required.add(c["scoring_labels"])
        if any(profile["backend"] in ("modern", "sift", "native_pretrained", "historical_compact", "direct_render") for profile in p["profiles"]) and c["positive"]:
            required.add(c["scoring_geometry"])
        if not required <= set(c["input_sha256"]):
            raise ValueError("Every actual image, reference map and scoring label must be pinned")
    for profile in p["profiles"]:
        if profile["backend"] == "solar":
            if sha(profile["checkpoint"]) != profile["configuration"]["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint mismatch")
        if profile["backend"] == "modern":
            controls = read(profile["pretrained_control"])
            if profile["pretrained_control"] not in p["input_sha256"] or not controls["passed"]:
                raise ValueError("Actual pinned native pretrained control must pass before lunar inference")
            provenance_path = profile["pretrained_provenance"]
            if provenance_path not in p["input_sha256"]:
                raise ValueError("Actual pretrained control provenance must be pinned")
            provenance = read(provenance_path)
            if provenance["method"] != profile["method"] or provenance["profile"] != profile["author_profile"]:
                raise ValueError("Pretrained control method/settings mismatch")
    if p["purpose"] == "cost_pilot":
        if p.get("case_cache_seals"): raise ValueError("Measured pilot must perform fresh inference, without replay caches")
        chosen = p["pilot_case_ids"]
        if len(chosen) < 4 or len(chosen) != len(set(chosen)) or not set(chosen) <= set(p["case_ids"]):
            raise ValueError("At least four explicit distinct pilot cases required")
        by = {c["case_id"]: c for c in cases}; pilot = [by[i] for i in chosen]
        if len({c["component"] for c in pilot}) < 2 or not any(c["positive"] for c in pilot) or not any(not c["positive"] for c in pilot):
            raise ValueError("Cost pilot must include two terrains, positives and nonoverlap")
        cases = pilot
    else:
        pilot = read(p["cost_pilot_summary"])
        if sha(p["cost_pilot_summary"]) != p["cost_pilot_summary_sha256"] or not pilot["complete"] or pilot["purpose"] != "cost_pilot":
            raise ValueError("Completed measured pilot, not a hand-written cost boolean, required")
        pilot_root = Path(p["cost_pilot_summary"]).parent
        seal = read(pilot_root / "COMPLETE.json")
        if (pilot_root / "FAILED.json").exists() or seal["summary_sha256"] != sha(p["cost_pilot_summary"]) or seal["protocol_sha256"] != pilot["protocol_sha256"]:
            raise ValueError("Pilot must have passed its external watchdog and completion seal")
        if pilot["software"] != software_identity() or pilot["comparison_identity"] != comparison_identity(p, m):
            raise ValueError("Pilot must use identical software, profiles, full case manifest and sources")
        cost = measured_cost(pilot["rows"], len(cases), p["tracks"], p["profiles"])
        if cost["conservative_total_seconds"] > p["max_calendar_seconds"]:
            raise ValueError("Measured full case/profile cost exceeds original admitted allowance")
    return p, m, cases


def comparison_identity(p, manifest):
    return digest(dict(profiles=p["profiles"], manifest_sha256=digest(manifest), tracks=p["tracks"],
        source_sha256=p["source_sha256"], cpu_threads=2))


def measured_cost(rows, cases, tracks, profiles):
    total = 0.; details = []
    for p in profiles:
        for track in tracks:
            r = [r for r in rows if r["profile"] == p["name"] and r["track"] == track]
            if len(r) < 4 or any(x["inference_failure"] is not None for x in r):
                raise ValueError("At least four successful measured pilot cases per profile/track required")
            # Worst observed complete inference+label+I/O and 50% margin. This
            # is an admission estimate, never an assertion of constant cost.
            peak = max(x["case_calendar_seconds"] for x in r)
            load = max(x["profile_load_seconds"] for x in r)
            estimate = 1.5 * (load + peak*cases)
            details.append(dict(profile=p["name"], track=track, cases=cases, estimate_seconds=estimate)); total += estimate
    return dict(conservative_total_seconds=int(np.ceil(total)), profile_tracks=details,
        rule="1.5 * (measured model load + worst measured case calendar time * all eligible cases)")


def cache_index(protocol):
    result = {}
    for path in sorted(protocol.get("case_cache_seals", {})):
        seal = read(path)
        if not seal["complete"] or sha(seal["row_path"]) != seal["row_sha256"] or sha(seal["arrays_path"]) != seal["arrays_sha256"]:
            raise ValueError("Completed case cache payload changed")
        row = read(seal["row_path"])
        if row["saved_arrays_sha256"] != seal["arrays_sha256"] or row["saved_arrays"] != seal["arrays_path"]:
            raise ValueError("Case-cache metadata and prediction arrays disagree")
        result.setdefault(seal["identity"], row)
    return result


class InputLoader:
    def __init__(self, publication_commit=None, blender_executable=None, fully_materialized_primary=False):
        self.image_key = self.geometry_key = None
        self.images = self.geometry = None
        self.last_setup = {}
        self.lookup = None; self.publication_commit = publication_commit; self.blender_executable = blender_executable
        self.fully_materialized_primary = fully_materialized_primary
    def close(self):
        if self.lookup is not None: self.lookup.close(); self.lookup = None
    def get(self, case, track):
        if self.fully_materialized_primary and track != "estimated_sun":
            raise ValueError("Portable remote archives support the primary estimated-Sun track only")
        # One current pair only. Dense ray tables and image arrays are not
        # retained for all cameras or mixed with scoring-only query geometry.
        self.last_setup = dict(image_decode_seconds=0., reference_geometry_load_seconds=0., reference_physics_setup_seconds=0.)
        key = (case["reference"], case["query"])
        if key != self.image_key:
            from PIL import Image
            tick = time.perf_counter()
            images = []
            for name in ("reference", "query"):
                with Image.open(case[name]) as im: image = np.asarray(im.convert("L")).copy()
                if image.shape != (512, 512) or image.dtype != np.uint8: raise ValueError("Native 512-pixel source content required")
                images.append(image)
            self.images = images; self.image_key = key
            self.last_setup["image_decode_seconds"] = time.perf_counter()-tick
        geometry_key = (case["reference_geometry"], case.get("solar_lookup_protocol"))
        if geometry_key != self.geometry_key:
            from solar_frontend_v63 import ReferenceGeometry
            self.close()
            tick = time.perf_counter(); self.geometry = ReferenceGeometry.load(case["reference_geometry"])
            self.geometry_key = geometry_key
            self.last_setup["reference_geometry_load_seconds"] = time.perf_counter()-tick
            if self.fully_materialized_primary:
                if track != "estimated_sun" or not self.geometry.solar_materialized_mask.all():
                    raise ValueError("Remote primary requires all physical columns; no Linux Blender fallback")
            elif case.get("solar_lookup_protocol"):
                from solar_lookup_v70 import ReferenceSunLookup
                tick = time.perf_counter()
                self.lookup = ReferenceSunLookup.from_protocol(case["solar_lookup_protocol"], self.publication_commit, self.blender_executable)
                self.last_setup["reference_physics_setup_seconds"] = time.perf_counter()-tick
        return InferenceInput(*self.images, self.geometry, np.asarray(case["query_intrinsics"], float),
            np.asarray(case["reference_sun_world"], float),
            None if track == "estimated_sun" else np.asarray(case["secondary_known_query_sun_world"], float), self.lookup)


def worker(protocol_path, commit, pre_reserved_ticket=None):
    if pre_reserved_ticket is not None:
        from solar_remote_relay_v75 import validate_worker_ticket, primary_scope
        validate_worker_ticket(pre_reserved_ticket, protocol_path)
        primary_scope(protocol_path, read(protocol_path))
    import torch, cv2
    torch.set_num_threads(2); cv2.setNumThreads(1)
    p, m, cases = admission(protocol_path, commit); root = Path(p["output"]); rows = []
    watchdog = read(root / "watchdog_admission.json"); charged = read(root / "reservation.json")
    if watchdog["protocol_sha256"] != sha(protocol_path) or watchdog["reservation_sha256"] != sha(root / "reservation.json") or charged["reservation"]["job_id"] != sha(protocol_path)+"/evaluation" or charged["reservation"]["seconds"] != p["max_calendar_seconds"]:
        raise ValueError("Worker requires the original admitted controller reservation/deadline")
    start = watchdog["start_monotonic"]; cap = p["max_calendar_seconds"]; deadline = datetime.fromisoformat(watchdog["absolute_deadline_utc"])
    def check():
        if time.monotonic()-start >= cap or datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Original evaluation allowance exhausted")
    check()
    memory_scope = worker_memory_scope(p["profiles"]); profile_memory = []; current_memory = None
    write_new(root / "execution.json", dict(protocol_sha256=sha(protocol_path), publication_commit=commit, software=software_identity(),
        memory_measurement=memory_scope, worker_peak_rss=worker_peak_rss_snapshot(), timing_scope=TIMING_SCOPE,
        purpose=p["purpose"], raw_uncalibrated=True, final_reserve_access=False, profile_order=[v["name"] for v in p["profiles"]],
        initialization_only_cost_controls=[v["name"] for v in p["profiles"] if v.get("initialization_only_cost_control",False)]))
    loader = InputLoader(commit, p.get("blender_executable"), fully_materialized_primary=bool(pre_reserved_ticket)); reusable = cache_index(p)
    try:
        for profile_index, profile in enumerate(p["profiles"]):
            check(); backend = None; load_seconds = 0.
            profile_rows_start = len(rows)
            current_memory = dict(profile=profile["name"], profile_index=profile_index, worker_pid=os.getpid(),
                before=worker_peak_rss_snapshot(), after=None, completed=False)
            profile_memory.append(current_memory)
            for track in p["tracks"]:
                for case in cases:
                    check(); before = time.monotonic(); model_load_this_case = 0.
                    for path, h in case["input_sha256"].items():
                        if sha(path) != h: raise ValueError("Frozen inference/scoring asset changed: "+path)
                    key = digest(dict(case=case, profile=profile, track=track, sources=p["source_sha256"]))
                    array_path = root / (key+".npz")
                    cached = reusable.get(key)
                    if cached is None:
                        if backend is None:
                            load_tick = time.monotonic(); backend = InferenceResultCapture(Backend(profile)); load_seconds = time.monotonic()-load_tick
                            model_load_this_case = load_seconds
                            write_new(root / (digest(profile)+".provenance.json"), backend.provenance)
                        inp = loader.get(case, track)
                        row, arrays = evaluate_case(backend, inp, case)
                        row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(backend, arrays)
                        row["reference_setup"] = loader.last_setup.copy()
                        row["complete_localization_cold_seconds"] = row["inference_seconds"] + sum(loader.last_setup.values())
                        row["timing_scope"] = TIMING_SCOPE["complete_localization_cold_seconds"]
                        with array_path.open("xb") as f: np.savez_compressed(f, **arrays)
                        row["reused_case_cache"] = False
                        row["profile_load_seconds"] = load_seconds
                    else:
                        row = dict(cached, reused_case_cache=True, original_case_calendar_seconds=cached["case_calendar_seconds"],
                            timing_scope="Inference and setup latency replayed from this exact pinned cached execution; current cache-copy time is not model latency")
                        with array_path.open("xb") as destination, Path(cached["saved_arrays"]).open("rb") as source: shutil.copyfileobj(source, destination)
                    row["raw_inference_execution_scope"] = ("Historical pinned prediction/trace from case cache; no fresh backend inference or module-use observation."
                        if row["reused_case_cache"] else "Fresh single original backend inference; requested flags alone do not prove module contribution.")
                    row["memory_measurement_scope"] = ("Cached execution memory fields, if present, are historical; this row has no fresh case/model RAM attribution. Current profile snapshots include all fresh and cached activity in this run."
                        if row["reused_case_cache"] else "No case/model RAM measured; current worker lifetime peaks are recorded before/after the profile.")
                    row.update(case_id=case["case_id"], component=case["component"], query_component=case["query_component"],
                        positive=case["positive"], profile=profile["name"], track=track,
                        initialization_only_cost_control=bool(profile.get("initialization_only_cost_control",False)),
                        unit_index=case["unit_index"], query_unit_index=case["query_unit_index"], light_a=case["light_a"], light_b=case["light_b"],
                        saved_arrays=str(array_path), saved_arrays_sha256=sha(array_path),
                        case_total_calendar_seconds=time.monotonic()-before,
                        case_calendar_seconds=time.monotonic()-before-model_load_this_case)
                    row_path = root / (key+".json"); write_new(row_path, row)
                    write_new(root / (key+".COMPLETE.json"), dict(complete=True, identity=key, row_path=str(row_path), row_sha256=sha(row_path),
                        arrays_path=str(array_path), arrays_sha256=sha(array_path), source_protocol_sha256=sha(protocol_path)))
                    rows.append(row); print(json.dumps(row), flush=True); check()
            del backend
            current_memory.update(after=worker_peak_rss_snapshot(), completed=True,
                fresh_inference_cases=sum(not row["reused_case_cache"] for row in rows[profile_rows_start:]),
                cached_cases=sum(row["reused_case_cache"] for row in rows[profile_rows_start:]))
        check()
        expected = len(cases)*len(p["profiles"])*len(p["tracks"])
        if len(rows) != expected: raise ValueError("Full matched case/profile/track denominator incomplete")
        summary = dict(complete=True, purpose=p["purpose"], rows=rows, software=software_identity(), comparison_identity=comparison_identity(p, m),
            memory_measurement=dict(memory_scope, profile_snapshots=profile_memory), timing_scope=TIMING_SCOPE,
            protocol_sha256=sha(protocol_path), manifest_sha256=sha(p["manifest"]), source_publication_commit=commit,
            calendar_seconds=time.monotonic()-start, profiles=p["profiles"], case_ids=[c["case_id"] for c in cases],
            fresh_inference_cases=sum(not r["reused_case_cache"] for r in rows), cached_cases=sum(r["reused_case_cache"] for r in rows),
            accepted_results=False, calibration_or_final_access=False, automatic_model_promotion=False)
        summary["scientific_model_evidence"] = not any(v.get("initialization_only_cost_control",False) for v in p["profiles"])
        summary["offline_reference_preparation"] = []
        for geometry in sorted({c["reference_geometry"] for c in cases}):
            receipt = m["reference_precompute_receipts"][geometry]
            count = sum(c["reference_geometry"] == geometry for c in cases)
            internal_seconds = read(receipt["receipt_path"])["calendar_seconds"]
            seconds = p["reference_preparation_costs"]["outer_calendar_seconds_by_archive"][geometry]
            summary["offline_reference_preparation"].append(dict(reference_geometry=geometry, **receipt,
                measured_calendar_seconds=seconds, internal_sidecar_calendar_seconds=internal_seconds,
                startup_inclusive_outer_calendar_seconds=seconds, queries_per_profile_per_track=count,
                fixed_study_amortized_seconds_per_query=seconds/count,
                note="Offline geometry/physical-table preparation includes external cold startup; internal sidecar time is reported separately. Fixed-study reuse does not specify a deployment query frequency."))
        summary["raw_results"] = aggregate_rows(rows)
        if p["purpose"] == "principal_ablations":
            summary["paired_terrain_intervals"] = {track: paired_component_bootstrap(rows, "A1B1C1", "A0B0C0", track) for track in p["tracks"]}
        if p["purpose"] == "checkpoint_selection": summary["selection"] = checkpoint_ranking(rows, p["profiles"], [c["case_id"] for c in cases])
        if p["purpose"] == "cost_pilot": summary["measured_full_cost"] = measured_cost(rows, len(m["cases"]), p["tracks"], p["profiles"])
        summary["worker_peak_rss"] = worker_peak_rss_snapshot()
        write_new(root / "summary.json", summary)
    except BaseException as e:
        if current_memory is not None and current_memory["after"] is None:
            current_memory.update(after=worker_peak_rss_snapshot(), completed=False,
                fresh_inference_cases=sum(not row["reused_case_cache"] for row in rows[profile_rows_start:]),
                cached_cases=sum(row["reused_case_cache"] for row in rows[profile_rows_start:]))
        write_new(root / "WORKER_FAILED.json", dict(error_type=type(e).__name__, error=str(e), completed_rows=len(rows), calendar_seconds=time.monotonic()-start,
            memory_measurement=dict(memory_scope, profile_snapshots=profile_memory), timing_scope=TIMING_SCOPE)); raise
    finally:
        loader.close()


def run(protocol_path, commit, python, pre_reserved_ticket=None):
    # The controller is independent of the scientific worker. Even an
    # uninterruptible long C/torch call cannot silently reset its wall limit.
    p, _, _ = admission(protocol_path, commit); root = Path(p["output"])
    if root.exists(): raise FileExistsError("Preserve every completed/partial run; no reset or automatic restart")
    from solar_evaluation_protocol_v74 import validate_runtime
    validate_runtime(p, protocol_path) # Recheck after source/asset admission work.
    if pre_reserved_ticket is None:
        from solar_compute_budget_v67 import reserve
        reservation = reserve(p["compute_ledger"], p["compute_ledger_sha256"], p["reservation_journal"], sha(protocol_path)+"/evaluation", "baseline", p["max_calendar_seconds"])
    else:
        from solar_remote_relay_v75 import consume_pre_reservation, primary_scope
        primary_scope(protocol_path, p)
        reservation = consume_pre_reservation(pre_reserved_ticket, protocol_path)
    root.mkdir(parents=True)
    write_new(root / "reservation.json", reservation)
    child = None; started = time.monotonic(); pending = []
    now = datetime.now(timezone.utc)
    original_deadline = datetime.fromisoformat(p["absolute_deadline_utc"])
    deadline = min(now+timedelta(seconds=p["max_calendar_seconds"]), original_deadline)
    def expired():
        return time.monotonic()-started >= p["max_calendar_seconds"] or datetime.now(timezone.utc) >= deadline
    write_new(root / "watchdog_admission.json", dict(protocol_sha256=sha(protocol_path), reservation_sha256=sha(root / "reservation.json"),
        start_monotonic=started, started_utc=now.isoformat(), absolute_deadline_utc=deadline.isoformat(), original_protocol_absolute_deadline_utc=p["absolute_deadline_utc"], controller_pid=os.getpid()))
    def interrupted(signum, frame): pending.append(signum)
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (root / "worker.log").open("x") as log:
            command = [python, str(Path(__file__)), "--protocol", protocol_path, "--publication-commit", commit, "--worker"]
            if pre_reserved_ticket is not None: command += ["--pre-reserved-ticket", pre_reserved_ticket]
            child = subprocess.Popen(command,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2"))
            while child.poll() is None:
                if pending: raise InterruptedError("Evaluation interrupted; preserve outputs and reservation")
                if expired(): raise TimeoutError("External evaluation watchdog exhausted original allowance")
                time.sleep(.1)
            if pending: raise InterruptedError("Evaluation interrupted at child exit")
            if expired(): raise TimeoutError("Evaluation exited after original allowance")
            if child.returncode: raise subprocess.CalledProcessError(child.returncode, child.args)
            summary = read(root / "summary.json")
            if not summary["complete"]: raise ValueError("Worker did not complete full comparison")
            if expired(): raise TimeoutError("Summary verification exceeded original allowance")
            write_new(root / "COMPLETE.json", dict(complete=True, summary_sha256=sha(root / "summary.json"), protocol_sha256=sha(protocol_path),
                external_watchdog_passed=True, observed_calendar_seconds=time.monotonic()-started, original_allowance_seconds=p["max_calendar_seconds"]))
            return summary
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        write_new(root / "FAILED.json", dict(error_type=type(e).__name__, error=str(e), calendar_seconds=time.monotonic()-started, original_allowance_seconds=p["max_calendar_seconds"], automatic_restart=False)); raise
    finally:
        for s, old_handler in old.items(): signal.signal(s, old_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--python", default=sys.executable); parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pre-reserved-ticket", help="Authenticated one-use remote relay only; local default still reserves normally")
    args = parser.parse_args()
    if args.worker: worker(args.protocol, args.publication_commit, args.pre_reserved_ticket)
    else: run(args.protocol, args.publication_commit, args.python, args.pre_reserved_ticket)
