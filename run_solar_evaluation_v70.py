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
}


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def software_identity():
    import torch, cv2, platform
    return dict(torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(), cpu_threads=2)


def admission(protocol_path, commit):
    p = read(protocol_path)
    if p.get("prepared_only", False): raise ValueError("Prepared template cannot start scientific inference")
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
            from native_pretrained_adapters_v73 import validate_native_profile, required_native_sources
            gate = validate_native_profile(profile)
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
    def __init__(self, publication_commit=None, blender_executable=None):
        self.image_key = self.geometry_key = None
        self.images = self.geometry = None
        self.last_setup = {}
        self.lookup = None; self.publication_commit = publication_commit; self.blender_executable = blender_executable
    def close(self):
        if self.lookup is not None: self.lookup.close(); self.lookup = None
    def get(self, case, track):
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
            if case.get("solar_lookup_protocol"):
                from solar_lookup_v70 import ReferenceSunLookup
                tick = time.perf_counter()
                self.lookup = ReferenceSunLookup.from_protocol(case["solar_lookup_protocol"], self.publication_commit, self.blender_executable)
                self.last_setup["reference_physics_setup_seconds"] = time.perf_counter()-tick
        return InferenceInput(*self.images, self.geometry, np.asarray(case["query_intrinsics"], float),
            np.asarray(case["reference_sun_world"], float),
            None if track == "estimated_sun" else np.asarray(case["secondary_known_query_sun_world"], float), self.lookup)


def worker(protocol_path, commit):
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
    write_new(root / "execution.json", dict(protocol_sha256=sha(protocol_path), publication_commit=commit, software=software_identity(),
        purpose=p["purpose"], raw_uncalibrated=True, final_reserve_access=False, profile_order=[v["name"] for v in p["profiles"]],
        initialization_only_cost_controls=[v["name"] for v in p["profiles"] if v.get("initialization_only_cost_control",False)]))
    loader = InputLoader(commit, p.get("blender_executable")); reusable = cache_index(p)
    try:
        for profile in p["profiles"]:
            check(); backend = None; load_seconds = 0.
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
                            load_tick = time.monotonic(); backend = Backend(profile); load_seconds = time.monotonic()-load_tick
                            model_load_this_case = load_seconds
                            write_new(root / (digest(profile)+".provenance.json"), backend.provenance)
                        inp = loader.get(case, track)
                        row, arrays = evaluate_case(backend, inp, case)
                        row["reference_setup"] = loader.last_setup.copy()
                        row["complete_localization_cold_seconds"] = row["inference_seconds"] + sum(loader.last_setup.values())
                        with array_path.open("xb") as f: np.savez_compressed(f, **arrays)
                        row["reused_case_cache"] = False
                        row["profile_load_seconds"] = load_seconds
                    else:
                        row = dict(cached, reused_case_cache=True, original_case_calendar_seconds=cached["case_calendar_seconds"],
                            timing_scope="Inference and setup latency replayed from this exact pinned cached execution; current cache-copy time is not model latency")
                        with array_path.open("xb") as destination, Path(cached["saved_arrays"]).open("rb") as source: shutil.copyfileobj(source, destination)
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
        check()
        expected = len(cases)*len(p["profiles"])*len(p["tracks"])
        if len(rows) != expected: raise ValueError("Full matched case/profile/track denominator incomplete")
        summary = dict(complete=True, purpose=p["purpose"], rows=rows, software=software_identity(), comparison_identity=comparison_identity(p, m),
            protocol_sha256=sha(protocol_path), manifest_sha256=sha(p["manifest"]), source_publication_commit=commit,
            calendar_seconds=time.monotonic()-start, profiles=p["profiles"], case_ids=[c["case_id"] for c in cases],
            fresh_inference_cases=sum(not r["reused_case_cache"] for r in rows), cached_cases=sum(r["reused_case_cache"] for r in rows),
            accepted_results=False, calibration_or_final_access=False, automatic_model_promotion=False)
        summary["scientific_model_evidence"] = not any(v.get("initialization_only_cost_control",False) for v in p["profiles"])
        summary["offline_reference_preparation"] = []
        for geometry in sorted({c["reference_geometry"] for c in cases}):
            receipt = m["reference_precompute_receipts"][geometry]
            count = sum(c["reference_geometry"] == geometry for c in cases)
            seconds = read(receipt["receipt_path"])["calendar_seconds"]
            summary["offline_reference_preparation"].append(dict(reference_geometry=geometry, **receipt,
                measured_calendar_seconds=seconds, queries_per_profile_per_track=count,
                fixed_study_amortized_seconds_per_query=seconds/count,
                note="Offline geometry/physical-table preparation; separate from warm online localization. Study reuse does not specify a deployment query frequency."))
        summary["raw_results"] = aggregate_rows(rows)
        if p["purpose"] == "principal_ablations":
            summary["paired_terrain_intervals"] = {track: paired_component_bootstrap(rows, "A1B1C1", "A0B0C0", track) for track in p["tracks"]}
        if p["purpose"] == "checkpoint_selection": summary["selection"] = checkpoint_ranking(rows, p["profiles"], [c["case_id"] for c in cases])
        if p["purpose"] == "cost_pilot": summary["measured_full_cost"] = measured_cost(rows, len(m["cases"]), p["tracks"], p["profiles"])
        write_new(root / "summary.json", summary)
    except BaseException as e:
        write_new(root / "WORKER_FAILED.json", dict(error_type=type(e).__name__, error=str(e), completed_rows=len(rows), calendar_seconds=time.monotonic()-start)); raise
    finally:
        loader.close()


def run(protocol_path, commit, python):
    # The controller is independent of the scientific worker. Even an
    # uninterruptible long C/torch call cannot silently reset its wall limit.
    p, _, _ = admission(protocol_path, commit); root = Path(p["output"])
    if root.exists(): raise FileExistsError("Preserve every completed/partial run; no reset or automatic restart")
    from solar_compute_budget_v67 import reserve
    reservation = reserve(p["compute_ledger"], p["compute_ledger_sha256"], p["reservation_journal"], sha(protocol_path)+"/evaluation", "baseline", p["max_calendar_seconds"])
    root.mkdir(parents=True)
    write_new(root / "reservation.json", reservation)
    child = None; started = time.monotonic(); pending = []
    now = datetime.now(timezone.utc)
    write_new(root / "watchdog_admission.json", dict(protocol_sha256=sha(protocol_path), reservation_sha256=sha(root / "reservation.json"),
        start_monotonic=started, started_utc=now.isoformat(), absolute_deadline_utc=(now+timedelta(seconds=p["max_calendar_seconds"])).isoformat(), controller_pid=os.getpid()))
    def interrupted(signum, frame): pending.append(signum)
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (root / "worker.log").open("x") as log:
            child = subprocess.Popen([python, str(Path(__file__)), "--protocol", protocol_path, "--publication-commit", commit, "--worker"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2"))
            while child.poll() is None:
                if pending: raise InterruptedError("Evaluation interrupted; preserve outputs and reservation")
                if time.monotonic()-started >= p["max_calendar_seconds"]: raise TimeoutError("External evaluation watchdog exhausted original allowance")
                time.sleep(.1)
            if pending: raise InterruptedError("Evaluation interrupted at child exit")
            if time.monotonic()-started >= p["max_calendar_seconds"]: raise TimeoutError("Evaluation exited after original allowance")
            if child.returncode: raise subprocess.CalledProcessError(child.returncode, child.args)
            summary = read(root / "summary.json")
            if not summary["complete"]: raise ValueError("Worker did not complete full comparison")
            if time.monotonic()-started >= p["max_calendar_seconds"]: raise TimeoutError("Summary verification exceeded original allowance")
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
    args = parser.parse_args()
    if args.worker: worker(args.protocol, args.publication_commit)
    else: run(args.protocol, args.publication_commit, args.python)
