"""Strict, raw development comparisons and pose-based checkpoint selection.

No calibration/final split is accepted. Scoring labels are decoded only after
the inference adapter returns. All matchers use the same reference-geometry
AP3P estimator. This module does not promote a model or assert acceptance.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import time
import numpy as np

LIGHT_PAIRS = ((0, 0), (16, 16), (0, 17), (16, 1), (8, 17), (8, 1))
CHECKPOINT_STEPS = (0, 500, 1000, 1500, 2000)
TRACKS = ("estimated_sun", "known_sun")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def principal_profiles(checkpoint, pooled_checkpoint, checkpoint_sha256, pooled_sha256):
    """The eight matched experiments differ ONLY in the declared A/B/C factors."""
    profiles = []
    for a, b, c in itertools.product((False, True), repeat=3):
        profiles.append(dict(name=f"A{int(a)}B{int(b)}C{int(c)}", backend="solar",
            checkpoint=checkpoint, configuration=dict(model_version="solar-experimental-v63",
            checkpoint_sha256=checkpoint_sha256, solar_mode="shared" if a else "off",
            response=b, active=c, pooled=False, budget=128, policy="active")))
    profiles.append(dict(name="pooled_A1B1C1", backend="solar", checkpoint=pooled_checkpoint,
        configuration=dict(profiles[-1]["configuration"], checkpoint_sha256=pooled_sha256, pooled=True)))
    return profiles


def completed_training_evidence(profile):
    """Verify that a saved checkpoint belongs to a completed bounded study.

    A training summary can be written immediately before the outer watchdog
    rejects a late process. Only the linked successful run receipt admits its
    checkpoints for scientific selection or learned ablation results.
    """
    evidence = profile.get("training_evidence", {})
    required = {key for name in ("protocol", "summary", "watchdog_receipt") for key in (name, name+"_sha256")}
    if not required <= set(evidence):
        raise ValueError("Completed source-bound training protocol, summary and external watchdog evidence required")
    for name in ("protocol", "summary", "watchdog_receipt"):
        if sha(evidence[name]) != evidence[name+"_sha256"]:
            raise ValueError("Frozen completed-training evidence changed: "+name)
    protocol = read(evidence["protocol"]); summary = read(evidence["summary"]); receipt = read(evidence["watchdog_receipt"])
    protocol_sha = sha(evidence["protocol"])
    if protocol["updates"] != 2000 or not summary["complete"] or summary["updates"] != 2000 or summary["protocol_sha256"] != protocol_sha:
        raise ValueError("Completed source-bound 2000-update training required for checkpoint selection")
    if protocol["seed"] not in (20261101, 20261102, 20261103) or summary["seed"] != protocol["seed"] or summary["pooled"] != protocol["pooled"] or protocol["pooled"] != profile["configuration"]["pooled"]:
        raise ValueError("Training seed/pooling differs from evaluated checkpoint")
    if "training_seed" in profile and profile["training_seed"] != protocol["seed"]:
        raise ValueError("Evaluated checkpoint declares a different training seed")
    if Path(protocol["output"]).resolve() != Path(evidence["summary"]).parent.resolve() or Path(protocol["watchdog_receipts"]["run"]).resolve() != Path(evidence["watchdog_receipt"]).resolve():
        raise ValueError("Training evidence paths do not belong to the published study")
    if receipt["complete"] is not True or receipt["mode"] != "run" or receipt["program"] != "train_solar_response_v63.py" or receipt["child_returncode"] != 0 or receipt.get("failure") is not None or receipt["protocol_sha256"] != protocol_sha:
        raise ValueError("Checkpoint study did not complete its external training watchdog")
    if receipt["source_sha256"] != protocol["source_sha256"]["run_solar_job_v70.py"]:
        raise ValueError("Executed training watchdog differs from the published source")
    started = datetime.fromisoformat(receipt["started_utc"]); deadline = datetime.fromisoformat(receipt["absolute_deadline_utc"])
    published_deadline = datetime.fromisoformat(protocol["absolute_deadline_utc"])
    if started.tzinfo is None or deadline.tzinfo is None or published_deadline.tzinfo is None:
        raise ValueError("Unambiguous original training deadlines required")
    cap = receipt["cap_seconds"]; elapsed = receipt["calendar_seconds"]; training_elapsed = summary["calendar_seconds"]
    if type(cap) is not int or cap != protocol["max_calendar_seconds"] or cap <= 0 or deadline != min(started+timedelta(seconds=cap), published_deadline):
        raise ValueError("Training watchdog original allowance or deadline changed")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (elapsed, training_elapsed)) or not 0 < training_elapsed <= elapsed <= min(cap, (deadline-started).total_seconds()):
        raise ValueError("Checkpoint study exceeded its measured external training deadline")
    root = Path(evidence["summary"]).parent
    if (root/"FAILED.json").exists():
        raise ValueError("A failed training study cannot supply selected checkpoints")
    checkpoint = Path(profile["checkpoint"])
    if checkpoint.parent.resolve() != root.resolve() or checkpoint.name not in summary["checkpoint_sha256"] or sha(checkpoint) != summary["checkpoint_sha256"][checkpoint.name] or sha(checkpoint) != profile["configuration"]["checkpoint_sha256"]:
        raise ValueError("Evaluated checkpoint does not belong to completed training evidence")
    if "training_step" in profile and checkpoint.name != f'step_{profile["training_step"]}.pt':
        raise ValueError("Evaluated training step does not match the checkpoint file")
    if sha(root/"history_2000.json") != summary["all_history_sha256"]:
        raise ValueError("Completed training history changed")
    return evidence


def validate_profile_training(profile, purpose):
    """Initialization-only cost measurements never become model evidence."""
    if profile["backend"] != "solar": return
    if profile.get("initialization_only_cost_control", False):
        if purpose != "cost_pilot":
            raise ValueError("Initialization-only checkpoints are restricted to cost pilots")
        return
    completed_training_evidence(profile)


def checkpoint_profiles(run_root, pooled=False, *, training_protocol=None, watchdog_receipt=None):
    """Requires completed, hash-sealed training; intermediate optimizer loss is not selection."""
    root = Path(run_root); summary = read(root / "summary.json")
    if not summary["complete"] or summary["updates"] != 2000 or summary["pooled"] != pooled:
        raise ValueError("A completed, matching 2000-update study is required")
    if training_protocol is None:
        raise ValueError("Published training protocol and successful external watchdog evidence required")
    protocol = read(training_protocol)
    watchdog_receipt = protocol["watchdog_receipts"]["run"] if watchdog_receipt is None else watchdog_receipt
    evidence = dict(protocol=str(training_protocol), protocol_sha256=sha(training_protocol),
        summary=str(root/"summary.json"), summary_sha256=sha(root/"summary.json"),
        watchdog_receipt=str(watchdog_receipt), watchdog_receipt_sha256=sha(watchdog_receipt))
    result = []
    for step in CHECKPOINT_STEPS:
        path = root / f"step_{step}.pt"; h = sha(path)
        if summary["checkpoint_sha256"][path.name] != h:
            raise ValueError("Training checkpoint no longer matches its completion summary")
        result.append(dict(name=f"step_{step}", backend="solar", checkpoint=str(path),
            training_step=step, training_seed=summary["seed"],
            training_evidence=evidence.copy(),
            configuration=dict(model_version="solar-experimental-v63", checkpoint_sha256=h,
                solar_mode="shared", response=True, active=True, pooled=pooled, budget=128, policy="active")))
    for profile in result: completed_training_evidence(profile)
    return result


def additional_profiles(checkpoint, checkpoint_sha256):
    """Decisive global-Sun control and all four acquisition policies/budgets."""
    base = dict(model_version="solar-experimental-v63", checkpoint_sha256=checkpoint_sha256,
        solar_mode="shared", response=True, active=True, pooled=False, budget=128, policy="active")
    result = [dict(name="independent_sun_A1B1C1", backend="solar", checkpoint=checkpoint,
        configuration=dict(base, solar_mode="independent"))]
    for budget in (32, 64, 128):
        for policy in ("active", "uniform", "appearance", "geometry"):
            result.append(dict(name=f"policy_{policy}_{budget}", backend="solar", checkpoint=checkpoint,
                configuration=dict(base, budget=budget, policy=policy)))
    return result


def validate_profiles(profiles, purpose):
    if not profiles or len({p["name"] for p in profiles}) != len(profiles):
        raise ValueError("Nonempty uniquely named profiles required")
    if purpose == "checkpoint_selection":
        if len(profiles) != 5 or sorted(p["training_step"] for p in profiles) != list(CHECKPOINT_STEPS):
            raise ValueError("All five preregistered checkpoints must be scored")
        expected = dict(profiles[0]["configuration"]); expected.pop("checkpoint_sha256")
        if any(p["backend"] != "solar" or {k: v for k, v in p["configuration"].items() if k != "checkpoint_sha256"} != expected for p in profiles):
            raise ValueError("Checkpoint selection cannot change inference configuration")
        if len({p["training_seed"] for p in profiles}) != 1:
            raise ValueError("Select checkpoints within one training seed")
    elif purpose == "principal_ablations":
        names = {f"A{int(a)}B{int(b)}C{int(c)}" for a, b, c in itertools.product((False, True), repeat=3)}
        byname = {p["name"]: p for p in profiles}
        if set(byname) != names | {"pooled_A1B1C1"}:
            raise ValueError("Exactly eight matched on/off profiles and the pooled control required")
        base = byname["A1B1C1"]
        for name in names:
            p = byname[name]; a, b, c = (bool(int(name[i])) for i in (1, 3, 5))
            expected = dict(base["configuration"], solar_mode="shared" if a else "off", response=b, active=c)
            if p["backend"] != "solar" or p["checkpoint"] != base["checkpoint"] or p["configuration"] != expected:
                raise ValueError("A/B/C controls must share the exact checkpoint and all other settings")
        pooled = byname["pooled_A1B1C1"]
        expected = dict(base["configuration"], pooled=True, checkpoint_sha256=pooled["configuration"]["checkpoint_sha256"])
        if pooled["backend"] != "solar" or pooled["configuration"] != expected:
            raise ValueError("Pooled control must change only its separately trained checkpoint and pooling")
    elif purpose not in ("cost_pilot", "additional_controls", "modern_comparison"):
        raise ValueError("Explicit development purpose required")
    for p in profiles:
        if p["backend"] not in ("solar", "sift", "modern", "native_pretrained", "historical_compact", "direct_render"):
            raise ValueError("Unimplemented baselines must be recorded as unavailable, never substituted")
        if p["backend"] == "native_pretrained":
            from native_pretrained_adapters_v73 import specification
            specification(p)
        if p["backend"] == "historical_compact":
            from historical_compact_adapters_v73 import METHODS
            if p["method"] not in METHODS: raise ValueError("Unknown immutable historical control")
        if p["backend"] == "direct_render":
            from direct_render_adapter_v73 import METHOD
            if p["method"] != METHOD: raise ValueError("Unknown direct-render method; no fallback")
        if p["backend"] == "solar":
            c = p["configuration"]
            if c["budget"] not in (32, 64, 128) or c["policy"] not in ("active", "uniform", "appearance", "geometry") or c["solar_mode"] not in ("shared", "off", "independent"):
                raise ValueError("Explicit finite solar and patch-policy controls required")


def bounds_disjoint(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all() or not (a[0] < a[2] and a[1] < a[3] and b[0] < b[2] and b[1] < b[3]):
        raise ValueError("Conservative finite physical footprint bounds required")
    return bool(a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def validate_manifest(m, audit):
    """Every eligible selection unit x all six lights, plus paired nonoverlap.

    Bounds are conservative axis-aligned footprints in the SAME declared lunar
    map frame. They must enclose the full image footprints, not patch centres.
    """
    if m["schema_version"] != 70 or m["split"] != "model_selection" or m["calibration_or_final_access"]:
        raise ValueError("Development-only v70 case manifest required")
    if not audit["passed"] or not audit["complete_dataset"] or audit["expected_units"] != 96 or audit["completed_units"] != 96 or audit["oracle_success"] < .95:
        raise ValueError("Complete independently passed 96-unit geometry gate required")
    if audit["oracle_pose_directions"] != 2 * sum(bool(u["eligible"]) for u in audit["units"]):
        raise ValueError("Geometry pose failures must remain in the oracle denominator")
    units = {int(u["unit_index"]): u for u in audit["units"] if u["split"] == "model_selection"}
    groups = sorted({u["component"] for u in units.values()})
    if len(groups) != 8 or len(units) != 32 or sorted(m["terrain_groups"]) != groups:
        raise ValueError("Original eight selection groups and four camera pairs per group required")
    if any(sum(u["component"] == g for u in units.values()) != 4 for g in groups):
        raise ValueError("Four original camera units per selection terrain required")
    eligible = {i for i, u in units.items() if u["eligible"]}
    cases = m["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case identity")
    positives = [c for c in cases if c["positive"]]; negatives = [c for c in cases if not c["positive"]]
    expected = {(i, a, b) for i in eligible for a, b in LIGHT_PAIRS}
    actual = [(int(c["unit_index"]), c["light_a"], c["light_b"]) for c in positives]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("All eligible cameras and all six fixed lighting pairs must remain in evaluation")
    pbyid = {c["case_id"]: c for c in positives}
    if len(negatives) != len(positives) or {c["matched_positive_case"] for c in negatives} != set(pbyid):
        raise ValueError("One matched nonoverlap pair per eligible positive required")
    for c in cases:
        ui = int(c["unit_index"]); qi = int(c["query_unit_index"])
        if c["split"] != "model_selection" or ui not in eligible or qi not in eligible or c["component"] != units[ui]["component"] or c["query_component"] != units[qi]["component"]:
            raise ValueError("Both inference endpoints must be eligible selection terrain")
        if c["positive"]:
            if ui != qi or c["component"] != c["query_component"]:
                raise ValueError("Positive pair endpoint inconsistency")
        else:
            p = pbyid[c["matched_positive_case"]]
            if (ui, c["light_a"], c["light_b"]) != (int(p["unit_index"]), p["light_a"], p["light_b"]) or c["component"] == c["query_component"]:
                raise ValueError("Negative must preserve its positive reference and lighting")
            proof = c["nonoverlap_certificate"]
            if proof["frame"] != m["footprint_frame"] or not proof["full_image_footprints"] or not bounds_disjoint(proof["reference_bounds"], proof["query_bounds"]):
                raise ValueError("Nonoverlap needs disjoint conservative full-image terrain footprints")
    return cases


@dataclass(frozen=True)
class InferenceInput:
    """Truth-free boundary, including per-image reference sunlight metadata."""
    reference: np.ndarray
    query: np.ndarray
    geometry: object
    query_intrinsics: np.ndarray
    reference_sun: np.ndarray
    known_query_sun: object = None
    solar_lookup: object = None


def common_pose(reference_xy, query_xy, inp):
    from native_baseline_pose_v7 import propose_pose
    p = np.asarray(reference_xy, float).reshape(-1, 2); q = np.asarray(query_xy, float).reshape(-1, 2)
    if len(p) != len(q) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Finite paired native coordinates required")
    if len(p) and (not ((p >= 0) & (p <= 511)).all() or not ((q >= 0) & (q <= 511)).all()):
        raise ValueError("Every returned correspondence must use native pixel centres")
    pose, info = propose_pose(p, q, inp.geometry.depth, inp.geometry.intrinsics, inp.geometry.cam2world, inp.query_intrinsics)
    return pose, dict(info, iterations_requested=2000, reprojection_threshold_native_px=3., solver="ap3p_reclassify")


def score_arrays(p, q, pose, truth, reference_range):
    """Model-free arithmetic scoring. Includes failed estimates in denominator."""
    p = np.asarray(p, float).reshape(-1, 2); q = np.asarray(q, float).reshape(-1, 2)
    if len(p) != len(q) or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(reference_range) or reference_range <= 0:
        raise ValueError("Finite matches and positive reference range required")
    positive = bool(truth["positive"]); pose_error = None
    if pose is not None:
        pose = np.asarray(pose, float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5):
            raise ValueError("Pose must be a finite camera-to-world rigid transform")
        if positive:
            target = np.asarray(truth["query_cam2world"], float)
            angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ target[:3, :3])-1)/2, -1, 1)))
            pose_error = dict(rotation_deg=float(angle), translation_m=float(np.linalg.norm(pose[:3, 3]-target[:3, 3])))
    correct = bool(positive and pose_error is not None and pose_error["rotation_deg"] <= 2 and pose_error["translation_m"] <= .01*reference_range)
    strict = bool(positive and pose_error is not None and pose_error["rotation_deg"] <= .5 and pose_error["translation_m"] <= .001*reference_range)
    correspondence = {}
    if positive:
        target = np.asarray(truth["query_xy"], float); visible = np.asarray(truth["visible"], bool)
        if target.shape != q.shape or visible.shape != (len(q),):
            raise ValueError("Independently projected scoring labels must align with returned reference points")
        errors = np.full(len(q), np.inf); errors[visible] = np.linalg.norm(q[visible]-target[visible], axis=1)
        for threshold in (1, 2, 4, 8, 12):
            correspondence[str(threshold)] = dict(correct=int((errors <= threshold).sum()), co_visible=int(visible.sum()), returned=len(q))
    return dict(positive=positive, correct=correct, strict=strict, pose_returned=pose is not None,
        wrong_positive=bool(positive and pose is not None and not correct), raw_nonoverlap=bool(not positive and pose is not None),
        pose_error=pose_error, reference_range_m=float(reference_range), correspondence=correspondence)


def load_scoring(case, p):
    """Called AFTER prediction. Dense depth labels support arbitrary baselines.

    Solar fixed-anchor exact labels are preferred; an arbitrary-point baseline
    must instead provide both query depth and camera data for the same common
    axial-depth correspondence routine. The latter uncertainty is disclosed.
    """
    if not case["positive"]:
        return dict(positive=False), "nonoverlap_full_footprint_certificate"
    p = np.asarray(p).reshape(-1, 2)
    with np.load(case["scoring_labels"], allow_pickle=False) as z:
        target_pose = z["query_cam2world"].copy()
        anchors = z["reference_xy"].copy(); target = z["query_xy"].copy(); visible = z["visible"].copy()
    lookup = {tuple(a): i for i, a in enumerate(anchors)}
    if len(lookup) != len(anchors):
        raise ValueError("Independent scoring anchors must be unique")
    if all(tuple(a) in lookup for a in p):
        indices = np.array([lookup[tuple(a)] for a in p], dtype=int)
        return dict(positive=True, query_cam2world=target_pose, query_xy=target[indices], visible=visible[indices]), "exact_frozen_projected_labels"
    # No interpolation of labels across terrain discontinuities. Modern/SIFT
    # reference points are independently projected from scoring-only depth.
    from lunar_geometry_v5 import correspondences_z
    with np.load(case["scoring_geometry"], allow_pickle=False) as z:
        args = [z[k].copy() for k in ("reference_depth", "query_depth", "reference_intrinsics", "reference_cam2world", "query_intrinsics", "query_cam2world")]
    _, target, _, visible = correspondences_z(*args, points=p)
    return dict(positive=True, query_cam2world=target_pose, query_xy=target, visible=visible), "common_axial_depth_projection_at_returned_points"


def checkpoint_ranking(rows, profiles, case_ids):
    validate_profiles(profiles, "checkpoint_selection")
    expected = {(p["name"], c, "estimated_sun") for p in profiles for c in case_ids}
    actual = [(r["profile"], r["case_id"], r["track"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Checkpoint selection requires complete, identical primary-track cases, including failures and negatives")
    table = []
    for p in profiles:
        part = [r for r in rows if r["profile"] == p["name"]]
        table.append(dict(profile=p["name"], step=p["training_step"],
            correct_poses=sum(r["metrics"]["correct"] for r in part),
            two_pixel_correspondences=sum(r["metrics"]["correspondence"].get("2", {}).get("correct", 0) for r in part),
            eligible_positive_cases=sum(r["positive"] for r in part),
            negative_cases=sum(not r["positive"] for r in part)))
    table.sort(key=lambda r: (-r["correct_poses"], -r["two_pixel_correspondences"], r["step"]))
    return dict(selected=table[0], ranking=table, criterion="unfiltered_correct_pose_count, then two_pixel_correspondence_count, then earlier_checkpoint", automatically_promoted=False)


def paired_component_bootstrap(rows, candidate, control, track="estimated_sun", seed=20261101, repetitions=2000):
    """Paired terrain multiplier bootstrap; negative weights use BOTH endpoints.

    Positive pairs receive their terrain multiplicity. A nonoverlap edge gets
    the product of the multiplicities of both endpoints. This is a declared
    multiway cluster resampling convention, not an independence claim.
    """
    selected = [r for r in rows if r["track"] == track and r["profile"] in (candidate, control)]
    by = {(r["profile"], r["case_id"]): r for r in selected}
    cases = sorted({r["case_id"] for r in selected})
    if len(by) != 2*len(cases) or any((p, c) not in by for p in (candidate, control) for c in cases):
        raise ValueError("Complete matched paired case table required")
    groups = sorted({r[k] for r in selected for k in ("component", "query_component")}); gi = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(seed); values = []; failures = []; negatives = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups)); pos_denom = neg_denom = 0.; gain = wrong = neg = 0.
        for c in cases:
            a, b = by[candidate, c], by[control, c]
            if (a["component"], a["query_component"], a["positive"]) != (b["component"], b["query_component"], b["positive"]):
                raise ValueError("Paired endpoints changed between methods")
            w = float(counts[gi[a["component"]]])
            if a["positive"]:
                pos_denom += w; gain += w*(int(a["metrics"]["correct"])-int(b["metrics"]["correct"]))
                wrong += w*(int(a["metrics"]["wrong_positive"])-int(b["metrics"]["wrong_positive"]))
            else:
                w *= counts[gi[a["query_component"]]]; neg_denom += w
                neg += w*(int(a["metrics"]["raw_nonoverlap"])-int(b["metrics"]["raw_nonoverlap"]))
        if pos_denom: values.append(gain/pos_denom); failures.append(wrong/pos_denom)
        if neg_denom: negatives.append(neg/neg_denom)
    interval = lambda a: None if not a else np.quantile(a, [.025, .5, .975]).tolist()
    return dict(raw_correct_pose_difference=interval(values), raw_wrong_pose_difference=interval(failures), raw_nonoverlap_difference=interval(negatives),
        terrain_components=len(groups), repetitions=repetitions, seed=seed,
        negative_resampling="product of both endpoint terrain multiplicities", accepted_performance_evaluated=False)


def aggregate_rows(rows):
    """Failures and nonoverlap always remain in the declared denominators."""
    table = []
    for profile, track in sorted({(r["profile"], r["track"]) for r in rows}):
        part = [r for r in rows if (r["profile"], r["track"]) == (profile, track)]
        pos = [r for r in part if r["positive"]]; neg = [r for r in part if not r["positive"]]
        def counts(items):
            return dict(cases=len(items), raw_correct_poses=sum(r["metrics"]["correct"] for r in items),
                raw_pose_outputs=sum(r["metrics"]["pose_returned"] for r in items),
                raw_wrong_positive=sum(r["metrics"]["wrong_positive"] for r in items),
                raw_nonoverlap_outputs=sum(r["metrics"]["raw_nonoverlap"] for r in items),
                inference_failures=sum(r["inference_failure"] is not None for r in items),
                unavailable_cases=sum((r["inference_failure"] or {}).get("status") == "unavailable" for r in items))
        table.append(dict(profile=profile, track=track, positive=counts(pos), nonoverlap=counts(neg),
            raw_pose_success=None if not pos else sum(r["metrics"]["correct"] for r in pos)/len(pos),
            median_complete_inference_seconds=float(np.median([r["inference_seconds"] for r in part])),
            mean_localization_seconds_including_measured_cold_setup=float(np.mean([r["complete_localization_cold_seconds"] for r in part])),
            cold_reference_setup_total_seconds=float(sum(r["reference_setup"]["reference_geometry_load_seconds"]+r["reference_setup"]["reference_physics_setup_seconds"] for r in part)),
            model_load_seconds_shared_across_tracks=float(max(r["profile_load_seconds"] for r in part)),
            module_execution=dict(
                response_requested_cases=sum(r.get("response_requested") is True for r in part),
                response_confirmed_cases=sum(r.get("response_usage_confirmed") is True for r in part),
                response_zero_cases=sum(r.get("response_usage_confirmed") is False for r in part),
                response_unreported_cases=sum(r.get("response_usage_confirmed") is None for r in part),
                response_refresh_attempted_cases=sum((r.get("response_refresh_calls") or 0)>0 for r in part),
                scope="Execution diagnostics only; a requested module is not proof that its learned responses were used or improved accuracy."),
            by_reference_terrain={g: counts([r for r in part if r["component"] == g]) for g in sorted({r["component"] for r in part})},
            by_lighting={f"{a},{b}": counts([r for r in part if (r["light_a"], r["light_b"]) == (a, b)]) for a, b in LIGHT_PAIRS},
            comparison_available=not any((r["inference_failure"] or {}).get("status") == "unavailable" for r in part),
            nonoverlap_evaluation_available=not any((r["inference_failure"] or {}).get("status") == "unavailable" for r in neg),
            accepted_performance=None))
    return table


class Backend:
    """One checkpoint/model construction per profile, reused across cases."""
    def __init__(self, profile):
        self.profile = profile; self.model = None; self.provenance = {}
        if profile["backend"] == "solar":
            from localize_terrain import load_matcher
            self.model = load_matcher(profile["checkpoint"], profile["configuration"])
        elif profile["backend"] == "modern":
            from modern_baselines_v2 import ModernMatcher
            self.model = ModernMatcher(profile["method"], profile["source"], profile["author_profile"], "cpu")
            self.provenance = self.model.provenance
        elif profile["backend"] == "native_pretrained":
            from native_pretrained_adapters_v73 import make_native_pretrained
            self.model = make_native_pretrained(profile)
            self.provenance = self.model.provenance
        elif profile["backend"] == "historical_compact":
            from historical_compact_adapters_v73 import make_historical_compact
            self.model = make_historical_compact(profile)
            self.provenance = self.model.provenance
        elif profile["backend"] == "direct_render":
            from direct_render_adapter_v73 import make_direct_render
            self.model = make_direct_render(profile)
            self.provenance = self.model.provenance
        elif profile["backend"] == "sift":
            import cv2
            self.model = cv2.SIFT_create(nfeatures=2048)
        else:
            raise ValueError("Unimplemented method cannot silently fall back")

    def infer(self, inp):
        p = self.profile
        if p["backend"] in ("historical_compact", "direct_render"):
            return self.model.infer(inp)
        if p["backend"] == "solar":
            c = p["configuration"]
            # Callback uses exactly the common estimator for EVERY method.
            def final_estimator(reference_xy, query_xy):
                return common_pose(reference_xy, query_xy, inp)
            return self.model.localize(inp.reference, inp.query, inp.geometry, inp.query_intrinsics,
                solar_mode=c["solar_mode"], response=c["response"], active=c["active"], budget=c["budget"], policy=c["policy"],
                known_sun=inp.known_query_sun, reference_sun=inp.reference_sun, final_estimator=final_estimator, solar_lookup=inp.solar_lookup)
        if p["backend"] in ("modern", "native_pretrained"):
            a, b = self.model.points(inp.reference, inp.query); self.model.sync()
        else:
            import cv2
            if min(inp.reference.std(), inp.query.std()) < 1:
                a = b = np.empty((0, 2))
            else:
                ka, da = self.model.detectAndCompute(inp.reference, None); kb, db = self.model.detectAndCompute(inp.query, None)
                pairs = [] if da is None or db is None or len(db) < 2 else cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
                matches = [x[0] for x in pairs if len(x) == 2 and x[0].distance < .75*x[1].distance]
                a = np.array([ka[m.queryIdx].pt for m in matches]).reshape(-1, 2); b = np.array([kb[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        pose, info = common_pose(a, b, inp)
        return dict(reference_xy=a, query_xy=b, pose=pose, reason=info["reason"], common_estimator=info)


def evaluate_case(backend, inp, case, scoring_loader=load_scoring, estimator=common_pose):
    """The labels loader is deliberately outside and after the measured call."""
    import cv2
    lookup_before = None if inp.solar_lookup is None else dict(inp.solar_lookup.stats)
    tick = time.perf_counter(); failure = None
    try:
        result = backend.infer(inp)
        p = np.asarray(result.get("reference_xy", []), float).reshape(-1, 2); q = np.asarray(result.get("query_xy", []), float).reshape(-1, 2)
        # Replay the SAME final solve even if a backend already returns a pose.
        # Report this extra verification cost separately; never time only its
        # inner neural forward and call that full localization latency.
        infer_seconds = time.perf_counter()-tick; replay_tick = time.perf_counter()
        pose, info = estimator(p, q, inp); replay_seconds = time.perf_counter()-replay_tick
        internal_pose = result.get("pose")
    except (RuntimeError, cv2.error) as exc:
        # Every eligible failure remains a row, not a skipped case.
        p = q = np.empty((0, 2)); pose = None; info = dict(reason="inference_failure", inliers=0); result = {}
        infer_seconds = time.perf_counter()-tick; replay_seconds = 0.
        failure = dict(type=type(exc).__name__, message=str(exc),
            status="unavailable" if type(exc).__name__.endswith("Unavailable") else "failed")
    if failure is None and ((pose is None) != (internal_pose is None) or (pose is not None and not np.allclose(pose, internal_pose, rtol=0, atol=1e-7))):
        # A broken common-estimator integration is an experiment-wide defect,
        # not an ordinary difficult image that may be counted as a model miss.
        raise ValueError("Backend pose differs from common estimator replay")
    truth, label_method = scoring_loader(case, p)
    from lunar_geometry_v5 import median_reference_range_z
    distance = median_reference_range_z(inp.geometry.depth, inp.geometry.intrinsics)
    metrics = score_arrays(p, q, pose, truth, distance)
    arrays = dict(reference_xy=p, query_xy=q, pose=np.empty((0, 4)) if pose is None else pose,
        scoring_query_xy=np.empty((0, 2)) if not case["positive"] else truth["query_xy"],
        scoring_visible=np.empty(0, bool) if not case["positive"] else truth["visible"],
        scoring_query_cam2world=np.empty((0, 4)) if not case["positive"] else truth["query_cam2world"])
    def json_values(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {key: json_values(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)): return [json_values(v) for v in value]
        return value
    lookup_after = None if inp.solar_lookup is None else dict(inp.solar_lookup.stats)
    lookup_stats = None if lookup_after is None else dict(
        policy="One reference-provider lifetime; online call work is inside inference_seconds; factory validation is cold reference setup.",
        before=lookup_before, after=lookup_after,
        numeric_delta={key: value-lookup_before.get(key, 0) for key, value in lookup_after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(lookup_before.get(key, 0), (int, float))})
    hypotheses = result.get("hypotheses")
    response_measurements = result.get("response_measurements")
    return dict(metrics=metrics, inference_seconds=infer_seconds, common_pose_replay_seconds=replay_seconds,
        inference_failure=failure, label_method=label_method, final_estimator=info,
        response_cost=result.get("response_cost"), native_encoder_forwards=result.get("native_encoder_forwards"),
        historical_control=json_values(result.get("historical_control")), render_search=json_values(result.get("render_search")),
        response_requested=getattr(backend, "profile", {}).get("configuration", {}).get("response"),
        response_measurements=response_measurements, response_refresh_calls=result.get("response_refresh_calls"),
        response_usage_confirmed=None if response_measurements is None else bool(response_measurements>0),
        rounds=json_values(result.get("rounds")), reason=result.get("reason", info.get("reason")),
        initialization=json_values(result.get("initialization")),
        initial_hypothesis_count=result.get("initial_hypothesis_count"),
        returned_hypothesis_count=None if hypotheses is None else len(hypotheses),
        live_hypothesis_count=None if hypotheses is None else sum(h.get("pose") is not None for h in hypotheses),
        solar_lookup_stats=json_values(lookup_stats)), arrays
