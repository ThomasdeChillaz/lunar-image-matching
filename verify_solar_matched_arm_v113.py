"""Model-free saved-array arithmetic replay; no matching/model imports.

Reopens original frozen labels AFTER loading saved predictions, verifies their
hashes, and independently recomputes every reported pose/correspondence count.
This does not claim independent rerendering or a fresh estimator reproduction.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from solar_evaluation_v70 import read, sha
from solar_aligned_scoring_v91 import load_aligned_scoring as load_scoring


from plan_solar_matched_arm_v113 import ARM_NAMES, COHORT, DIRECT_NAME, TRACK


def exact_case_ids(protocol, cohort):
    if (
        protocol.get("case_ids") is None or len(protocol["case_ids"]) != 384
        or len(set(protocol["case_ids"])) != 384
        or protocol.get("manifest") != cohort["manifest"]
        or protocol.get("geometry_audit") != cohort["geometry_audit"]
        or protocol.get("tracks") != [TRACK] or protocol.get("cpu_threads") != 2
        or protocol.get("calibration_or_final_access") is not False
        or protocol.get("accepted_results") is not False
    ):
        raise ValueError("Original 384-case development authority and raw-only primary policy required")
    names = tuple(profile["name"] for profile in protocol["profiles"])
    if protocol.get("purpose") == "cost_pilot" and protocol.get("stage_mode") == "matched_direct_arm_cost_pilot_v113":
        ids = cohort["pilot_case_ids"]
        if protocol.get("pilot_case_ids") != ids or names != (DIRECT_NAME,):
            raise ValueError("Exact historical eight-case Direct-only qualification required")
        return ids
    if protocol.get("purpose") == "matched_development_comparison" and protocol.get("stage_mode") == "matched_modern_direct_arm_v113":
        size = str(protocol.get("comparison_case_count"))
        if size not in ("96", "64", "48"):
            raise ValueError("A predeclared matched cohort size is required")
        ids = cohort["subsets"][size]["case_ids"]
        if protocol.get("comparison_case_ids") != ids or names != (*ARM_NAMES, DIRECT_NAME):
            raise ValueError("Exact ordered v103 cohort and five fixed profiles required")
        return ids
    raise ValueError("Unsupported v113 replay purpose")


def replay(root, protocol_path, output):
    root = Path(root); summary = read(root / "summary.json"); p = read(protocol_path); manifest = read(p["manifest"])
    for name in ("verify_solar_matched_arm_v113.py", "plan_solar_matched_arm_v113.py", "solar_evaluation_v70.py", "solar_aligned_scoring_v91.py"):
        if p["source_sha256"][name] != sha(Path(__file__).with_name(name)):
            raise ValueError("Replay/scoring source differs from the pre-inference freeze")
    if not summary["complete"] or summary["protocol_sha256"] != sha(protocol_path) or summary["manifest_sha256"] != sha(p["manifest"]):
        raise ValueError("Complete frozen development summary required")
    seal = read(root / "COMPLETE.json")
    if (root / "FAILED.json").exists() or not seal["external_watchdog_passed"] or seal["summary_sha256"] != sha(root / "summary.json"):
        raise ValueError("Original external watchdog completion seal required")
    bycase = {c["case_id"]: c for c in manifest["cases"]}
    if p["case_ids"] != [case["case_id"] for case in manifest["cases"]]:
        raise ValueError("Original manifest eligibility/order changed")
    cohort_path, cohort_digest = COHORT
    if sha(cohort_path) != cohort_digest or p["input_sha256"].get(cohort_path) != cohort_digest:
        raise ValueError("Exact published v103 cohort identity must be frozen")
    ids = exact_case_ids(p, read(cohort_path))
    expected = {(name, case, track) for name in [x["name"] for x in p["profiles"]] for case in ids for track in p["tracks"]}
    actual = [(r["profile"], r["case_id"], r["track"]) for r in summary["rows"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Independent replay requires the full matched case/profile/track grid")
    replayed = []
    for row in summary["rows"]:
        c = bycase[row["case_id"]]
        for path, h in c["input_sha256"].items():
            if sha(path) != h: raise ValueError("Scoring/inference source asset changed")
        if row["component"] != c["component"] or row["query_component"] != c["query_component"] or row["positive"] != c["positive"]:
            raise ValueError("Case terrain endpoint metadata changed")
        if sha(row["saved_arrays"]) != row["saved_arrays_sha256"]:
            raise ValueError("Saved predictions/labels changed")
        with np.load(row["saved_arrays"], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
        points = a["reference_xy"]; query = a["query_xy"]
        if points.shape != query.shape or points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or not np.isfinite(query).all():
            raise ValueError("Saved native correspondence schema invalid")
        truth, label_method = load_scoring(c, points)
        if label_method != row["label_method"]: raise ValueError("Scoring projection method changed")
        if c["positive"] and (not np.array_equal(truth["query_xy"], a["scoring_query_xy"], equal_nan=True) or not np.array_equal(truth["visible"], a["scoring_visible"]) or not np.array_equal(truth["query_cam2world"], a["scoring_query_cam2world"])):
            raise ValueError("Saved labels differ from independently reopened source labels")
        if "scoring_query_sun_world"in a and not np.array_equal(a["scoring_query_sun_world"],np.asarray(c["secondary_known_query_sun_world"],float)):
            raise ValueError("Saved scoring Sun differs from the frozen post-inference label")
        if "estimated_sun_world"in a and(a["estimated_sun_world"].shape!=(3,)or not np.isfinite(a["estimated_sun_world"]).all()or not np.isclose(np.linalg.norm(a["estimated_sun_world"]),1.,atol=1e-6,rtol=0)):
            raise ValueError("Saved estimated Sun is not a unit reference-world vector")
        if "coarse_reference_xy"in a:
            coarse=a["coarse_reference_xy"];candidates=a["coarse_query_candidates"]
            if coarse.ndim!=2 or coarse.shape[1]!=2 or candidates.shape!=(len(coarse),8,2)or not np.isfinite(coarse).all()or not np.isfinite(candidates).all():
                raise ValueError("Saved raw coarse candidate schema invalid")
            coarse_truth,_=load_scoring(c,coarse)
            if c["positive"]and(not np.array_equal(coarse_truth["query_xy"],a["scoring_coarse_query_xy"],equal_nan=True)or not np.array_equal(coarse_truth["visible"],a["scoring_coarse_visible"])):
                raise ValueError("Saved candidate labels differ from independently reopened source labels")
        # Explicit native camera-ray norm, independent of the metric helper.
        with np.load(c["reference_geometry"], allow_pickle=False) as z: depth = z["depth"].copy(); k = z["intrinsics"].copy()
        y, x = np.mgrid[:512, :512]; rays = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)]) @ np.linalg.inv(k).T
        valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0)
        distance = float(np.median(depth.ravel()[valid] * np.linalg.norm(rays[valid], axis=1)))
        metrics = row["metrics"]
        if not np.isclose(distance, metrics["reference_range_m"], rtol=0, atol=1e-9): raise ValueError("Reference range arithmetic differs")
        pose = a["pose"]; has_pose = pose.shape == (4, 4)
        if not has_pose and pose.shape != (0, 4): raise ValueError("Explicit pose-null sentinel required")
        angle = shift = None; correct = strict = False
        if has_pose and c["positive"]:
            target = truth["query_cam2world"]
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(pose[:3, :3].T @ target[:3, :3])-1)/2, -1, 1))))
            shift = float(np.linalg.norm(pose[:3, 3]-target[:3, 3]))
            correct = angle <= 2 and shift <= .01*distance; strict = angle <= .5 and shift <= .001*distance
            if metrics["pose_error"] is None or not np.isclose(angle, metrics["pose_error"]["rotation_deg"], rtol=0, atol=1e-9) or not np.isclose(shift, metrics["pose_error"]["translation_m"], rtol=0, atol=1e-9):
                raise ValueError("Camera pose arithmetic differs")
        elif metrics["pose_error"] is not None: raise ValueError("No pose error may be manufactured for a missing/negative pose")
        checks = dict(correct=bool(correct), strict=bool(strict), pose_returned=has_pose,
            wrong_positive=bool(c["positive"] and has_pose and not correct), raw_nonoverlap=bool(not c["positive"] and has_pose))
        if any(metrics[key] != value for key, value in checks.items()): raise ValueError("Raw pose outcome arithmetic differs")
        if c["positive"]:
            error = np.full(len(query), np.inf); mask = truth["visible"]
            error[mask] = np.sqrt(((query[mask]-truth["query_xy"][mask])**2).sum(axis=1))
            for threshold in (1, 2, 4, 8, 12):
                expected_counts = dict(correct=int(np.count_nonzero(error <= threshold)), co_visible=int(mask.sum()), returned=len(query))
                if metrics["correspondence"][str(threshold)] != expected_counts: raise ValueError("Correspondence arithmetic differs")
        elif metrics["correspondence"]: raise ValueError("Nonoverlap may not claim positive correspondences")
        replayed.append(dict(case_id=c["case_id"], profile=row["profile"], track=row["track"], **checks))
    result = dict(passed=True, replayed_rows=len(replayed), rows=replayed, summary_sha256=sha(root / "summary.json"),
        replay_source_sha256=sha(__file__), scope="Independent saved-array/source-label/native-pixel arithmetic replay; no model inference or new scientific result")
    with Path(output).open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run"); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    a = parser.parse_args(); print(json.dumps(replay(a.run, a.protocol, a.output), indent=2))
