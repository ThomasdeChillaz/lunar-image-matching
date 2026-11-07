"""Independent saved-geometry and branch-accounting replay; no matcher imported."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("direct", "legacy_no_extra", "legacy_active32", "single_uniform128",
           "beam_uniform128", "beam_geometry128", "beam_active128")
NEW = METHODS[3:]


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_accounting(result):
    hypotheses = result["initialization"]["photometric_hypotheses"]
    all_ids = [h["branch_id"] for h in hypotheses]
    lights = {h["branch_id"]: h["light_id"] for h in hypotheses}
    require(len(all_ids) == len(set(all_ids)) <= 8, "Bounded distinct camera/light hypotheses required")
    cap = result["common_effective_native_fields"]
    if all_ids:
        require(cap == (min(128, result["shared_reference_candidate_capacity"])//len(all_ids))*len(all_ids),
                "Common complete-round field budget differs")
    for name in NEW:
        row = result["methods"][name]
        acquisition = row["acquisition"]
        if not all_ids:
            require(row["pose"] is None and acquisition["attempted_native_fields"] == 0,
                    "Missing initializer cannot create a pose")
            continue
        ids = all_ids[:1] if name == "single_uniform128" else all_ids
        require(acquisition["retained_branch_ids"] == ids and acquisition["pruning"] is False
                and acquisition["requested_native_fields"] == cap
                and acquisition["query_fields_before_selection"] == 0,
                "Branch retention or native-field budget changed")
        selected, trace = acquisition["selected_indices"], acquisition["trace"]
        require(len(set(selected)) == len(selected) == len(trace) == acquisition["rounds"],
                "Repeated or unrecorded patch acquisition")
        require(acquisition["attempted_native_fields"] == len(selected)*len(ids) == cap,
                "Every branch-specific attempted field must be charged")
        for i, step in enumerate(trace):
            require(step["index"] == selected[i] and [b["branch_id"] for b in step["branches"]] == ids,
                    "Branch dropped during patch selection")
            require(all(b["light_id"] == lights[b["branch_id"]] and b["attempted_fields"] == i+1 for b in step["branches"]),
                    "Shared light changed or failed search was uncharged")
            for key in ("weights_before", "weights_after"):
                w = np.asarray(step[key], float)
                require(w.shape == (len(ids),) and np.isfinite(w).all() and (w > 0).all() and np.isclose(w.sum(), 1.),
                        "Provisional branch weights cannot prune an alternative")
        candidates = row["branch_candidates"]
        require({c["branch_id"] for c in candidates} == set(ids)
                and all(c["light_id"] == lights[c["branch_id"]] for c in candidates),
                "Final geometry must evaluate all retained shared-light alternatives")
        chosen = next(c for c in candidates if c["branch_id"] == row["branch_id"])
        require(all(row[key] == chosen[key] for key in ("reference_xy", "query_xy", "pose", "estimator", "light_id")),
                "Final output mixes branches")


def verify(protocol, checked, read):
    path = ROOT/"reports/solar-active-v137/replay_saved.py"
    spec = spec_from_file_location("independent_beam_geometry_v143", path)
    independent = module_from_spec(spec); spec.loader.exec_module(independent)
    inputs = independent.read(checked(protocol["inference_manifest"]))
    labels = independent.read(checked(protocol["scoring_manifest"]))
    raw, scores = read("predictions.json"), read("scores.json")
    historical = independent.read(checked(protocol["historical_predictions"]))
    require(len(raw["rows"]) == len(scores) == len(inputs) == len(labels) == 8
            and sum(l["positive"] for l in labels) == 4
            and raw["query_truth_used"] is False and raw["accepted_results"] is False,
            "Complete unchanged eight-case uncalibrated study required")
    counts = {n: dict(correct=0, strict=0, wrong_positive=0, raw_nonoverlap=0) for n in METHODS}
    cells, identity = 0, 0
    for record, scored, inp, label, old in zip(raw["rows"], scores, inputs, labels, historical["rows"], strict=True):
        require(record["id"] == scored["id"] == inp["id"] == label["id"] == old["id"], "Case order differs")
        require(tuple(record["result"]["methods"]) == METHODS, "Seven fixed methods required")
        verify_accounting(record["result"])
        for name, previous in (("direct", "direct"), ("legacy_no_extra", "joint_no_extra"), ("legacy_active32", "active")):
            for key in ("reference_xy", "query_xy", "pose", "estimator"):
                require(record["result"]["methods"][name][key] == old["result"]["methods"][previous][key],
                        "Frozen baseline decision changed: " + name + "/" + key)
            identity += 1
        geometry = None
        if label["positive"]:
            with np.load(checked(inp["reference_geometry"]["path"]), allow_pickle=False) as a:
                da, ka, ca = (a[n].copy() for n in ("axial_depth", "intrinsics", "cam2world"))
            with np.load(checked(label["query_geometry"]["path"]), allow_pickle=False) as b:
                db, kb, cb = (b[n].copy() for n in ("axial_depth", "intrinsics", "cam2world"))
            geometry = da, db, ka, ca, kb, cb
        for name, row in record["result"]["methods"].items():
            for key in ("reference_xy", "query_xy"):
                points = np.asarray(row[key]).reshape(-1, 2)
                require(np.isfinite(points).all() and ((points >= 0) & (points <= 511)).all(), "Invalid native coordinates")
            actual, expected = independent.arithmetic(row, label["positive"], geometry), scored["methods"][name]
            for key in ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap", "correspondence"):
                require(actual[key] == expected[key], "Independent metric differs: " + key)
            if actual["pose_error"] is not None:
                for key, value in actual["pose_error"].items():
                    require(abs(value-expected["pose_error"][key]) < 1e-5, "Pose arithmetic differs: " + key)
            sun = row["estimated_catalogue_sun_proposal"]
            if sun is not None and label["positive"]:
                a, b = np.asarray(sun), np.asarray(label["query_sun_scoring_only"])
                angle = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))
                require(abs(angle-expected["sun_proposal_error_deg"]) < 1e-5, "Sun arithmetic differs")
            else:
                require(expected["sun_proposal_error_deg"] is None, "Unexpected Sun error")
            for key in counts[name]:
                counts[name][key] += int(actual[key])
            cells += 1
    return dict(passed=True, method_case_cells=cells, methods=counts,
        positive_pairs=4, nonoverlap_pairs=4, historical_control_identities=identity,
        independent_geometry_arithmetic=True, branch_retention_and_field_accounting=True,
        accepted_pose_study=False, calibration_or_final_access=False)
