"""Read-only independent geometry replay; no matcher or launch authority imported."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np

from analyze_saved import METHODS, ROOT, RUN, OUT, digest, read, summarize


def require(value, message):
    if not value:
        raise ValueError(message)


def compare_score(actual, expected):
    for key in ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap", "correspondence"):
        require(actual[key] == expected[key], "Independent arithmetic differs: " + key)
    require((actual["pose_error"] is None) == (expected["pose_error"] is None), "Pose error presence differs")
    if actual["pose_error"] is not None:
        for key, value in actual["pose_error"].items():
            require(abs(value-expected["pose_error"][key]) < 1e-5, "Pose arithmetic differs: " + key)


def replay():
    protocol = read(ROOT/"reports/solar-blind-v140/protocol.json")
    for name, expected in {**protocol["sources"], **protocol["inputs"]}.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                "Missing or changed frozen dependency: " + name)
    report = read(OUT/"results.json")
    for name, expected in report["saved_evidence_sha256"].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and digest(path) == expected,
                "Changed saved evidence: " + name)
    require(report["source_protocol_sha256"] == digest(ROOT/"reports/solar-blind-v140/protocol.json"),
            "Protocol changed")
    raw, scores = read(RUN/"predictions.json"), read(RUN/"scores.json")
    require(read(RUN/"PREDICTIONS_SEALED.json") == dict(rows=8, sha256=digest(RUN/"predictions.json")),
            "Prediction seal differs")
    inputs, labels = read(ROOT/protocol["inference_manifest"]), read(ROOT/protocol["scoring_manifest"])
    path = ROOT/"reports/solar-active-v137/replay_saved.py"
    spec = spec_from_file_location("independent_native_geometry_v142", path)
    arithmetic = module_from_spec(spec); spec.loader.exec_module(arithmetic)
    cells = 0
    for row, scored, inp, label in zip(raw["rows"], scores, inputs, labels, strict=True):
        require(row["id"] == scored["id"] == inp["id"] == label["id"], "Case order differs")
        geometry = None
        if label["positive"]:
            with np.load(ROOT/inp["reference_geometry"]["path"], allow_pickle=False) as a:
                da, ka, ca = (a[n].copy() for n in ("axial_depth", "intrinsics", "cam2world"))
            with np.load(ROOT/label["query_geometry"]["path"], allow_pickle=False) as b:
                db, kb, cb = (b[n].copy() for n in ("axial_depth", "intrinsics", "cam2world"))
            geometry = da, db, ka, ca, kb, cb
        acquisition = [row["result"]["methods"][name].get("acquisition")
                       for name in ("uniform", "appearance", "geometry", "active")]
        if any(a is None for a in acquisition):
            require(all(a is None for a in acquisition) and not row["result"]["initialization"]["photometric_hypotheses"],
                    "Missing acquisition despite a valid initializer")
        else:
            require(len({a["attempted_extra_patches"] for a in acquisition}) == 1 and
                    len({tuple(a["cell_schedule"]) for a in acquisition}) == 1,
                    "Unequal acquisition budget or spatial slots")
            for a in acquisition:
                require(a["requested_extra_patches"] == 32 and 0 <= a["attempted_extra_patches"] <= 32 and
                        len(set(a["selected_indices"])) == len(a["selected_indices"]) == a["attempted_extra_patches"],
                        "Changed or duplicated acquisition searches")
        for name in METHODS:
            method, expected = row["result"]["methods"][name], scored["methods"][name]
            for field in ("reference_xy", "query_xy"):
                points = np.asarray(method[field], float).reshape(-1, 2)
                require(np.isfinite(points).all() and ((points >= 0) & (points <= 511)).all(), "Invalid native coordinates")
            compare_score(arithmetic.arithmetic(method, label["positive"], geometry), expected)
            sun = method["estimated_catalogue_sun_proposal"]
            if sun is not None and label["positive"]:
                a, b = np.asarray(sun), np.asarray(label["query_sun_scoring_only"])
                angle = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))
                require(abs(angle-expected["sun_proposal_error_deg"]) < 1e-5, "Sun proposal arithmetic differs")
            else:
                require(expected["sun_proposal_error_deg"] is None, "Unexpected Sun error")
            cells += 1
    recomputed = summarize(raw, scores, labels)
    for key, value in recomputed.items():
        require(value == report[key], "Summary or terrain analysis differs: " + key)
    receipt = read(OUT/"completion-receipt.json")
    require(receipt["complete"] is True and receipt["fixed_charge_seconds"] == 900 and
            receipt["protocol_sha256"] == report["source_protocol_sha256"], "Completion scope differs")
    return dict(passed=True, method_case_cells=cells, sources=len(protocol["sources"]), inputs=len(protocol["inputs"]),
        independently_recomputed_geometry=True, summary_and_component_analysis_exact=True,
        no_matching_or_fitting=True, private_authority_used=False, accepted_pose_study=False)


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
