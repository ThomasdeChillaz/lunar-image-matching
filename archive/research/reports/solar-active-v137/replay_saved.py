"""Independent arithmetic and terrain-group analysis of the sealed six-arm study.

No matcher, pose estimator, controller, private journal or launch authority is
imported. True query geometry is used only to score saved predictions.
"""

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import statistics

import cv2
import numpy as np

HOME = Path("reports/solar-active-v137")
RUN = Path("runs/solar-active-v137/comparison")
THRESHOLDS = (1, 2, 4, 8, 12)
FRESH = ("direct", "joint_no_extra", "uniform", "appearance", "geometry", "active")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_range(depth, k):
    y, x = np.indices(depth.shape)
    rays = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]@np.linalg.inv(k).T
    good = np.isfinite(depth.ravel()) & (depth.ravel() > 0)
    return float(np.median(depth.ravel()[good]*np.linalg.norm(rays[good], axis=1)))


def visible_targets(points, fields):
    da, db, ka, ca, kb, cb = fields
    if not len(points):
        return np.empty((0, 2)), np.zeros(0, bool)
    pixel = np.rint(points).astype(int)
    good = ((pixel >= 0) & (pixel < da.shape[::-1])).all(axis=1)
    at = np.clip(pixel, [0, 0], np.array(da.shape[::-1])-1)
    z = da[at[:, 1], at[:, 0]]
    good &= np.isfinite(z) & (z > 0)
    rays = np.c_[points, np.ones(len(points))]@np.linalg.inv(ka).T
    world = (rays*np.where(good, z, 0.)[:, None])@ca[:3, :3].T+ca[:3, 3]
    camera = (world-cb[:3, 3])@cb[:3, :3]
    xyh = camera@kb.T
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyh[:, :2]/xyh[:, 2, None]
    good &= (camera[:, 2] > 0) & np.isfinite(xy).all(axis=1)
    good &= ((xy >= 0) & (xy < np.array(db.shape[::-1])-1)).all(axis=1)
    sensor = np.where(np.isfinite(xy), xy, -1.).astype(np.float32)
    sampled = cv2.remap(np.where(np.isfinite(db), db, 0.).astype(np.float32),
                        sensor[:, 0, None], sensor[:, 1, None], cv2.INTER_LINEAR).ravel()
    coverage = cv2.remap((np.isfinite(db) & (db > 0)).astype(np.float32),
                         sensor[:, 0, None], sensor[:, 1, None], cv2.INTER_LINEAR).ravel()
    good &= coverage >= .9999
    good &= np.abs(sampled-camera[:, 2]) <= .05+.01*camera[:, 2]
    return xy[good], good


def arithmetic(method, positive, fields=None):
    p, q = (np.asarray(method[name], float).reshape(-1, 2) for name in ("reference_xy", "query_xy"))
    require(p.shape == q.shape and np.isfinite(p).all() and np.isfinite(q).all(), "Invalid native endpoints")
    returned = method["pose"] is not None
    correct = strict = False; error = None; counts = {}
    if positive:
        target, visible = visible_targets(p, fields)
        residual = np.linalg.norm(q[visible]-target, axis=1)
        counts = {str(t):dict(correct=int((residual <= t).sum()), visible=int(visible.sum()), returned=len(p)) for t in THRESHOLDS}
        distance = reference_range(fields[0], fields[2])
        if returned:
            camera = np.asarray(method["pose"], float); truth = fields[-1]
            require(camera.shape == (4, 4) and np.isfinite(camera).all(), "Invalid saved camera")
            # Independent trace/arccos implementation, rather than the scorer's
            # skew/atan2 expression. Numerical angle differences are audited.
            relative = camera[:3, :3].T@truth[:3, :3]
            angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1.)/2., -1., 1.))))
            shift = float(np.linalg.norm(camera[:3, 3]-truth[:3, 3]))
            correct, strict = angle <= 2. and shift <= .01*distance, angle <= .5 and shift <= .001*distance
            error = dict(rotation_deg=angle, translation_m=shift, reference_range_m=distance)
    return dict(correct=bool(correct), strict=bool(strict), pose_returned=returned,
                wrong_positive=positive and returned and not correct,
                raw_nonoverlap=not positive and returned, correspondence=counts, pose_error=error)


def key(case):
    return tuple(case[n] for n in ("reference_index", "query_index", "reference_seed", "query_seed", "positive"))


def interval(values):
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def summarize():
    protocol = read(HOME/"protocol.json")
    complete, seal, raw, saved = (read(RUN/name) for name in ("COMPLETE.json", "PREDICTIONS_SEALED.json", "predictions.json", "scores.json"))
    require(complete["complete"] and not (RUN/"FAILED.json").exists(), "Complete successful study required")
    require(complete["protocol_sha256"] == raw["protocol_sha256"] == sha(HOME/"protocol.json")
        and seal["sha256"] == sha(RUN/"predictions.json") and seal["rows"] == len(raw["rows"]) == 32
        and raw["query_truth_used"] is False and [r["case"] for r in raw["rows"]] == protocol["cases"],
        "Changed seal, source or incomplete grid")
    for name, digest in {**protocol["sources"], **protocol["input_sha256"]}.items():
        require(sha(name) == digest, "Changed replay dependency: "+name)
    old = {key(r["case"]):r for r in read(protocol["cached_predictions"])["rows"]}
    joint = {key(r["case"]):r for r in read(protocol["joint_pilot_predictions"])["rows"]}
    names = tuple(raw["rows"][0]["result"]["methods"])
    require(names[:6] == FRESH and len(names) == 26, "Missing fresh/historical methods")
    angle_difference = 0.; comparisons = 0; joint_checks = 0
    for row, scored in zip(raw["rows"], saved, strict=True):
        case = row["case"]
        require(scored["case"] == case and tuple(row["result"]["methods"]) == names, "Method/case grid changed")
        before = old[key(case)]
        require(row["fresh_initializer"] == before["result"]["photometric_initialization"], "Fresh initializer control changed")
        for field in ("reference_xy", "query_xy", "pose", "estimator"):
            require(row["result"]["methods"]["direct"][field] == before["result"]["methods"]["framed_direct"][field], "Fresh Direct changed")
            if key(case) in joint:
                require(row["result"]["methods"]["joint_no_extra"][field] == joint[key(case)]["result"]["methods"]["joint"][field], "No-extra control changed")
        joint_checks += key(case) in joint
        for name, method in before["result"]["methods"].items():
            require(row["result"]["methods"]["cached_"+name] == method, "Historical control changed")
        schedules, attempted = [], []
        for name in ("uniform", "appearance", "geometry", "active"):
            method = row["result"]["methods"][name]
            acquisition = method.get("acquisition")
            if acquisition is None:
                require(not row["fresh_initializer"] and method["pose"] is None,
                        "Acquisition trace missing despite an initializer")
                attempted.append(0); schedules.append(())
            else:
                selected = acquisition["selected_indices"]
                require(len(selected) == len(set(selected)) == acquisition["attempted_extra_patches"]
                        and len(selected) <= 32 and acquisition["requested_extra_patches"] == 32,
                        "Changed, duplicated or uncharged patch searches")
                attempted.append(len(selected)); schedules.append(tuple(acquisition["cell_schedule"]))
        require(len(set(attempted)) == len(set(schedules)) == 1,
                "Acquisition policies did not receive equal budgets and spatial slots")
        fields = None
        if case["positive"]:
            label = protocol["labels"][case["reference_index"]]["joint_scoring_geometry"]
            require(sha(label["path"]) == label["sha256"], "Scoring labels changed")
            with np.load(label["path"], allow_pickle=False) as data:
                fields = [data[n].copy() for n in ("reference_depth", "query_depth", "reference_intrinsics", "reference_cam2world", "query_intrinsics", "query_cam2world")]
        for name in names:
            value = arithmetic(row["result"]["methods"][name], case["positive"], fields)
            expected = scored["methods"][name]
            for field in ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap", "correspondence"):
                require(value[field] == expected[field], "Independent metric decision differs: "+name+"/"+field)
            if value["pose_error"] is not None:
                for field, tolerance in (("rotation_deg", 1e-5), ("translation_m", 1e-6), ("reference_range_m", 1e-6)):
                    difference = abs(value["pose_error"][field]-expected["pose_error"][field])
                    require(difference <= tolerance, "Independent pose arithmetic differs")
                    if field == "rotation_deg": angle_difference = max(angle_difference, difference)
            comparisons += 1
    positives = [i for i,r in enumerate(saved) if r["case"]["positive"]]
    negatives = [i for i,r in enumerate(saved) if not r["case"]["positive"]]
    groups = sorted({saved[i]["case"]["component"] for i in positives})
    active_correct = sum(saved[i]["methods"]["active"]["correct"] for i in positives)
    result = dict(schema="sequential_acquisition_results_v137", passed=True, method_case_cells=comparisons,
        positive_pairs=len(positives), nonoverlap_pairs=len(negatives), terrain_components=groups,
        independent_arithmetic=True, maximum_angle_formula_difference_deg=angle_difference,
        all20_historical_controls_exact=True, fresh_initializer_and_direct_exact=True,
        no_extra_joint_pilot_rows_exact=joint_checks, accepted_pose_study=False,
        calibration_or_final_access=False, methods={}, comparisons={}, by_condition={})
    for name in names:
        def count(field, indices): return sum(bool(saved[i]["methods"][name][field]) for i in indices)
        record = dict(correct=count("correct", positives), strict=count("strict", positives),
            wrong_positive=count("wrong_positive", positives), raw_nonoverlap=count("raw_nonoverlap", negatives),
            returned_positive=count("pose_returned", positives), matches=sum(len(raw["rows"][i]["result"]["methods"][name]["reference_xy"]) for i in positives),
            one_pixel_matches=sum(saved[i]["methods"][name]["correspondence"]["1"]["correct"] for i in positives),
            two_pixel_matches=sum(saved[i]["methods"][name]["correspondence"]["2"]["correct"] for i in positives),
            timing_scope="fresh_online" if name in FRESH else "historical_not_retimed")
        if name in FRESH:
            record["median_positive_online_seconds"] = statistics.median(raw["rows"][i]["result"]["methods"][name]["online_seconds"] for i in positives)
            record["median_all_online_seconds"] = statistics.median(row["result"]["methods"][name]["online_seconds"] for row in raw["rows"])
        result["methods"][name] = record
        if name != "active":
            differences, negative_rates, empty = [], [], 0
            for draw in product(groups, repeat=len(groups)):
                w = Counter(draw)
                denominator = sum(w[saved[i]["case"]["component"]] for i in positives)
                gain = sum(w[saved[i]["case"]["component"]]*(int(saved[i]["methods"]["active"]["correct"])-int(saved[i]["methods"][name]["correct"])) for i in positives)
                differences.append(100*gain/denominator)
                edge = [w[saved[i]["case"]["component"]]*w[saved[i]["case"]["query_component"]] for i in negatives]
                if sum(edge): negative_rates.append(sum(weight*saved[i]["methods"]["active"]["raw_nonoverlap"] for weight,i in zip(edge, negatives, strict=True))/sum(edge))
                else: empty += 1
            result["comparisons"][name] = dict(active_minus_control_pp=100*(active_correct-record["correct"])/len(positives),
                descriptive_terrain95_interval_pp=interval(differences), active_nonoverlap_endpoint_interval=interval(negative_rates), negative_draws_without_edges=empty)
    controls = [n for n in names if n.startswith("cached_")]+["direct", "joint_no_extra"]
    strongest = max(result["methods"][n]["correct"] for n in controls)
    active, direct = result["methods"]["active"], result["methods"]["direct"]
    result["strongest_completed_control_correct"] = strongest
    result["preliminary_signal"] = (active["correct"] >= strongest+2 and active["wrong_positive"] <= direct["wrong_positive"]
        and active["raw_nonoverlap"] <= direct["raw_nonoverlap"]
        and all(active["correct"] > result["methods"][n]["correct"] for n in ("uniform", "appearance", "geometry")))
    for index in range(4):
        ids = [i for i in positives if saved[i]["case"]["reference_index"] == index]
        result["by_condition"][str(index)] = {n:dict(correct=sum(saved[i]["methods"][n]["correct"] for i in ids), wrong=sum(saved[i]["methods"][n]["wrong_positive"] for i in ids)) for n in FRESH}
    result["limitations"] = "Four exposed known-light acquisition conditions on three terrains. Repeated noise is not new geography. Raw results are uncalibrated; source/worker memory and cache preparation are not per-model memory rankings. No novelty, orbital, final-test or1% deployment-risk claim."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
