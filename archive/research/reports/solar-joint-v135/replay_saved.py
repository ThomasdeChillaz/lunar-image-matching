"""Replay the bounded joint-image pilot and all cached controls without inference."""

import argparse
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("joint_v135_frozen_controller", HERE/"experiment.py")
controller = module_from_spec(spec); spec.loader.exec_module(controller)
base = controller.base
HOME, RUN = controller.HOME, controller.WORK/"comparison"


def percentile(values, fraction):
    ordered = sorted(values); at = (len(ordered)-1)*fraction; low = int(at)
    return ordered[low]+(ordered[min(low+1, len(ordered)-1)]-ordered[low])*(at-low)


def summarize():
    import numpy as np
    p = base.read(HOME/"protocol.json")
    complete, seal, raw, scores = [base.read(RUN/n) for n in
        ("COMPLETE.json", "PREDICTIONS_SEALED.json", "predictions.json", "scores.json")]
    base.check(complete["complete"] and not (RUN/"FAILED.json").exists()
        and complete["protocol_sha256"] == raw["protocol_sha256"] == base.sha(HOME/"protocol.json"),
        "Completed frozen comparison required")
    base.check(seal["sha256"] == base.sha(RUN/"predictions.json") and seal["rows"] == len(raw["rows"]) == 8
        and raw["query_truth_used"] is False and [r["case"] for r in raw["rows"]] == p["cases"], "Changed seal or grid")
    for name, digest in {**p["sources"], **p["input_sha256"]}.items():
        base.check(base.sha(name) == digest, "Changed dependency: "+name)
    replay = [dict(case=r["case"], methods=base.score_result(r["result"], r["case"], p["inputs"], p["labels"])) for r in raw["rows"]]
    base.check(replay == scores, "Saved metric replay failed")
    old = {controller.base_key(r["case"]): r for r in base.read(controller.CACHE)["rows"]}
    for row in raw["rows"]:
        before = old[controller.base_key(row["case"])]
        for name, method in before["result"]["methods"].items():
            base.check(row["result"]["methods"]["cached_"+name] == method, "Cached control changed")
        for key in ("reference_xy", "query_xy", "pose", "estimator"):
            base.check(row["result"]["methods"]["hard"][key] == before["result"]["methods"]["framed_native_terrain_subpixel"][key],
                       "Fresh hard control changed")
        field = row["result"]["native_correlation_fields"]
        base.check(base.sha(field["path"]) == field["sha256"], "Changed field archive")
        with np.load(field["path"], allow_pickle=False) as data:
            a, b = data["scores"], data["support"]; n = len(data["anchor_indices"])
            base.check(a.shape == b.shape == (n, 49, 49) and n <= 36 and b.dtype == bool
                       and np.isfinite(a).all() and (a[~b] == 0).all(), "Invalid field/support encoding")
    names = tuple(raw["rows"][0]["result"]["methods"])
    base.check(len(names) == 24 and all(tuple(r["result"]["methods"]) == names for r in raw["rows"]), "Incomplete methods")
    positive = [i for i,r in enumerate(scores) if r["case"]["positive"]]
    negative = [i for i,r in enumerate(scores) if not r["case"]["positive"]]
    groups = sorted({scores[i]["case"]["component"] for i in positive})
    reference = p["baseline"]
    result = dict(schema="joint_correlation_potential_results_v135", passed=True,
        protocol_sha256=base.sha(HOME/"protocol.json"), predictions_sha256=seal["sha256"],
        method_case_cells=192, positive_pairs=4, negative_pairs=4, terrain_components=groups,
        hard_control_exact=True, cached20_controls_exact=True, calibration_or_final_access=False,
        accepted_pose_study=False, complete_inference_latency_measured=False, methods={})
    for name in names:
        def count(key, indices):
            return sum(bool(scores[i]["methods"][name][key]) for i in indices)
        differences, rates, empty = [], [], 0
        for draw in product(groups, repeat=len(groups)):
            w = Counter(draw)
            denominator = sum(w[scores[i]["case"]["component"]] for i in positive)
            difference = sum(w[scores[i]["case"]["component"]]*(int(scores[i]["methods"][name]["correct"])-
                int(scores[i]["methods"][reference]["correct"])) for i in positive)
            differences.append(100*difference/denominator)
            edges = [w[scores[i]["case"]["component"]]*w[scores[i]["case"]["query_component"]] for i in negative]
            if sum(edges):
                rates.append(sum(weight*scores[i]["methods"][name]["raw_nonoverlap"]
                    for weight,i in zip(edges,negative,strict=True))/sum(edges))
            else:
                empty += 1
        record = dict(correct=count("correct",positive), strict=count("strict",positive),
            wrong_positive=count("wrong_positive",positive), raw_nonoverlap=count("raw_nonoverlap",negative),
            returned_positive_poses=count("pose_returned",positive),
            matches=sum(len(raw["rows"][i]["result"]["methods"][name]["reference_xy"]) for i in positive),
            native_one_pixel_matches=sum(scores[i]["methods"][name]["correspondence"]["1"]["correct"] for i in positive),
            native_two_pixel_matches=sum(scores[i]["methods"][name]["correspondence"]["2"]["correct"] for i in positive),
            paired_against=reference, paired_difference_pp=100*(count("correct",positive)-
                sum(scores[i]["methods"][reference]["correct"] for i in positive))/len(positive),
            descriptive_terrain_95pct_interval_pp=[percentile(differences,.025),percentile(differences,.975)],
            negative_endpoint_interval=[percentile(rates,.025),percentile(rates,.975)],
            negative_draws_without_edges=empty)
        if name in ("hard","joint","joint_camera_proposal"):
            record["median_conditional_stage_seconds"] = statistics.median(raw["rows"][i]["result"]["methods"][name]["conditional_inference_seconds"] for i in positive)
        result["methods"][name] = record
    controls = [n for n in names if n.startswith("cached_") and n != "cached_camera_proposal"]+["hard"]
    strongest = max(result["methods"][n]["correct"] for n in controls)
    candidate, direct = result["methods"]["joint"],result["methods"][reference]
    result["preliminary_signal"] = candidate["correct"] >= strongest+1 and candidate["wrong_positive"] <= direct["wrong_positive"] and candidate["raw_nonoverlap"] <= direct["raw_nonoverlap"]
    result["strongest_completed_control_correct"] = strongest
    result["by_condition"] = {}
    for index in range(4):
        selected = [i for i in positive if scores[i]["case"]["reference_index"] == index]
        result["by_condition"][str(index)] = {n:dict(correct=sum(scores[i]["methods"][n]["correct"] for i in selected),
            wrong=sum(scores[i]["methods"][n]["wrong_positive"] for i in selected)) for n in ("hard","joint",reference)}
    result["interpretation_limits"] = "Four consumed positive conditions on three terrains, matching light supplied, cached predicted initializer. Conditional grid correspondences are not independent landmarks. No1% risk, full-runtime, learned-gain, novelty or final-test claim."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); base.put(args.output, summarize())
