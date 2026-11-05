"""Replay sealed scores and summarize the matched development comparison; no inference."""

import argparse
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "solar-local-v127"))
spec = spec_from_file_location("saved_v127_metrics", HERE.parent / "solar-local-v127/experiment.py")
metric = module_from_spec(spec); spec.loader.exec_module(metric)
HOME = HERE.relative_to(ROOT)
RUN = Path("runs/solar-local-v129/comparison")


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    low = int(index)
    return ordered[low] + (ordered[min(low+1, len(ordered)-1)]-ordered[low])*(index-low)


def summarize():
    p = metric.read(HOME / "protocol.json")
    complete = metric.read(RUN / "COMPLETE.json")
    sealed = metric.read(RUN / "PREDICTIONS_SEALED.json")
    raw = metric.read(RUN / "predictions.json")
    scores = metric.read(RUN / "scores.json")
    metric.check(complete["complete"] and not (RUN / "FAILED.json").exists(), "Completed comparison required")
    metric.check(complete["protocol_sha256"] == raw["protocol_sha256"] == metric.sha(HOME / "protocol.json"), "Changed protocol")
    metric.check(sealed["sha256"] == metric.sha(RUN / "predictions.json") and sealed["rows"] == len(raw["rows"]) == 32, "Changed or incomplete predictions")
    metric.check(raw["query_truth_used"] is False and [r["case"] for r in raw["rows"]] == p["cases"], "Changed truth isolation or roster")
    for name, digest in {**p["sources"], **p["input_sha256"]}.items():
        metric.check(metric.sha(name) == digest, "Changed replay dependency: " + name)
    replay = [dict(case=r["case"], methods=metric.score_result(r["result"], r["case"], p["inputs"], p["labels"])) for r in raw["rows"]]
    metric.check(replay == scores, "Saved score replay failed")
    methods = tuple(raw["rows"][0]["result"]["methods"])
    metric.check(all(tuple(row["result"]["methods"]) == methods for row in raw["rows"]), "Incomplete method grid")
    positive = [i for i, row in enumerate(scores) if row["case"]["positive"]]
    negative = [i for i, row in enumerate(scores) if not row["case"]["positive"]]
    groups = sorted({scores[i]["case"]["component"] for i in positive})
    result = dict(schema="sealed_native_measurement_repair_results_v129", passed=True,
                  protocol_sha256=metric.sha(HOME / "protocol.json"), predictions_sha256=sealed["sha256"],
                  scores_sha256=metric.sha(RUN / "scores.json"), positive_pairs=len(positive), negative_pairs=len(negative),
                  terrain_components=groups, independent_camera_conditions=4, calibrated_acceptance_evaluated=False,
                  known_matching_illumination=True, calibration_or_final_access=False, methods={}, by_condition={})
    for name in methods:
        def count(key, indices):
            return sum(bool(scores[i]["methods"][name][key]) for i in indices)
        pairs = [raw["rows"][i]["result"]["methods"][name] for i in positive]
        failures = Counter(f["reason"] for row in pairs for f in row.get("failures", []))
        result["methods"][name] = dict(correct=count("correct", positive), strict=count("strict", positive),
            wrong_positive=count("wrong_positive", positive), raw_nonoverlap=count("raw_nonoverlap", negative),
            returned_positive_poses=count("pose_returned", positive), retained_positive_matches=sum(len(row["reference_xy"]) for row in pairs),
            native_two_pixel_matches=sum(scores[i]["methods"][name]["correspondence"]["2"]["correct"] for i in positive),
            median_positive_inference_seconds=statistics.median(row["inference_seconds"] for row in pairs),
            patch_failures=dict(sorted(failures.items())))
        wins = sum(scores[i]["methods"][name]["correct"] and not scores[i]["methods"]["direct"]["correct"] for i in positive)
        losses = sum(scores[i]["methods"]["direct"]["correct"] and not scores[i]["methods"][name]["correct"] for i in positive)
        differences = []
        endpoint_negative_rates = []
        empty_endpoint_resamples = 0
        for draw in product(groups, repeat=len(groups)):
            weights = Counter(draw)
            denominator = sum(weights[scores[i]["case"]["component"]] for i in positive)
            differences.append(100*sum(weights[scores[i]["case"]["component"]]*(
                int(scores[i]["methods"][name]["correct"])-int(scores[i]["methods"]["direct"]["correct"])) for i in positive)/denominator)
            edge_weights = [weights[scores[i]["case"]["component"]]*weights[scores[i]["case"]["query_component"]] for i in negative]
            if sum(edge_weights):
                endpoint_negative_rates.append(sum(w*scores[i]["methods"][name]["raw_nonoverlap"] for w, i in zip(edge_weights, negative, strict=True))/sum(edge_weights))
            else:
                empty_endpoint_resamples += 1
        row = result["methods"][name]
        row["paired_against_direct"] = dict(wins=wins, losses=losses,
            difference_percentage_points=100*(wins-losses)/len(positive),
            descriptive_95pct_component_bootstrap_interval_pp=[percentile(differences, .025), percentile(differences, .975)],
            enumerated_component_draws=len(differences),
            negative_endpoint_weighted_interval=[percentile(endpoint_negative_rates, .025), percentile(endpoint_negative_rates, .975)],
            negative_draws_with_no_represented_edges=empty_endpoint_resamples)
        row["pilot_improvement_rule_passed"] = name in ("halo", "pooled_uniform", "pooled_parallax") and wins-losses >= 2 and row["wrong_positive"] <= count_direct(scores, positive, "wrong_positive") and row["raw_nonoverlap"] <= count_direct(scores, negative, "raw_nonoverlap")
    for index, item in enumerate(p["inputs"]):
        chosen = [i for i in positive if scores[i]["case"]["reference_index"] == index]
        result["by_condition"][item["tag"]] = dict(case_id=p["labels"][index]["case_id"], purpose=p["labels"][index]["purpose"],
            pairs=len(chosen), methods={name: dict(correct=sum(scores[i]["methods"][name]["correct"] for i in chosen),
               returned_matches=sum(len(raw["rows"][i]["result"]["methods"][name]["reference_xy"]) for i in chosen)) for name in methods})
    result["uncertainty_scope"] = "Exact enumeration of the empirical three-component paired bootstrap; negatives weight both endpoints. Too few terrain groups for a reliable population interval or a 1% risk claim."
    result["latency_scope"] = "Actual reference-cache misses and hits in frozen case order; query work always rerun, complete coarse initialization and shared patch preparation included. These are development workload measurements."
    result["elapsed_seconds"] = complete["elapsed_seconds"]
    return result


def count_direct(scores, indices, key):
    return sum(bool(scores[i]["methods"]["direct"][key]) for i in indices)


if __name__ == "__main__":
    metric.check(Path.cwd() == ROOT, "Replay from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize()
    if args.output:
        metric.put(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
