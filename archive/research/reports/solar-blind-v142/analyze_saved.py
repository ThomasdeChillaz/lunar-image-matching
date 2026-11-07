"""Summarize sealed v140 outputs without running any matcher or fitting policy."""
from collections import Counter
from itertools import product
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/solar-blind-v140"
OUT = Path(__file__).resolve().parent
METHODS = ("direct", "joint_no_extra", "uniform", "appearance", "geometry", "active")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    """Linear empirical quantile, including singleton and endpoint cases."""
    values = sorted(values)
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Nonempty finite sample and unit-interval quantile required")
    position = fraction * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def positive_interval(labels, scores, method, control):
    indices = [i for i, label in enumerate(labels) if label["positive"]]
    groups = sorted({labels[i]["component"] for i in indices})
    values = []
    for draw in product(groups, repeat=len(groups)):
        weights = Counter(draw)
        denominator = sum(weights[labels[i]["component"]] for i in indices)
        numerator = sum(weights[labels[i]["component"]] *
            (int(scores[i]["methods"][method]["correct"]) -
             int(scores[i]["methods"][control]["correct"])) for i in indices)
        values.append(100 * numerator / denominator)
    return dict(groups=groups, enumerated_draws=len(values),
        interval_pp=[quantile(values, .025), quantile(values, .975)],
        interpretation="Descriptive component bootstrap; three groups cannot establish population reliability")


def negative_endpoint_interval(labels, scores, method):
    indices = [i for i, label in enumerate(labels) if not label["positive"]]
    groups = sorted({labels[i][field] for i in indices for field in ("component", "query_component")})
    values, empty = [], 0
    for draw in product(groups, repeat=len(groups)):
        multiplicity = Counter(draw)
        weights = [multiplicity[labels[i]["component"]] *
                   multiplicity[labels[i]["query_component"]] for i in indices]
        denominator = sum(weights)
        if not denominator:
            empty += 1
            continue
        numerator = sum(w * int(scores[i]["methods"][method]["raw_nonoverlap"])
                        for i, w in zip(indices, weights, strict=True))
        values.append(100 * numerator / denominator)
    return dict(groups=groups, enumerated_draws=len(values) + empty,
        undefined_no_edge_draws=empty,
        interval_percent=None if not values else [quantile(values, .025), quantile(values, .975)],
        interpretation="Both terrain endpoints weighted; empirical interval is not an upper deployment-risk bound")


def summarize(raw, scores, labels):
    if (len(raw["rows"]) != len(scores) or len(scores) != len(labels)
        or len(labels) != 8 or sum(r["positive"] for r in labels) != 4
        or raw["query_truth_used"] is not False or raw["accepted_results"] is not False):
        raise ValueError("Complete eight-case truth-free uncalibrated study required")
    for record, scored, label in zip(raw["rows"], scores, labels, strict=True):
        if not record["id"] == scored["id"] == label["id"]:
            raise ValueError("Case order changed")
        if tuple(record["result"]["methods"]) != METHODS or tuple(scored["methods"]) != METHODS:
            raise ValueError("All six methods in frozen order required")
    positive = [i for i, label in enumerate(labels) if label["positive"]]
    output = {}
    for method in METHODS:
        records = [s["methods"][method] for s in scores]
        times = [raw["rows"][i]["result"]["methods"][method]["online_seconds"] for i in positive]
        angles = [records[i]["sun_proposal_error_deg"] for i in positive
                  if records[i]["sun_proposal_error_deg"] is not None]
        output[method] = dict(
            **{field: sum(int(r[field]) for r in records)
               for field in ("correct", "strict", "wrong_positive", "raw_nonoverlap")},
            positive_pairs=len(positive), nonoverlap_pairs=len(labels)-len(positive),
            median_positive_online_seconds=median(times),
            positive_online_seconds=times,
            native_correspondences={str(threshold): {
                field: sum(records[i]["correspondence"][str(threshold)][field] for i in positive)
                for field in ("correct", "visible", "returned")}
                for threshold in (1, 2, 4, 8, 12)},
            sun_proposals=len(angles),
            median_sun_proposal_error_deg=None if not angles else median(angles))
    comparisons = {}
    for control in METHODS[:-1]:
        comparisons[control] = dict(
            paired_gain_pp=100*(output["active"]["correct"]-output[control]["correct"])/len(positive),
            paired_component_interval=positive_interval(labels, scores, "active", control))
    return dict(schema="sealed_unknown_light_summary_v142", methods=output,
        active_minus_control=comparisons,
        active_negative_endpoint_interval=negative_endpoint_interval(labels, scores, "active"),
        positive_terrain_components=sorted({labels[i]["component"] for i in positive}),
        case_results=[dict(id=label["id"], source_case=label["source_case"], positive=label["positive"],
            reference_component=label["component"], query_component=label["query_component"],
            methods=scored["methods"]) for label, scored in zip(labels, scores, strict=True)],
        accepted_pose_study=False, new_trainable_parameters=0,
        query_sun_input=False, continuous_sun_estimation=False,
        source="Frozen v140 six-method eight-case experiment; discrete reference light catalogue",
        claim_limits="Exposed development terrain; no calibrated acceptance, fresh test, learned gain or modern-model ranking")


def main():
    protocol = read(ROOT/"reports/solar-blind-v140/protocol.json")
    seal, raw = read(RUN/"PREDICTIONS_SEALED.json"), read(RUN/"predictions.json")
    replay = read(RUN/"replay.json")
    if (seal != dict(rows=8, sha256=digest(RUN/"predictions.json")) or
        replay["passed"] is not True or replay["method_case_cells"] != 48 or
        read(RUN/"COMPLETE.json")["complete"] is not True):
        raise ValueError("Successful complete, sealed and independently replayed run required")
    result = summarize(raw, read(RUN/"scores.json"), read(ROOT/protocol["scoring_manifest"]))
    for method in METHODS:
        for key in ("correct", "strict", "wrong_positive", "raw_nonoverlap"):
            if result["methods"][method][key] != replay["methods"][method][key]:
                raise ValueError("Summary disagrees with independent replay")
    paths = [RUN/name for name in ("predictions.json", "PREDICTIONS_SEALED.json", "scores.json",
        "replay.json", "timing.json", "timing-runtime.json", "predict-runtime.json",
        "score-runtime.json", "replay-runtime.json")]
    result["saved_evidence_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result["source_protocol_sha256"] = digest(ROOT/"reports/solar-blind-v140/protocol.json")
    result["runtime"] = {operation: read(RUN/(operation+"-runtime.json"))
                         for operation in ("timing", "predict", "score", "replay")}
    result["timing"] = read(RUN/"timing.json")
    result["actual_controller_elapsed_seconds"] = read(RUN/"COMPLETE.json")["elapsed_seconds"]
    result["fixed_comparison_charge_seconds"] = 900
    (OUT/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({name: {key: record[key] for key in
        ("correct", "strict", "wrong_positive", "raw_nonoverlap", "median_positive_online_seconds")}
        for name, record in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
