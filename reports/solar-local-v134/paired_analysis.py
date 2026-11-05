"""Descriptive paired uncertainty versus the protocol's framed Direct baseline.

Saved metrics only. Requires completed, sealed, independently replayable outputs.
No model selection, confidence fitting or new matcher inference is performed.
"""

import argparse
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("native_v134_saved_summary", HERE/"replay_saved.py")
replay = module_from_spec(spec); spec.loader.exec_module(replay)


def paired(scores, method, comparator):
    positive = [row for row in scores if row["case"]["positive"]]
    groups = sorted({row["case"]["component"] for row in positive})
    values = [int(row["methods"][method]["correct"])-int(row["methods"][comparator]["correct"])
              for row in positive]
    estimates = []
    for draw in product(groups, repeat=len(groups)):
        weights = Counter(draw)
        denominator = sum(weights[row["case"]["component"]] for row in positive)
        estimates.append(100*sum(weights[row["case"]["component"]]*difference
                                for row, difference in zip(positive, values, strict=True))/denominator)
    interval = [replay.saved.percentile(estimates, .025), replay.saved.percentile(estimates, .975)]
    return dict(comparator=comparator, positive_pairs=len(positive), terrain_components=groups,
                wins=sum(v > 0 for v in values), losses=sum(v < 0 for v in values),
                difference_percentage_points=100*sum(values)/len(positive),
                descriptive_95pct_component_interval_pp=interval,
                component_draws=len(estimates),
                positive_interval_lower_bound_above_zero=interval[0] > 0.)


def analyze():
    summary = replay.summarize()
    scores = replay.saved.metric.read("runs/solar-local-v134/comparison/scores.json")
    comparator = summary["principal_baseline"]
    return dict(schema="paired_reference_frame_native_results_v134", passed=True,
                protocol_sha256=summary["protocol_sha256"], predictions_sha256=summary["predictions_sha256"],
                principal_method=summary["principal_method"], principal_baseline=comparator,
                raw_not_calibrated=True, new_matcher_inference=False, reserve_access=False,
                comparisons={name: paired(scores, name, comparator) for name in summary["methods"]},
                limitations="Three exposed terrain components and repeated acquisition noise: intervals are descriptive and cannot establish a population improvement or 1% risk. Negative endpoint uncertainty is retained in results.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay.saved.metric.put(args.output, analyze())
