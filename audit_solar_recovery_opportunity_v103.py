"""Join frozen pilot cases to previously saved development outcomes."""

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

from audit_supplied_sun_support_v102 import verified_json
from run_solar_radiometric_pilot_v101 import POSITIVE_IDS


SCARCITY = "reports/solar_match_scarcity_v102.json"
SCARCITY_SHA = "f7c92864ef78666e5362597f396e192290d2acfd3fe18fc1cf6988c66ff7fbec"
REPLAY = "reports/solar_full_crosslight_rescue_replay_v99.json"
REPLAY_SHA = "8347b60efa4d5f3721f2e118087475ec791339c595f43e2cd204b2d678312182"


def summarize(rows):
    return dict(
        distinct_conditions=len(rows),
        scheduled_rows=sum(len(row["aliases"]) for row in rows),
        terrain_groups=dict(sorted(Counter(row["terrain"] for row in rows).items())),
    )


def audit():
    conditions = verified_json(SCARCITY, SCARCITY_SHA)["rows"]
    trace = verified_json(REPLAY, REPLAY_SHA)["paired_mechanism_trace"]
    alias = {case: row for row in conditions for case in row["aliases"]}
    positive = {row["case_id"]: row for row in trace if row["positive"]}
    if (len(conditions) != 128 or len(alias) != 192 or alias.keys() != positive.keys()
            or any(row["correct"] != positive[case]["direct_correct"] for case, row in alias.items())):
        raise ValueError("The deduplicated rows must match the exact direct replay")

    pilot = [alias[case] for case in POSITIVE_IDS]
    if len(pilot) != 24 or len({row["query"] for row in pilot}) != 24:
        raise ValueError("Require 24 distinct frozen positive pilot conditions")
    pilot_scarce = [row for row in pilot if row["best_saved_candidate"] < 6]
    pilot_dark_zero = [row for row in pilot_scarce if row["query_p95"] <= 3 and row["best_saved_candidate"] == 0]
    dark_misses = [
        row for row in conditions
        if not row["correct"] and max(row["query_p95"], row["exact_template_p95"]) <= 3
    ]
    correct = sum(row["direct_correct"] for row in positive.values())
    target = math.ceil(.70 * len(positive))
    dark_schedule = summarize(dark_misses)["scheduled_rows"]

    negatives = [row for row in trace if not row["positive"]]
    edges = Counter(tuple(sorted((row["component"], row["query_component"]))) for row in negatives)
    if len(negatives) != 192 or any(a == b for a, b in edges):
        raise ValueError("Require the exact inter-terrain nonoverlap cohort")
    endpoint_nodes = {item for edge in edges for item in edge}
    disjoint = max(
        n for n in range(1, 1 + len(endpoint_nodes)//2)
        if any(len({item for edge in group for item in edge}) == 2*n
               for group in combinations(edges, n))
    )

    return dict(
        scope="Saved development records only. No new image access, inference, acceptance, or cohort selection.",
        sources={SCARCITY: SCARCITY_SHA, REPLAY: REPLAY_SHA},
        positive_pilot=dict(
            frozen_cases=len(pilot), distinct_scarce=summarize(pilot_scarce)["distinct_conditions"],
            scarce_terrain_groups=summarize(pilot_scarce)["terrain_groups"],
            scarce_geometrically_available=sum(row["valid_patch_centres"] >= 230 for row in pilot_scarce),
            zero_matches_query_p95_at_most_3=len(pilot_dark_zero),
            net_correct_for_10_point_pilot_signal=math.ceil(.10 * len(pilot)),
        ),
        full_24_positive=dict(
            eligible=len(positive), raw_correct=correct, minimum_correct_for_70_percent=target,
            shortfall=target-correct, both_images_p95_at_most_3_misses=summarize(dark_misses),
            required_fraction_of_these_scheduled_misses_if_no_other_change=(target-correct)/dark_schedule,
        ),
        nonoverlap=dict(
            raw_poses=sum(row["direct_raw_nonoverlap"] for row in negatives),
            scheduled_rows=len(negatives), terrain_endpoints=len(endpoint_nodes),
            unordered_endpoint_edges=[dict(terrain_a=a, terrain_b=b, rows=n) for (a,b), n in sorted(edges.items())],
            maximum_terrain_disjoint_edges=disjoint,
            scope="Terrain-disjoint endpoint sets are descriptive and do not certify independent errors.",
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    with args.output.open("x") as stream:
        json.dump(audit(), stream, indent=2)
        stream.write("\n")
