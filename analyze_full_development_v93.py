"""Report sealed v92 relighting and optional v93 native baseline outcomes."""

import argparse
from collections import defaultdict
from itertools import combinations
import math
from pathlib import Path
from statistics import median

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, read, sha, utc, validate_dispatch, write,
)
from run_solar_full_native_baselines_v93 import seal_rows
from run_solar_matched_direct_render_v91 import BASIS, BASIS_SHA256, budget
from solar_evaluation_v70 import aggregate_rows, paired_component_bootstrap


DIRECT = "reference_camera_render_match"
SIFT = "SIFT2048"
MATCHANYTHING = "MatchAnything-EfficientLoFTR-native512"
OUTCOMES = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap")
CASE_FIELDS = ("component", "query_component", "positive", "unit_index", "query_unit_index", "light_a", "light_b")
TIME_FIELDS = ("inference_seconds", "complete_localization_cold_seconds", "case_calendar_seconds", "profile_load_seconds")


def completed_run(protocol_path, replay_path, purpose, profiles, basis, cases):
    p = read(protocol_path)
    ids = [case["case_id"] for case in cases]
    if (p["purpose"] != purpose or p["case_ids"] != ids or p["manifest"] != basis["manifest"]
            or p["input_sha256"][p["manifest"]] != sha(p["manifest"])
            or p["split"] != "model_selection" or p["tracks"] != ["estimated_sun"]
            or p["calibration_or_final_access"] is not False or p["accepted_results"] is not False
            or p["cpu_threads"] != 2 or [profile["name"] for profile in p["profiles"]] != profiles
            or p["scoring_revision"] != "v91_full_reference_point_mask"):
        raise ValueError("Exact full primary v91-aligned development protocol required")
    root = Path(p["output"])
    terminal = ("summary.json", "COMPLETE.json", "watchdog.json", "reservation.json", "execution.json")
    if any(not (root / filename).is_file() for filename in terminal):
        raise ValueError("Full terminal evidence is unavailable for " + str(root))
    summary, seal, watchdog, reservation, execution = (read(root / filename) for filename in terminal)
    digest = sha(protocol_path)
    if (not summary["complete"] or not seal["external_watchdog_passed"] or not watchdog["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(record["protocol_sha256"] != digest for record in (summary, seal, watchdog, execution))
            or seal["summary_sha256"] != sha(root / "summary.json")
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or summary["profiles"] != p["profiles"] or summary["purpose"] != purpose
            or summary["software"] != p["expected_software"] or summary["calibration_or_final_access"]
            or summary["selection_performed"]):
        raise ValueError("Exact full external completion and software evidence required")
    start = utc(watchdog["started_utc"])
    cap = p["max_calendar_seconds"]
    deadline = validate_dispatch(p, start, 12000)
    entries, _ = budget(p)
    charge = reservation["reservation"]
    if (deadline != utc(watchdog["hard_deadline_utc"]) or not start <= utc(watchdog["ended_utc"]) < deadline
            or watchdog["cap_seconds"] != cap or not 0 < watchdog["calendar_seconds"] <= cap
            or charge not in entries or reservation != watchdog["compute_reservation"]
            or charge["job_id"] != digest + JOB_SUFFIX or charge["seconds"] != cap
            or charge["category"] != "baseline" or charge["refund_on_failure"]):
        raise ValueError("Original full clock or canonical baseline charge differs")
    commit = watchdog["publication_commit"]
    if execution["publication_commit"] != commit or summary["publication_commit"] != commit:
        raise ValueError("Full source-publication identity differs")
    check_publication(protocol_path, p["source_sha256"], p["metadata_sha256"], commit, lambda: None)
    rows, _ = seal_rows(root, p, ids)
    if rows != summary["rows"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Original full row grid or aggregate differs from the seals")
    replay = read(replay_path)
    if (replay["passed"] is not True or replay["replayed_rows"] != len(rows) or len(replay["rows"]) != len(rows)
            or replay["protocol_sha256"] != digest or replay["summary_sha256"] != seal["summary_sha256"]
            or replay["watchdog_sha256"] != seal["watchdog_sha256"] or replay["table"] != summary["table"]
            or replay["accepted_results"] is not False):
        raise ValueError("Exact independent replay of every full row required")
    by_case = {case["case_id"]: case for case in cases}
    for row, check in zip(rows, replay["rows"], strict=True):
        case = by_case[row["case_id"]]
        if (any(row[field] != case[field] for field in CASE_FIELDS) or check["case_id"] != row["case_id"]
                or (len(profiles) > 1 and check.get("profile") != row["profile"])
                or any(check[key] != row["metrics"][key] for key in OUTCOMES + ("correspondence",))):
            raise ValueError("Paired endpoints or independently replayed row outcomes differ")
        for field in TIME_FIELDS:
            timing = row[field]
            if isinstance(timing, bool) or not isinstance(timing, (float, int)) or not math.isfinite(timing) or timing < 0:
                raise ValueError("Saved row timing must be finite and nonnegative")
    evidence = dict(
        protocol=str(protocol_path), protocol_sha256=digest, replay=str(replay_path),
        replay_sha256=sha(replay_path), summary_sha256=seal["summary_sha256"],
        watchdog_sha256=seal["watchdog_sha256"], publication_commit=commit,
        original_calendar_seconds=watchdog["calendar_seconds"], original_reserved_seconds=cap,
        original_software=summary["software"], rows=len(rows),
    )
    return rows, evidence


def counts(rows):
    pos = [row for row in rows if row["positive"]]
    neg = [row for row in rows if not row["positive"]]
    correct = sum(row["metrics"]["correct"] for row in pos)
    wrong = sum(row["metrics"]["wrong_positive"] for row in pos)
    return dict(
        positives=len(pos), negatives=len(neg), correct=correct, strict=sum(row["metrics"]["strict"] for row in pos),
        wrong_positive=wrong, no_positive_pose=len(pos) - correct - wrong,
        raw_nonoverlap=sum(row["metrics"]["raw_nonoverlap"] for row in neg),
        inference_failures=sum(row["inference_failure"] is not None for row in rows),
        median_inference_seconds=median(row["inference_seconds"] for row in rows) if rows else None,
        median_input_plus_inference_seconds=median(row["complete_localization_cold_seconds"] for row in rows) if rows else None,
        median_case_calendar_seconds=median(row["case_calendar_seconds"] for row in rows) if rows else None,
    )


def strata(rows):
    return dict(
        overall=rows, same_light_index=[r for r in rows if r["light_a"] == r["light_b"]],
        changed_light_index=[r for r in rows if r["light_a"] != r["light_b"]],
    )


def method_summary(rows):
    positives, negatives = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["positive"]:
            positives[row["component"]].append(row)
        else:
            negatives[row["component"], row["query_component"]].append(row)
    return dict(
        strata={name: counts(subset) for name, subset in strata(rows).items()},
        positive_by_terrain={group: counts(values) for group, values in sorted(positives.items())},
        negative_by_directed_terrain_pair=[
            dict(reference_terrain=a, query_terrain=b, **counts(values))
            for (a, b), values in sorted(negatives.items())
        ],
        profile_load_seconds=max(row["profile_load_seconds"] for row in rows),
    )


def paired(rows_a, rows_b):
    name_a, name_b = rows_a[0]["profile"], rows_b[0]["profile"]
    result = {}
    for name, a in strata(rows_a).items():
        b = strata(rows_b)[name]
        if [row["case_id"] for row in a] != [row["case_id"] for row in b]:
            raise ValueError("Exact paired case order differs between profiles")
        pairs = list(zip(a, b, strict=True))
        pos = [(x, y) for x, y in pairs if x["positive"]]
        neg = [(x, y) for x, y in pairs if not x["positive"]]
        ac, bc = counts(a), counts(b)
        interval = paired_component_bootstrap(a + b, name_a, name_b)
        if interval["terrain_components"] != 8:
            raise ValueError("Preserve all eight original terrain groups in paired analysis")
        result[name] = dict(
            positive_cases=len(pos), negative_cases=len(neg),
            correct_pose_delta=(ac["correct"] - bc["correct"]) / len(pos),
            wrong_positive_delta=(ac["wrong_positive"] - bc["wrong_positive"]) / len(pos),
            nonoverlap_pose_delta=(ac["raw_nonoverlap"] - bc["raw_nonoverlap"]) / len(neg),
            positive_gains=sum(x["metrics"]["correct"] and not y["metrics"]["correct"] for x, y in pos),
            positive_losses=sum(y["metrics"]["correct"] and not x["metrics"]["correct"] for x, y in pos),
            new_nonoverlap_poses=sum(x["metrics"]["raw_nonoverlap"] and not y["metrics"]["raw_nonoverlap"] for x, y in neg),
            removed_nonoverlap_poses=sum(y["metrics"]["raw_nonoverlap"] and not x["metrics"]["raw_nonoverlap"] for x, y in neg),
            percentile_2_5_50_97_5=interval,
        )
    return dict(first=name_a, second=name_b, strata=result)


def analyze(direct_protocol, direct_replay, native_protocol=None, native_replay=None):
    if (native_protocol is None) != (native_replay is None):
        raise ValueError("Provide the full native protocol and independent replay together")
    if sha(BASIS) != BASIS_SHA256:
        raise ValueError("Original immutable case basis required")
    basis, manifest, cases = fixed_v86_cases(BASIS, full=True)
    groups = sorted({case[field] for case in cases for field in ("component", "query_component")})
    if len(groups) != 8 or len(cases) != 384 or sum(case["positive"] for case in cases) != 192:
        raise ValueError("Exact 192 positive, 192 nonoverlap and eight terrain groups required")
    direct, evidence_direct = completed_run(
        direct_protocol, direct_replay, "full_direct_render_development_v92", [DIRECT], basis, cases,
    )
    all_rows = {DIRECT: direct}
    evidence = {"direct_v92": evidence_direct}
    if native_protocol is not None:
        native, evidence_native = completed_run(
            native_protocol, native_replay, "full_sift_matchanything_development_v93", [SIFT, MATCHANYTHING], basis, cases,
        )
        all_rows.update({name: [row for row in native if row["profile"] == name] for name in (SIFT, MATCHANYTHING)})
        evidence["native_v93"] = evidence_native
    return dict(
        complete=True, accepted_results=False, split="model_selection", track="estimated_sun",
        manifest=basis["manifest"], manifest_sha256=sha(basis["manifest"]), terrain_groups=groups,
        cases=384, positive_cases=192, negative_cases=192,
        method_results={name: method_summary(rows) for name, rows in all_rows.items()},
        paired_comparisons=[paired(a, b) for a, b in combinations(all_rows.values(), 2)],
        evidence=evidence,
        light_grouping="Equality of frozen light indices. The same index can correspond to different ENU Sun vectors across terrain.",
        timing_scope="Medians retain every case. Inference includes the matcher and common pose solve; input-plus-inference includes measured loader setup. Calendar time also includes replay, scoring and I/O.",
        interval_scope="Descriptive percentile bootstrap with eight terrain groups. A positive uses its terrain multiplicity and a negative uses the product of both endpoint multiplicities.",
    )


def markdown(report):
    lines = [
        "# Lunar localization on the full development set", "",
        "The fixed primary track includes 192 positive and 192 nonoverlap pairs across eight terrain groups. "
        "All reported poses are raw outputs. Same-light and changed-light slices use the declared light indices. "
        "Correct poses meet 2° rotation and 1% relative translation thresholds. Strict poses meet 0.5° and 0.1%.", "",
        "| Method | Light indices | Correct / positive | Strict / positive | Wrong / positive | No pose / positive | Raw pose / nonoverlap | Median inference (s) | Median input + inference (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["method_results"].items():
        for stratum, value in result["strata"].items():
            pos, neg = value["positives"], value["negatives"]
            lines.append(
                f"| {method} | {stratum.replace('_', ' ')} | {value['correct']}/{pos} | {value['strict']}/{pos} | "
                f"{value['wrong_positive']}/{pos} | {value['no_positive_pose']}/{pos} | {value['raw_nonoverlap']}/{neg} | "
                f"{value['median_inference_seconds']:.3f} | {value['median_input_plus_inference_seconds']:.3f} |"
            )
    if report["paired_comparisons"]:
        lines.extend(["", "| First minus second | Light indices | Correct-pose difference | 95% terrain interval | Raw nonoverlap difference | 95% two-endpoint interval |",
                      "|---|---|---:|---:|---:|---:|"])
        for comparison in report["paired_comparisons"]:
            pair = comparison["first"] + " − " + comparison["second"]
            for stratum, value in comparison["strata"].items():
                ci = value["percentile_2_5_50_97_5"]
                pos, neg = ci["raw_correct_pose_difference"], ci["raw_nonoverlap_difference"]
                lines.append(
                    f"| {pair} | {stratum.replace('_', ' ')} | {100 * value['correct_pose_delta']:+.2f} pp | "
                    f"[{100 * pos[0]:+.2f}, {100 * pos[2]:+.2f}] pp | {100 * value['nonoverlap_pose_delta']:+.2f} pp | "
                    f"[{100 * neg[0]:+.2f}, {100 * neg[2]:+.2f}] pp |"
                )
    lines.extend(["", report["timing_scope"], ""])
    if report["paired_comparisons"]:
        lines.extend([report["interval_scope"], ""])
    lines.extend(["Terrain and directed negative-endpoint tables, source identities and original timing receipts are in the JSON companion.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("direct-protocol", "direct-replay", "output-json", "output-markdown"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--native-protocol")
    parser.add_argument("--native-replay")
    args = parser.parse_args()
    report = analyze(args.direct_protocol, args.direct_replay, args.native_protocol, args.native_replay)
    write(args.output_json, report)
    with Path(args.output_markdown).open("x") as stream:
        stream.write(markdown(report))


if __name__ == "__main__":
    main()
