"""Describe the completed, independently replayed v115 raw development comparison."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np

from plan_solar_matched_arm_v113 import ARM_NAMES, COHORT, DIRECT_NAME, RUN_CAP, TRACK, _seconds, pinned


NAMES = (*ARM_NAMES, DIRECT_NAME)
CASE_COUNT = 96
SEED = 20261101
REPETITIONS = 2000
OUTCOMES = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap")
CASE_FIELDS = ("component", "query_component", "positive", "unit_index", "query_unit_index", "light_a", "light_b")
PURPOSE = "matched_development_comparison"
STAGE = "matched_modern_direct_arm_v113"
REPLAY_SOURCE = "verify_solar_matched_arm_v113.py"


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(data), sha256(data).hexdigest()


def _key(row):
    return row["profile"], row["case_id"], row["track"]


def _digest_json(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cohort_cases(cohort, manifest):
    declared = cohort["subsets"][str(CASE_COUNT)]
    ids = declared["case_ids"]
    lookup = {case["case_id"]: case for case in manifest["cases"]}
    if (len(ids) != CASE_COUNT or len(set(ids)) != CASE_COUNT or len(lookup) != 384
            or len(manifest["cases"]) != len(lookup) or not set(ids) <= lookup.keys()):
        raise ValueError("Exact frozen development cohort required")
    cases = [lookup[case_id] for case_id in ids]
    groups = set(cohort["terrain_groups"])
    positives = [case for case in cases if case["positive"]]
    negatives = [case for case in cases if not case["positive"]]
    expected_lights = {(row["reference"], row["query"]): row["cases"] for row in declared["positive_lighting"]}
    edge_counts = Counter((case["component"], case["query_component"]) for case in negatives)
    expected_endpoints = [dict(case_id=case["case_id"], reference=case["component"],
                               query=case["query_component"]) for case in negatives]
    matched = {case["matched_positive_case"]: case for case in negatives}
    paired_fields = ("unit_index", "component", "light_a", "light_b")
    if (manifest["split"] != "model_selection" or manifest["calibration_or_final_access"] is not False
            or any(case["split"] != "model_selection" for case in cases)
            or len(groups) != 8 or set(manifest["terrain_groups"]) != groups
            or len(positives) != declared["positives"] or len(negatives) != declared["nonoverlaps"]
            or len(positives) != 48 or len(negatives) != 48
            or any(case["component"] != case["query_component"] for case in positives)
            or Counter(case["component"] for case in positives) != dict.fromkeys(groups, 6)
            or Counter((c["component"], c["light_a"], c["light_b"]) for c in positives)
            != {(g, a, b): 1 for g in groups for a, b in expected_lights}
            or Counter((case["light_a"], case["light_b"]) for case in positives) != expected_lights
            or set(matched) != {c["case_id"] for c in positives}
            or any(matched[c["case_id"]][key] != c[key] for c in positives for key in paired_fields)
            or expected_endpoints != declared["negative_endpoints"] or len(edge_counts) != 8
            or any(a == b or n != 6 for (a, b), n in edge_counts.items())
            or {a for a, _ in edge_counts} != groups or {b for _, b in edge_counts} != groups):
        raise ValueError("Frozen lighting and both terrain endpoints must be preserved")
    return cases, sorted(groups)


def _worker(summary):
    software = summary["software"]
    memory = summary["memory_measurement"]
    snapshots = memory["profile_snapshots"]
    pid = memory["worker_pid"]
    peak = summary["worker_peak_rss"]
    if (software["machine"] != "aarch64" or not software["platform"].startswith("Linux-")
            or software["cpu_threads"] != 2 or not software["torch"].endswith("+cpu")
            or type(pid) is not int or pid <= 0 or memory["profile_order"] != list(NAMES)
            or memory["job_scope"] != "shared_multi_profile_worker_process"
            or memory["per_model_memory_ranking_supported"] is not False
            or peak["worker_pid"] != pid or len(snapshots) != len(NAMES)):
        raise ValueError("Five fresh profiles must share one qualified two-thread Linux ARM CPU worker")
    for i, (item, name) in enumerate(zip(snapshots, NAMES, strict=True)):
        if (item["profile"] != name or item["profile_index"] != i or item["worker_pid"] != pid
                or item["completed"] is not True or item["fresh_inference_cases"] != CASE_COUNT
                or item["cached_cases"] != 0 or any(item[k]["worker_pid"] != pid for k in ("before", "after"))):
            raise ValueError("Shared worker identity or fresh profile execution differs")
    return dict(software=software, worker_pid=pid, process_lifetime_peak_rss=peak,
                isolated_model_memory_available=False)


def validate_rows(summary, replay, cases):
    expected = [(name, case["case_id"], TRACK) for name in NAMES for case in cases]
    rows = summary["rows"]
    replay_rows = replay["rows"]
    if (summary["complete"] is not True or summary["cached_cases"] != 0
            or summary["fresh_inference_cases"] != len(expected)
            or summary["accepted_results"] is not False or summary["calibration_or_final_access"] is not False
            or summary["automatic_model_promotion"] is not False or summary["scientific_model_evidence"] is not True
            or summary["case_ids"] != [case["case_id"] for case in cases]
            or [_key(row) for row in rows] != expected
            or replay["passed"] is not True or replay["replayed_rows"] != len(expected)
            or [_key(row) for row in replay_rows] != expected):
        raise ValueError("Complete fresh blind rows and exact independent replay required")
    by_case = {case["case_id"]: case for case in cases}
    replayed = {_key(row): row for row in replay_rows}
    for row in rows:
        case = by_case[row["case_id"]]
        metrics = row["metrics"]
        failure = row["inference_failure"]
        if (any(row[key] != case[key] for key in CASE_FIELDS) or type(row["positive"]) is not bool
                or row["reused_case_cache"] is not False or row["initialization_only_cost_control"] is not False
                or any(type(metrics[key]) is not bool or replayed[_key(row)][key] is not metrics[key]
                       for key in OUTCOMES)
                or metrics["positive"] is not case["positive"]):
            raise ValueError("Fresh outcome, independent replay or frozen endpoint metadata differs")
        positive, pose, correct = (metrics[key] for key in ("positive", "pose_returned", "correct"))
        if (correct and (not positive or not pose) or metrics["strict"] and not correct
                or metrics["wrong_positive"] != (positive and pose and not correct)
                or metrics["raw_nonoverlap"] != (not positive and pose)
                or failure is not None and (pose or failure["status"] not in ("failed", "timeout", "unavailable"))):
            raise ValueError("Inconsistent raw outcome or operational failure")
        inferred = _seconds(row["inference_seconds"], "inference", allow_zero=failure is not None)
        complete = _seconds(row["complete_localization_cold_seconds"], "input plus inference", allow_zero=failure is not None)
        if complete < inferred:
            raise ValueError("Input-plus-inference time cannot be below inference alone")
    return {_key(row): row for row in rows}


def _counts(rows):
    positive = [row for row in rows if row["positive"]]
    negative = [row for row in rows if not row["positive"]]
    correct = sum(row["metrics"]["correct"] for row in positive)
    wrong = sum(row["metrics"]["wrong_positive"] for row in positive)
    nonoverlap = sum(row["metrics"]["raw_nonoverlap"] for row in negative)
    failures = Counter(row["inference_failure"]["status"] for row in rows if row["inference_failure"] is not None)
    return dict(positive_cases=len(positive), nonoverlap_cases=len(negative), correct=correct,
                strict=sum(row["metrics"]["strict"] for row in positive), wrong_positive=wrong,
                no_positive_pose=len(positive) - correct - wrong, raw_nonoverlap=nonoverlap,
                correct_rate=correct / len(positive) if positive else None,
                raw_nonoverlap_rate=nonoverlap / len(negative) if negative else None,
                operational_failures=dict(failures),
                median_inference_seconds=median(row["inference_seconds"] for row in rows),
                median_input_plus_inference_seconds=median(row["complete_localization_cold_seconds"] for row in rows))


def _strata(cases):
    return dict(overall=cases,
                same_light_index=[case for case in cases if case["light_a"] == case["light_b"]],
                changed_light_index=[case for case in cases if case["light_a"] != case["light_b"]])


def _contrasts(pairs, metric):
    diffs = [int(a["metrics"][metric]) - int(b["metrics"][metric]) for a, b in pairs]
    return dict(cases=len(diffs), first_only=diffs.count(1), second_only=diffs.count(-1),
                delta_percentage_points=100 * sum(diffs) / len(diffs))


def _paired(pairs, groups, draws):
    pos = [(a, b) for a, b in pairs if a["positive"]]
    neg = [(a, b) for a, b in pairs if not a["positive"]]
    totals = Counter(a["component"] for a, _ in pos)
    gains = Counter()
    for a, b in pos:
        gains[a["component"]] += int(a["metrics"]["correct"]) - int(b["metrics"]["correct"])
    draws_deltas = 100 * (draws @ [gains[g] for g in groups]) / (draws @ [totals[g] for g in groups])
    positive = _contrasts(pos, "correct")
    positive.update(bootstrap_2_5_50_97_5_percentage_points=np.quantile(draws_deltas, [.025, .5, .975]).tolist(),
                    net_correct_by_terrain={g: gains[g] for g in groups})
    negative = _contrasts(neg, "raw_nonoverlap")
    negative["leave_one_terrain_out"] = {
        g: _contrasts([(a, b) for a, b in neg if g not in (a["component"], a["query_component"])], "raw_nonoverlap")
        for g in groups
    }
    comparable = all(row["inference_failure"] is None for pair in pairs for row in pair)
    return dict(positive_correct=positive, positive_wrong=_contrasts(pos, "wrong_positive"),
                negative_raw_pose=negative, complete_without_operational_failures=comparable)


def analyze(summary, replay, cases, groups):
    worker = _worker(summary)
    rows = validate_rows(summary, replay, cases)
    slices = _strata(cases)
    draws = np.random.default_rng(SEED).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=REPETITIONS)

    def select(name, subset):
        return [rows[name, case["case_id"], TRACK] for case in subset]

    methods = {}
    for name in NAMES:
        part = select(name, cases)
        pos = [row for row in part if row["positive"]]
        neg = [row for row in part if not row["positive"]]
        edges = sorted({(row["component"], row["query_component"]) for row in neg})
        lighting = sorted({(row["light_a"], row["light_b"]) for row in part})
        methods[name] = dict(
            strata={label: _counts(select(name, subset)) for label, subset in slices.items()},
            by_positive_terrain={g: _counts([row for row in pos if row["component"] == g]) for g in groups},
            by_directed_negative_terrain=[dict(reference=a, query=b, **_counts(
                [row for row in neg if (row["component"], row["query_component"]) == (a, b)])) for a, b in edges],
            by_lighting={f"{a},{b}": _counts([row for row in part if (row["light_a"], row["light_b"]) == (a, b)])
                         for a, b in lighting},
        )
    paired = [dict(first=a, second=b, strata={label: _paired(
        list(zip(select(a, subset), select(b, subset), strict=True)), groups, draws)
        for label, subset in slices.items()}) for a, b in combinations(NAMES, 2)]
    return dict(schema="solar_five_arm_paired_summary_v115", complete=True, track=TRACK,
                split="model_selection", accepted_results=False, calibration_or_final_access=False,
                case_count=len(cases), method_case_cells=len(rows), terrain_groups=groups, worker=worker,
                methods=methods, paired=paired, bootstrap_seed=SEED, bootstrap_repetitions=REPETITIONS,
                inference_scope="All eligible rows include geometric no-pose outcomes and any operational failures.",
                timing_scope="Same-worker model-resident times. Input-plus-inference includes actual reused-loader setup. Worker RSS is a lifetime peak, with no model memory ranking.",
                positive_interval_scope="Descriptive paired percentile bootstrap with eight reused development terrain groups. Each resampled terrain retains its two cameras and all its lighting variants. No multiplicity adjustment or fresh-test claim.",
                negative_scope="The 48 nonoverlap cases reuse both endpoints on eight directed terrain edges. Omit both incident edges for each leave-one-terrain-out calculation. These sensitivities do not establish an independent-trial confidence bound or calibrated 1% risk.",
                lighting_scope="Same and changed refer to frozen light-index equality. They do not imply equal physical Sun vectors across terrain.")


def _validate_protocol(protocol, cohort, manifest):
    case_ids = cohort["subsets"][str(CASE_COUNT)]["case_ids"]
    profiles = protocol["profiles"]
    cap = protocol["max_calendar_seconds"]
    if (protocol["stage_mode"] != STAGE or protocol["purpose"] != PURPOSE
            or protocol["comparison_case_count"] != CASE_COUNT or protocol["comparison_case_ids"] != case_ids
            or protocol["case_ids"] != [c["case_id"] for c in manifest["cases"]]
            or protocol["manifest"] != cohort["manifest"] or protocol["geometry_audit"] != cohort["geometry_audit"]
            or protocol["tracks"] != [TRACK] or protocol["cpu_threads"] != 2 or cap != RUN_CAP
            or protocol["accepted_results"] is not False or protocol["calibration_or_final_access"] is not False
            or protocol["case_cache_seals"] or tuple(p["name"] for p in profiles) != NAMES
            or protocol["input_sha256"].get(COHORT[0]) != COHORT[1]
            or protocol["input_sha256"].get(cohort["manifest"]) != cohort["manifest_sha256"]
            or protocol["input_sha256"].get(cohort["geometry_audit"]) != cohort["geometry_audit_sha256"]):
        raise ValueError("Exact published 96-case protocol and five original arm names required")


def validate_evidence(protocol, protocol_sha, summary, summary_sha, seal, replay, watchdog, cohort, manifest):
    _validate_protocol(protocol, cohort, manifest)
    profiles = protocol["profiles"]
    cap = protocol["max_calendar_seconds"]
    if (summary["purpose"] != PURPOSE or summary["profiles"] != profiles
            or summary["protocol_sha256"] != protocol_sha or summary["manifest_sha256"] != cohort["manifest_sha256"]
            or seal["complete"] is not True or seal["external_watchdog_passed"] is not True
            or seal["summary_sha256"] != summary_sha or seal["protocol_sha256"] != protocol_sha
            or seal["original_allowance_seconds"] != cap
            or replay["summary_sha256"] != summary_sha
            or replay["replay_source_sha256"] != protocol["source_sha256"][REPLAY_SOURCE]):
        raise ValueError("Summary, completion seal and independent replay must bind the same frozen execution")
    if (watchdog["complete"] is not True or watchdog["protocol_sha256"] != protocol_sha
            or watchdog["source_publication_commit"] != summary["source_publication_commit"]
            or watchdog["run_attempt"] != 1 or type(watchdog["workflow_run_id"]) is not int
            or watchdog["workflow_run_id"] <= 0 or watchdog["cap_seconds"] != cap
            or watchdog["child_returncode"] != 0 or watchdog["failure"] is not None
            or watchdog["archive_failure"] is not None or watchdog["primary_only"] is not True
            or watchdog["automatic_retry"] is not False or watchdog["reservation_refunded"] is not False
            or watchdog["result_artifact"]["scientific_output"] != protocol["output"]):
        raise ValueError("Successful primary-only outer watchdog on the original worker is required")
    charge = watchdog["original_reservation"]["reservation"]
    if (charge["job_id"] != protocol_sha + "/evaluation" or charge["category"] != "baseline"
            or charge["seconds"] != cap or charge["refund_on_failure"] is not False):
        raise ValueError("Original cumulative baseline budget charge required")
    inner = _seconds(summary["calendar_seconds"], "worker calendar")
    controller = _seconds(seal["observed_calendar_seconds"], "controller calendar")
    outer = _seconds(watchdog["calendar_seconds"], "outer watchdog calendar")
    if not inner <= controller <= outer <= cap:
        raise ValueError("Original nested calendar limits were not preserved")


def _one_member(archive, name):
    if archive.namelist().count(name) != 1:
        raise ValueError("Result archive must contain exactly one " + name)
    return archive.open(name)


def _archive_controls(archive, protocol, protocol_sha):
    controls = {}
    for profile in protocol["profiles"]:
        name = profile["name"]
        if name == DIRECT_NAME:
            filename = "actual_ARM_Direct_control.json"
        elif name in ARM_NAMES[2:]:
            filename = _digest_json(profile) + ".actual_ARM_control.json"
        else:
            continue
        with _one_member(archive, f"{protocol['output']}/{filename}") as source:
            raw = source.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError("Fresh control receipt is too large")
        control = json.loads(raw)
        if control["passed"] is not True or (name == DIRECT_NAME and (
                control["protocol_sha256"] != protocol_sha or control["fresh_actual_arm"] is not True
                or control["injected_query_sun_rejected"] is not True)):
            raise ValueError("Fresh same-worker control did not pass for " + name)
        controls[name] = sha256(raw).hexdigest()
    return controls


def summarize(root, run, protocol_path, replay_path, watchdog_path, archive_path):
    cohort = pinned(root, COHORT)
    manifest = pinned(root, (cohort["manifest"], cohort["manifest_sha256"]))
    cases, groups = cohort_cases(cohort, manifest)
    run = Path(run)
    if (run / "FAILED.json").exists():
        raise ValueError("A failed run cannot produce a completed comparison")
    protocol, protocol_sha = _read(protocol_path)
    with Path(__file__).open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != protocol["source_sha256"][Path(__file__).name]:
            raise ValueError("Reporter source differs from the pre-inference freeze")
    summary, summary_sha = _read(run / "summary.json")
    seal, seal_sha = _read(run / "COMPLETE.json")
    replay, replay_sha = _read(replay_path)
    watchdog, watchdog_sha = _read(watchdog_path)
    validate_evidence(protocol, protocol_sha, summary, summary_sha, seal, replay, watchdog, cohort, manifest)
    with Path(archive_path).open("rb") as source:
        archive_sha = file_digest(source, "sha256").hexdigest()
    if archive_sha != watchdog["result_artifact"]["sha256"]:
        raise ValueError("Original outer-watchdog result archive hash differs")
    with ZipFile(archive_path) as archive:
        for name, digest in (("summary.json", summary_sha), ("COMPLETE.json", seal_sha)):
            member = f"{protocol['output']}/{name}"
            with _one_member(archive, member) as source:
                if file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError("Locally unpacked result differs from outer-watchdog archive")
        controls = _archive_controls(archive, protocol, protocol_sha)
    report = analyze(summary, replay, cases, groups)
    report["evidence"] = dict(protocol_sha256=protocol_sha, summary_sha256=summary_sha, seal_sha256=seal_sha,
                              replay_sha256=replay_sha, watchdog_sha256=watchdog_sha, archive_sha256=archive_sha,
                              workflow_run_id=watchdog["workflow_run_id"], source_commit=watchdog["source_publication_commit"],
                              cohort_sha256=COHORT[1], manifest_sha256=cohort["manifest_sha256"],
                              fresh_control_sha256=controls)
    return report


def recover(root, run, protocol_path):
    """Inventory individually sealed partial cells without certifying their scores."""
    cohort = pinned(root, COHORT)
    manifest = pinned(root, (cohort["manifest"], cohort["manifest_sha256"]))
    cases, _ = cohort_cases(cohort, manifest)
    protocol, protocol_sha = _read(protocol_path)
    _validate_protocol(protocol, cohort, manifest)
    return recover_rows(run, protocol, protocol_sha, cases)


def recover_rows(run, protocol, protocol_sha, cases):
    run = Path(run)
    expected = {(p["name"], c["case_id"], TRACK): (p, c) for p in protocol["profiles"] for c in cases}
    found = {}
    invalid_markers = []
    for marker in sorted(run.glob("*.COMPLETE.json")):
        try:
            seal, _ = _read(marker)
            identity = seal["identity"]
            if (len(identity) != 64 or any(ch not in "0123456789abcdef" for ch in identity)
                    or marker.name != identity + ".COMPLETE.json" or seal["complete"] is not True
                    or seal["source_protocol_sha256"] != protocol_sha):
                raise ValueError("Cell seal identity differs")
            row_name, arrays_name = identity + ".json", identity + ".npz"
            row, row_sha = _read(run / row_name)
            key = _key(row)
            profile, case = expected[key]
            if (key in found or row["reused_case_cache"] is not False
                    or identity != _digest_json(dict(case=case, profile=profile, track=TRACK, sources=protocol["source_sha256"]))
                    or any(row[k] != case[k] for k in CASE_FIELDS) or seal["row_sha256"] != row_sha
                    or seal["row_path"] != f"{protocol['output']}/{row_name}"
                    or seal["arrays_path"] != f"{protocol['output']}/{arrays_name}"
                    or row["saved_arrays"] != seal["arrays_path"] or row["saved_arrays_sha256"] != seal["arrays_sha256"]):
                raise ValueError("Cell identity or stored bytes differ")
            with (run / arrays_name).open("rb") as source:
                if file_digest(source, "sha256").hexdigest() != seal["arrays_sha256"]:
                    raise ValueError("Stored prediction digest differs")
            failure = row["inference_failure"]
            found[key] = "evaluated" if failure is None else failure["status"]
        except (KeyError, TypeError, ValueError, OSError) as error:
            invalid_markers.append(dict(marker=marker.name, error=type(error).__name__))
    return dict(schema="solar_five_arm_partial_recovery_v115", complete=False, expected_cells=len(expected),
                sealed_cells=len(found), protocol_sha256=protocol_sha, invalid_markers=invalid_markers,
                methods={name: dict(expected_cells=len(cases), sealed_cells=sum(key[0] == name for key in found),
                                    statuses=dict(Counter(value for key, value in found.items() if key[0] == name)),
                                    missing_case_ids=[c["case_id"] for c in cases if (name, c["case_id"], TRACK) not in found])
                         for name in NAMES},
                scope="Execution inventory only. Individually sealed cells have matching byte hashes. Pose scores, replay and paired comparison remain uncertified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partial", action="store_true")
    for name in ("replay", "watchdog", "archive"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.partial:
        report = recover(args.root, args.run, args.protocol)
    else:
        if None in (args.replay, args.watchdog, args.archive):
            parser.error("--replay, --watchdog and --archive are required for a completed comparison")
        report = summarize(args.root, args.run, args.protocol, args.replay, args.watchdog, args.archive)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
