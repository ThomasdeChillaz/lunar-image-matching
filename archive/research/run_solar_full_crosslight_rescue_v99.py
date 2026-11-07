"""Fresh matched 384-case comparison of direct matching and cross-light rescue."""

import argparse
from datetime import timedelta
import math
from pathlib import Path
from statistics import median
import sys

from run_solar_crosslight_mechanism_v98 import (
    DIRECT, TIMING, PoseTraceBackend, exact as exact_pilot, paired_trace, verify_saved_final_call,
)
from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, guarded_run, read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_full_direct_render_v92 import (
    FULL_CEILING, PILOT_REPLAY as V91_REPLAY, completion as v92_completion,
)
from run_solar_full_native_baselines_v93 import seal_rows
from run_solar_matched_direct_render_v91 import BASIS, budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, guarded_aligned_worker
from solar_crosslight_pool_v96 import NAME as RESCUE


SOURCE = Path(__file__).name
PURPOSE = "full_fresh_paired_crosslight_rescue_v99"
PILOT = "reports/solar-crosslight-mechanism-dispatch-v98.json"
PILOT_SHA256 = "de83a2d240a0346b8c96b71037710d169301652b5e0056fd10e671730744a877"
PILOT_REPLAY = "reports/solar_crosslight_mechanism_results_v98.json"
HISTORICAL = "reports/solar-full-direct-render-v92.json"
HISTORICAL_SHA256 = "9421f06bdaad9d3b8b9da8e424d4d21b6e5ceda90f6a1d228848842d24c0f909"
HISTORICAL_REPLAY = "reports/solar_full_direct_render_replay_v92.json"
HISTORICAL_REPLAY_SHA256 = "b11cc65d41367d82c85fc770b109a83fcd4fb65dca2a7da7c56ae039725af558"
PROFILES = (DIRECT, RESCUE)
CASE_FIELDS = ("component", "query_component", "positive", "unit_index", "query_unit_index", "light_a", "light_b")
OUTCOMES = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap", "correspondence")
TERMINAL = ("summary.json", "watchdog.json", "COMPLETE.json", "execution.json", "reservation.json")


def completion(path, p):
    """Verify fresh ordered rows and raw arrays against the public protocol and original clock."""
    from solar_evaluation_v70 import aggregate_rows

    root = Path(p["output"])
    files = [root / name for name in TERMINAL]
    summary, watchdog, seal, execution, reservation = [read(item) for item in files]
    protocol_hash = sha(path)
    if (not summary["complete"] or not seal["external_watchdog_passed"] or not watchdog["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(item["protocol_sha256"] != protocol_hash for item in (summary, watchdog, seal, execution))
            or seal["summary_sha256"] != sha(files[0]) or seal["watchdog_sha256"] != sha(files[1])
            or summary["purpose"] != p["purpose"] or summary["profiles"] != p["profiles"]
            or summary["software"] != p["expected_software"] or summary["calibration_or_final_access"]
            or summary["selection_performed"]):
        raise ValueError("Exact complete two-arm primary-development receipt required")
    start = utc(watchdog["started_utc"])
    deadline = validate_dispatch(p, start, FULL_CEILING)
    entries, _ = budget(p)
    charge = reservation["reservation"]
    cap = p["max_calendar_seconds"]
    if (deadline != utc(watchdog["hard_deadline_utc"]) or not start < utc(watchdog["ended_utc"]) < deadline
            or not 0 < watchdog["calendar_seconds"] <= cap or watchdog["cap_seconds"] != cap
            or any(execution[k] != watchdog[k] for k in ("started_utc", "work_deadline_utc", "hard_deadline_utc"))
            or reservation != watchdog["compute_reservation"] or charge not in entries
            or charge["job_id"] != protocol_hash + JOB_SUFFIX or charge["category"] != "baseline"
            or charge["seconds"] != cap or charge["refund_on_failure"]):
        raise ValueError("Exact original terminal clock and canonical charge required")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit or execution["publication_commit"] != commit:
        raise ValueError("Published completion source identity differs")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, lambda: None)
    rows, _ = seal_rows(root, p, p["case_ids"])
    if rows != summary["rows"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Published summary differs from the original sealed grid")
    return summary, watchdog, seal, rows, files


def verify_replayed(path, protocol_path, summary, seal, rows, *, two_arms):
    independent = read(path)
    checked = independent["rows"]
    if (independent["passed"] is not True or independent["replayed_rows"] != len(rows)
            or independent["accepted_results"] is not False or independent["table"] != summary["table"]
            or independent["protocol_sha256"] != sha(protocol_path)
            or independent["summary_sha256"] != seal["summary_sha256"]
            or independent["watchdog_sha256"] != seal["watchdog_sha256"]):
        raise ValueError("Exact independently replayed metrics required")
    for replayed, raw in zip(checked, rows, strict=True):
        if (replayed["case_id"] != raw["case_id"] or (two_arms and replayed["profile"] != raw["profile"])
                or any(replayed[k] != raw["metrics"][k] for k in OUTCOMES)):
            raise ValueError("Independent row replay differs from the sealed outcomes")
    return independent


def timing_projection(rows, case_count, profiles, calendar):
    from run_solar_evaluation_v70 import measured_cost

    for row in rows:
        for field in ("case_calendar_seconds", "profile_load_seconds"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Measured case and model wall times must be finite")
    rule = measured_cost(rows, case_count, ["estimated_sun"], profiles)
    observed = sum(row["case_calendar_seconds"] + row["profile_load_seconds"] for row in rows)
    overhead = max(0.0, calendar - observed)
    return dict(original_rule=rule, pilot_outer_calendar_seconds=calendar, pilot_unallocated_seconds=overhead,
                conservative_controller_calendar_seconds=rule["conservative_total_seconds"] + math.ceil(1.5 * overhead + 5))


def historical(cases):
    if sha(HISTORICAL) != HISTORICAL_SHA256 or sha(HISTORICAL_REPLAY) != HISTORICAL_REPLAY_SHA256:
        raise ValueError("Exact original v92 full control and independent replay required")
    p = read(HISTORICAL)
    summary, _, seal, rows, _, files = v92_completion(HISTORICAL, p)
    independent = verify_replayed(HISTORICAL_REPLAY, HISTORICAL, summary, seal, rows, two_arms=False)
    if (p["case_ids"] != [case["case_id"] for case in cases] or p["accepted_results"]
            or [profile["name"] for profile in p["profiles"]] != [DIRECT]
            or any(row[k] != case[k] for row, case in zip(rows, cases, strict=True) for k in CASE_FIELDS)):
        raise ValueError("Original historical v92 profile and paired case identities differ")
    return p, summary, independent["rows"], {path: sha(path) for path in files.values()}


def pilot_proof(direct, case_count):
    if sha(PILOT) != PILOT_SHA256:
        raise ValueError("Exact dispatched v98 two-arm pilot required")
    p = read(PILOT)
    cases = exact_pilot(p)
    summary, watchdog, seal, rows, files = completion(PILOT, p)
    independent = verify_replayed(PILOT_REPLAY, PILOT, summary, seal, rows, two_arms=True)
    if (p["profiles"][0] != direct or [profile["name"] for profile in p["profiles"]] != list(PROFILES)
            or p["full_case_count"] != case_count or len(cases) != 8 or len(rows) != 16
            or any(row["inference_failure"] is not None for row in rows)):
        raise ValueError("Original v98 pilot must measure both exact production arms on all eight cases")
    for row in rows:
        verify_saved_final_call(row)
    traces = [dict(case_id=a["case_id"], positive=a["positive"], verified=paired_trace(a, b))
              for a, b in zip(rows[:8], rows[8:], strict=True)]
    repeated = [dict(case_id=old["case_id"], exact_raw_metrics=old == {
        k: v for k, v in fresh.items() if k != "profile"
    }) for old, fresh in zip(read(V91_REPLAY)["rows"], independent["rows"][:8], strict=True)]
    if (not all(item["exact_raw_metrics"] for item in repeated)
            or independent["original_v91_direct_repeatability"] != repeated
            or independent["paired_mechanism_trace"] != traces):
        raise ValueError("Fresh v98 repeatability and independently replayed mechanism trace required")
    projection = timing_projection(rows, case_count, p["profiles"], watchdog["calendar_seconds"])
    reported = independent["original_full_cost_projection"]
    if (reported["available"] is not True or reported["full_run_authorized"] is not False
            or any(reported.get(k) != value for k, value in projection.items())):
        raise ValueError("Original v98 complete measured cost must independently reproduce")
    return p, independent, projection, {str(path): sha(path) for path in files}


def frozen(settings):
    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Provide only a fresh output and lease; v98 measured cost fixes the cap")
    output = relative(settings["output"])
    lease = settings["publication_dispatch_lease_seconds"]
    if not output.is_relative_to("runs/solar-full-crosslight-rescue-v99") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Versioned v99 runs path and original 1800-second lease limit required")
    _, _, cases = fixed_v86_cases(BASIS, full=True)
    if len(cases) != 384 or sum(case["positive"] for case in cases) != 192:
        raise ValueError("Retain all 192 positives and all 192 nonoverlaps")
    prior, summary, historical_rows, prior_files = historical(cases)
    pilot, independent, projection, pilot_files = pilot_proof(prior["profiles"][0], len(cases))
    cap = projection["conservative_controller_calendar_seconds"]
    if not 5 < cap <= FULL_CEILING or summary["software"] != pilot["expected_software"]:
        raise ValueError("Fresh two-arm study exceeds original ceiling or measured software changed")
    sources, metadata, assets = (dict(prior[k]) for k in ("source_sha256", "metadata_sha256", "input_sha256"))
    merge_pins(sources, pilot["source_sha256"])
    merge_pins(sources, {SOURCE: sha(Path(__file__))})
    merge_pins(metadata, pilot["metadata_sha256"])
    merge_pins(metadata, {PILOT: PILOT_SHA256, PILOT_REPLAY: sha(PILOT_REPLAY),
                          HISTORICAL: HISTORICAL_SHA256, HISTORICAL_REPLAY: HISTORICAL_REPLAY_SHA256})
    for pins in (pilot["input_sha256"], metadata, prior_files, pilot_files):
        merge_pins(assets, pins)
    p = {k: prior[k] for k in ("basis", "manifest", "geometry_audit", "full_case_count", "case_ids", "compute_ledger",
                              "compute_ledger_sha256", "reservation_journal", "blender_executable", "scoring_revision")}
    p.update(**settings, settings=settings, purpose=PURPOSE, profiles=pilot["profiles"], tracks=["estimated_sun"],
             split="model_selection", cpu_threads=2, calibration_or_final_access=False, accepted_results=False,
             automatic_model_promotion=False, expected_software=pilot["expected_software"], source_sha256=sources,
             metadata_sha256=metadata, input_sha256=assets, max_calendar_seconds=cap,
             maximum_full_calendar_seconds=FULL_CEILING, measured_cost_admission=projection, timing_scope=TIMING,
             historical_control=HISTORICAL, pilot_control=PILOT, comparison="two fresh arms; v92 preserved separately",
             interval=dict(seed=20261101, repetitions=2000, cluster="both terrain endpoints for nonoverlap"),
             scope="All 384 paired observed development cases; no calibration, final inference or generalization claim.")
    return p, cases, historical_rows, independent["rows"]


def prepare(settings, path):
    relative(path)
    p, *_ = frozen(read(settings))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve every earlier comparison attempt")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Fresh paired measured cost exceeds original baseline allowance")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None,
             latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def exact(p):
    expected, cases, history, pilot = frozen(p["settings"])
    staged = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p.get("prepared_only") is not False or set(p) != set(expected) | staged
            or any(p[k] != value for k, value in expected.items())):
        raise ValueError("Exact published v99 source, grid and measured-cap identity required")
    return cases, history, pilot


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original paired-study compute allowance exhausted")

    check()
    cases, _, _ = exact(p)
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, digest in p["input_sha256"].items():
        check()
        if sha(asset) != digest:
            raise ValueError("Original development evidence changed: " + asset)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    initial, cap = p["initial_reservations"], p["max_calendar_seconds"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical baseline history changed")
    own = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if len(own) > 1 or any(e["category"] != "baseline" or e["seconds"] != cap or e["refund_on_failure"] for e in own):
        raise ValueError("Existing charge differs from the measured cap")
    if not own and cap > remaining:
        raise ValueError("Fresh paired measured cost exceeds original baseline allowance")
    check()
    return p, cases, deadline


def outcomes(rows):
    positive = [r for r in rows if r["positive"]]
    negative = [r for r in rows if not r["positive"]]
    metric = lambda group, field: sum(r["metrics"][field] for r in group)
    return dict(positives=len(positive), nonoverlap=len(negative), correct=metric(positive, "correct"),
                strict=metric(positive, "strict"), wrong_positive=metric(positive, "wrong_positive"),
                missing_positive_pose=len(positive) - metric(positive, "pose_returned"),
                raw_nonoverlap=metric(negative, "raw_nonoverlap"),
                inference_failures_positive=sum(r["inference_failure"] is not None for r in positive),
                inference_failures_nonoverlap=sum(r["inference_failure"] is not None for r in negative),
                median_complete_inference_seconds=median(r["inference_seconds"] for r in rows))


def compare(rows, cases, interval):
    from solar_evaluation_v70 import paired_component_bootstrap

    case_count = len(cases)
    direct, rescue = rows[:case_count], rows[case_count:]
    traces = []
    for a, b, case in zip(direct, rescue, cases, strict=True):
        for row, profile in ((a, DIRECT), (b, RESCUE)):
            if (row["profile"] != profile or row["case_id"] != case["case_id"]
                    or any(row[k] != case[k] for k in CASE_FIELDS)):
                raise ValueError("Exact paired case ID, order and terrain endpoints required")
            if row["inference_failure"] is None:
                verify_saved_final_call(row)
        both_succeeded = a["inference_failure"] is None and b["inference_failure"] is None
        trace = paired_trace(a, b) if both_succeeded else None
        traces.append(dict(case_id=case["case_id"], positive=case["positive"], light_a=case["light_a"],
                           light_b=case["light_b"], component=case["component"], query_component=case["query_component"],
                           direct_inference_failure=a["inference_failure"], rescue_inference_failure=b["inference_failure"],
                           verified=trace, direct_correct=a["metrics"]["correct"], rescue_correct=b["metrics"]["correct"],
                           direct_raw_nonoverlap=a["metrics"]["raw_nonoverlap"],
                           rescue_raw_nonoverlap=b["metrics"]["raw_nonoverlap"],
                           rescue_wrong_positive=b["metrics"]["wrong_positive"]))

    def mechanisms(items):
        verified = [item for item in items if item["verified"] is not None]
        selected = [item for item in verified if item["verified"]["selected_pooled"]]
        return dict(cases=len(items), verified_traces=len(verified), trace_unavailable=len(items) - len(verified),
                    eligible=sum(item["verified"]["eligible"] for item in verified),
                    attempted=sum(item["verified"]["attempted"] for item in verified), selected_pooled=len(selected),
                    selected_correct=sum(item["rescue_correct"] and not item["direct_correct"] for item in selected),
                    selected_wrong_positive=sum(item["rescue_wrong_positive"] for item in selected),
                    selected_new_nonoverlap=sum(item["rescue_raw_nonoverlap"] and not item["direct_raw_nonoverlap"]
                                                for item in selected),
                    improved_correct=sum(item["rescue_correct"] and not item["direct_correct"] for item in items),
                    lost_correct=sum(item["direct_correct"] and not item["rescue_correct"] for item in items),
                    new_nonoverlap=sum(item["rescue_raw_nonoverlap"] and not item["direct_raw_nonoverlap"] for item in items))

    groups = {
        "all": lambda r: True,
        "same_light": lambda r: r["light_a"] == r["light_b"],
        "changed_light": lambda r: r["light_a"] != r["light_b"],
    }
    strata = {}
    for name, include in groups.items():
        a, b = ([r for r in arm if include(r)] for arm in (direct, rescue))
        selected = [item for item in traces if include(item)]
        strata[name] = dict(direct=outcomes(a), rescue=outcomes(b),
            positive_mechanism=mechanisms([item for item in selected if item["positive"]]),
            nonoverlap_mechanism=mechanisms([item for item in selected if not item["positive"]]),
            paired_interval=paired_component_bootstrap(a + b, RESCUE, DIRECT, seed=interval["seed"],
                                                       repetitions=interval["repetitions"]))
    return strata, traces


def repeatability(current, previous):
    index = {(item.get("profile", DIRECT), item["case_id"]): item for item in current}
    discrepancies = []
    for old in previous:
        key = old.get("profile", DIRECT), old["case_id"]
        fresh = index[key]
        differences = [field for field in OUTCOMES if fresh[field] != old[field]]
        if differences:
            discrepancies.append(dict(profile=key[0], case_id=key[1], differing_fields=differences))
    return dict(checked=len(previous), exact_raw_metrics=len(previous) - len(discrepancies), differences=discrepancies)


def replay(path, output):
    relative(path)
    relative(output)
    p = read(path)
    cases, historical_rows, pilot_rows = exact(p)
    summary, watchdog, seal, rows, _ = completion(path, p)
    case_index = {case["case_id"]: case for case in cases}
    for case in cases:
        for asset, digest in case["input_sha256"].items():
            if sha(asset) != digest:
                raise ValueError("Exact frozen primary-development scoring data required")
    checked = [dict(profile=row["profile"], **_replay_metrics(row, case_index[row["case_id"]])) for row in rows]
    strata, traces = compare(rows, cases, p["interval"])
    write(output, dict(
        passed=True, replayed_rows=len(checked), rows=checked, table=summary["table"], strata=strata,
        paired_mechanism_trace=traces, v92_historical_direct_repeatability=repeatability(checked, historical_rows),
        v98_pilot_repeatability=repeatability(checked, pilot_rows), protocol_sha256=sha(path),
        summary_sha256=seal["summary_sha256"], watchdog_sha256=seal["watchdog_sha256"],
        watchdog_calendar_seconds=watchdog["calendar_seconds"], reserved_seconds=p["max_calendar_seconds"],
        accepted_results=False, comparison=p["comparison"], timing_scope=p["timing_scope"],
        scope="Independent saved-only replay of all 768 fresh cells. Intervals group the eight observed terrains."
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        command = commands.add_parser(action)
        command.add_argument("--protocol", required=True)
        if action in ("run", "worker"):
            command.add_argument("--publication-commit", required=True)
        if action == "prepare":
            command.add_argument("--settings", required=True)
        elif action == "run":
            command.add_argument("--python", default=sys.executable)
        elif action == "worker":
            command.add_argument("--controller-pid", required=True, type=int)
        else:
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    elif args.action == "worker":
        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, PoseTraceBackend)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
