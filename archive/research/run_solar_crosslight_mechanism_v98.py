"""Fresh paired direct and cross-light mechanism pilot on the exact v91 cases."""

import argparse
from datetime import timedelta
import hashlib
import math
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, guarded_run, read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_full_direct_render_v92 import (
    FULL_CEILING, PILOT, PILOT_REPLAY, PILOT_REPLAY_SHA256, PILOT_SHA256, completion,
)
from run_solar_full_native_baselines_v93 import seal_rows
from run_solar_matched_direct_render_v91 import budget, identities, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, guarded_aligned_worker


SOURCE = "run_solar_crosslight_mechanism_v98.py"
BACKEND = "solar_crosslight_pool_v96.py"
PURPOSE = "crosslight_direct_mechanism_cost_pilot_v98"
DIRECT = "reference_camera_render_match"
NATIVE = "reports/solar-full-native-baselines-dispatch-v94.json"
NATIVE_SHA = "82e51b9a3e325e8eafc3022c7ac613dbce1a2400b71405a32eef4aca3dd8f1bc"
OFFCATALOGUE = "reports/solar-offcatalogue-direct-dispatch-v95.json"
OFFCATALOGUE_SHA = "2eb77cccb34b602c8b355af58190890cf8ef446ec0788d3ea8949d6ce3307899"
PILOT_CEILING = 240
TIMING = (
    "Both profiles run fresh on the same eight cases with identical pose-call instrumentation. "
    "Inference includes nine original solver calls and the observer. The evaluator's independent "
    "final pose replay is outside inference and included in case calendar time."
)


def historical():
    if sha(PILOT) != PILOT_SHA256 or sha(PILOT_REPLAY) != PILOT_REPLAY_SHA256:
        raise ValueError("Exact independently replayed v91 pilot required")
    p = read(PILOT)
    expected, cases = identities(p["settings"])
    if any(p.get(name) != value for name, value in expected.items()):
        raise ValueError("Original v91 case and profile identities changed")
    summary, _, seal, rows, sealed, files = completion(PILOT, p)
    replay = read(PILOT_REPLAY)
    if (replay["passed"] is not True or replay["replayed_rows"] != 8 or replay["accepted_results"]
            or replay["protocol_sha256"] != PILOT_SHA256 or replay["summary_sha256"] != seal["summary_sha256"]
            or replay["watchdog_sha256"] != seal["watchdog_sha256"]
            or [item["case_id"] for item in replay["rows"]] != p["case_ids"]):
        raise ValueError("Original v91 replay is incomplete")
    evidence = {path: sha(path) for path in files.values()}
    for row, marker in sealed:
        receipt = read(marker)
        merge_pins(evidence, {str(marker): sha(marker), receipt["row_path"]: receipt["row_sha256"],
                              row["saved_arrays"]: row["saved_arrays_sha256"]})
    for fields in (("reference", "query"), ("reference_geometry", "solar_lookup_protocol")):
        if tuple(cases[0].get(k) for k in fields) == tuple(cases[-1].get(k) for k in fields):
            raise ValueError("Frozen arm boundary would retain the previous loader cache key")
    return p, cases, summary["software"], evidence


def serial_protocols(settings):
    if sha(NATIVE) != NATIVE_SHA or sha(OFFCATALOGUE) != OFFCATALOGUE_SHA:
        raise ValueError("Original published predecessor protocol changed")
    path = str(relative(settings["offcatalogue_predecessor"]))
    original, selected = read(OFFCATALOGUE), read(path)
    local = {"output", "settings", "prepared_only", "created_utc", "latest_start_utc",
             "absolute_deadline_utc", "initial_reservations"}
    if selected["prepared_only"] is not False or any(selected.get(k) != v for k, v in original.items() if k not in local):
        raise ValueError("The serial off-catalogue predecessor must preserve the exact v95 experiment")
    return {NATIVE: NATIVE_SHA, path: sha(path)}


def terminal_predecessors(protocol):
    entries, _ = budget(protocol)
    result = {}
    for path, checksum in protocol["serial_predecessors"].items():
        previous = read(path)
        root = Path(previous["output"])
        watchdog_path = root / "watchdog.json"
        if not watchdog_path.is_file():
            raise ValueError("Wait for the designated serial predecessor to finish: " + path)
        watchdog = read(watchdog_path)
        reservation = read(root / "reservation.json")
        charge = reservation["reservation"]
        if (sha(path) != checksum or watchdog["protocol_sha256"] != checksum
                or reservation != watchdog["compute_reservation"] or charge not in entries
                or charge["job_id"] != checksum + JOB_SUFFIX or charge["category"] != "baseline"
                or charge["seconds"] != previous["max_calendar_seconds"] or charge["refund_on_failure"]
                or not utc(watchdog["started_utc"]) < utc(watchdog["ended_utc"]) <= utc_now()
                or type(watchdog["child_returncode"]) is not int):
            raise ValueError("A verified terminal worker receipt is required: " + path)
        result[str(watchdog_path)] = sha(watchdog_path)
    return result


def frozen(settings):
    from control_historical_native_v74 import source_closure
    from solar_crosslight_pool_v96 import NAME, POLICY

    if not isinstance(settings, dict) or set(settings) != {
        "output", "max_calendar_seconds", "publication_dispatch_lease_seconds", "offcatalogue_predecessor",
    }:
        raise ValueError("Specify output, cap, lease and exact off-catalogue serial predecessor")
    output = relative(settings["output"])
    cap, lease = settings["max_calendar_seconds"], settings["publication_dispatch_lease_seconds"]
    if (not output.is_relative_to("runs/solar-crosslight-mechanism-v98") or type(cap) is not int
            or not 5 < cap <= PILOT_CEILING or type(lease) is not int or not 0 < lease <= 1800):
        raise ValueError("New v98 run path and original finite pilot limits required")
    prior, cases, software, evidence = historical()
    predecessors = serial_protocols(settings)
    sources, public, assets = (dict(prior[name]) for name in ("source_sha256", "metadata_sha256", "input_sha256"))
    merge_pins(sources, source_closure({SOURCE, BACKEND}))
    merge_pins(public, {PILOT: PILOT_SHA256, PILOT_REPLAY: PILOT_REPLAY_SHA256})
    merge_pins(public, predecessors)
    merge_pins(assets, public)
    merge_pins(assets, evidence)
    direct, = prior["profiles"]
    if direct["name"] != DIRECT or direct["configuration"]["top_pose_candidates"] != 8:
        raise ValueError("Original direct profile and eight-candidate budget required")
    crosslight = dict(direct, name=NAME, crosslight_pooling=dict(POLICY))
    p = {name: prior[name] for name in (
        "basis", "manifest", "geometry_audit", "full_case_count", "case_ids", "compute_ledger",
        "compute_ledger_sha256", "reservation_journal", "blender_executable", "scoring_revision",
    )}
    p.update(
        **settings, settings=settings, purpose=PURPOSE, profiles=[direct, crosslight], tracks=["estimated_sun"],
        split="model_selection", cpu_threads=2, calibration_or_final_access=False, accepted_results=False,
        automatic_model_promotion=False, expected_software=software, source_sha256=sources,
        metadata_sha256=public, input_sha256=assets, serial_predecessors=predecessors,
        maximum_full_calendar_seconds=FULL_CEILING, timing_scope=TIMING,
        scope="Fresh two-arm cost and mechanism pilot on the original eight observed development cases.",
    )
    return p, cases


def prepare(settings_path, path):
    relative(path)
    p, _ = frozen(read(settings_path))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve all previous attempts")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Original canonical allowance cannot admit this pilot")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None,
             latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def exact(p):
    expected, cases = frozen(p["settings"])
    staged = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p.get("prepared_only") is not False or set(p) != set(expected) | staged
            or any(p[name] != value for name, value in expected.items())):
        raise ValueError("Exact published cross-light pilot identity required")
    return cases


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, PILOT_CEILING)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original cross-light pilot allowance exhausted")

    check()
    cases = exact(p)
    terminal_predecessors(p)
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, digest in p["input_sha256"].items():
        check()
        if sha(asset) != digest:
            raise ValueError("Original frozen evidence changed: " + asset)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original measured CPU/software stratum required")
    entries, remaining = budget(p)
    initial = p["initial_reservations"]
    cap = p["max_calendar_seconds"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation history changed")
    ours = [row for row in entries if row["job_id"] == sha(path) + JOB_SUFFIX]
    if len(ours) > 1 or any(row["category"] != "baseline" or row["seconds"] != cap or row["refund_on_failure"] for row in ours):
        raise ValueError("Existing pilot charge differs from original cap")
    if not ours and cap > remaining:
        raise ValueError("Original cap exceeds remaining baseline allowance")
    check()
    return p, cases, deadline


def pair_fingerprint(p, q):
    import numpy as np

    digest = hashlib.sha256()
    for points in (p, q):
        values = np.ascontiguousarray(points, dtype="<f8")
        digest.update(repr(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class PoseTraceBackend:
    """Instrument the actual production solver calls without another inference."""

    def __init__(self, profile):
        from solar_crosslight_pool_v96 import CrossLightBackend, NAME
        from solar_evaluation_v70 import Backend

        if profile["name"] not in (DIRECT, NAME):
            raise ValueError("Exactly the frozen direct and cross-light profiles are allowed")
        self.profile = profile
        self.backend = CrossLightBackend(profile) if profile["name"] == NAME else Backend(profile)
        self.provenance = dict(self.backend.provenance, observed_inference_pose_call_limit=9)
        self.calls = []
        original = self.backend.model.estimator

        def observed(p, q, inp):
            if len(self.calls) >= 9:
                raise ValueError("Inference exceeded the original nine pose calls")
            checksum = pair_fingerprint(p, q)
            pose, info = original(p, q, inp)
            self.calls.append(dict(pairs=len(p), input_sha256=checksum, pose_returned=pose is not None))
            return pose, info

        self.backend.model.estimator = observed

    def infer(self, inp):
        self.calls = []
        result = self.backend.infer(inp)
        if len(self.calls) != 9:
            raise ValueError("Expected eight actual candidate calls and one actual final call")
        search = dict(result["render_search"], observed_pose_calls=list(self.calls))
        return dict(result, render_search=search)


def paired_trace(direct, crosslight):
    a, b = direct["render_search"], crosslight["render_search"]
    a_calls, b_calls = a["observed_pose_calls"], b["observed_pose_calls"]
    if len(a_calls) != 9 or len(b_calls) != 9 or a_calls[:7] != b_calls[:7]:
        raise ValueError("Exact nine-call limit and first seven paired solver inputs required")
    trace = b["crosslight_pooling"]
    eligible = a_calls[7]["pairs"] < 6 and not any(call["pose_returned"] for call in a_calls[:7])
    if trace["eligible"] != eligible or trace["original_eighth_pairs"] != a_calls[7]["pairs"]:
        raise ValueError("Pool eligibility differs from the original candidate trace")
    if trace["attempted"]:
        if (not eligible or a_calls[7]["pose_returned"] or b_calls[7]["pairs"] < 6
                or len(trace["evidence"]["contributing_reference_light_indices"]) < 2):
            raise ValueError("Pool displaced a potentially viable original pose slot")
    elif a_calls != b_calls:
        raise ValueError("An unattempted pool changed the original direct calls")
    if trace["selected_pooled"]:
        if (not trace["attempted"] or not b_calls[7]["pose_returned"] or not b_calls[8]["pose_returned"]
                or b_calls[7]["input_sha256"] != b_calls[8]["input_sha256"]
                or a_calls[8]["pose_returned"] or b["selected_light_index"] is not None
                or crosslight["raw_inference_result"]["estimated_sun"] is not None):
            raise ValueError("Pooled output must be a reproduced rescue without a single Sun")
    elif (a_calls[-1] != b_calls[-1] or a["selected_light_index"] != b["selected_light_index"]
          or direct["reason"] != crosslight["reason"] or direct["metrics"] != crosslight["metrics"]):
        raise ValueError("Fallback failed to preserve the exact direct output")
    return dict(eligible=trace["eligible"], attempted=trace["attempted"],
                selected_pooled=trace["selected_pooled"], direct_correct=direct["metrics"]["correct"],
                crosslight_correct=crosslight["metrics"]["correct"], evidence=trace.get("evidence"),
                original_eighth_pairs=a_calls[7]["pairs"], original_final_pairs=a_calls[8]["pairs"],
                crosslight_final_pairs=b_calls[8]["pairs"])


def verify_saved_final_call(row):
    import numpy as np

    calls = row["render_search"]["observed_pose_calls"]
    with np.load(row["saved_arrays"], allow_pickle=False) as data:
        reference, query = data["reference_xy"], data["query_xy"]
    expected = dict(pairs=len(reference), input_sha256=pair_fingerprint(reference, query),
                    pose_returned=row["metrics"]["pose_returned"])
    if len(calls) != 9 or calls[-1] != expected:
        raise ValueError("Observed final pose call differs from saved inference output")


def replay(path, output):
    from run_solar_evaluation_v70 import measured_cost
    from solar_evaluation_v70 import aggregate_rows

    relative(path)
    relative(output)
    p = read(path)
    cases = exact(p)
    predecessors = terminal_predecessors(p)
    root = Path(p["output"])
    summary, watchdog, seal, execution, reservation = (read(root / name) for name in (
        "summary.json", "watchdog.json", "COMPLETE.json", "execution.json", "reservation.json",
    ))
    checksum = sha(path)
    if (not summary["complete"] or not seal["external_watchdog_passed"] or not watchdog["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(item["protocol_sha256"] != checksum for item in (summary, watchdog, seal, execution))
            or seal["summary_sha256"] != sha(root / "summary.json")
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or summary["software"] != p["expected_software"] or summary["profiles"] != p["profiles"]):
        raise ValueError("Original complete matched pilot seal required")
    start = utc(watchdog["started_utc"])
    deadline = validate_dispatch(p, start, PILOT_CEILING)
    entries, remaining = budget(p)
    cap = p["max_calendar_seconds"]
    charge = reservation["reservation"]
    if (deadline != utc(watchdog["hard_deadline_utc"]) or not start <= utc(watchdog["ended_utc"]) < deadline
            or watchdog["cap_seconds"] != cap or not 0 < watchdog["calendar_seconds"] <= cap
            or any(execution[k] != watchdog[k] for k in ("started_utc", "hard_deadline_utc", "work_deadline_utc"))
            or reservation != watchdog["compute_reservation"] or charge not in entries
            or charge["job_id"] != checksum + JOB_SUFFIX or charge["seconds"] != cap
            or charge["category"] != "baseline" or charge["refund_on_failure"]):
        raise ValueError("Original pilot deadline and exact baseline charge required")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit or execution["publication_commit"] != commit:
        raise ValueError("Original pilot source-publication identity differs")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, lambda: None)
    rows, _ = seal_rows(root, p, p["case_ids"])
    if rows != summary["rows"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Original raw grid or aggregate differs from the seals")
    case_index = {case["case_id"]: case for case in cases}
    for case in cases:
        for asset, expected in case["input_sha256"].items():
            if sha(asset) != expected:
                raise ValueError("Exact original scoring and inference asset required")
    for row in rows:
        if row["inference_failure"] is None:
            verify_saved_final_call(row)
        for field in ("case_calendar_seconds", "profile_load_seconds"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Finite complete pilot wall times are required")
    results = [dict(profile=row["profile"], **_replay_metrics(row, case_index[row["case_id"]])) for row in rows]
    traces = []
    for direct, crosslight in zip(rows[:8], rows[8:], strict=True):
        if direct["case_id"] != crosslight["case_id"]:
            raise ValueError("Exact paired case IDs required")
        both_succeeded = direct["inference_failure"] is None and crosslight["inference_failure"] is None
        checks = paired_trace(direct, crosslight) if both_succeeded else None
        traces.append(dict(case_id=direct["case_id"], positive=direct["positive"], verified=checks))
    if any(row["inference_failure"] is not None for row in rows):
        projection = dict(available=False, reason="The measured pilot has inference failures.")
    else:
        rule = measured_cost(rows, p["full_case_count"], p["tracks"], p["profiles"])
        included = sum(row["case_calendar_seconds"] + row["profile_load_seconds"] for row in rows)
        overhead = max(0.0, watchdog["calendar_seconds"] - included)
        estimated = rule["conservative_total_seconds"] + math.ceil(1.5 * overhead + 5)
        projection = dict(
            available=True, original_rule=rule, maximum_full_calendar_seconds=FULL_CEILING,
            pilot_outer_calendar_seconds=watchdog["calendar_seconds"], pilot_unallocated_seconds=overhead,
            conservative_controller_calendar_seconds=estimated,
            controller_rule="Original case rule plus 1.5× observed unallocated pilot time and five seal seconds.",
            fits_original_ceiling=estimated <= FULL_CEILING, fits_current_baseline_allowance=estimated <= remaining,
            full_run_authorized=False,
        )
    original_replay = read(PILOT_REPLAY)["rows"]
    repeated_direct = [result for result in results if result["profile"] == DIRECT]
    direct_repeatability = [dict(case_id=old["case_id"], exact_raw_metrics=old == {
        key: value for key, value in fresh.items() if key != "profile"
    }) for old, fresh in zip(original_replay, repeated_direct, strict=True)]
    write(output, dict(
        passed=True, replayed_rows=len(results), rows=results, paired_mechanism_trace=traces, table=summary["table"],
        original_v91_direct_repeatability=direct_repeatability,
        original_full_cost_projection=projection, predecessor_watchdogs=predecessors,
        protocol_sha256=checksum, summary_sha256=seal["summary_sha256"], watchdog_sha256=seal["watchdog_sha256"],
        accepted_results=False, timing_scope=TIMING,
        scope="Independent saved pose and trace replay on two fresh arms and eight observed development cases.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        cmd = sub.add_parser(action)
        cmd.add_argument("--protocol", required=True)
        if action in ("run", "worker"):
            cmd.add_argument("--publication-commit", required=True)
        if action == "prepare":
            cmd.add_argument("--settings", required=True)
        elif action == "run":
            cmd.add_argument("--python", default=sys.executable)
        elif action == "worker":
            cmd.add_argument("--controller-pid", required=True, type=int)
        else:
            cmd.add_argument("--output", required=True)
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
