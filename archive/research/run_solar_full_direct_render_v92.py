"""Measured-cost continuation of the original direct-render pilot on all 384 cases."""

import argparse
import math
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, guarded_run, read, sha, utc, utc_now,
    validate_dispatch, write,
)
from run_solar_matched_direct_render_v91 import BASIS, budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import (
    _replay_metrics, _sealed_rows, guarded_aligned_worker,
)


PILOT = "reports/solar-matched-direct-render-pilot-v91.json"
PILOT_SHA256 = "2c070870a4ca63ede04c83e7bd12bce916ff339297e2c3fbc46157083f1cb303"
PILOT_REPLAY = "reports/solar_matched_direct_render_results_v91.json"
PILOT_REPLAY_SHA256 = "5623a8f6b0ada8c5f53c831a227d612dc70e780a738efa12618080f705fa7ad6"
PURPOSE = "full_direct_render_development_v92"
FULL_CEILING = 12000


def completion(path, protocol):
    """Verify saved external completion against the original clock and charge."""
    root = Path(protocol["output"])
    files = {name: str(root / name) for name in (
        "summary.json", "watchdog.json", "COMPLETE.json", "execution.json", "reservation.json",
    )}
    summary, watchdog, seal, execution, reservation = [read(files[name]) for name in files]
    digest = sha(path)
    if (not seal["external_watchdog_passed"] or not watchdog["complete"] or not summary["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(row["protocol_sha256"] != digest for row in (summary, watchdog, seal, execution))
            or seal["summary_sha256"] != sha(files["summary.json"])
            or seal["watchdog_sha256"] != sha(files["watchdog.json"])
            or summary["profiles"] != protocol["profiles"]
            or summary["calibration_or_final_access"] or summary["selection_performed"]
            or summary["purpose"] != protocol["purpose"]):
        raise ValueError("Original complete, source-bound development outputs required")
    start, end = utc(watchdog["started_utc"]), utc(watchdog["ended_utc"])
    hard_deadline = validate_dispatch(protocol, start, FULL_CEILING)
    cap = protocol["max_calendar_seconds"]
    if (watchdog["cap_seconds"] != cap or hard_deadline != utc(watchdog["hard_deadline_utc"])
            or not start <= end < hard_deadline or not 0 < watchdog["calendar_seconds"] <= cap
            or execution["started_utc"] != watchdog["started_utc"]
            or execution["hard_deadline_utc"] != watchdog["hard_deadline_utc"]
            or execution["work_deadline_utc"] != watchdog["work_deadline_utc"]):
        raise ValueError("Original startup-inclusive baseline deadline differs")
    entries, _ = budget(protocol)
    charge = reservation["reservation"]
    if (reservation != watchdog["compute_reservation"] or charge not in entries
            or charge["job_id"] != digest + JOB_SUFFIX or charge["category"] != "baseline"
            or charge["seconds"] != cap or charge["refund_on_failure"]):
        raise ValueError("The original exact canonical baseline charge must be retained")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit or execution["publication_commit"] != commit:
        raise ValueError("Original comparison publication identity differs")
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, lambda: None)
    sealed = _sealed_rows(root, protocol, protocol["case_ids"])
    rows = [row for row, _ in sealed]
    if rows != summary["rows"]:
        raise ValueError("Summary differs from the original sealed case grid")
    return summary, watchdog, seal, rows, sealed, files


def frozen(settings):
    from run_solar_evaluation_v70 import measured_cost

    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("Set only a new output and lease; the measured pilot determines the cap")
    root = relative(settings["output"])
    lease = settings["publication_dispatch_lease_seconds"]
    if root == Path("runs") or not root.is_relative_to("runs") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Use a new runs directory and an original lease of at most 1800 seconds")
    if sha(PILOT) != PILOT_SHA256 or sha(PILOT_REPLAY) != PILOT_REPLAY_SHA256:
        raise ValueError("Exact original direct-render pilot and independent replay required")
    pilot = read(PILOT)
    if str(root) == pilot["output"] or pilot["split"] != "model_selection" or pilot["calibration_or_final_access"]:
        raise ValueError("Preserve the original pilot and keep evaluation development-only")
    basis, manifest, cases = fixed_v86_cases(BASIS, full=True)
    if (pilot["case_ids"] != basis["pilot_case_ids"] or pilot["manifest"] != basis["manifest"]
            or pilot["scoring_revision"] != "v91_full_reference_point_mask"):
        raise ValueError("Original pilot case/profile/scoring identities differ")
    summary, _, seal, rows, sealed, files = completion(PILOT, pilot)
    independent = read(PILOT_REPLAY)
    if (independent["passed"] is not True or independent["replayed_rows"] != 8
            or independent["protocol_sha256"] != PILOT_SHA256
            or independent["summary_sha256"] != seal["summary_sha256"]
            or independent["watchdog_sha256"] != seal["watchdog_sha256"]
            or [row["case_id"] for row in independent["rows"]] != pilot["case_ids"]):
        raise ValueError("Actual independent replay must bind the original eight completed outcomes")
    for row in rows:
        for field in ("case_calendar_seconds", "profile_load_seconds"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Actual finite original pilot timing required")
    cost = measured_cost(rows, len(cases), pilot["tracks"], pilot["profiles"])
    cap = cost["conservative_total_seconds"]
    if not 5 < cap <= FULL_CEILING:
        raise ValueError("Original 12000-second ceiling cannot admit this full measured cost")
    sources = dict(pilot["source_sha256"])
    this_source = Path(__file__).name
    merge_pins(sources, {this_source: sha(this_source)})
    public = dict(pilot["metadata_sha256"])
    merge_pins(public, {PILOT: PILOT_SHA256})
    inputs = dict(pilot["input_sha256"])
    evidence = {path: sha(path) for path in files.values()}
    merge_pins(evidence, {PILOT: PILOT_SHA256, PILOT_REPLAY: PILOT_REPLAY_SHA256})
    for row, marker in sealed:
        case_seal = read(marker)
        merge_pins(evidence, {str(marker): sha(marker), case_seal["row_path"]: case_seal["row_sha256"],
                              row["saved_arrays"]: row["saved_arrays_sha256"]})
    merge_pins(inputs, evidence)
    protocol = dict(pilot)
    for field in ("prepared_only", "created_utc", "latest_start_utc", "absolute_deadline_utc", "initial_reservations"):
        protocol.pop(field)
    protocol.update(
        settings=settings, output=settings["output"], purpose=PURPOSE, max_calendar_seconds=cap,
        publication_dispatch_lease_seconds=lease, source_sha256=sources, metadata_sha256=public,
        input_sha256=inputs, case_ids=[case["case_id"] for case in cases],
        full_case_count=len(manifest["cases"]), measured_cost_admission=cost, maximum_full_calendar_seconds=FULL_CEILING,
        original_pilot_artifacts=evidence, expected_software=summary["software"],
        scope="Fresh raw primary model-selection results on the complete original 384-case manifest.",
    )
    return protocol, cases


def prepare(settings_path, path):
    relative(path)
    protocol, _ = frozen(read(settings_path))
    if Path(protocol["output"]).exists():
        raise FileExistsError("Preserve all prior attempts")
    entries, remaining = budget(protocol)
    if protocol["max_calendar_seconds"] > remaining:
        raise ValueError("Full measured cost exceeds the remaining canonical allowance")
    protocol.update(prepared_only=True, initial_reservations=entries,
                    created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    write(path, protocol)


def admission(path, commit, started):
    from direct_render_adapter_v73 import validate_render_profile
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    protocol = read(path)
    deadline = validate_dispatch(protocol, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline:
            raise TimeoutError("Original full direct-render allowance exhausted")

    check()
    expected, cases = frozen(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Published full evaluation identities differ")
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, check)
    for asset, digest in protocol["input_sha256"].items():
        check()
        if sha(asset) != digest:
            raise ValueError("Original direct-render input or pilot evidence changed: " + asset)
    validate_render_profile(protocol["profiles"][0])
    if software_identity() != protocol["expected_software"]:
        raise ValueError("Full evaluation requires the identical measured software/CPU stratum")
    entries, remaining = budget(protocol)
    initial = protocol["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation prefix changed")
    cap = protocol["max_calendar_seconds"]
    owned = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if len(owned) > 1 or any(entry["category"] != "baseline" or entry["seconds"] != cap for entry in owned):
        raise ValueError("Existing full direct-render charge differs from the measured cap")
    if not owned and cap > remaining:
        raise ValueError("Full measured cost exceeds remaining baseline allowance")
    check()
    return protocol, cases, deadline


def replay(path, output):
    from solar_evaluation_v70 import aggregate_rows

    relative(path)
    relative(output)
    protocol = read(path)
    expected, cases = frozen(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Original full comparison identities differ")
    summary, _, seal, rows, _, _ = completion(path, protocol)
    if summary["software"] != protocol["expected_software"]:
        raise ValueError("Actual full study changed the measured CPU/software stratum")
    result = []
    for row, case in zip(rows, cases, strict=True):
        for asset, digest in case["input_sha256"].items():
            if sha(asset) != digest:
                raise ValueError("Original held-out case input changed")
        result.append(_replay_metrics(row, case))
    table = aggregate_rows(rows)
    if summary["table"] != table:
        raise ValueError("Original aggregate table differs from the sealed rows")
    write(output, dict(passed=True, replayed_rows=len(result), rows=result, table=table,
                       protocol_sha256=sha(path), summary_sha256=seal["summary_sha256"],
                       watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
                       scope="Independent saved metrics on all 384 original development cases."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        sub = commands.add_parser(action)
        sub.add_argument("--protocol", required=True)
        if action in ("run", "worker"):
            sub.add_argument("--publication-commit", required=True)
        if action == "prepare":
            sub.add_argument("--settings", required=True)
        elif action == "run":
            sub.add_argument("--python", default=sys.executable)
        elif action == "worker":
            sub.add_argument("--controller-pid", type=int, required=True)
        else:
            sub.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, Path(__file__).name)
    elif args.action == "worker":
        from solar_evaluation_v70 import Backend

        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, Backend)
    else:
        replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
