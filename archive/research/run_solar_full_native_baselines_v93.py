"""Bounded full development comparison of SIFT2048 and native MatchAnything."""

import argparse
from datetime import timedelta
import math
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, fixed_v86_cases, guarded_run, read, sha, utc, utc_now,
    validate_dispatch, write,
)
from run_solar_full_direct_render_v92 import FULL_CEILING, completion
from run_solar_matched_direct_render_v91 import BASIS, BASIS_SHA256, budget, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _original, _replay_metrics, guarded_aligned_worker


SIFT_REPLAY = "reports/solar-preselection-compact-controls-cost-pilot-replay-v86.json"
SIFT_REPLAY_SHA = "448ed91f3fe5ed9ead6f26564acee1b4967f843e7740d60c6608b2224d04abad"
MA_RECOVERY = "reports/solar-matched-matchanything-recovery-v91.json"
MA_RECOVERY_SHA = "e902f75e017d611659b6fffe4e825dcb3c6164c01223c1c3a373250a8249489f"
MA_ASSEMBLY = "reports/solar_matched_matchanything_recovery_results_v91.json"
MA_ASSEMBLY_SHA = "e2043060f513e3e84509e719bb1d3d91694c461322d0d29e3e87ffa4eadff91a"
PURPOSE = "full_sift_matchanything_development_v93"


def seal_rows(root, protocol, ids):
    """Open exactly one sealed row for every ordered profile/case cell."""
    from solar_evaluation_v70 import digest

    root = Path(root)
    cases = {c["case_id"]: c for c in read(protocol["manifest"])["cases"]}
    profiles = {p["name"]: p for p in protocol["profiles"]}
    ordered = [(name, case) for name in profiles for case in ids]
    by_key, pins = {}, {}
    for marker in sorted(root.glob("*.COMPLETE.json")):
        seal = read(marker)
        row_path, array_path = (Path(seal[name]) for name in ("row_path", "arrays_path"))
        if row_path.parent != root or array_path.parent != root:
            raise ValueError("Original saved payloads must stay inside their run directory")
        row = read(row_path)
        name, case_id = row["profile"], row["case_id"]
        key = digest(dict(case=cases[case_id], profile=profiles[name], track="estimated_sun",
                          sources=protocol["source_sha256"]))
        if (not seal["complete"] or seal["identity"] != key or marker != root / (key + ".COMPLETE.json")
                or row_path != root / (key + ".json") or array_path != root / (key + ".npz")
                or row["track"] != "estimated_sun" or row["saved_arrays"] != str(array_path)
                or row["saved_arrays_sha256"] != seal["arrays_sha256"]
                or sha(row_path) != seal["row_sha256"] or sha(array_path) != seal["arrays_sha256"]
                or (name, case_id) in by_key):
            raise ValueError("Original saved baseline cell differs from its seal or case identity")
        by_key[name, case_id] = row
        merge_pins(pins, {str(marker): sha(marker), str(row_path): seal["row_sha256"],
                          str(array_path): seal["arrays_sha256"]})
    if set(by_key) != set(ordered):
        raise ValueError("Exact complete profile/case grid required")
    return [by_key[key] for key in ordered], pins


def sift_pilot(basis):
    """Validate the original v86 four-profile pilot and retain its SIFT costs."""
    from solar_evaluation_protocol_v74 import validate_lease

    if sha(BASIS) != BASIS_SHA256 or sha(SIFT_REPLAY) != SIFT_REPLAY_SHA:
        raise ValueError("Exact original SIFT cost pilot and independent replay required")
    root = Path(basis["output"])
    names = ("summary.json", "COMPLETE.json", "watchdog_admission.json", "execution.json", "reservation.json")
    files = [root / name for name in names]
    summary, seal, watchdog, execution, reservation = [read(path) for path in files]
    independent = read(SIFT_REPLAY)
    if (not summary["complete"] or not seal["complete"] or not seal["external_watchdog_passed"]
            or any(record["protocol_sha256"] != BASIS_SHA256 for record in (summary, seal, watchdog, execution))
            or seal["summary_sha256"] != sha(files[0]) or summary["profiles"] != basis["profiles"]
            or summary["cached_cases"] != 0 or summary["case_ids"] != basis["pilot_case_ids"]
            or summary["software"] != execution["software"] or execution["final_reserve_access"]
            or independent["passed"] is not True or independent["summary_sha256"] != seal["summary_sha256"]
            or independent["replayed_rows"] != len(summary["rows"])):
        raise ValueError("Sealed and independently replayed v86 pilot required")
    cap, latest, absolute = validate_lease(basis)
    started, deadline = utc(watchdog["started_utc"]), utc(watchdog["absolute_deadline_utc"])
    if (not utc(basis["created_utc"]) <= started <= latest
            or deadline != min(started + timedelta(seconds=cap), absolute)
            or not 0 < summary["calendar_seconds"] <= seal["observed_calendar_seconds"] <= (deadline-started).total_seconds()
            or watchdog["reservation_sha256"] != sha(files[-1])
            or seal["original_allowance_seconds"] != cap):
        raise ValueError("Original v86 pilot deadline differs")
    entries, _ = budget(basis)
    charge = reservation["reservation"]
    if (charge not in entries or charge["job_id"] != BASIS_SHA256 + "/evaluation"
            or charge["category"] != "baseline" or charge["seconds"] != cap or charge["refund_on_failure"]):
        raise ValueError("Original v86 SIFT pilot charge is absent")
    commit = summary["source_publication_commit"]
    if execution["publication_commit"] != commit:
        raise ValueError("Original v86 publication differs")
    check_publication(BASIS, basis["source_sha256"], {}, commit, lambda: None)
    rows, pins = seal_rows(root, basis, basis["pilot_case_ids"])
    if rows != summary["rows"] or len(rows) != 32:
        raise ValueError("Original v86 summary differs from its 32 sealed measurements")
    profile, = [item for item in basis["profiles"] if item["name"] == "SIFT2048" and item["backend"] == "sift"]
    merge_pins(pins, {str(path): sha(path) for path in files})
    merge_pins(pins, {BASIS: BASIS_SHA256, SIFT_REPLAY: SIFT_REPLAY_SHA})
    return profile, [row for row in rows if row["profile"] == "SIFT2048"], summary["software"], pins


def matchanything_pilot(basis):
    """Verify both original MatchAnything phases and their independent assembly."""
    if sha(MA_RECOVERY) != MA_RECOVERY_SHA or sha(MA_ASSEMBLY) != MA_ASSEMBLY_SHA:
        raise ValueError("Exact completed MatchAnything recovery and assembly required")
    from run_solar_matched_matchanything_recovery_v91 import ORIGINAL

    original, failed, carried, pins = _original()
    recovery, assembly = read(MA_RECOVERY), read(MA_ASSEMBLY)
    ids = basis["pilot_case_ids"]
    if (original["case_ids"] != ids or recovery["case_ids"] != ids[4:] or recovery["combined_case_ids"] != ids
            or recovery["original_files_sha256"] != pins or recovery["profiles"] != original["profiles"]
            or recovery["manifest"] != basis["manifest"] or recovery["tracks"] != ["estimated_sun"]):
        raise ValueError("Original MatchAnything cases, inputs or profile changed")
    summary, _, seal, fresh, fresh_seals, files = completion(MA_RECOVERY, recovery)
    rows = [row for row, _ in carried] + fresh
    if (assembly["complete"] is not True or assembly["independent_saved_metric_replay_passed"] is not True
            or assembly["original_run_complete"] is not False or assembly["carried_rows"] != 4
            or assembly["fresh_rows"] != 4 or assembly["rows"] != rows
            or assembly["original_protocol_sha256"] != sha(ORIGINAL)
            or assembly["recovery_protocol_sha256"] != MA_RECOVERY_SHA
            or assembly["recovery_watchdog_sha256"] != seal["watchdog_sha256"]
            or assembly["original_watchdog_sha256"] != pins[str(Path(original["output"]) / "watchdog.json")]):
        raise ValueError("Independent assembled MatchAnything measurements differ from the sealed phases")
    entries, _ = budget(recovery)
    charge = failed["compute_reservation"]["reservation"]
    if (charge not in entries or charge["job_id"] != sha(ORIGINAL) + JOB_SUFFIX
            or charge["seconds"] != original["max_calendar_seconds"] or charge["category"] != "baseline"):
        raise ValueError("Original failed MatchAnything charge must be retained")
    check_publication(ORIGINAL, original["source_sha256"], original["metadata_sha256"],
                      failed["publication_commit"], lambda: None)
    profile, = recovery["profiles"]
    provenance = profile["name"] + ".provenance.json"
    before = Path(original["output"]) / provenance
    after = Path(recovery["output"]) / provenance
    if read(before) != read(after):
        raise ValueError("Actual native model provenance changed across the measured pilot phases")
    merge_pins(pins, {path: sha(path) for path in (*files.values(), ORIGINAL, MA_RECOVERY, MA_ASSEMBLY, str(after))})
    for row, marker in fresh_seals:
        seal_row = read(marker)
        merge_pins(pins, {str(marker): sha(marker), seal_row["row_path"]: seal_row["row_sha256"],
                          row["saved_arrays"]: row["saved_arrays_sha256"]})
    return recovery, profile, rows, summary["software"], pins


def frozen(settings):
    from run_solar_evaluation_v70 import measured_cost

    if not isinstance(settings, dict) or set(settings) != {"output", "publication_dispatch_lease_seconds"}:
        raise ValueError("A new output and lease are required; the measured pilots determine the cap")
    root = relative(settings["output"])
    lease = settings["publication_dispatch_lease_seconds"]
    if root == Path("runs") or not root.is_relative_to("runs") or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("New runs directory and finite original publication lease required")
    basis, manifest, cases = fixed_v86_cases(BASIS, full=True)
    sift, sift_rows, sift_software, sift_pins = sift_pilot(basis)
    recovery, ma, ma_rows, ma_software, ma_pins = matchanything_pilot(basis)
    if (sift_software != ma_software or basis["baseline_admission"]["maximum_full_calendar_seconds"] != FULL_CEILING
            or str(root) in {basis["output"], recovery["output"], recovery["original_output"]}):
        raise ValueError("Identical measured software, original full ceiling and new output required")
    for group in (sift_rows, ma_rows):
        if [row["case_id"] for row in group] != basis["pilot_case_ids"]:
            raise ValueError("Both methods must retain the same eight pilot rows")
        for row in group:
            if any(isinstance(row[name], bool) or not isinstance(row[name], (int, float))
                   or not math.isfinite(row[name]) or row[name] < 0
                   for name in ("case_calendar_seconds", "profile_load_seconds")) or row["case_calendar_seconds"] <= 0:
                raise ValueError("Actual finite positive measured case costs required")
    profiles = [sift, ma]
    cost = measured_cost(sift_rows + ma_rows, len(cases), ["estimated_sun"], profiles)
    cap = cost["conservative_total_seconds"]
    if not 5 < cap <= FULL_CEILING:
        raise ValueError("Original 12000-second ceiling cannot admit this paired full cost")
    sources = dict(recovery["source_sha256"])
    merge_pins(sources, basis["source_sha256"])
    for name in (Path(__file__).name, "run_solar_full_direct_render_v92.py"):
        merge_pins(sources, {name: sha(name)})
    public = dict(recovery["metadata_sha256"])
    merge_pins(public, {BASIS: BASIS_SHA256, SIFT_REPLAY: SIFT_REPLAY_SHA,
                        MA_RECOVERY: MA_RECOVERY_SHA, MA_ASSEMBLY: MA_ASSEMBLY_SHA})
    assets = dict(recovery["input_sha256"])
    for pins in (sift_pins, ma_pins, public):
        merge_pins(assets, pins)
    protocol = dict(
        **settings, settings=settings, purpose=PURPOSE, split="model_selection", cpu_threads=2,
        tracks=["estimated_sun"], calibration_or_final_access=False, accepted_results=False,
        profiles=profiles, basis=BASIS, manifest=basis["manifest"], geometry_audit=basis["geometry_audit"],
        full_case_count=len(manifest["cases"]), case_ids=[case["case_id"] for case in cases],
        compute_ledger=basis["compute_ledger"], compute_ledger_sha256=basis["compute_ledger_sha256"],
        reservation_journal=basis["reservation_journal"], blender_executable=basis["blender_executable"],
        source_sha256=sources, metadata_sha256=public, input_sha256=assets,
        max_calendar_seconds=cap, maximum_full_calendar_seconds=FULL_CEILING, measured_cost_admission=cost,
        expected_software=ma_software, scoring_revision="v91_full_reference_point_mask",
        scope="Two fresh profiles on the complete original model-selection manifest.",
    )
    return protocol, cases


def prepare(settings_path, path):
    relative(path)
    p, _ = frozen(read(settings_path))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve prior attempts")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Measured full cost exceeds the original remaining allowance")
    p.update(prepared_only=True, initial_reservations=entries, created_utc=None,
             latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def admission(path, commit, started):
    from native_pretrained_adapters_v73 import validate_native_profile
    from run_solar_evaluation_v70 import software_identity

    relative(path)
    p = read(path)
    deadline = validate_dispatch(p, started, FULL_CEILING)

    def check():
        if utc_now() >= deadline:
            raise TimeoutError("Original paired baseline allowance exhausted")

    expected, cases = frozen(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen paired comparison identity changed")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, digest in p["input_sha256"].items():
        check()
        if sha(asset) != digest:
            raise ValueError("Original paired baseline evidence changed: " + asset)
    validate_native_profile(p["profiles"][1])
    if software_identity() != p["expected_software"]:
        raise ValueError("Actual full job differs from the measured CPU/software stratum")
    entries, remaining = budget(p)
    initial = p["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation prefix changed")
    cap = p["max_calendar_seconds"]
    own = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if len(own) > 1 or any(entry["category"] != "baseline" or entry["seconds"] != cap for entry in own):
        raise ValueError("Existing charge differs from the original paired cap")
    if not own and cap > remaining:
        raise ValueError("Measured full cost exceeds remaining original allowance")
    check()
    return p, cases, deadline


def replay(path, output):
    from solar_evaluation_v70 import aggregate_rows

    relative(path)
    relative(output)
    p = read(path)
    expected, cases = frozen(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Original paired full comparison identities differ")
    root = Path(p["output"])
    summary, watchdog, seal, execution, reservation = [read(root / name) for name in (
        "summary.json", "watchdog.json", "COMPLETE.json", "execution.json", "reservation.json",
    )]
    if (not summary["complete"] or not seal["external_watchdog_passed"] or not watchdog["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(record["protocol_sha256"] != sha(path) for record in (summary, watchdog, seal, execution))
            or seal["summary_sha256"] != sha(root / "summary.json")
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or summary["profiles"] != p["profiles"] or summary["software"] != p["expected_software"]
            or summary["calibration_or_final_access"] or summary["selection_performed"]):
        raise ValueError("Original paired external completion evidence differs")
    started = utc(watchdog["started_utc"])
    deadline = validate_dispatch(p, started, FULL_CEILING)
    cap = p["max_calendar_seconds"]
    entries, _ = budget(p)
    charge = reservation["reservation"]
    if (utc(watchdog["hard_deadline_utc"]) != deadline or not started <= utc(watchdog["ended_utc"]) < deadline
            or watchdog["cap_seconds"] != cap or not 0 < watchdog["calendar_seconds"] <= cap
            or reservation != watchdog["compute_reservation"] or charge not in entries
            or charge["job_id"] != sha(path) + JOB_SUFFIX or charge["category"] != "baseline"
            or charge["seconds"] != cap or charge["refund_on_failure"]):
        raise ValueError("Original paired full deadline or canonical charge differs")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit or execution["publication_commit"] != commit:
        raise ValueError("Original paired full publication identity differs")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, lambda: None)
    rows, _ = seal_rows(root, p, p["case_ids"])
    if rows != summary["rows"] or summary["table"] != aggregate_rows(rows):
        raise ValueError("Original paired raw rows or aggregate table differ")
    case_map = {case["case_id"]: case for case in cases}
    for case in cases:
        for asset, digest in case["input_sha256"].items():
            if sha(asset) != digest:
                raise ValueError("Reopened exact development case bytes differ")
    checks = [dict(profile=row["profile"], **_replay_metrics(row, case_map[row["case_id"]])) for row in rows]
    write(output, dict(passed=True, replayed_rows=len(checks), rows=checks, table=summary["table"],
                       protocol_sha256=sha(path), summary_sha256=seal["summary_sha256"],
                       watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
                       scope="Independent saved metrics on both complete paired development grids."))


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
