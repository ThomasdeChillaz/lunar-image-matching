"""Bounded direct-render/SIFT comparison on the exact eight v86 cases."""

import argparse
from pathlib import Path
import sys


BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
BASIS_SHA256 = "9e14f9c1b95eb12a796ec0533e8c9e50af292a86895e763912a91f786f6b0912"
CONTROL = "reports/solar-direct-render-native-control-v90.json"
LIBRARY = "reports/solar-direct-render-library-v90.json"
PURPOSE = "matched_direct_render_development_v91"
WORKER_SOURCE = "run_solar_matched_matchanything_recovery_v91.py"
SCORER_SOURCE = "solar_aligned_scoring_v91.py"


def relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Canonical repository-relative path required")
    path = Path(value)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != value
            or path == Path(".") or not path.resolve().is_relative_to(Path.cwd())):
        raise ValueError("Canonical repository-relative path required")
    return path


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {
        "output", "max_calendar_seconds", "publication_dispatch_lease_seconds",
    }:
        raise ValueError("Exact output, cap and publication lease settings required")
    output = relative(settings["output"])
    if output == Path("runs") or not output.is_relative_to("runs"):
        raise ValueError("Use a versioned output directory below runs")
    cap, lease = settings["max_calendar_seconds"], settings["publication_dispatch_lease_seconds"]
    if type(cap) is not int or not 5 < cap <= 18000 or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Explicit cap and lease within the original baseline limits required")


def merge_pins(target, incoming):
    for path, digest in incoming.items():
        if path in target and target[path] != digest:
            raise ValueError("Conflicting bytes in frozen evidence for " + path)
        target[path] = digest


def render_profile():
    from control_direct_render_v90 import verify_completed_control
    from direct_render_adapter_v73 import METHOD, validate_render_profile
    from run_solar_distributional_evaluation_v90 import read, sha

    native = verify_completed_control(CONTROL)
    control = read(CONTROL)
    library = read(LIBRARY)
    root = relative(library["output"])
    manifest_path = str(root / "manifest.json")
    receipt_path = str(root / "COMPLETE.json")
    manifest, receipt = read(manifest_path), read(receipt_path)
    if (receipt.get("libraries") != 32 or receipt.get("entries") != 768
            or manifest.get("publication_commit") != receipt.get("publication_commit")
            or manifest.get("protocol_sha256") != sha(LIBRARY)
            or manifest.get("incremental_render_seconds") != 0.0):
        raise ValueError("Complete original 32-camera/768-render inventory required")
    evidence = {}
    for field, path in dict(protocol=LIBRARY, manifest=manifest_path, receipt=receipt_path,
                            controls=native["controls"]).items():
        evidence[field] = path
        evidence[field + "_sha256"] = sha(path)
    profile = dict(
        name=METHOD, backend="direct_render", method=METHOD, prepared_only=False, available=True,
        cpu_threads=2, native_size=512, calibration_or_final_access=False,
        configuration=control["profiles"][0]["configuration"],
        source_sha256=native["source_sha256"], artifact_sha256={},
        publication_commit=receipt["publication_commit"], render_evidence=evidence,
    )
    _, libraries, _ = validate_render_profile(profile)
    assets = dict(native["metadata_sha256"])
    merge_pins(assets, {evidence[key]: evidence[key + "_sha256"] for key in ("protocol", "manifest", "receipt", "controls")})
    # Both predecessor protocols are public. Their original completion
    # artifacts and the large immutable manifest are transitive input pins.
    public = {path: assets[path] for path in (CONTROL, LIBRARY)}
    sources = dict(profile["source_sha256"])
    merge_pins(sources, library["source_sha256"])
    return profile, public, sources, assets, libraries


def identities(settings):
    from control_historical_native_v74 import source_closure as control_source_closure
    from run_solar_distributional_evaluation_v90 import fixed_v86_cases, sha
    from solar_evaluation_protocol_v74 import source_closure

    validate_settings(settings)
    if sha(BASIS) != BASIS_SHA256:
        raise ValueError("Original exact v86 cohort protocol required")
    basis, manifest, cases = fixed_v86_cases(BASIS)
    if len(cases) != 8 or sum(case["positive"] for case in cases) != 4:
        raise ValueError("Retain the original four positive and four nonoverlap cases")
    profile, public, render_sources, assets, libraries = render_profile()
    public[BASIS] = BASIS_SHA256
    sources = source_closure(".")
    merge_pins(sources, render_sources)
    merge_pins(sources, control_source_closure({Path(__file__).name, WORKER_SOURCE, SCORER_SOURCE}))
    for case in cases:
        merge_pins(assets, case["input_sha256"])
        key = case["reference_asset_metadata"]["source_reference_geometry_sha256"]
        library = libraries[key]
        merge_pins(assets, {library["source_summary"]: library["source_summary_sha256"]})
        merge_pins(assets, {entry["image"]: entry["image_sha256"] for entry in library["entries"]})
    for key in ("manifest", "geometry_audit", "learning_contract", "compute_ledger"):
        path = basis[key]
        merge_pins(assets, {path: basis["input_sha256"].get(path) or sha(path)})
    merge_pins(assets, public)
    frozen = dict(
        **settings, settings=settings, purpose=PURPOSE, split="model_selection",
        calibration_or_final_access=False, accepted_results=False, automatic_model_promotion=False,
        cpu_threads=2, tracks=["estimated_sun"], profiles=[profile], basis=BASIS,
        manifest=basis["manifest"], geometry_audit=basis["geometry_audit"],
        full_case_count=len(manifest["cases"]), case_ids=[case["case_id"] for case in cases],
        compute_ledger=basis["compute_ledger"], compute_ledger_sha256=basis["compute_ledger_sha256"],
        reservation_journal=basis["reservation_journal"], blender_executable=basis.get("blender_executable"),
        source_sha256=sources, metadata_sha256=public, input_sha256=assets,
        scoring_revision="v91_full_reference_point_mask",
        scope="Raw primary development outcomes from the exact eight v86 cases and the frozen reference-camera render library.",
    )
    return frozen, cases


def budget(protocol):
    from solar_learning_pipeline_v73 import journal_state

    ledger, journal, charged = journal_state(
        protocol["compute_ledger"], protocol["compute_ledger_sha256"], protocol["reservation_journal"],
    )
    return journal["reservations"], ledger["remaining_baseline_wall_seconds"] - charged["baseline"]


def prepare(settings_path, path):
    from run_solar_distributional_evaluation_v90 import read, write

    relative(path)
    protocol, _ = identities(read(settings_path))
    if Path(protocol["output"]).exists():
        raise FileExistsError("Preserve every previous attempt")
    entries, remaining = budget(protocol)
    if protocol["max_calendar_seconds"] > remaining:
        raise ValueError("Original cap exceeds the remaining canonical baseline allowance")
    protocol.update(prepared_only=True, initial_reservations=entries,
                    created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    write(path, protocol)
    return protocol


def admission(path, commit, started):
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, check_publication, read, sha, utc_now, validate_dispatch

    relative(path)
    protocol = read(path)
    deadline = validate_dispatch(protocol, started, 18000)

    def check():
        if utc_now() >= deadline:
            raise TimeoutError("Original direct-render allowance exhausted during admission")

    check()
    expected, cases = identities(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen matched comparison identity differs")
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, check)
    for name, expected_sha in protocol["input_sha256"].items():
        check()
        if sha(name) != expected_sha:
            raise ValueError("Frozen render, control or case input changed: " + name)
    entries, remaining = budget(protocol)
    initial = protocol["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Frozen canonical reservation history changed")
    cap = protocol["max_calendar_seconds"]
    owned = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if len(owned) > 1 or any(row["category"] != "baseline" or row["seconds"] != cap for row in owned):
        raise ValueError("Existing direct-render charge differs from its original cap")
    if not owned and cap > remaining:
        raise ValueError("Original cap exceeds the remaining canonical baseline allowance")
    check()
    return protocol, cases, deadline


def replay(path, output):
    from run_solar_distributional_evaluation_v90 import JOB_SUFFIX, check_publication, read, sha, utc, validate_dispatch, write
    from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, _sealed_rows

    relative(path)
    relative(output)
    protocol = read(path)
    expected, cases = identities(protocol["settings"])
    if protocol.get("prepared_only") is not False or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen replay identities differ")
    root = Path(protocol["output"])
    seal, watchdog, summary = (read(root / name) for name in ("COMPLETE.json", "watchdog.json", "summary.json"))
    if (not seal["external_watchdog_passed"] or not watchdog["complete"] or not summary["complete"]
            or watchdog["failure"] is not None or watchdog["child_returncode"] != 0
            or any(record["protocol_sha256"] != sha(path) for record in (seal, watchdog, summary))
            or seal["watchdog_sha256"] != sha(root / "watchdog.json")
            or seal["summary_sha256"] != sha(root / "summary.json")):
        raise ValueError("Complete original direct-render evaluation proof required")
    started = utc(watchdog["started_utc"])
    deadline = validate_dispatch(protocol, started, 18000)
    cap = protocol["max_calendar_seconds"]
    if (utc(watchdog["hard_deadline_utc"]) != deadline or not started <= utc(watchdog["ended_utc"]) < deadline
            or watchdog["cap_seconds"] != cap or not 0 < watchdog["calendar_seconds"] <= cap):
        raise ValueError("Original direct-render deadline differs")
    entries, _ = budget(protocol)
    charge = watchdog["compute_reservation"]["reservation"]
    if (charge not in entries or charge["job_id"] != sha(path) + JOB_SUFFIX
            or charge["category"] != "baseline" or charge["seconds"] != cap):
        raise ValueError("Original baseline charge is absent")
    commit = watchdog["publication_commit"]
    if summary["publication_commit"] != commit:
        raise ValueError("Original publication identity changed")
    check_publication(path, protocol["source_sha256"], protocol["metadata_sha256"], commit, lambda: None)
    for asset, checksum in protocol["input_sha256"].items():
        if sha(asset) != checksum:
            raise ValueError("Reopened frozen input differs: " + asset)
    sealed = _sealed_rows(root, protocol, protocol["case_ids"])
    rows = [row for row, _ in sealed]
    if summary["rows"] != rows:
        raise ValueError("Summary differs from the original sealed case grid")
    checks = [_replay_metrics(row, case) for row, case in zip(rows, cases, strict=True)]
    write(output, dict(
        passed=True, replayed_rows=len(checks), rows=checks,
        protocol_sha256=sha(path), summary_sha256=seal["summary_sha256"],
        watchdog_sha256=seal["watchdog_sha256"], accepted_results=False,
        scope="Independent saved-array replay on the eight paired development cases.",
    ))


def main():
    from run_solar_distributional_evaluation_v90 import guarded_run

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        command = sub.add_parser(action)
        command.add_argument("--protocol", required=True)
        if action == "prepare":
            command.add_argument("--settings", required=True)
        elif action in ("run", "worker"):
            command.add_argument("--publication-commit", required=True)
        if action == "run":
            command.add_argument("--python", default=sys.executable)
        elif action == "worker":
            command.add_argument("--controller-pid", required=True, type=int)
        elif action == "replay":
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, Path(__file__).name)
    elif args.action == "replay":
        replay(args.protocol, args.output)
    else:
        from run_solar_matched_matchanything_recovery_v91 import guarded_aligned_worker
        from solar_evaluation_v70 import Backend

        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, Backend)


if __name__ == "__main__":
    main()
