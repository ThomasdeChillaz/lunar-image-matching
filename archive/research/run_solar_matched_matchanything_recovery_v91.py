"""Complete the four unsealed v90 cases after a label-alignment error."""
import argparse
from datetime import timedelta
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_common_protocol, check_publication, guarded_run, guarded_worker,
    read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_matched_matchanything_v90 import identities, relative_path

ORIGINAL = "reports/solar-matched-matchanything-pilot-v90.json"
ORIGINAL_SHA = "debb173f2b4a0b3281b46ba7e6d62843492a4daa89f851f0ce03cb1e533b50a5"
PURPOSE = "matched_matchanything_scoring_recovery_v91"
SCORER = "solar_aligned_scoring_v91.py"
RUNNER = "run_solar_matched_matchanything_recovery_v91.py"


def _sealed_rows(root, protocol, expected_ids):
    from solar_evaluation_v70 import digest

    root = Path(root)
    profile, = protocol["profiles"]
    source = read(protocol["manifest"])
    cases = {case["case_id"]: case for case in source["cases"]}
    found = {}
    for seal_path in sorted(root.glob("*.COMPLETE.json")):
        seal = read(seal_path)
        row_path = Path(seal["row_path"])
        array_path = Path(seal["arrays_path"])
        if row_path.parent != root or array_path.parent != root:
            raise ValueError("Sealed artifacts must stay in their original run directory")
        row = read(row_path)
        case = cases[row["case_id"]]
        key = digest(dict(case=case, profile=profile, track="estimated_sun", sources=protocol["source_sha256"]))
        if (not seal["complete"] or seal["identity"] != key or row_path != root / (key + ".json")
                or array_path != root / (key + ".npz") or seal_path != root / (key + ".COMPLETE.json")
                or seal["row_sha256"] != sha(row_path) or seal["arrays_sha256"] != sha(array_path)
                or row["saved_arrays"] != str(array_path) or row["saved_arrays_sha256"] != seal["arrays_sha256"]
                or row["profile"] != profile["name"] or row["track"] != "estimated_sun"
                or row["case_id"] in found):
            raise ValueError("Original case seal, identity or payload differs")
        found[row["case_id"]] = (row, seal_path)
    if set(found) != set(expected_ids):
        raise ValueError("The exact sealed case inventory differs")
    return [found[case_id] for case_id in expected_ids]


def _original():
    if sha(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError("Original published failing protocol differs")
    p = read(ORIGINAL)
    root = Path(p["output"])
    watchdog = read(root / "watchdog.json")
    log = (root / "worker.log").read_text()
    if (watchdog["complete"] or watchdog["child_returncode"] != 1
            or watchdog["protocol_sha256"] != ORIGINAL_SHA
            or (root / "COMPLETE.json").exists() or (root / "summary.json").exists()
            or "ValueError: Independently projected scoring labels must align with returned reference points" not in log
            or not 0 < watchdog["calendar_seconds"] < p["max_calendar_seconds"]
            or not utc(watchdog["started_utc"]) < utc(watchdog["ended_utc"]) < utc(watchdog["work_deadline_utc"])):
        raise ValueError("Specific terminal v90 label-alignment failure required")
    execution = read(root / "execution.json")
    reservation = read(root / "reservation.json")
    if (execution["protocol_sha256"] != ORIGINAL_SHA or reservation != watchdog["compute_reservation"]
            or execution["publication_commit"] != watchdog["publication_commit"]
            or watchdog["cap_seconds"] != p["max_calendar_seconds"]):
        raise ValueError("Original watchdog, execution or charge differs")
    rows = _sealed_rows(root, p, p["case_ids"][:4])
    inventory = {str(path): sha(path) for path in sorted(root.iterdir()) if path.is_file()}
    return p, watchdog, rows, inventory


def binding(settings):
    if set(settings) != {"output", "max_calendar_seconds", "publication_dispatch_lease_seconds"}:
        raise ValueError("Declare only a fresh recovery output and separate cap/lease")
    if not relative_path(settings["output"]).startswith("runs/"):
        raise ValueError("Recovery output must be a fresh run under runs/")
    p, watchdog, rows, original_files = _original()
    if settings["output"] == p["output"]:
        raise ValueError("Original partial run must remain immutable")
    basis, manifest, cases, profile, sources, metadata, assets = identities(p["settings"])
    if (p["profiles"] != [profile] or p["case_ids"] != [case["case_id"] for case in cases]
            or p["manifest"] != basis["manifest"] or p["source_sha256"] != sources
            or p["input_sha256"] != assets or p["metadata_sha256"] != metadata):
        raise ValueError("Original native control, sources or case identity changed")
    sources = {**sources, SCORER: sha(SCORER), RUNNER: sha(RUNNER)}
    return dict(
        prepared_only=True, purpose=PURPOSE, split=p["split"], settings=settings,
        profiles=p["profiles"], output=settings["output"], max_calendar_seconds=settings["max_calendar_seconds"],
        publication_dispatch_lease_seconds=settings["publication_dispatch_lease_seconds"],
        created_utc=None, latest_start_utc=None, absolute_deadline_utc=None,
        cpu_threads=2, tracks=p["tracks"], calibration_or_final_access=False, accepted_results=False,
        manifest=p["manifest"], basis=p["basis"], full_case_count=len(manifest["cases"]),
        case_ids=p["case_ids"][4:], combined_case_ids=p["case_ids"],
        compute_ledger=p["compute_ledger"], compute_ledger_sha256=p["compute_ledger_sha256"],
        reservation_journal=p["reservation_journal"], blender_executable=p["blender_executable"],
        source_sha256=sources, metadata_sha256={**metadata, ORIGINAL: ORIGINAL_SHA}, input_sha256=assets,
        original_protocol=ORIGINAL, original_output=p["output"],
        original_publication_commit=watchdog["publication_commit"],
        original_files_sha256=original_files, original_baseline_charge=watchdog["compute_reservation"],
        carried_seals={row["case_id"]: str(seal_path) for row, seal_path in rows},
        scope="Only the four unsealed original cases receive fresh inference. The first four retain v90 bytes.",
    ), cases[4:], basis["learning_contract"], p["max_calendar_seconds"]


def prepare(settings_path, protocol_path):
    p, _, _, ceiling = binding(read(settings_path))
    cap = p["max_calendar_seconds"]
    if type(cap) is not int or not 5 < cap <= ceiling:
        raise ValueError("Recovery cap must fit within the original attempt ceiling")
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve all previous output directories")
    write(protocol_path, p)


def admission(path, commit, started):
    from solar_learning_pipeline_v73 import journal_state, load_contract, remaining

    p = read(path)
    check_common_protocol(p)
    expected, cases, contract_path, ceiling = binding(p["settings"])
    mutable = {"prepared_only", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if set(expected) != set(p) or any(p[name] != value for name, value in expected.items() if name not in mutable):
        raise ValueError("Published recovery changed the original partial run, case set or settings")
    deadline = validate_dispatch(p, started, ceiling)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original recovery preflight allowance exhausted")

    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, checksum in p["input_sha256"].items():
        check()
        if sha(asset) != checksum:
            raise ValueError("Original model or case input changed: " + asset)
    contract = load_contract(contract_path)
    _, journal, _ = journal_state(p["compute_ledger"], p["compute_ledger_sha256"], p["reservation_journal"])
    if p["original_baseline_charge"]["reservation"] not in journal["reservations"]:
        raise ValueError("The original failed charge must remain in the canonical ledger")
    ours = [entry for entry in journal["reservations"] if entry["job_id"] == sha(path) + JOB_SUFFIX]
    cap = p["max_calendar_seconds"]
    if len(ours) > 1 or any(entry["category"] != "baseline" or entry["seconds"] != cap for entry in ours):
        raise ValueError("Recovery charge identity differs")
    if not ours and cap > remaining(contract)["baseline"]:
        raise ValueError("Separate recovery charge exceeds remaining baseline allowance")
    check()
    return p, cases, deadline


def guarded_aligned_worker(path, commit, pid, admit, backend_factory):
    # Only the independent post-inference label loader is substituted, and
    # only inside this dedicated, bounded subprocess. Matcher and pose calls
    # are still the exact shared v90 implementation.
    from functools import partial
    from unittest.mock import patch

    import solar_evaluation_v70 as evaluator
    from solar_aligned_scoring_v91 import load_aligned_scoring

    score_after_inference = partial(evaluator.evaluate_case, scoring_loader=load_aligned_scoring)
    with patch.object(evaluator, "evaluate_case", score_after_inference):
        return guarded_worker(path, commit, pid, admit, backend_factory)


def _replay_metrics(row, case):
    import numpy as np

    from solar_aligned_scoring_v91 import load_aligned_scoring

    with np.load(row["saved_arrays"], allow_pickle=False) as source:
        a = {key: source[key] for key in ("reference_xy", "query_xy", "pose",
                                        "scoring_query_xy", "scoring_visible", "scoring_query_cam2world")}
    p, q = a["reference_xy"], a["query_xy"]
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Paired finite native correspondence arrays required")
    truth, method = load_aligned_scoring(case, p)
    if method != row["label_method"]:
        raise ValueError("Frozen label method differs")
    metrics = row["metrics"]
    pose = a["pose"]
    has_pose = pose.shape == (4, 4)
    if not has_pose and pose.shape != (0, 4):
        raise ValueError("Explicit pose-null shape required")
    positive = case["positive"]
    correct = strict = False
    if positive:
        for field in ("query_xy", "visible", "query_cam2world"):
            if not np.array_equal(truth[field], a["scoring_" + field], equal_nan=True):
                raise ValueError("Saved scoring labels differ from reopened source")
        visible = truth["visible"]
        error = np.linalg.norm(q[visible] - truth["query_xy"][visible], axis=1)
        for threshold in (1, 2, 4, 8, 12):
            expected = dict(correct=int((error <= threshold).sum()), co_visible=int(visible.sum()), returned=len(q))
            if metrics["correspondence"][str(threshold)] != expected:
                raise ValueError("Replayed correspondence counts differ")
        if has_pose:
            target = truth["query_cam2world"]
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(pose[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))))
            shift = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            with np.load(case["reference_geometry"], allow_pickle=False) as g:
                y, x = np.where(np.isfinite(g["depth"]) & (g["depth"] > 0))
                rays = np.column_stack((x, y, np.ones(len(y)))) @ np.linalg.inv(g["intrinsics"]).T
                reference_range = float(np.median(g["depth"][y, x] * np.linalg.norm(rays, axis=1)))
            expected = [angle, shift, reference_range]
            observed = [metrics["pose_error"]["rotation_deg"], metrics["pose_error"]["translation_m"], metrics["reference_range_m"]]
            if not np.allclose(expected, observed, rtol=0, atol=1e-9):
                raise ValueError("Replayed pose geometry differs")
            correct = angle <= 2 and shift <= 0.01 * reference_range
            strict = angle <= 0.5 and shift <= 0.001 * reference_range
    elif metrics["correspondence"]:
        raise ValueError("Nonoverlap cannot claim correspondence truth")
    if (not positive or not has_pose) and metrics["pose_error"] is not None:
        raise ValueError("Missing or nonoverlap pose cannot have a positive pose error")
    outcome = dict(correct=correct, strict=strict, pose_returned=has_pose,
                   wrong_positive=positive and has_pose and not correct, raw_nonoverlap=not positive and has_pose)
    if any(metrics[key] != value for key, value in outcome.items()):
        raise ValueError("Replayed raw pose outcomes differ")
    return dict(case_id=case["case_id"], **outcome, correspondence=metrics["correspondence"])


def assemble(path, output):
    from solar_evaluation_v70 import aggregate_rows

    p = read(path)
    orig, _, carried, original_files = _original()
    if p["original_files_sha256"] != original_files:
        raise ValueError("Original v90 partial run changed")
    root = Path(p["output"])
    marker = read(root / "COMPLETE.json")
    watchdog = read(root / "watchdog.json")
    summary = read(root / "summary.json")
    if (not marker["external_watchdog_passed"] or not watchdog["complete"] or not summary["complete"]
            or any(record["protocol_sha256"] != sha(path) for record in (marker, watchdog, summary))
            or marker["watchdog_sha256"] != sha(root / "watchdog.json")
            or marker["summary_sha256"] != sha(root / "summary.json")):
        raise ValueError("Complete original external recovery receipt required")
    fresh = _sealed_rows(root, p, p["case_ids"])
    if summary["rows"] != [row for row, _ in fresh] or p["combined_case_ids"] != orig["case_ids"]:
        raise ValueError("Sealed fresh case inventory differs")
    cases = {case["case_id"]: case for case in read(p["manifest"])["cases"]}
    if sha(p["manifest"]) != p["input_sha256"][p["manifest"]]:
        raise ValueError("Original case manifest changed")
    rows = []
    provenance = []
    replayed = []
    for phase, group in (("original_v90_sealed", carried), ("recovered_v91_fresh", fresh)):
        for row, seal_path in group:
            case = cases[row["case_id"]]
            for asset, checksum in case["input_sha256"].items():
                if sha(asset) != checksum:
                    raise ValueError("Reopened original case input changed")
            replayed.append(_replay_metrics(row, case))
            provenance.append(dict(case_id=row["case_id"], phase=phase, seal_path=str(seal_path), seal_sha256=sha(seal_path)))
            rows.append(row)
    if [row["case_id"] for row in rows] != orig["case_ids"]:
        raise ValueError("The exact original eight paired cases are required")
    write(output, dict(
        complete=True, independent_saved_metric_replay_passed=True, original_run_complete=False,
        original_protocol_sha256=ORIGINAL_SHA, original_watchdog_sha256=original_files[str(Path(orig["output"]) / "watchdog.json")],
        recovery_protocol_sha256=sha(path), recovery_watchdog_sha256=marker["watchdog_sha256"],
        carried_rows=len(carried), fresh_rows=len(fresh), rows=rows, provenance=provenance,
        replayed=replayed, table=aggregate_rows(rows), accepted_results=False,
        timing_scope="Per-row model-resident inference timings span two separately charged runs; no single-run eight-case calendar claim.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--settings", required=True)
    prep.add_argument("--protocol", required=True)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--protocol", required=True)
    assembly.add_argument("--output", required=True)
    for action in ("run", "worker"):
        sub = commands.add_parser(action)
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--publication-commit", required=True)
        if action == "run":
            sub.add_argument("--python", default=sys.executable)
        else:
            sub.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "assemble":
        assemble(args.protocol, args.output)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, RUNNER)
    else:
        from solar_evaluation_v70 import Backend

        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, Backend)


if __name__ == "__main__":
    main()
