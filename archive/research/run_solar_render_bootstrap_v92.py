"""Bounded five-arm mechanism pilot on the exact eight v86 cases."""
import argparse
from datetime import timedelta
from pathlib import Path
import sys

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, check_publication, guarded_run, read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_matched_direct_render_v91 import budget, identities as direct_identities, merge_pins, relative
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics, guarded_aligned_worker

RUNNER = "run_solar_render_bootstrap_v92.py"
ADAPTER = "solar_render_bootstrap_v92.py"
DIRECT = "reports/solar-matched-direct-render-pilot-v91.json"
DIRECT_REPLAY = "reports/solar_matched_direct_render_results_v91.json"
STUDY = "reports/solar-distributional2000-study-v90.json"
PURPOSE = "physics_first_render_bootstrap_mechanism_cost_pilot_v92"
ARMS = (
    ("reference_only_A1B0C1", "reference_only", True, False),
    ("pose_only_A1B0C1", "pose_only", True, False),
    ("full_A1B0C1", "full", True, False),
    ("full_A1B0C0_uniform", "full", False, False),
    ("full_A1B1C1_unselected_step2000", "full", True, True),
)


def _study_checkpoint():
    from solar_distributional_study_runner_v90 import verify_output

    protocol = read(STUDY)
    root = Path(protocol["output"])
    marker, watchdog, worker = [read(root / name) for name in ("COMPLETE.json", "watchdog.json", "worker_summary.json")]
    summary = root / "science/summary.json"
    if (not marker["external_watchdog_passed"] or not watchdog["complete"] or not worker["complete"]
            or any(x["protocol_sha256"] != sha(STUDY) for x in (marker, watchdog, worker))
            or marker["watchdog_sha256"] != sha(root / "watchdog.json")
            or marker["worker_summary_sha256"] != sha(root / "worker_summary.json")
            or marker["science_summary_sha256"] != verify_output(protocol, root)):
        raise ValueError("Completed, unselected, source-sealed 2,000-update study required")
    pilot = read(protocol["pilot_protocol"])
    checkpoint = root / "science/step_2000.pt"
    checkpoint_sha = read(summary)["artifacts"][checkpoint.name]
    cfg = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, expected_step=2000,
        prior_path=pilot["bank_prior"], prior_sha256=pilot["bank_prior_sha256"],
        teacher_directory=pilot["teacher_directory"], frozen_descriptor_sha256=pilot["expected_frozen_descriptor_sha256"],
    )
    local = {str(path): sha(path) for path in (root / "COMPLETE.json", root / "watchdog.json", root / "worker_summary.json", summary)}
    local.update({str(checkpoint): checkpoint_sha, cfg["prior_path"]: cfg["prior_sha256"]})
    return cfg, protocol["source_sha256"], local


def binding(settings):
    p, cases = direct_identities(settings)
    if p["max_calendar_seconds"] > 600:
        raise ValueError("Explicit mechanism cost-pilot cap cannot exceed 600 seconds")
    original = read(DIRECT)
    direct_replay = read(DIRECT_REPLAY)
    if (p["profiles"] != original["profiles"] or p["case_ids"] != original["case_ids"]
            or p["manifest"] != original["manifest"] or p["geometry_audit"] != original["geometry_audit"]
            or not direct_replay["passed"] or direct_replay["replayed_rows"] != 8
            or direct_replay["protocol_sha256"] != sha(DIRECT)):
        raise ValueError("Reuse the exact completed v91 library, control and eight cases")
    basis = read(p["basis"])
    learning = read(basis["learning_contract"])
    teacher = learning["initializer"]
    frozen = learning["expected_frozen_descriptor_sha256"]
    response, response_sources, response_inputs = _study_checkpoint()
    if (teacher, frozen) != (response["teacher_directory"], response["frozen_descriptor_sha256"]):
        raise ValueError("Matched B-on/B-off arms must use the same frozen teacher")
    sources = p["source_sha256"]
    merge_pins(sources, response_sources)
    for name in (RUNNER, ADAPTER, "solar_distributional_evaluation_v90.py",
                 "run_solar_distributional_evaluation_v90.py"):
        merge_pins(sources, {name: sha(name)})
    direct_profile, = p["profiles"]
    p["profiles"] = [
        dict(name=name, backend="physics_render_bootstrap", direct_profile=direct_profile,
             teacher_directory=teacher, frozen_descriptor_sha256=frozen,
             coarse_response_checkpoint=response if use_b else None,
             configuration=dict(active=active, policy="active" if active else "uniform",
                                budget=128, render_feedback=feedback))
        for name, feedback, active, use_b in ARMS
    ]
    public = {name: sha(name) for name in (DIRECT, DIRECT_REPLAY, STUDY)}
    merge_pins(p["metadata_sha256"], public)
    merge_pins(p["input_sha256"], public)
    merge_pins(p["input_sha256"], response_inputs)
    merge_pins(p["input_sha256"], {str(Path(teacher) / n): h for n, h in learning["initializer_sha256"].items()})
    p.update(
        purpose=PURPOSE, formal_eight_ablation_complete=False, model_selection=False,
        checkpoint_selection=False, inference_revision="render_bootstrap_coarse128_v92",
        scope="Five matched mechanism arms on eight development cases. Fixed 128 patch, 4096 proposal and 2000 final budgets. B-on uses fixed unselected step2000.",
    )
    return p, cases, basis["learning_contract"]


def prepare(settings_path, path):
    relative(path)
    p, _, _ = binding(read(settings_path))
    if Path(p["output"]).exists():
        raise FileExistsError("Preserve every previous attempt")
    entries, remaining = budget(p)
    if p["max_calendar_seconds"] > remaining:
        raise ValueError("Mechanism pilot exceeds original remaining baseline budget")
    p.update(prepared_only=True, initial_reservations=entries,
             created_utc=None, latest_start_utc=None, absolute_deadline_utc=None)
    write(path, p)


def admission(path, commit, started):
    from solar_learning_pipeline_v73 import load_contract, remaining

    p = read(path)
    deadline = validate_dispatch(p, started, 600)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original mechanism allowance exhausted during admission")

    check()
    expected, cases, contract_path = binding(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(key) != value for key, value in expected.items()):
        raise ValueError("Published fixed mechanism profiles, cases or artifact identity changed")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    for asset, checksum in p["input_sha256"].items():
        check()
        if sha(asset) != checksum:
            raise ValueError("Original mechanism input changed: " + asset)
    entries, available = budget(p)
    initial = p["initial_reservations"]
    if entries[:len(initial)] != initial:
        raise ValueError("Original canonical reservation history changed")
    ours = [entry for entry in entries if entry["job_id"] == sha(path) + JOB_SUFFIX]
    cap = p["max_calendar_seconds"]
    if len(ours) > 1 or any(entry["category"] != "baseline" or entry["seconds"] != cap for entry in ours):
        raise ValueError("Mechanism pilot charge identity differs")
    if not ours and cap > min(available, remaining(load_contract(contract_path))["baseline"]):
        raise ValueError("Mechanism pilot exceeds original cumulative baseline allowance")
    check()
    return p, cases, deadline


def replay(path, output):
    """Check original saved rows and source labels without rerunning any model."""
    from solar_evaluation_v70 import digest

    p = read(path)
    expected, cases, _ = binding(p["settings"])
    if p.get("prepared_only") is not False or any(p.get(key) != value for key, value in expected.items()):
        raise ValueError("Mechanism replay protocol differs")
    root = Path(p["output"])
    marker, watchdog, summary = [read(root / n) for n in ("COMPLETE.json", "watchdog.json", "summary.json")]
    if (not marker["external_watchdog_passed"] or not watchdog["complete"] or not summary["complete"]
            or any(x["protocol_sha256"] != sha(path) for x in (marker, watchdog, summary))
            or marker["watchdog_sha256"] != sha(root / "watchdog.json")
            or marker["summary_sha256"] != sha(root / "summary.json")):
        raise ValueError("Successful original external watchdog proof required")
    started = utc(watchdog["started_utc"])
    deadline = validate_dispatch(p, started, 600)
    charge = watchdog["compute_reservation"]["reservation"]
    entries, _ = budget(p)
    if (utc(watchdog["hard_deadline_utc"]) != deadline or not started <= utc(watchdog["ended_utc"]) < deadline
            or charge not in entries or charge["job_id"] != sha(path) + JOB_SUFFIX
            or charge["seconds"] != p["max_calendar_seconds"]):
        raise ValueError("Original mechanism deadline or ledger charge differs")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], watchdog["publication_commit"], lambda: None)
    for asset, checksum in p["input_sha256"].items():
        if sha(asset) != checksum:
            raise ValueError("Original replay input changed: " + asset)
    rows = summary["rows"]
    checks = []
    expected_grid = [(profile, case) for profile in p["profiles"] for case in cases]
    for row, (profile, case) in zip(rows, expected_grid, strict=True):
        key = digest(dict(case=case, profile=profile, track="estimated_sun", sources=p["source_sha256"]))
        row_path, array_path = root / (key + ".json"), root / (key + ".npz")
        seal = read(root / (key + ".COMPLETE.json"))
        if (not seal["complete"] or seal["identity"] != key or seal["row_sha256"] != sha(row_path)
                or seal["arrays_sha256"] != sha(array_path) or row != read(row_path)
                or row["saved_arrays"] != str(array_path) or row["case_id"] != case["case_id"]
                or row["profile"] != profile["name"]):
            raise ValueError("Original per-case mechanism row or arrays differ")
        checks.append(dict(profile=profile["name"], **_replay_metrics(row, case)))
    write(output, dict(
        passed=True, replayed_rows=len(checks), rows=checks, protocol_sha256=sha(path),
        summary_sha256=marker["summary_sha256"], watchdog_sha256=marker["watchdog_sha256"],
        formal_eight_ablation_complete=False, model_selection=False, accepted_results=False,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run", "worker", "replay"):
        command = sub.add_parser(action)
        command.add_argument("--protocol", required=True)
        if action == "prepare":
            command.add_argument("--settings", required=True)
        elif action == "replay":
            command.add_argument("--output", required=True)
        else:
            command.add_argument("--publication-commit", required=True)
            if action == "run":
                command.add_argument("--python", default=sys.executable)
            else:
                command.add_argument("--controller-pid", type=int, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.settings, args.protocol)
    elif args.action == "replay":
        replay(args.protocol, args.output)
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, RUNNER)
    else:
        from solar_render_bootstrap_v92 import RenderBootstrapBackend

        guarded_aligned_worker(args.protocol, args.publication_commit, args.controller_pid, admission, RenderBootstrapBackend)


if __name__ == "__main__":
    main()
