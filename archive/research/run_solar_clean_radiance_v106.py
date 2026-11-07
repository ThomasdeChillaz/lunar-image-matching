"""Run the fixed development clean-radiance control under one original budget reservation."""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX,
    JOURNAL,
    LEDGER,
    TRACKS,
    check_publication,
    guarded_run,
    read,
    sha,
    utc,
    utc_now,
    validate_dispatch,
    write,
)
from run_solar_matched_direct_render_v91 import budget, merge_pins


SOURCE = Path(__file__).name
RENDERER = "render_solar_clean_radiance_v106.py"
MATCHER = "solar_clean_pose_v106.py"
SELECTOR = "prepare_solar_clean_rerender_v106.py"
COHORT = "reports/solar-clean-rerender-cohort-v106.json"
DISPATCH = "reports/solar-clean-radiance-dispatch-v106.json"
OUTPUT = Path("runs/solar-clean-radiance-v106")
RENDER = OUTPUT / "render"
MATCH = OUTPUT / "pose"
ORIGINAL = "reports/solar-radiometric-pilot-dispatch-v101.json"
ORIGINAL_SHA = "b7e821211828566d7d60d5250fb230d02e1e73d6092a96418ce5435ab3306b28"
ORIGINAL_REPLAY = "reports/solar_radiometric_pilot_replay_v101.json"
ORIGINAL_REPLAY_SHA = "4cb8d85c39cabcf84aff8f805bc9160406a268bf47a130acfe346c4f6da258f9"
FINGERPRINT = "reports/solar_dither_fingerprint_result_v105.json"
FINGERPRINT_SHA = "4ebbf0043283c56d2fceb42cd7aca034db5421982c9da11f8d35cf1ec5829232"
RENDER_CAP = 900
CAP = 1200
LEASE = 1800
PURPOSE = "matched_clean_radiance_dither_control_v106"
CASE_IDS = (
    "u013-l00-00-positive", "u013-to-u021-l00-00-nonoverlap",
    "u025-l00-00-positive", "u025-to-u057-l00-00-nonoverlap",
    "u027-l08-01-positive", "u027-to-u059-l08-01-nonoverlap",
    "u061-l00-17-positive", "u061-to-u069-l00-17-nonoverlap",
)
LOCAL_ADDITIONS = (
    SOURCE, RENDERER, MATCHER, SELECTOR,
    "control_solar_png_dither_v105.py",
    "render_solar_development_v67.py", "render_solar_completion_v68.py",
    "render_solar_data_v65.py", "north_camera_scene_v59.py",
    "north_camera_correspondences_v59.py", "blender_camera_control_v31.py",
    "north_terrain_pilot_v32.py", "solar_direct_arm_bridge_v103.py", "solar_portable_evidence_v75.py",
)


def frozen():
    from solar_clean_pose_v106 import profile_specs

    for path, expected in ((ORIGINAL, ORIGINAL_SHA), (ORIGINAL_REPLAY, ORIGINAL_REPLAY_SHA),
                           (FINGERPRINT, FINGERPRINT_SHA)):
        if sha(path) != expected:
            raise ValueError("Frozen predecessor differs: " + path)
    prior = read(ORIGINAL)
    cohort = read(COHORT)
    cases = cohort["cases"]
    if ([row["case_id"] for row in cases] != list(CASE_IDS)
            or not all(case_id in prior["case_ids"] for case_id in CASE_IDS)
            or cohort["output_root"] != str(RENDER)
            or cohort["cap_seconds"] != RENDER_CAP
            or len(cohort["renders"]) != 104
            or cohort["split"] != "model_selection"
            or cohort["calibration_or_final_access"] is not False
            or prior["tracks"] != TRACKS
            or prior["calibration_or_final_access"]
            or prior["compute_ledger"] != LEDGER or prior["reservation_journal"] != JOURNAL):
        raise ValueError("Exact observed eight-case, 104-exposure primary cohort required")
    sources = dict(prior["source_sha256"])
    merge_pins(sources, {path: sha(path) for path in LOCAL_ADDITIONS})
    metadata = dict(prior["metadata_sha256"])
    cohort_sha = sha(COHORT)
    merge_pins(metadata, {ORIGINAL: ORIGINAL_SHA, ORIGINAL_REPLAY: ORIGINAL_REPLAY_SHA,
                          FINGERPRINT: FINGERPRINT_SHA, COHORT: cohort_sha})
    return dict(
        purpose=PURPOSE, output=str(OUTPUT), cohort=COHORT, cohort_sha256=cohort_sha,
        original_dispatch=ORIGINAL, original_replay=ORIGINAL_REPLAY,
        case_ids=list(CASE_IDS), profiles=profile_specs(prior["profiles"][0]),
        source_sha256=sources, metadata_sha256=metadata, expected_software=prior["expected_software"],
        manifest=prior["manifest"], input_sha256=prior["input_sha256"],
        split="model_selection", tracks=TRACKS, native_size=512, cpu_threads=2,
        calibration_or_final_access=False, accepted_results=False, automatic_model_promotion=False,
        compute_ledger=LEDGER, compute_ledger_sha256=prior["compute_ledger_sha256"], reservation_journal=JOURNAL,
        publication_dispatch_lease_seconds=LEASE, max_calendar_seconds=CAP, render_cap_seconds=RENDER_CAP,
        scope="Targeted previously observed development cases. Fresh archived-style RGB8, dither-free RGB8 and RGB16, and an independent-noise sensitivity arm. Full blind 24-template search and unchanged common pose estimator.",
    ), cases


def freeze():
    p, _ = frozen()
    if OUTPUT.exists() or Path(DISPATCH).exists():
        raise FileExistsError("Preserve existing protocol and every partial attempt")
    entries, remaining = budget(p)
    if CAP > remaining:
        raise ValueError("Original canonical allowance cannot admit this bounded comparison")
    created = utc_now()
    latest = created + timedelta(seconds=LEASE)
    p.update(prepared_only=False, initial_reservations=entries, created_utc=created.isoformat(),
             latest_start_utc=latest.isoformat(), absolute_deadline_utc=(latest + timedelta(seconds=CAP)).isoformat())
    write(DISPATCH, p)


def admission(path, commit, started):
    from run_solar_evaluation_v70 import software_identity

    if path != DISPATCH:
        raise ValueError("Use the one fixed prospectively published dispatch path")
    p = read(path)
    deadline = validate_dispatch(p, started, CAP)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original bounded comparison deadline exhausted")

    expected, cases = frozen()
    fields = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p.get("prepared_only") is not False or set(p) != set(expected) | fields
            or any(p[k] != v for k, v in expected.items())):
        raise ValueError("Published dispatch differs from exact fixed study")
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    if software_identity() != p["expected_software"]:
        raise ValueError("Original local CPU/software stratum required")
    entries, remaining = budget(p)
    initial = p["initial_reservations"]
    own = [row for row in entries if row["job_id"] == sha(path) + JOB_SUFFIX]
    if entries[:len(initial)] != initial or len(own) > 1 or any(
        row["category"] != "baseline" or row["seconds"] != CAP or row["refund_on_failure"] for row in own
    ) or (not own and CAP > remaining):
        raise ValueError("Original canonical reservation history or allowance differs")
    check()
    return p, cases, deadline


def worker(path, commit, parent_pid):
    import cv2
    import torch

    from control_solar_png_dither_v105 import renderer_path
    from solar_clean_pose_v106 import evaluate, replay

    execution = read(OUTPUT / "execution.json")
    reservation = read(OUTPUT / "reservation.json")["reservation"]
    if (os.getppid() != parent_pid or execution["controller_pid"] != parent_pid
            or execution["protocol_sha256"] != sha(path) or execution["publication_commit"] != commit
            or reservation["job_id"] != sha(path) + JOB_SUFFIX or reservation["category"] != "baseline"
            or reservation["seconds"] != CAP or reservation["refund_on_failure"]):
        raise ValueError("Original live controller and exact canonical reservation required")
    p, _, hard = admission(path, commit, utc(execution["started_utc"]))
    entries, _ = budget(p)
    if reservation not in entries or hard != utc(execution["hard_deadline_utc"]):
        raise ValueError("Worker cannot change reserved cap or original deadline")
    soft = utc(execution["work_deadline_utc"])

    def check():
        if utc_now() >= soft or time.monotonic() - execution["start_monotonic"] >= CAP - 5:
            raise TimeoutError("Original worker allowance exhausted")

    check()
    with (OUTPUT / "WORKER_STARTED").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    blender = renderer_path()
    render_deadline = min(soft, utc_now() + timedelta(seconds=RENDER_CAP))
    print("Starting 104 render-once development exposures", flush=True)
    subprocess.run(
        [str(blender), "--background", "--factory-startup", "--disable-autoexec", "--threads", "2",
         "--python-exit-code", "2", "--python", RENDERER, "--", "--protocol", COHORT,
         "--publication-commit", commit, "--output", str(RENDER), "--deadline-utc", render_deadline.isoformat()],
        timeout=max(0., (render_deadline - utc_now()).total_seconds()), check=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    )
    check()
    render_receipt = RENDER / "COMPLETE.json"
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    print("Starting blind 24-template pose comparisons", flush=True)
    MATCH.mkdir(exist_ok=False)
    rows, profile_memory = evaluate(p, COHORT, render_receipt, MATCH, check)
    check()
    independent = replay(p, COHORT, render_receipt, MATCH, rows)
    if independent.get("passed") is not True:
        raise ValueError("The independent clean-radiance pose replay did not pass")
    write(OUTPUT / "independent_replay.json", independent)
    check()
    write(OUTPUT / "summary.json", dict(
        complete=True, purpose=PURPOSE, protocol_sha256=sha(path), publication_commit=commit,
        render_receipt=str(render_receipt), render_receipt_sha256=sha(render_receipt),
        independent_replay_sha256=sha(OUTPUT / "independent_replay.json"),
        rows=rows, profile_memory=profile_memory, accepted_results=False, calibration_or_final_access=False,
    ))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    commands.add_parser("freeze")
    for action in ("run", "worker"):
        command = commands.add_parser(action)
        command.add_argument("--protocol", default=DISPATCH)
        command.add_argument("--publication-commit", required=True)
        if action == "run":
            command.add_argument("--python", default=sys.executable)
        else:
            command.add_argument("--controller-pid", type=int, required=True)
    args = cli.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, admission, SOURCE)
    else:
        worker(args.protocol, args.publication_commit, args.controller_pid)


if __name__ == "__main__":
    main()
