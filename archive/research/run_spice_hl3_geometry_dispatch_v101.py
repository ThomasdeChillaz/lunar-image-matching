"""One-time source-published v67 admission for the 24-pair SPICE-HL3 diagnostic."""
import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from run_solar_crosslight_mechanism_v98 import terminal_predecessors
from run_solar_distributional_evaluation_v90 import (
    JOB_SUFFIX, JOURNAL, LEDGER, TRACKS, check_publication, guarded_run,
    read, sha, utc, utc_now, validate_dispatch, write,
)
from run_solar_matched_direct_render_v91 import budget, merge_pins, relative
import spice_hl3_optical_transfer_v100 as optical


SOURCE = Path(__file__).name
CORE = "spice_hl3_optical_transfer_v100.py"
CORE_SHA = "011f4f5ac3b793f4d816ddf744152fc48d9afba8d827221ad0d645309d15e8fa"
PREPARED_MANIFEST = "spice_hl3_c_geometry_transfer_prepared.json"
PREPARED_SHA = "d4d9d151be40a80e8fbc628138717141a6279605b7125fd03512731f2454a435"
LEDGER_SHA = "51680600269e37fe745ded5181d04ef61de8713bf01b3c1792415c1a13802a5c"
CROSS_LIGHT = "reports/solar-full-crosslight-rescue-dispatch-v99.json"
CROSS_LIGHT_SHA = "5ca6a4a5de7c7dcd3987db92007d7485074485bb746ebcecf5efef8b08612817"
DISPATCH = "reports/spice-hl3-c-geometry-dispatch-v101.json"
PREPARATION = "reports/spice-hl3-c-geometry-preparation-v101.json"
OUTPUT = "runs/spice-hl3-c-geometry-v101"
ASSET_ROOT = "runs/spice-hl3-c-assets-v101"
PROFILE = "sift_nominal_relative_geometry"
TRACK = "qualified_mocap"
SPLIT = "supplementary_analog_diagnostic"
PURPOSE = "spice_hl3_c_native_zed2_geometry_transfer_v101"
CAP_SECONDS = 120
SCOPE = ("Supplementary physical lunar-analog optics. Twenty-four within-traverse ZED2 left pairs "
         "at author-rectified native 640×360 on one laboratory site. Mocap only scores nominal relative "
         "rotation, signed translation direction and epipolar consistency. No calibration or final "
         "reserve, principal 512-pixel comparison, metric translation, or illumination-transfer claim.")


def manifest_digest(value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def software():
    import cv2
    import numpy as np

    return dict(python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
                opencv=cv2.__version__, numpy=np.__version__, cpu_thread_ceiling=2)


def binding(settings):
    from control_historical_native_v74 import source_closure

    if not isinstance(settings, dict) or set(settings) != {"asset_root", "publication_dispatch_lease_seconds"}:
        raise ValueError("Specify only the exact local input bundle and an explicit publication lease")
    lease = settings["publication_dispatch_lease_seconds"]
    if settings["asset_root"] != ASSET_ROOT or type(lease) is not int or not 0 < lease <= 1800:
        raise ValueError("Fixed repository-relative asset directory and finite original dispatch lease required")
    root = Path.cwd() / ASSET_ROOT
    if not root.is_dir() or root.resolve() != root:
        raise ValueError("Stage the exact bundle in the checkout with no directory symlink")
    source_path = optical.asset(root, dict(path=PREPARED_MANIFEST, sha256=PREPARED_SHA))
    if sha(source_path) != PREPARED_SHA or sha(optical.__file__) != CORE_SHA:
        raise ValueError("Exact original prepared manifest and optical program required")
    if sha(LEDGER) != LEDGER_SHA or sha(CROSS_LIGHT) != CROSS_LIGHT_SHA:
        raise ValueError("Original canonical budget and serial predecessor required")
    m, frames, size, _, distortion, truth = optical.validate_manifest(source_path, allow_prepared=True)
    cases = m["cases"]
    stream_counts = Counter(frames[c["reference"]]["stream_id"] for c in cases)
    endpoints = [c[key] for c in cases for key in ("reference", "query")]
    if (m["prepared_only"] is not True or m["trajectory"] != "C" or m["track"] != TRACK
            or m["camera_key"] != "zed2:left" or tuple(m["crop"]) != (0, 0, 640, 360)
            or size != (640, 360) or distortion.any() or len(cases) != 24
            or len(set(endpoints)) != 48 or set(endpoints) != set(frames) or len(truth) != 48
            or stream_counts != {"fast": 12, "slow": 12}
            or any(set(c) != {"id", "reference", "query"} for c in cases)):
        raise ValueError("Exact 24-pair native ZED2 optical transfer contract required")
    enabled = dict(m, prepared_only=False)
    sources = source_closure({
        "run_solar_distributional_evaluation_v90.py", "run_solar_crosslight_mechanism_v98.py",
        "run_solar_matched_direct_render_v91.py", "solar_compute_budget_v67.py", "solar_learning_pipeline_v73.py",
        "control_historical_native_v74.py",
    })
    merge_pins(sources, {SOURCE: sha(__file__), CORE: CORE_SHA})
    predecessors = {CROSS_LIGHT: CROSS_LIGHT_SHA}
    return dict(
        purpose=PURPOSE, scope=SCOPE, output=OUTPUT, settings=settings, split=SPLIT, tracks=[TRACK],
        profiles=[dict(name=PROFILE)], cpu_threads=2, calibration_or_final_access=False,
        accepted_results=False, automatic_model_promotion=False, native_size=list(size),
        case_ids=[c["id"] for c in cases], case_counts_by_stream=dict(stream_counts),
        expected_software=software(), compute_ledger=LEDGER, compute_ledger_sha256=LEDGER_SHA,
        reservation_journal=JOURNAL, source_sha256=sources,
        metadata_sha256={LEDGER: LEDGER_SHA, **predecessors}, serial_predecessors=predecessors,
        prepared_manifest_sha256=PREPARED_SHA, optical_manifest=enabled, optical_manifest_sha256=manifest_digest(enabled),
        publication_dispatch_lease_seconds=lease, max_calendar_seconds=CAP_SECONDS,
        publication_ref="origin/refs/heads/main", automatic_execution=False,
    ), cases


def prepare(settings_path, path, *, freeze=False):
    if str(relative(path)) != (DISPATCH if freeze else PREPARATION):
        raise ValueError("Use the single versioned preparation or dispatch protocol path")
    p, _ = binding(read(settings_path))
    if Path(p["output"]).exists() or Path(DISPATCH).exists():
        raise FileExistsError("Preserve the original dispatch and every prior attempt")
    initial, remaining = budget(p)
    if CAP_SECONDS > remaining:
        raise ValueError("Original baseline allowance cannot admit this diagnostic")
    created = latest = absolute = None
    if freeze:
        terminal_predecessors(p)
        created = utc_now()
        latest = created + timedelta(seconds=p["publication_dispatch_lease_seconds"])
        absolute = latest + timedelta(seconds=CAP_SECONDS)
    p.update(prepared_only=not freeze, initial_reservations=initial,
             created_utc=created.isoformat() if created else None,
             latest_start_utc=latest.isoformat() if latest else None,
             absolute_deadline_utc=absolute.isoformat() if absolute else None)
    write(path, p)


def exact(p):
    expected, cases = binding(p["settings"])
    state = {"prepared_only", "initial_reservations", "created_utc", "latest_start_utc", "absolute_deadline_utc"}
    if (p["prepared_only"] is not False or set(p) != set(expected) | state
            or any(p[key] != value for key, value in expected.items())):
        raise ValueError("Exact source-frozen optical dispatch required")
    return cases


def published_main(commit, check):
    """Require reachability from the currently served public branch, beyond git show."""
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Full published source commit required")
    check()
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        response = subprocess.check_output(["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
                                           stderr=subprocess.DEVNULL, timeout=10, env=environment).decode().split()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Could not verify current GitHub publication") from exc
    check()
    if len(response) != 2 or response[1] != "refs/heads/main" or not re.fullmatch("[0-9a-f]{40}", response[0]):
        raise ValueError("Expected exact origin main publication proof")
    head = response[0]
    if head != commit:
        status = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], timeout=5,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if status:
            raise ValueError("Dispatch commit is not verified as published on origin main")
    check()
    return head


def admission(path, commit, started, *, verify_remote):
    if str(relative(path)) != DISPATCH:
        raise ValueError("Single published dispatch path required")
    p = read(path)
    if p["prepared_only"] is not False:
        raise ValueError("A preparation cannot start optical inference")
    deadline = validate_dispatch(p, started, CAP_SECONDS)

    def check():
        if utc_now() >= deadline - timedelta(seconds=5):
            raise TimeoutError("Original optical dispatch allowance exhausted")

    check()
    cases = exact(p)
    terminal_predecessors(p)
    check_publication(path, p["source_sha256"], p["metadata_sha256"], commit, check)
    if verify_remote:
        published_main(commit, check)
    reservations, remaining = budget(p)
    initial = p["initial_reservations"]
    owned = [entry for entry in reservations if entry["job_id"] == sha(path) + JOB_SUFFIX]
    if (reservations[:len(initial)] != initial or len(owned) > 1
            or any(entry["category"] != "baseline" or entry["seconds"] != CAP_SECONDS or entry["refund_on_failure"]
                   for entry in owned)
            or (not owned and CAP_SECONDS > remaining)):
        raise ValueError("Exact canonical nonrefundable baseline allowance required")
    check()
    return p, cases, deadline


def controller_admission(path, commit, started):
    p, cases, deadline = admission(path, commit, started, verify_remote=True)
    # The shared v90 watchdog requires these fields solely to check the
    # order of the completed worker rows. The published protocol, optical
    # manifest and scientific outputs retain their true supplementary track.
    controller_view = dict(p, split="model_selection", tracks=TRACKS)
    return controller_view, [dict(case_id=c["id"]) for c in cases], deadline


def verify_ticket(p, path, commit, controller_pid):
    root = Path(p["output"])
    ticket = read(root / "execution.json")
    receipt = read(root / "reservation.json")["reservation"]
    entries, _ = budget(p)
    if (os.getppid() != controller_pid or ticket["controller_pid"] != controller_pid
            or ticket["protocol_sha256"] != sha(path) or ticket["publication_commit"] != commit
            or receipt not in entries or receipt["job_id"] != sha(path) + JOB_SUFFIX
            or receipt["category"] != "baseline" or receipt["seconds"] != CAP_SECONDS or receipt["refund_on_failure"]):
        raise ValueError("Original live controller, published lease and nonrefundable canonical charge required")
    return ticket


def worker(path, commit, controller_pid):
    if str(relative(path)) != DISPATCH:
        raise ValueError("Single published dispatch path required")
    p = read(path)
    ticket = verify_ticket(p, path, commit, controller_pid)
    started = utc(ticket["started_utc"])
    p, cases, hard = admission(path, commit, started, verify_remote=False)
    soft = utc(ticket["work_deadline_utc"])
    if hard != utc(ticket["hard_deadline_utc"]) or soft != hard - timedelta(seconds=5):
        raise ValueError("Original worker deadline cannot change")

    def remaining():
        wall = (soft - utc_now()).total_seconds()
        clock = CAP_SECONDS - 5 - (time.monotonic() - ticket["start_monotonic"])
        return min(wall, clock)

    if remaining() <= 0:
        raise TimeoutError("Original optical lease exhausted before inference")
    root = Path(p["output"])
    with (root / "WORKER_STARTED").open("x") as marker:
        marker.write(str(os.getpid()) + "\n")
    enabled = Path(ASSET_ROOT) / ("spice_hl3_enabled_v101_" + sha(path) + ".json")
    optical.write(enabled, p["optical_manifest"])
    if sha(enabled) != p["optical_manifest_sha256"]:
        raise ValueError("Enabled optical manifest differs from exact published bytes")
    if remaining() <= 0:
        raise TimeoutError("Original optical lease exhausted during setup")
    result = root / "optical"
    optical.run(enabled, result, remaining())
    replay_path = root / "independent_replay.json"
    optical.replay(enabled, result, replay_path)
    replay = read(replay_path)
    scientific = read(result / "summary.json")
    if (replay["passed"] is not True or [r["case_id"] for r in replay["rows"]] != [c["id"] for c in cases]
            or any(r["arm"] != "reference" or r["independently_scored"] is None for r in replay["rows"])
            or scientific["track"] != TRACK or scientific["native_size"] != p["native_size"]
            or remaining() <= 0):
        raise ValueError("Full original independently replayed optical outcome required within deadline")
    write(root / "summary.json", dict(
        complete=True, protocol_sha256=sha(path), publication_commit=commit, purpose=PURPOSE, scope=SCOPE,
        scientific_split=SPLIT, scientific_track=TRACK, calibration_or_final_access=False, accepted_results=False,
        enabled_manifest=str(enabled), enabled_manifest_sha256=sha(enabled),
        optical_summary=str(result / "summary.json"), optical_summary_sha256=sha(result / "summary.json"),
        independent_replay=str(replay_path), independent_replay_sha256=sha(replay_path),
        predecessor_watchdogs=terminal_predecessors(p),
        rows=[dict(case_id=c["id"], profile=PROFILE, track=TRACKS[0]) for c in cases],
        rows_scope="Watchdog ordering keys only. Scientific results are in optical/ and independent_replay.json.",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "freeze", "run", "worker"):
        command = commands.add_parser(action)
        command.add_argument("--protocol", required=True)
        if action in ("prepare", "freeze"):
            command.add_argument("--settings", required=True)
        else:
            command.add_argument("--publication-commit", required=True)
        if action == "run":
            command.add_argument("--python", default=sys.executable)
        elif action == "worker":
            command.add_argument("--controller-pid", required=True, type=int)
    args = parser.parse_args()
    if args.action in ("prepare", "freeze"):
        prepare(args.settings, args.protocol, freeze=args.action == "freeze")
    elif args.action == "run":
        guarded_run(args.protocol, args.publication_commit, args.python, controller_admission, SOURCE)
    else:
        worker(args.protocol, args.publication_commit, args.controller_pid)


if __name__ == "__main__":
    main()
