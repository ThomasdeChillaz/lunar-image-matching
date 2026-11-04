"""Run the fixed four-slot development test of a simulator attitude prior."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from importlib.metadata import version
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import monotonic

from render_solar_clean_radiance_v106 import digest, read, write
from render_solar_raw_acquisition_v109 import publication
from run_solar_secondary_v112 import JOURNAL, LEDGER, checked, run_step, utc
from solar_compute_budget_v67 import reserve
from solar_secondary_sentry_v112 import DIR as PREVIOUS_BLIND, INPUT_SEAL, normalized_u8, request


ROOT = Path("runs/solar-pose-cone-v113")
DISPATCH = "reports/solar-pose-cone-dispatch-v113.json"
PREVIOUS_DISPATCH = "reports/solar-secondary-sentry-dispatch-v112.json"
PREVIOUS_DISPATCH_SHA = "82d2a0cb780318cfa2303af9945caba1e1ff119c30b541febb5dd52b6fc267fd"
CONTROL = "reports/solar-secondary-sentry-replay-v112.json"
CONTROL_SHA = "66b76e4e38ac9c1b9b13d11cea2a45d7273491d92de33e3f014d94e1f67aeeff"
CAP = 180
LEASE = 3600
SUFFIX = "/pose-cone-v113"
SLOTS = tuple(f"r{i:02d}" for i in range(4))
SOURCE = (
    "run_solar_pose_cone_v113.py",
    "solar_pose_cone_v113.py",
    "solar_blind_beams_v113.py",
    "replay_solar_pose_cone_v113.py",
    "replay_solar_secondary_sentry_v112.py",
)


def local_runtime():
    return dict(platform=platform.system(), machine=platform.machine(), python=platform.python_version(),
                torch=version("torch"), numpy=version("numpy"), opencv=version("opencv-python"))


def opaque_inputs():
    spec = request()
    if tuple(row["id"] for row in spec["slots"]) != SLOTS:
        raise ValueError("Require the original four opaque slots")
    prof = spec["profile"]
    cfg = prof["configuration"]
    if (cfg["pose_initialization"] != "quantization_aware"
            or cfg["inference_revision"] != "solar-v84-quantized-init-exact-cache"
            or (cfg["solar_mode"], cfg["response"], cfg["active"], cfg["budget"], cfg["policy"])
            != ("shared", False, True, 128, "active")):
        raise ValueError("Require the frozen v112 candidate profile")
    ref_table = PREVIOUS_BLIND / "reference-table.json"
    entries = [read(INPUT_SEAL)["request"], read(ref_table)["archive"]]
    entries += [row[role] for row in spec["slots"] for role in ("reference_exr", "query_exr")]
    entries += [dict(path=prof["checkpoint"], sha256=cfg["checkpoint_sha256"])]
    pins = {entry["path"]: entry["sha256"] for entry in entries}
    if len(pins) != 9:
        raise ValueError("Require six exact EXRs, one reference table, one request and one checkpoint")
    for path, checksum in pins.items():
        checked(path, checksum)
    pins.update({str(path): digest(path) for path in (INPUT_SEAL, ref_table)})
    return spec, dict(sorted(pins.items()))


def build(created):
    prior = read(checked(PREVIOUS_DISPATCH, PREVIOUS_DISPATCH_SHA))
    control = read(checked(CONTROL, CONTROL_SHA))
    if (control["coverage_gate_pass"] or not control["passed"]
            or control["summary"]["candidate_oracle_correct"] != 0
            or tuple(row["id"] for row in control["rows"]) != SLOTS
            or control["calibration_or_final_access"]):
        raise ValueError("Only the completed v112 development stop admits this paired diagnostic")
    sources = dict(prior["source_sha256"], **{name: digest(name) for name in SOURCE})
    for name, sha in sources.items():
        checked(name, sha)
    return dict(schema=113, phase="paired_simulator_attitude_cone", split="model_selection",
                root=str(ROOT), cap_seconds=CAP, cpu_threads=2, runtime=local_runtime(),
                calibration_or_final_access=False, cohort=list(SLOTS), prior="reference_ENU_52deg_off_nadir",
                intervention="Restrict only v73 sampled AP3P roots and their LM refits",
                score_after_sealing=True, poses_are_raw=True,
                gate="At least one correct candidate in both positive renderer replicates and no increase in nonoverlap raw or final pose outputs",
                source_sha256=dict(sorted(sources.items())),
                metadata_sha256={LEDGER: prior["metadata_sha256"][LEDGER],
                                 PREVIOUS_DISPATCH: PREVIOUS_DISPATCH_SHA, CONTROL: CONTROL_SHA},
                opaque_input_sha256=opaque_inputs()[1],
                created_utc=created.isoformat(), latest_start_utc=(created + timedelta(seconds=LEASE)).isoformat())


def active(commit, deadline):
    plan = read(DISPATCH)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Require the full published dispatch commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    published = subprocess.check_output(["git", "show", f"{commit}:{DISPATCH}"], timeout=5)
    if published != Path(DISPATCH).read_bytes():
        raise ValueError("Worker dispatch differs from the published Git object")
    if (plan["root"] != str(ROOT) or plan["cohort"] != list(SLOTS)
            or plan["cap_seconds"] != CAP or plan["runtime"] != local_runtime()
            or plan["calibration_or_final_access"]):
        raise ValueError("Wrong frozen worker contract")
    for name, checksum in plan["source_sha256"].items():
        checked(name, checksum)
    spec, pins = opaque_inputs()
    if pins != plan["opaque_input_sha256"]:
        raise ValueError("Opaque scientific inputs changed")
    journal = read(JOURNAL)
    checksum = digest(DISPATCH)
    own, = [row for row in journal["reservations"] if row["job_id"] == checksum + SUFFIX]
    booked = utc(own["reserved_utc"])
    if (journal["ledger_sha256"] != plan["metadata_sha256"][LEDGER]
            or own["seconds"] != CAP or own["category"] != "baseline"
            or own["refund_on_failure"] or deadline != booked + timedelta(seconds=CAP)
            or not booked <= datetime.now(timezone.utc) < deadline):
        raise ValueError("Worker needs the original unexpired reservation")
    return spec, own


def blind(commit, deadline):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2
    import numpy as np
    import torch
    from localize_terrain import jsonable, load_matcher
    from replay_raw_acquisition_v109 import decode, luminance
    from solar_blind_beams_v113 import localize_with_blind_cone_v113
    from solar_frontend_v63 import ReferenceGeometry

    spec, reservation = active(commit, deadline)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prof = spec["profile"]
    cfg = prof["configuration"]
    matcher = load_matcher(prof["checkpoint"], cfg, allow_untrained=True)
    table = ReferenceGeometry.load(read(PREVIOUS_BLIND / "reference-table.json")["archive"]["path"])
    intrinsics = np.asarray(spec["query_intrinsics"])
    saved = []
    images = {}
    bank_dir = ROOT / "banks"
    bank_dir.mkdir()
    for slot in spec["slots"]:
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Frozen worker deadline elapsed")
        for role in ("reference_exr", "query_exr"):
            path = slot[role]["path"]
            if path not in images:
                raw = luminance(decode(Path(path).read_bytes(), exr_size=512))
                images[path] = normalized_u8(raw)
        started = monotonic()
        result, bank, diagnostics = localize_with_blind_cone_v113(
            matcher, images[slot["reference_exr"]["path"]], images[slot["query_exr"]["path"]],
            table, intrinsics, response=cfg["response"], active=cfg["active"],
            budget=cfg["budget"], policy=cfg["policy"])
        output = bank_dir / (slot["id"] + ".json")
        write(output, dict(id=slot["id"], blind_bank=bank.to_dict(),
                           compact_final_pose=jsonable(result.get("pose")),
                           compact_final_reason=result.get("reason"), diagnostics=jsonable(diagnostics),
                           elapsed_seconds=monotonic() - started))
        saved.append(dict(path=str(output), sha256=digest(output)))
    if len(images) != 6:
        raise ValueError("Expected six independently sealed exposures")
    write(ROOT / "COMPLETE.json", dict(schema="v113_sealed_blind_cone", complete=True,
                                      query_truth_opened=False, banks=saved,
                                      dispatch_sha256=digest(DISPATCH), publication_commit=commit,
                                      absolute_deadline_utc=deadline.isoformat(), reservation=reservation))


def launch(commit):
    plan = read(DISPATCH)
    if plan != build(utc(plan["created_utc"])):
        raise ValueError("Frozen dispatch differs from validated experiment")
    publication(DISPATCH, plan, commit)
    if not utc(plan["created_utc"]) <= datetime.now(timezone.utc) <= utc(plan["latest_start_utc"]):
        raise TimeoutError("The one-use publication lease has elapsed")
    ROOT.mkdir()
    checksum = digest(DISPATCH)
    charge = reserve(LEDGER, plan["metadata_sha256"][LEDGER], JOURNAL, checksum + SUFFIX, "baseline", CAP)
    deadline = utc(charge["reservation"]["reserved_utc"]) + timedelta(seconds=CAP)
    finished, error = [], None
    try:
        run_step([sys.executable, __file__, "blind", "--publication-commit", commit,
                  "--deadline-utc", deadline.isoformat()], ROOT / "blind.log", deadline)
        seal = read(ROOT / "COMPLETE.json")
        if seal["complete"] is not True or len(seal["banks"]) != len(SLOTS):
            raise ValueError("Incomplete blind seal")
        finished.append("blind")
    except BaseException as exc:
        error = dict(type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        write(ROOT / "watchdog.json", dict(complete=error is None, error=error,
                                          finished_steps=finished, dispatch_sha256=checksum,
                                          publication_commit=commit, absolute_deadline_utc=deadline.isoformat(),
                                          **charge))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("freeze")
    for name in ("launch", "blind"):
        action = actions.add_parser(name)
        action.add_argument("--publication-commit", required=True)
        if name == "blind":
            action.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        write(DISPATCH, build(datetime.now(timezone.utc)))
    elif args.action == "blind":
        blind(args.publication_commit, utc(args.deadline_utc))
    else:
        with Path("runs/solar-secondary-v112.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.publication_commit)


if __name__ == "__main__":
    main()
