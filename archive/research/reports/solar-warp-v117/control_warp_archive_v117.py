"""Admit one procedural timing run and one post-v115 archival warp run."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.request import Request, urlopen

from solar_compute_budget_v67 import reserve
from solar_learning_pipeline_v73 import load_contract, remaining
from run_solar_secondary_v112 import run_step
from prepare_warp_roster_v117 import roster as original_roster

HOME = Path("reports/solar-warp-v117")
WORK = Path("runs/solar-warp-v117")
LEDGER = "reports/solar_compute_admission_v67.json"
LEDGER_SHA = "51680600269e37fe745ded5181d04ef61de8713bf01b3c1792415c1a13802a5c"
JOURNAL = "runs/solar-compute-reservations-v67.json"
CONTRACT = "reports/solar_learning_contract_v79.json"
V115 = "https://api.github.com/repos/ThomasdeChillaz/lunar-image-matching/actions/runs/41941904269"
V115_HEAD = "a95d290d9ecd001b2316e90285ae7bb2bff28c87"
V115_PROTOCOL = "c43b7a0a5236b27055a7179caf1cfa1a274b18743a9fd868d0054fa96d23878c"
V115_TICKET = "5394f9ed554357c3c5e20c0a4d9d98132c6e3b5a71ac330e15a43970212b7c36"
CAPS = {"timing": 60, "archive": 180}
LEASE = timedelta(hours=1)
LOCAL_SOURCES = (
    "control_warp_archive_v117.py", "warp_candidates.py", "archived_warp_worker.py",
    "prepare_warp_roster_v117.py", "smoke_warp_candidate_v117.py", "verify_warp_archive_v117.py",
)
REUSED_SOURCES = (
    "lunar_geometry_v2.py", "lunar_geometry_v5.py", "north_camera_variation_v59.py",
    "north_render_plan_v31.py", "replay_raw_acquisition_v109.py", "solar_compute_budget_v67.py",
    "run_solar_secondary_v112.py", "solar_learning_pipeline_v73.py",
)
ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "VECLIB_MAXIMUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1", "OPENCV_IO_ENABLE_OPENEXR": "1"}


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc)


def stamp(value):
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise ValueError("An original UTC timestamp is required")
    return t.astimezone(timezone.utc)


def official(url):
    with urlopen(Request(url, headers={"Accept": "application/vnd.github+json",
                                      "User-Agent": "lunar-warp-v117-admission"}), timeout=12) as response:
        if response.geturl() != url:
            raise ValueError("Unexpected GitHub metadata redirect")
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("GitHub metadata exceeded its read bound")
    return json.loads(raw)


def terminal(receipt_path, get=official):
    run = get(V115)
    if (run["id"] != 41941904269 or run["head_sha"] != V115_HEAD or run["run_attempt"] != 1
            or run["event"] != "workflow_dispatch"):
        raise ValueError("Authoritative status does not identify the original five-arm workflow")
    if run["status"] != "completed":
        raise RuntimeError("The original v115 comparison is still live")
    if not run.get("conclusion"):
        raise ValueError("Official terminal conclusion is missing")
    metadata = get(V115 + "/artifacts?per_page=100")
    artifacts = metadata["artifacts"]
    if len(artifacts) != metadata["total_count"]:
        raise ValueError("Incomplete official v115 artifact inventory")
    if receipt_path is None:
        if artifacts or run["conclusion"] == "success":
            raise ValueError("Archive the original v115 artifacts before the next scientific job")
    else:
        receipt = read(receipt_path)
        if (receipt["schema"] != "solar_five_arm_official_artifact_audit_v115"
                or receipt["run_id"] != run["id"] or receipt["conclusion"] != run["conclusion"]
                or receipt["protocol_sha256"] != V115_PROTOCOL or receipt["ticket_sha256"] != V115_TICKET
                or receipt["new_inference"] is not False
                or (run["conclusion"] == "success" and
                    (not receipt["complete_archive_verified"] or not receipt["independently_replayed"]))):
            raise ValueError("Expected official v115 archived evidence and replay")
        checked = {(a["id"], a["digest"]) for a in receipt["archived_downloads"].values()}
        advertised = {(a["id"], a["digest"]) for a in artifacts}
        if not checked or checked != advertised:
            raise ValueError("Preserve all artifacts of the original v115 attempt")
        for record in receipt["records"].values():
            path = Path(receipt_path).parent / record["path"]
            if not path.resolve().is_relative_to(Path(receipt_path).parent.resolve()) or sha(path) != record["sha256"]:
                raise ValueError("A preserved original v115 artifact or replay changed")
    return dict(run_id=run["id"], head_sha=V115_HEAD, run_attempt=1, conclusion=run["conclusion"],
                artifact_audit_sha256=sha(receipt_path) if receipt_path else None)


def timing_proof():
    output = WORK / "timing"
    receipt_path = output / "fixture" / "receipt.json"
    receipt = read(receipt_path)
    complete = read(output / "COMPLETE.json")
    forecast = receipt["full_scope_forecast"]
    seconds = forecast["projected_seconds"]
    if (complete["phase"] != "timing" or complete.get("complete") is not True
            or complete["timing_receipt_sha256"] != sha(receipt_path) or receipt.get("passed") is not True
            or forecast.get("admitted") is not True or not 0 < seconds <= CAPS["archive"]
            or forecast["projected_uncompressed_evidence_bytes"] > 512 * 1024**2):
        raise ValueError("Native timing failed or lacks a complete frozen receipt")
    return dict(receipt_sha256=sha(receipt_path), completion_sha256=sha(output / "COMPLETE.json"),
                projected_archive_seconds=seconds)


def immutable(phase, verify_rosters=True):
    inputs = HOME / "inference-inputs.json"
    labels = HOME / "scoring-inputs.json"
    if verify_rosters:
        expected = original_roster(Path.cwd(), verify_scoring=False)
        if (read(inputs), read(labels)) != expected:
            raise ValueError("Published inference and scoring rosters are not the exact archived split")
    sources = [*(str(HOME / n) for n in LOCAL_SOURCES), *REUSED_SOURCES, CONTRACT, LEDGER]
    if sha(LEDGER) != LEDGER_SHA:
        raise ValueError("Original compute ledger changed")
    return dict(schema="solar_warp_archival_gate_v117", phase=phase, split="model_selection",
                cpu_threads=2, cap_seconds=CAPS[phase], calibration_or_final_access=False,
                primary_cell_pixels=32, sensitivity_cell_pixels=16, known_reference_illumination=True,
                query_pose_read_after_prediction_seal=True, native_size=[512, 512], expected_prediction_rows=32,
                source_sha256={name: sha(name) for name in sources},
                inputs=dict(path=str(inputs), sha256=sha(inputs)),
                labels=dict(path=str(labels), sha256=sha(labels)),
                software=dict(platform=platform.system(), machine=platform.machine(),
                              python=platform.python_version(), numpy=version("numpy"), opencv=version("opencv-python")),
                budget_ledger_sha256=LEDGER_SHA, reservation_journal=JOURNAL)


def publication(path, protocol, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Require the exact public Git commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    names = (str(path), *protocol["source_sha256"], protocol["inputs"]["path"], protocol["labels"]["path"])
    for name in names:
        contents = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=5)
        if contents != Path(name).read_bytes():
            raise ValueError("Execution byte differs from the published revision: " + name)


def freeze(phase, predecessor, path, *, get=official):
    if path.exists():
        raise FileExistsError("Preserve the original one-use dispatch")
    verified = terminal(predecessor, get=get)
    protocol = immutable(phase)
    timing = timing_proof() if phase == "archive" else None
    start = now()
    put(path, dict(protocol, prepared_only=False, predecessor=verified, timing=timing,
                   created_utc=start.isoformat(), latest_start_utc=(start + LEASE).isoformat()))


def worker_ticket(args, expected_mode=None):
    ticket_path = Path(args.ticket).resolve()
    ticket = read(ticket_path)
    mode = expected_mode or args.mode
    phase = "timing" if mode == "timing" else "archive"
    output = (WORK / phase).resolve()
    if (ticket_path.parent != output or ticket_path.name != f"{mode}-ticket.json"
            or ticket["controller_pid"] != os.getppid() or ticket["phase"] != phase or ticket["mode"] != mode
            or not stamp(ticket["reserved_utc"]) <= now() < stamp(ticket["deadline_utc"])
            or (stamp(ticket["deadline_utc"]) - stamp(ticket["reserved_utc"])).total_seconds() != CAPS[phase]):
        raise ValueError("Worker requires the live controller and original reservation deadline")
    booking = read(JOURNAL)
    rows = [r for r in booking["reservations"] if r["job_id"] == ticket["protocol_sha256"] + "/warp-" + phase]
    if (booking["ledger_sha256"] != LEDGER_SHA or len(rows) != 1 or rows[0]["seconds"] != CAPS[phase]
            or rows[0]["category"] != "baseline" or rows[0]["refund_on_failure"] is not False
            or rows[0]["reserved_utc"] != ticket["reserved_utc"]):
        raise ValueError("Worker requires the exact original canonical charge")
    if phase == "archive":
        expected = ticket["arguments"]
        keys = ("repo", "inputs", "inputs_sha256", "output", "labels", "labels_sha256",
                "predictions", "predictions_sha256")
        actual = {key: str(getattr(args, key)) if getattr(args, key) is not None else None for key in keys}
        if actual != expected:
            raise ValueError("Worker arguments are outside the sealed stage")
    if any(os.environ.get(k) != v for k, v in ENV.items()):
        raise ValueError("Thread and decoder settings must be supplied by the original controller")
    return ticket


def archive(protocol, output, issue, step):
    from archived_warp_worker import ROWS, evidence_path

    prediction = output / "predictions.json"
    scores = output / "scores.json"
    replay = output / "independent-replay.json"
    script = HOME / "archived_warp_worker.py"
    base = dict(repo=str(Path.cwd()), inputs=protocol["inputs"]["path"], inputs_sha256=protocol["inputs"]["sha256"],
                labels=None, labels_sha256=None, predictions=None, predictions_sha256=None)
    operations = lambda a: [arg for key, value in a.items() if value is not None
                            for arg in ("--" + key.replace("_", "-"), value)]
    predict_args = dict(base, output=str(prediction))
    ticket = issue("predict", predict_args)
    step([sys.executable, str(script), "predict", "--ticket", str(ticket), *operations(predict_args)], "predict")
    prediction_sha = sha(prediction)
    raw = read(prediction)
    if len(raw["rows"]) != ROWS or raw["evidence_bytes"] > 512 * 1024**2:
        raise ValueError("Incomplete or oversized evidence cannot be scored")
    evidence = [evidence_path(output, row) for row in raw["rows"]]
    if len(set(evidence)) != ROWS or sum(p.stat().st_size for p in evidence) != raw["evidence_bytes"]:
        raise ValueError("The full prediction archive must seal each evidence cell")
    put(output / "PREDICTIONS_SEALED.json", dict(rows=ROWS, sha256=prediction_sha, evidence_bytes=raw["evidence_bytes"]))
    score_args = dict(base, output=str(scores), labels=protocol["labels"]["path"],
                      labels_sha256=protocol["labels"]["sha256"], predictions=str(prediction),
                      predictions_sha256=prediction_sha)
    ticket = issue("score", score_args)
    step([sys.executable, str(script), "score", "--ticket", str(ticket), *operations(score_args)], "score")
    replay_args = dict(score_args, output=str(replay), scores=str(scores), scores_sha256=sha(scores))
    step([sys.executable, str(HOME / "verify_warp_archive_v117.py"), *operations(replay_args)], "independent-replay")
    replayed = read(replay)
    if (replayed["passed"] is not True or replayed["prediction_sha256"] != prediction_sha
            or replayed["scores_sha256"] != sha(scores) or replayed["replayed_rows"] != ROWS):
        raise ValueError("Expected complete independent saved-cell and scoring replay")
    return dict(predictions_sha256=prediction_sha, scores_sha256=sha(scores),
                independent_replay_sha256=sha(replay), advancement_gate=replayed["gate"])


def run(phase, commit, predecessor, path, *, get=official):
    p = read(path)
    output = WORK / phase
    expected = immutable(phase)
    if (p.get("prepared_only") is not False or any(p[k] != v for k, v in expected.items())
            or stamp(p["latest_start_utc"]) != stamp(p["created_utc"]) + LEASE):
        raise ValueError("Require the exact published execution freeze")
    publication(path, p, commit)
    if p["predecessor"] != terminal(predecessor, get=get) or p["timing"] != (timing_proof() if phase == "archive" else None):
        raise ValueError("Predecessor or original timing evidence differs from the freeze")
    if not stamp(p["created_utc"]) <= now() <= stamp(p["latest_start_utc"]):
        raise TimeoutError("The original one-use launch lease has expired")
    if remaining(load_contract(CONTRACT))["baseline"] < CAPS[phase]:
        raise ValueError("The original cumulative compute allowance cannot cover the cap")
    output.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha(path)
    charge = reserve(LEDGER, LEDGER_SHA, JOURNAL, protocol_sha + "/warp-" + phase, "baseline", CAPS[phase])
    reserved = charge["reservation"]["reserved_utc"]
    deadline = stamp(reserved) + timedelta(seconds=CAPS[phase])
    os.environ.update(ENV, PYTHONPATH=str(Path.cwd()))
    steps = []
    status = dict(phase=phase, protocol_sha256=protocol_sha, source_commit=commit,
                  deadline_utc=deadline.isoformat(), charge=charge, calibration_or_final_access=False)

    def issue(mode, arguments=None):
        target = output / f"{mode}-ticket.json"
        put(target, dict(phase=phase, mode=mode, controller_pid=os.getpid(), reserved_utc=reserved,
                         deadline_utc=deadline.isoformat(), protocol_sha256=protocol_sha, arguments=arguments))
        return target

    def step(command, name):
        run_step(command, output / (name + ".log"), deadline)
        steps.append(name)

    try:
        if phase == "timing":
            ticket = issue("timing")
            step([sys.executable, __file__, "timing-worker", "--ticket", str(ticket)], "timing")
            extra = {"timing_receipt_sha256": sha(output / "fixture" / "receipt.json")}
        else:
            extra = archive(p, output, issue, step)
        if now() >= deadline:
            raise TimeoutError("Original external deadline elapsed before completion")
        put(output / "COMPLETE.json", dict(status, **extra, steps=steps, complete=True))
    except BaseException as e:
        put(output / "FAILED.json", dict(status, steps=steps, complete=False, error_type=type(e).__name__, error=str(e)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "run"):
        command = children.add_parser(name)
        command.add_argument("phase", choices=tuple(CAPS))
        command.add_argument("--v115-audit", type=Path)
        if name == "run":
            command.add_argument("--public-commit", required=True)
    status = children.add_parser("status")
    status.add_argument("--v115-audit", type=Path)
    timing = children.add_parser("timing-worker")
    timing.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(terminal(args.v115_audit), indent=2))
    elif args.action == "timing-worker":
        ticket = worker_ticket(args, "timing")
        from smoke_warp_candidate_v117 import measure_512

        def check():
            if now() >= stamp(ticket["deadline_utc"]):
                raise TimeoutError("The original 60-second procedural deadline elapsed")

        measure_512(check, args.ticket.parent / "fixture")
    else:
        dispatch = HOME / f"{args.phase}-dispatch.json"
        if args.action == "freeze":
            freeze(args.phase, args.v115_audit, dispatch)
        else:
            with (WORK.parent / "solar-warp-v117.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(args.phase, args.public_commit, args.v115_audit, dispatch)


if __name__ == "__main__":
    main()
