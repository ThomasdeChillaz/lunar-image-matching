"""Bounded v125 projection timing and archive controller over frozen v117 work."""

from time import perf_counter

_START = perf_counter()

import argparse
from datetime import timedelta
import fcntl
import math
import os
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[2]
HOME = Path("reports/solar-warp-v125")
WORK = Path("runs/solar-warp-v125")
V117 = Path("reports/solar-warp-v117")
OLD_RUN = Path("runs/solar-warp-v117/timing")
OLD_DISPATCH = V117 / "timing-dispatch.json"
OLD_RECEIPT = V117 / "timing-procedural-v123.json"
OLD_FAILURE = OLD_RUN / "FAILED.json"
OLD_PINS = {
    OLD_DISPATCH: "6043d1f54efc1e834985737998c5da920084686de0508f1fdfa915f4195ab7be",
    OLD_RECEIPT: "09c42572f7c9b0daca04860074a22ba2023796eb8db16a795b70d6014b81f8a9",
    OLD_FAILURE: "6abbd5cfdce1bdca8cf11b3d7d479249c0a90fa44c740745be08666071dda1e9",
}
RUNNER = HOME / "control_warp_archive_v125.py"
ENGINE = HOME / "fast_projection.py"
CAPS = {"timing": 60, "archive": 180}
ROWS = 32
STORAGE = 512 * 1024**2
ENV = {"NUMEXPR_MAX_THREADS": "2"}


for directory in (REPO, REPO / V117, REPO / HOME):
    sys.path.insert(0, str(directory))

import control_warp_archive_v117 as legacy
from solar_learning_pipeline_v73 import journal_state


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def dispatch_path(phase):
    return HOME / f"{phase}-dispatch.json"


def old_failure():
    need(all(legacy.sha(path) == digest for path, digest in OLD_PINS.items()),
         "Preserve the exact failed original v117 timing artifacts")
    plan = legacy.read(OLD_DISPATCH)
    receipt = legacy.read(OLD_RECEIPT)
    failure = legacy.read(OLD_FAILURE)
    charge = failure["charge"]["reservation"]
    need(failure["complete"] is False and failure["phase"] == "timing"
         and failure["protocol_sha256"] == OLD_PINS[OLD_DISPATCH]
         and legacy.sha(OLD_RUN / "fixture/receipt.json") == OLD_PINS[OLD_RECEIPT]
         and not (OLD_RUN / "COMPLETE.json").exists()
         and not (OLD_RUN.parent / "archive").exists()
         and charge["job_id"] == OLD_PINS[OLD_DISPATCH] + "/warp-timing"
         and charge["category"] == "baseline" and charge["seconds"] == CAPS["timing"]
         and charge["refund_on_failure"] is False and receipt["passed"] is False
         and receipt["full_scope_forecast"]["admitted"] is False
         and receipt["full_scope_forecast"]["projected_seconds"] > CAPS["archive"],
         "Original 60-second timing failure and nonrefundable charge must remain intact")
    return plan, dict(dispatch_sha256=OLD_PINS[OLD_DISPATCH], receipt_sha256=OLD_PINS[OLD_RECEIPT],
                      failure_sha256=OLD_PINS[OLD_FAILURE], archive_started=False,
                      projected_archive_seconds=receipt["full_scope_forecast"]["projected_seconds"], charge=charge)


def immutable(phase):
    historical, prior = old_failure()
    original = legacy.immutable(phase, verify_rosters=False)
    need(original["inputs"] == historical["inputs"] and original["labels"] == historical["labels"]
         and original["source_sha256"] == historical["source_sha256"] and original["software"] == historical["software"]
         and original["cap_seconds"] == CAPS[phase] and original["expected_prediction_rows"] == ROWS,
         "Original sources, native runtime and rosters may not change")
    engine = dict(path=str(ENGINE), sha256=legacy.sha(ENGINE), binding="FastProjectionBank",
                  scope="Only projection-cache winner construction changes. v117 rank, scoring, and replay remain byte-identical.")
    original["source_sha256"].update({str(path): legacy.sha(path) for path in (RUNNER, ENGINE, OLD_DISPATCH, OLD_RECEIPT)})
    return dict(original, controller_revision="v125", projection_engine=engine, original_v117_timing=prior)


def timing_proof(engine):
    output = WORK / "timing"
    receipt_path = output / "fixture" / "receipt.json"
    complete = legacy.read(output / "COMPLETE.json")
    receipt = legacy.read(receipt_path)
    forecast = receipt["full_scope_forecast"]
    seconds = forecast["projected_seconds"]
    charge = complete["charge"]["reservation"]
    timing_dispatch = dispatch_path("timing")
    timing_protocol = legacy.read(timing_dispatch)
    reserved = legacy.stamp(charge["reserved_utc"])
    deadline = legacy.stamp(complete["deadline_utc"])
    finished = legacy.stamp(complete["ended_utc"])
    samples = [receipt["cold_projection_sample"], *receipt["warm_samples"]]
    names = ["cell-32-cold.npz", "cell-32-warm-0.npz", "cell-16-warm-0.npz",
             "cell-32-warm-1.npz", "cell-16-warm-1.npz"]
    need(not (output / "FAILED.json").exists() and complete["schema"] == "solar_warp_projection_completion_v125"
         and complete["phase"] == "timing" and complete["complete"] is True and complete["steps"] == ["timing"]
         and complete["projection_engine"] == engine == timing_protocol["projection_engine"]
         and timing_protocol["controller_revision"] == "v125" and timing_protocol["cap_seconds"] == CAPS["timing"]
         and complete["protocol_sha256"] == legacy.sha(timing_dispatch)
         and complete["timing_receipt_sha256"] == legacy.sha(receipt_path)
         and (deadline - reserved).total_seconds() == CAPS["timing"] and reserved < finished < deadline
         and charge["job_id"] == complete["protocol_sha256"] + "/warp-v125-timing"
         and charge["seconds"] == CAPS["timing"] and charge["category"] == "baseline"
         and charge["refund_on_failure"] is False and receipt["schema"] == "solar_warp_procedural_timing_v117"
         and receipt["passed"] is True and receipt["procedural_only"] is True
         and receipt["archived_lunar_images_opened"] is False and receipt["native_shape"] == [512, 512]
         and receipt["posed_candidates"] == 32 and receipt["fixture_translation_native_pixels"] == [40, 32]
         and forecast["admitted"] is True and 0 < seconds <= CAPS["archive"]
         and forecast["projected_uncompressed_evidence_bytes"] <= STORAGE
         and forecast["full_pairings"] == 16 and forecast["native_scales"] == [32, 16]
         and forecast["theoretical_candidates_per_pairing"] == 2048 and forecast["timing_margin"] == 1.5
         and forecast["all_candidates_retained_for_write_forecast"] is True
         and forecast["cap_seconds"] == CAPS["archive"] and forecast["evidence_storage_cap_bytes"] == STORAGE
         and [sample["evidence_file"] for sample in samples] == names,
         "New v125 timing has not passed the complete unchanged 180-second archive forecast")
    legacy.publication(timing_dispatch, timing_protocol, complete["source_commit"])
    for sample, name in zip(samples, names, strict=True):
        path = output / "fixture" / name
        need(sample["control"]["passed"] is True and path.stat().st_size == sample["evidence_bytes"]
             and legacy.sha(path) == sample["evidence_sha256"], "Procedural timing sample or control changed")
    return dict(protocol_sha256=complete["protocol_sha256"], source_commit=complete["source_commit"],
                receipt_sha256=legacy.sha(receipt_path), completion_sha256=legacy.sha(output / "COMPLETE.json"),
                projected_archive_seconds=seconds, charge=charge)


def predecessor(audit):
    known = legacy.read(OLD_DISPATCH)["predecessor"]
    actual = legacy.terminal(audit)
    need(actual == known and actual["conclusion"] == "success", "Original verified v115 predecessor differs")
    return actual


def freeze(phase, audit):
    path = dispatch_path(phase)
    if path.exists() or (WORK / phase).exists():
        raise FileExistsError("Keep all original v125 phase attempts")
    p = immutable(phase)
    prior = predecessor(audit)
    timing = timing_proof(p["projection_engine"]) if phase == "archive" else None
    created = legacy.now()
    legacy.put(path, dict(p, prepared_only=False, predecessor=prior, timing=timing,
                          created_utc=created.isoformat(), latest_start_utc=(created + legacy.LEASE).isoformat()))


def ensure_time(deadline, monotonic_deadline):
    if legacy.now() >= deadline or time.monotonic() >= monotonic_deadline:
        raise TimeoutError("Original v125 calendar/monotonic deadline reached")


def admission(phase, audit, commit):
    p = legacy.read(dispatch_path(phase))
    static = immutable(phase)
    need(p.get("prepared_only") is False
         and set(p) == {*static, "prepared_only", "predecessor", "timing", "created_utc", "latest_start_utc"}
         and all(p[key] == value for key, value in static.items())
         and legacy.stamp(p["latest_start_utc"]) == legacy.stamp(p["created_utc"]) + legacy.LEASE,
         "Require exact immutable published v125 phase")
    legacy.publication(dispatch_path(phase), p, commit)
    need(p["predecessor"] == predecessor(audit)
         and p["timing"] == (timing_proof(p["projection_engine"]) if phase == "archive" else None),
         "Original v115 or v125 timing admission evidence changed")
    if not legacy.stamp(p["created_utc"]) <= legacy.now() <= legacy.stamp(p["latest_start_utc"]):
        raise TimeoutError("Original one-use publication lease elapsed")
    _, journal, _ = journal_state(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL)
    charges = journal["reservations"]
    need(p["original_v117_timing"]["charge"] in charges
         and (p["timing"] is None or p["timing"]["charge"] in charges)
         and legacy.remaining(legacy.load_contract(legacy.CONTRACT))["baseline"] >= CAPS[phase],
         "Original charges must remain and cumulative allowance must cover this phase")
    return p


def worker_ticket(mode, path):
    phase = "timing" if mode == "timing" else "archive"
    output = (WORK / phase).resolve()
    ticket = legacy.read(path)
    frozen_path = dispatch_path(phase)
    p = legacy.read(frozen_path)
    need(path.resolve() == output / f"{mode}-ticket.json"
         and ticket["mode"] == mode and ticket["phase"] == phase and ticket["controller_pid"] == os.getppid()
         and ticket["protocol_sha256"] == legacy.sha(frozen_path)
         and p["schema"] == "solar_warp_archival_gate_v117" and p["controller_revision"] == "v125"
         and p["projection_engine"]["sha256"] == legacy.sha(ENGINE)
         and p["projection_engine"]["path"] == str(ENGINE)
         and all(os.environ.get(k) == v for k, v in {**legacy.ENV, **ENV}.items()),
         "Require exact v125 controller, projection source and two-thread environment")
    deadline = legacy.stamp(ticket["deadline_utc"])
    reserved = legacy.stamp(ticket["charge"]["reserved_utc"])
    monotonic_deadline = ticket["monotonic_deadline"]
    need((deadline - reserved).total_seconds() == CAPS[phase]
         and reserved <= legacy.now()
         and math.isfinite(monotonic_deadline) and 0 < monotonic_deadline - time.monotonic() <= CAPS[phase],
         "The original unextended two-clock cap is required")
    _, journal, _ = journal_state(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL)
    charge = ticket["charge"]
    need(charge in journal["reservations"]
         and charge["job_id"] == ticket["protocol_sha256"] + "/warp-v125-" + phase
         and charge["category"] == "baseline" and charge["seconds"] == CAPS[phase]
         and charge["refund_on_failure"] is False, "Require original v125 canonical charge")
    ensure_time(deadline, monotonic_deadline)
    # The controller already bound both rosters to publication. The worker
    # verifies the exact dispatch and code without opening the scoring roster.
    legacy.publication(frozen_path, {**p, "labels": p["inputs"]}, ticket["source_commit"])
    ensure_time(deadline, monotonic_deadline)
    return p, ticket, output, lambda: ensure_time(deadline, monotonic_deadline)


def archived_arguments(mode, p, output, ticket):
    data = ticket["arguments"]
    prediction, scores = output / "predictions.json", output / "scores.json"
    expected = dict(repo=str(REPO), inputs=p["inputs"]["path"], inputs_sha256=p["inputs"]["sha256"],
                    labels=None, labels_sha256=None, predictions=None, predictions_sha256=None,
                    output=str(prediction.relative_to(REPO)))
    if mode == "score":
        seal = legacy.read(output / "PREDICTIONS_SEALED.json")
        prediction_sha = legacy.sha(prediction)
        need(seal["rows"] == ROWS and seal["sha256"] == prediction_sha
             and 0 < seal["evidence_bytes"] <= STORAGE, "Scoring requires complete original predictions sealed")
        expected.update(labels=p["labels"]["path"], labels_sha256=p["labels"]["sha256"],
                        predictions=str(prediction.relative_to(REPO)), predictions_sha256=prediction_sha,
                        output=str(scores.relative_to(REPO)))
    need(data == expected, "Worker may only receive the original mode-specific input and output paths")
    return data


def worker(mode, path):
    from unittest.mock import patch

    p, ticket, output, check = worker_ticket(mode, path)
    if mode == "timing":
        from fast_projection import FastProjectionBank
        import smoke_warp_candidate_v117 as original

        need(ticket["arguments"] is None, "Native timing receives no archive data")
        startup = max(original._IMPORT_SECONDS, perf_counter() - _START)
        with patch.object(original, "ProjectionBank", FastProjectionBank), patch.object(original, "_IMPORT_SECONDS", startup):
            original.measure_512(check, output / "fixture")
        return
    import archived_warp_worker as original

    args = archived_arguments(mode, p, output, ticket)
    inputs = original.roster(Path(args["inputs"]), args["inputs_sha256"], "inference")
    if mode == "predict":
        from fast_projection import FastProjectionBank

        with patch.object(original, "ProjectionBank", FastProjectionBank):
            original.infer(REPO, inputs, Path(args["output"]))
    else:
        labels = original.roster(Path(args["labels"]), args["labels_sha256"], "scoring_only")
        original.score(REPO, inputs, labels, Path(args["predictions"]), args["predictions_sha256"], Path(args["output"]))
    check()


def run(phase, audit, commit):
    p = admission(phase, audit, commit)
    output = WORK / phase
    output.mkdir(parents=True, exist_ok=False)
    protocol_sha = legacy.sha(dispatch_path(phase))
    charge = legacy.reserve(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL, protocol_sha + "/warp-v125-" + phase,
                            "baseline", CAPS[phase])
    deadline = legacy.stamp(charge["reservation"]["reserved_utc"]) + timedelta(seconds=CAPS[phase])
    monotonic_deadline = time.monotonic() + min(CAPS[phase], (deadline - legacy.now()).total_seconds())
    os.environ.update(legacy.ENV, **ENV, PYTHONPATH=str(REPO))
    status = dict(schema="solar_warp_projection_completion_v125", phase=phase, protocol_sha256=protocol_sha,
                  source_commit=commit, projection_engine=p["projection_engine"], deadline_utc=deadline.isoformat(),
                  charge=charge, calibration_or_final_access=False,
                  original_v117_timing_failure_sha256=p["original_v117_timing"]["failure_sha256"])
    steps = []

    def issue(mode, arguments=None):
        target = output / f"{mode}-ticket.json"
        legacy.put(target, dict(mode=mode, phase=phase, controller_pid=os.getpid(), protocol_sha256=protocol_sha,
                                source_commit=commit,
                                charge=charge["reservation"], deadline_utc=deadline.isoformat(),
                                monotonic_deadline=monotonic_deadline, arguments=arguments))
        return target

    def step(command, name):
        ensure_time(deadline, monotonic_deadline)
        if name in ("predict", "score"):
            command = [sys.executable, str(RUNNER), "worker", name, "--ticket", str(output / f"{name}-ticket.json")]
        remaining = monotonic_deadline - time.monotonic()
        legacy.run_step(command, output / f"{name}.log", min(deadline, legacy.now() + timedelta(seconds=remaining)))
        ensure_time(deadline, monotonic_deadline)
        steps.append(name)

    try:
        if phase == "timing":
            path = issue("timing")
            step([sys.executable, str(RUNNER), "worker", "timing", "--ticket", str(path)], "timing")
            extra = dict(timing_receipt_sha256=legacy.sha(output / "fixture" / "receipt.json"))
        else:
            extra = legacy.archive(p, output, issue, step)
        ensure_time(deadline, monotonic_deadline)
        legacy.put(output / "COMPLETE.json", dict(status, **extra, steps=steps, complete=True,
                                                 ended_utc=legacy.now().isoformat()))
    except BaseException as error:
        legacy.put(output / "FAILED.json", dict(status, steps=steps, complete=False,
                                               error_type=type(error).__name__, error=str(error)))
        raise


def main():
    need(Path.cwd() == REPO, "Use the original repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "run"):
        command = commands.add_parser(action)
        command.add_argument("phase", choices=tuple(CAPS))
        command.add_argument("--v115-audit", type=Path, required=True)
        if action == "run":
            command.add_argument("--public-commit", required=True)
    command = commands.add_parser("worker")
    command.add_argument("mode", choices=("timing", "predict", "score"))
    command.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.mode, args.ticket)
    elif args.action == "freeze":
        freeze(args.phase, args.v115_audit)
    else:
        with (WORK.parent / "solar-warp-v125.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.phase, args.v115_audit, args.public_commit)


if __name__ == "__main__":
    main()
