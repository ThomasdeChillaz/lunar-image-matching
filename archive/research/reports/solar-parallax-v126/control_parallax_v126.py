"""Prospective one-use control for plane-initialized native patch measurements."""

import argparse
from datetime import timedelta
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
HOME = Path("reports/solar-parallax-v126")
WORK = Path("runs/solar-parallax-v126")
SCRIPT = HOME / "control_parallax_v126.py"
V117 = Path("reports/solar-warp-v117")
V120 = Path("reports/solar-warp-handoff-v120")
V125 = Path("reports/solar-warp-v125")
PARENT = Path("runs/solar-warp-v125/archive")
PREDICTIONS = PARENT / "predictions.json"
PREDICTION_SHA = "d72ce03e6ec00dcc4a9119f6c168b462eaa48ee159fc40ef54033345de2e1430"
PINNED = {
    V125 / "archive-complete-v125.json": "73bd207346aff215ecbbd1146c7397ec4143828f70655c64b32de8ff36dc09cb",
    V125 / "archive-dispatch.json": "1483a5838b9a1781e3f2a1e601ed47e0423fb0cd9656de46719588c717be85ec",
    V125 / "archive-independent-replay-v125.json": "9b190892c109164801e283a0af8c9b9a5f96c298b5a1e757ff30535674f65913",
    PARENT / "PREDICTIONS_SEALED.json": "73df566d7a7aad51f5d88f7b3829039a67e44a4a375fe40028c9cf3a7e96cea5",
    V117 / "inference-inputs.json": "471a2176f54cebe62432cbf78aaf6ff37515566bb66df9ca2b6d96f719697049",
    V117 / "scoring-inputs.json": "4a8549742ecdd9663a7eb5616c5a16b512e56a0d03eab0ccbdd625b29351b8b7",
}
CAPS = {"timing": 60, "archive": 180}
TAGS = ("exposure-0", "exposure-1", "exposure-2", "exposure-3")
IDS = ("u061-l00-17-positive", "u013-l00-00-positive", "u025-l00-00-positive", "u027-l08-01-positive")
ARMS = ("parallax", "uniform")
SEEDS = 8
ANCHORS = 36
BUDGET = 12
RADIUS = 16
ENV = {"NUMEXPR_MAX_THREADS": "2"}

for directory in (ROOT, ROOT / V117, ROOT / V120, ROOT / HOME):
    sys.path.insert(0, str(directory))

import control_warp_archive_v117 as legacy
from solar_learning_pipeline_v73 import journal_state


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    path = Path(path)
    need(not path.is_symlink() and path.resolve().is_relative_to(ROOT) and legacy.sha(path) == expected,
         f"Expected exact scoped evidence at {path}")
    return path


def predecessor():
    for path, digest in PINNED.items():
        checked(path, digest)
    complete = legacy.read(V125 / "archive-complete-v125.json")
    replay = legacy.read(V125 / "archive-independent-replay-v125.json")
    seal = legacy.read(PARENT / "PREDICTIONS_SEALED.json")
    need(complete["complete"] is True and complete["advancement_gate"]["passed"] is False
         and complete["steps"] == ["predict", "score", "independent-replay"]
         and complete["predictions_sha256"] == replay["prediction_sha256"] == seal["sha256"] == PREDICTION_SHA
         and complete["independent_replay_sha256"] == PINNED[V125 / "archive-independent-replay-v125.json"]
         and complete["protocol_sha256"] == PINNED[V125 / "archive-dispatch.json"]
         and complete["calibration_or_final_access"] is False and replay["passed"] is True
         and replay["replayed_rows"] == seal["rows"] == 32 and replay["gate"]["passed"] is False,
         "A completed v125 replay with the original failed advancement gate is mandatory")
    return complete["charge"]["reservation"]


def forecast_scope(receipt):
    samples = receipt.get("row_wall_seconds", [])
    rows = receipt.get("rows", [])
    need(receipt.get("schema") == "solar_parallax_procedural_v126"
         and receipt.get("passed") is True and receipt.get("procedural_only") is True
         and receipt.get("archived_lunar_images_opened") is False and receipt.get("native_shape") == [512, 512]
         and len(samples) == len(rows) == 2 and [row["multidepth"] for row in rows] == [False, True]
         and [row["row_wall_seconds"] for row in rows] == samples
         and all(row["passed"] is True and set(row["arms"]) == set(ARMS)
                 and all(result["passed"] is True and result["patch_searches"] == BUDGET for result in row["arms"].values())
                 for row in rows)
         and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
         "Passed independent generated measurement controls and native timing required")
    forecasts = {count: 1.5 * (max(samples) * count + 20.) for count in (16, 8)}
    count = next((n for n in (16, 8) if forecasts[n] <= CAPS["archive"]), None)
    need(count is not None, "Full eight-row primary scope exceeds the 180-second archive cap")
    return count, {str(n): value for n, value in forecasts.items()}


def pairing_grid(count):
    need(count in (8, 16), "Scope must retain every seed pairing of each admitted case")
    return [[tag, r, q] for tag in TAGS[:count // 4] for r in (0, 1) for q in (0, 1)]


def timing_evidence(sources):
    path = WORK / "timing/receipt.json"
    complete = legacy.read(WORK / "timing/COMPLETE.json")
    dispatch = HOME / "timing-dispatch.json"
    old = legacy.read(dispatch)
    receipt = legacy.read(path)
    need(complete["complete"] is True and complete["phase"] == "timing"
         and complete["protocol_sha256"] == legacy.sha(dispatch) and complete["receipt_sha256"] == legacy.sha(path)
         and complete["completed_stages"] == ["timing"]
         and complete["charge"]["job_id"] == complete["protocol_sha256"] + "/parallax-v126-timing"
         and complete["charge"]["seconds"] == CAPS["timing"] and complete["charge"]["category"] == "baseline"
         and complete["charge"]["refund_on_failure"] is False
         and legacy.stamp(complete["charge"]["reserved_utc"]) < legacy.stamp(complete["ended_utc"])
         < legacy.stamp(complete["deadline_utc"])
         and legacy.stamp(complete["deadline_utc"]) - legacy.stamp(complete["charge"]["reserved_utc"])
         == timedelta(seconds=CAPS["timing"])
         and old["source_sha256"] == sources and not (WORK / "timing/FAILED.json").exists(),
         "Timing must come from the same published native implementation")
    legacy.publication(dispatch, old, complete["source_commit"])
    count, forecast = forecast_scope(receipt)
    return dict(protocol_sha256=complete["protocol_sha256"], receipt_sha256=complete["receipt_sha256"],
                completed_sha256=legacy.sha(WORK / "timing/COMPLETE.json"),
                charge=complete["charge"], row_count=count, projected_seconds=forecast)


def build(phase):
    prior = predecessor()
    source = (*legacy.REUSED_SOURCES, legacy.CONTRACT, legacy.LEDGER,
              "localization_robust_pose_v4.py", "localization_pose_refit_v3.py", "solar_active_v63.py",
              "localization_diagnostics_v3.py", "dataset.py", "lunargeo_dataset.py",
              str(V117 / "control_warp_archive_v117.py"),
              str(V117 / "archived_warp_worker.py"), str(V117 / "prepare_warp_roster_v117.py"),
              str(V117 / "warp_candidates.py"),
              *(str(V120 / n) for n in ("reference_patch_warp.py", "handoff_patch.py", "handoff_localizer.py", "handoff_pipeline.py")),
              str(SCRIPT), *(str(HOME / n) for n in ("measure_v126.py", "smoke_v126.py", "parallax_policy.py",
                                                    "score_parallax_v126.py")))
    sources = {name: legacy.sha(name) for name in source}
    for path, digest in PINNED.items():
        if path.is_relative_to("reports"):
            sources[str(path)] = digest
    timing = timing_evidence(sources) if phase == "archive" else None
    return dict(schema="solar_plane_initialized_parallax_pilot_v126", phase=phase, split="model_selection",
                prepared_only=True, calibration_or_final_access=False, cpu_threads=2, cap_seconds=CAPS[phase],
                known_reference_illumination=True, accepted_results=False, v125_gate_passed=False,
                predecessor_v125_charge=prior, original_predictions_sha256=PREDICTION_SHA,
                software=legacy.read(V125 / "archive-dispatch.json")["software"],
                original_prediction_seal_sha256=PINNED[PARENT / "PREDICTIONS_SEALED.json"],
                reference_anchor_count=ANCHORS, patch_budget_per_arm=BUDGET, patch_radius_native=RADIUS,
                seeds=dict(detector_scale=32, ranking="planar", max_ordered_hypotheses=SEEDS, warps=1),
                arms=list(ARMS), timing=timing, pairings=pairing_grid(timing["row_count"]) if timing else [],
                scope_policy="All 16 pairings if measured conservative cost fits, otherwise fixed faint/bright 8, otherwise refuse.",
                follow_on=dict(automatic=False, requires_new_frozen_matched_nonoverlaps=True,
                               requires_estimated_sun_with_no_query_sun_label=True,
                               calibration_and_final_remain_closed=True),
                feasibility_gate=dict(faint_min_paired_wins=2, no_faint_or_bright_paired_losses=True,
                                      both_pose_and_independent_2px_support=True, exact_shared_detector_cell_quotas=True,
                                      no_benchmark_or_deployment_promotion=True),
                source_sha256=sources,
                inputs=dict(path=str(V117 / "inference-inputs.json"), sha256=PINNED[V117 / "inference-inputs.json"]),
                labels=dict(path=str(V117 / "scoring-inputs.json"), sha256=PINNED[V117 / "scoring-inputs.json"]))


def plan_path(phase):
    return HOME / f"{phase}-dispatch.json"


def read_admitted(phase, commit):
    dispatch = plan_path(phase)
    p = legacy.read(dispatch)
    expected = dict(build(phase), prepared_only=False)
    start, latest = (legacy.stamp(p[k]) for k in ("created_utc", "latest_start_utc"))
    need(p == dict(expected, created_utc=p["created_utc"], latest_start_utc=p["latest_start_utc"])
         and latest == start + legacy.LEASE and start <= legacy.now() <= latest, "Published v126 dispatch or lease differs")
    verify_runtime(p)
    legacy.publication(dispatch, p, commit)
    return p


def verify_runtime(p):
    need(p["software"] == dict(platform=platform.system(), machine=platform.machine(), python=platform.python_version(),
                               numpy=version("numpy"), opencv=version("opencv-python")),
         "Use the exact preserved native runtime")


def seed_bank(predictions, tag, r, q):
    import numpy as np

    matching = [row for row in predictions["rows"] if
                (row["tag"], row["reference_seed"], row["query_seed"], row["cell_pixels"]) == (tag, r, q, 32)]
    row, = matching
    order = row["orders"]["planar"][:SEEDS]
    bank = np.asarray(predictions["banks"][tag], dtype=float)
    need(predictions["calibration_or_final_access"] is False and bank.ndim == 3 and bank.shape[1:] == (4, 4)
         and row["candidate_count"] == len(bank) and len(order) == len(set(order))
         and all(type(i) is int and 0 <= i < len(bank) for i in order), "Invalid sealed v125 plane seed ordering")
    return order, bank[order]


def predict(p, output):
    import cv2
    import numpy as np
    from archived_warp_worker import asset, roster
    from measure_v126 import run as measure
    from replay_raw_acquisition_v109 import decode, luminance

    cv2.setNumThreads(1)
    source = legacy.read(checked(PREDICTIONS, p["original_predictions_sha256"]))
    inputs = {row["tag"]: row for row in roster(Path(p["inputs"]["path"]), p["inputs"]["sha256"], "inference")}
    rows = []
    for tag, r, q in p["pairings"]:
        item = inputs[tag]
        need(set(item) == {"tag", "reference_geometry", "reference_radiance", "query_radiance", "query_intrinsics"},
             "Only original reference-only predictor inputs are allowed")
        started = time.perf_counter()
        order, seeds = seed_bank(source, tag, r, q)
        if len(order):
            with np.load(asset(ROOT, item["reference_geometry"]), allow_pickle=False) as z:
                depth, rk, ref, valid = (z[k].copy() for k in ("axial_depth", "intrinsics", "cam2world", "valid"))
            image = lambda role, index: luminance(decode(asset(ROOT, item[role][index]).read_bytes(), exr_size=512))
            result = measure(image("reference_radiance", r), depth, rk, ref, image("query_radiance", q),
                             np.asarray(item["query_intrinsics"], dtype=float), seeds, reference_valid=valid)
        else:
            empty = lambda: dict(raw_pose=None, reason="empty_plane_seed_bank", measurements=[], inliers=0,
                                 inlier_indices=[], pose_score=None, selected_anchors=[], failures=[], patch_searches=0,
                                 evaluated_positions=0, measured_patch_count=0, measurement_seconds=0., pose_seconds=0.)
            result = dict(arms={arm: empty() for arm in ARMS}, selection=None, shared_geometry_seconds=0.,
                          policy_seconds=0., seed_count=0, raw_only=True, calibrated_acceptance=False,
                          patch_budget=BUDGET, radius_pixels=RADIUS)
        rows.append(dict(pairing=[tag, r, q], seed_indices=order, native_elapsed_seconds=time.perf_counter()-started, **result))
    legacy.put(output, dict(schema="solar_plane_initialized_blind_native_v126", rows=rows,
                            original_predictions_sha256=PREDICTION_SHA, query_truth_opened=False, accepted_results=False))


def read_blind(p, path, seal_path):
    seal = legacy.read(seal_path)
    raw = legacy.read(checked(path, seal["sha256"]))
    need(raw["schema"] == "solar_plane_initialized_blind_native_v126" and raw["query_truth_opened"] is False
         and raw["accepted_results"] is False and raw["original_predictions_sha256"] == PREDICTION_SHA
         and seal["rows"] == len(raw["rows"]) and [row["pairing"] for row in raw["rows"]] == p["pairings"]
         and all(set(row["arms"]) == set(ARMS) and 0 <= row["native_elapsed_seconds"] <= CAPS["archive"]
                 for row in raw["rows"]), "Complete hash-sealed blind pairing and timing grid required")
    return raw


def score(p, raw):
    from score_parallax_v126 import replay_scores

    return replay_scores(p, raw)


def execute_stage(stage, p, root):
    predictions = root / "predictions.json"
    seal = root / "PREDICTIONS_SEALED.json"
    if stage == "predict":
        predict(p, predictions)
    else:
        scored = score(p, read_blind(p, predictions, seal))
        if stage == "score":
            legacy.put(root / "scores.json", scored)
        else:
            need(scored == legacy.read(root / "scores.json"), "Independent sealed pose and anchor arithmetic differs")
            legacy.put(root / "independent-replay.json", dict(
                passed=True, rows=len(scored["rows"]), predictions_sha256=legacy.sha(predictions),
                scores_sha256=legacy.sha(root / "scores.json"), new_measurements=False,
            ))


def worker(stage, ticket_path):
    ticket = legacy.read(ticket_path)
    p = legacy.read(plan_path("archive"))
    root = (WORK / "archive").resolve()
    need(ticket_path.resolve() == root / f"{stage}-ticket.json" and ticket["stage"] == stage
         and ticket["controller_pid"] == os.getppid() and ticket["protocol_sha256"] == legacy.sha(plan_path("archive"))
         and all(os.environ.get(k) == v for k, v in {**legacy.ENV, **ENV}.items()),
         "Require original controller, phase, source and worker environment")
    _, journal, _ = journal_state(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL)
    charge = ticket["charge"]
    deadline = legacy.stamp(ticket["deadline_utc"])
    need(charge in journal["reservations"] and charge["job_id"] == ticket["protocol_sha256"] + "/parallax-v126-archive"
         and charge["seconds"] == CAPS["archive"] and charge["category"] == "baseline" and charge["refund_on_failure"] is False
         and (deadline - legacy.stamp(charge["reserved_utc"])).total_seconds() == CAPS["archive"]
         and legacy.now() < deadline, "Worker requires original live canonical reservation")
    legacy.publication(plan_path("archive"), {**p, "labels": p["inputs"]}, ticket["commit"])
    verify_runtime(p)
    import cv2

    cv2.setNumThreads(1)
    execute_stage(stage, p, root)


def launch(phase, commit):
    p = read_admitted(phase, commit)
    _, journal, _ = journal_state(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL)
    need(p["predecessor_v125_charge"] in journal["reservations"]
         and (p["timing"] is None or p["timing"]["charge"] in journal["reservations"])
         and legacy.remaining(legacy.load_contract(legacy.CONTRACT))["baseline"] >= CAPS[phase],
         "Canonical original charges and current phase allowance required")
    root = WORK / phase
    root.mkdir(parents=True, exist_ok=False)
    digest = legacy.sha(plan_path(phase))
    charge = legacy.reserve(legacy.LEDGER, legacy.LEDGER_SHA, legacy.JOURNAL, digest + "/parallax-v126-" + phase,
                            "baseline", CAPS[phase])["reservation"]
    deadline = legacy.stamp(charge["reserved_utc"]) + timedelta(seconds=CAPS[phase])
    monotonic_deadline = time.monotonic() + (deadline - legacy.now()).total_seconds()
    os.environ.update(legacy.ENV, **ENV, PYTHONPATH=os.pathsep.join(map(str, (ROOT, ROOT / V117, ROOT / V120))))
    status = dict(phase=phase, source_commit=commit, protocol_sha256=digest, charge=charge,
                  deadline_utc=deadline.isoformat(), completed_stages=[], accepted_results=False)

    def step(command, name):
        now = legacy.now()
        hard = min(deadline, now + timedelta(seconds=monotonic_deadline - time.monotonic()))
        if now >= hard:
            raise TimeoutError("Original v126 deadline elapsed")
        legacy.run_step(command, root / f"{name}.log", hard)
        status["completed_stages"].append(name)

    try:
        if phase == "timing":
            receipt = root / "receipt.json"
            step([sys.executable, str(HOME / "smoke_v126.py"), "--output", str(receipt)], "timing")
            forecast_scope(legacy.read(receipt))
            status["receipt_sha256"] = legacy.sha(receipt)
        else:
            for stage in ("predict", "score", "replay"):
                ticket = root / f"{stage}-ticket.json"
                legacy.put(ticket, dict(stage=stage, controller_pid=os.getpid(), protocol_sha256=digest,
                                        commit=commit, charge=charge, deadline_utc=deadline.isoformat()))
                step([sys.executable, str(SCRIPT), "worker", stage, "--ticket", str(ticket)], stage)
                if stage == "predict":
                    path = root / "predictions.json"
                    seal = root / "PREDICTIONS_SEALED.json"
                    legacy.put(seal, dict(rows=len(p["pairings"]), sha256=legacy.sha(path)))
                    read_blind(p, path, seal)
            for name in ("predictions", "scores", "independent-replay"):
                status[name.replace("-", "_") + "_sha256"] = legacy.sha(root / f"{name}.json")
        if legacy.now() >= deadline or time.monotonic() >= monotonic_deadline:
            raise TimeoutError("Original v126 deadline elapsed before terminal seal")
        legacy.put(root / "COMPLETE.json", dict(status, complete=True, ended_utc=legacy.now().isoformat()))
    except BaseException as error:
        legacy.put(root / "FAILED.json", dict(status, complete=False, error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == "__main__":
    need(Path.cwd() == ROOT, "Run from repository root")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "run"):
        command = commands.add_parser(action)
        command.add_argument("phase", choices=tuple(CAPS))
        if action == "run":
            command.add_argument("--public-commit", required=True)
    command = commands.add_parser("worker")
    command.add_argument("stage", choices=("predict", "score", "replay"))
    command.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        p = build(args.phase)
        now = legacy.now()
        legacy.put(plan_path(args.phase), dict(p, prepared_only=False, created_utc=now.isoformat(),
                                              latest_start_utc=(now + legacy.LEASE).isoformat()))
    elif args.action == "run":
        launch(args.phase, args.public_commit)
    else:
        worker(args.stage, args.ticket)
