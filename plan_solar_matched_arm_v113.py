"""Read-only, exact-source budget forecast for a five-arm ARM comparison."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

COHORT = ("reports/solar_matched_arm_cohorts_v103.json", "3f5754829bcef72d90574975b509bf9354a108531848cb2961c5590d914aa9aa")
ARM_PUBLISHED = ("reports/solar_arm_v110_pilot_results.json", "79680cae639c50af4a6130fee3231547fa3d60af9011ab2f2b04fb532d0c0921")
ARM_PROTOCOL = ("reports/solar-modern-arm-v110/pilot-protocol.json", "5bb40d750744b887f0919803cb0de1184a6c94c5c7b66f2ef6fc0c842ea6a6fe")
ARM_SUMMARY = ("runs/solar-modern-arm-v110/pilot/summary.json", "8f415b2bea94c3109313edc4d58d43b2d12d1878a33a0f318946f51a87583d4b")
ARM_WATCHDOG = ("reports/solar_arm_v110_remote_watchdog.json", "ab589410f6c342a2662ddf23f6df0b4989d67d15d08e1482b987a1d8e3e1760d")
ARM_REPLAY = ("reports/solar_arm_v110_independent_replay.json", "22a0a5c40b2984b4e41b467a0509347991791ccafc68746bccb26da1a39761da")
DIRECT_HISTORICAL_PROTOCOL = ("reports/solar-full-direct-render-v92.json", "9421f06bdaad9d3b8b9da8e424d4d21b6e5ceda90f6a1d228848842d24c0f909")
DIRECT_TRANSPORT_PREFLIGHT = ("reports/solar_direct_arm_payload_preflight_v104.json", "cf252e2a44629a1b699d902921372abe99146ff6aaa08a36582e169eeb58ae0a")
ARM_NAMES = (
    "frozen_step0_A1B0C1_quantization_aware", "SIFT2048", "loma_r_common", "romav2_common",
)
DIRECT_NAME = "reference_camera_render_match"
TRACK = "estimated_sun"
MARGIN = 1.5
RUN_CAP = 18_000
DIRECT_QUALIFICATION_CAP = 1_800
V112_PROTECTED = 8_400
V112_STOP = ("reports/solar-secondary-sentry-replay-v112.json", "66b76e4e38ac9c1b9b13d11cea2a45d7273491d92de33e3f014d94e1f67aeeff")
V112_CHARGES = {
    "e68018d94949e7668af1f6b7820451ee928e46c168123737b74072c22d85250c/secondary-v112": 500,
    "82d2a0cb780318cfa2303af9945caba1e1ff119c30b541febb5dd52b6fc267fd/secondary-v112": 1200,
}


@dataclass(frozen=True)
class ArmTiming:
    name: str
    case_peak_seconds: float
    model_load_seconds: float
    fresh_controls_seconds: float


@dataclass(frozen=True)
class MeasuredTiming:
    arms: tuple
    watchdog_unattributed_seconds: float
    workflow_before_watchdog_seconds: float


def read(path):
    return json.loads(Path(path).read_bytes())


def pinned(root, path_digest):
    relative, digest = path_digest
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Evidence paths must be relative to their audited evidence root")
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping evidence file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frozen evidence hash changed for " + relative)
    return json.loads(data)


def _seconds(value, field, *, allow_zero=False):
    if (
        type(value) not in (int, float) or not math.isfinite(value)
        or value < 0 or (value == 0 and not allow_zero)
    ):
        raise ValueError("Invalid measured timing for " + field)
    return float(value)


def _clock_gap(later, earlier):
    def parse(value):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Recorded workflow clocks require explicit time zones")
        return stamp
    return _seconds((parse(later) - parse(earlier)).total_seconds(), "pre-watchdog setup", allow_zero=True)


def validate_rows(summary, replay, names, case_ids, control_arms):
    rows = summary["rows"]
    expected = {(name, case_id, TRACK) for name in names for case_id in case_ids}
    grid = lambda data: {(row["profile"], row["case_id"], row["track"]) for row in data}
    replay_rows = replay["rows"]
    if (
        summary.get("complete") is not True or summary.get("cached_cases") != 0
        or summary.get("fresh_inference_cases") != len(expected)
        or summary.get("accepted_results") is not False or summary.get("calibration_or_final_access") is not False
        or len(rows) != len(expected) or grid(rows) != expected
        or replay.get("passed") is not True or replay.get("replayed_rows") != len(expected)
        or len(replay_rows) != len(expected) or grid(replay_rows) != expected
        or any(row.get("reused_case_cache") is not False or row.get("inference_failure") is not None for row in rows)
    ):
        raise ValueError("Exact independently replayed fresh primary cohort required")
    details = {(r["profile"], r["case_id"], r["track"]): r for r in replay_rows}
    for row in rows:
        replayed = details[(row["profile"], row["case_id"], row["track"])]
        if any(replayed[key] != row["metrics"][key] for key in (
            "correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap",
        )):
            raise ValueError("Independent result replay and pilot metrics differ")
    timings = []
    measured_case_sum = 0.0
    for name in names:
        part = [row for row in rows if row["profile"] == name]
        case_times = [_seconds(row["case_calendar_seconds"], name + " case") for row in part]
        measured_case_sum += sum(case_times)
        controls = [_seconds(row["fresh_control_calendar_seconds"], name + " control", allow_zero=True) for row in part]
        if name in control_arms and sum(c > 0 for c in controls) != 1:
            raise ValueError("One real pretrained or Direct control timing required for " + name)
        loads = [_seconds(row["profile_load_seconds"], name + " model load", allow_zero=True) for row in part]
        timings.append(ArmTiming(name, max(case_times), max(loads), sum(controls)))
    return tuple(timings), measured_case_sum


def timing(summary, replay, watchdog, names, case_ids, control_arms):
    arms, case_sum = validate_rows(summary, replay, names, case_ids, control_arms)
    wall = _seconds(watchdog["calendar_seconds"], "outer completed watchdog")
    worker = _seconds(summary["calendar_seconds"], "worker calendar")
    explicit = case_sum + sum(arm.model_load_seconds + arm.fresh_controls_seconds for arm in arms)
    residue = _seconds(wall - explicit, "unattributed parent/worker setup")
    if not explicit <= worker <= wall or residue < _seconds(watchdog["setup_seconds"], "capsule setup"):
        raise ValueError("Recorded parent/setup/worker timings are inconsistent")
    return MeasuredTiming(
        arms, residue, _clock_gap(watchdog["started_utc"], watchdog["workflow_started_utc"]),
    )


def read_v110(root):
    protocol = pinned(root, ARM_PROTOCOL)
    summary = pinned(root, ARM_SUMMARY)
    watchdog = pinned(root, ARM_WATCHDOG)
    replay = pinned(root, ARM_REPLAY)
    published = pinned(root, ARM_PUBLISHED)
    cohort = pinned(root, COHORT)
    if (
        published["summary_sha256"] != ARM_SUMMARY[1] or published["remote_watchdog_sha256"] != ARM_WATCHDOG[1]
        or published["protocol_sha256"] != ARM_PROTOCOL[1] or published["independent_replay_sha256"] != ARM_REPLAY[1]
        or summary["protocol_sha256"] != ARM_PROTOCOL[1] or replay["summary_sha256"] != ARM_SUMMARY[1]
        or watchdog["protocol_sha256"] != ARM_PROTOCOL[1]
        or watchdog.get("workflow_run_id") != 41930272861 or watchdog.get("run_attempt") != 1
        or watchdog.get("complete") is not True or watchdog.get("child_returncode") != 0
        or watchdog.get("failure") is not None or watchdog["result_artifact"]["file_count"] <= 0
        or protocol["pilot_case_ids"] != cohort["pilot_case_ids"]
        or tuple(profile["name"] for profile in protocol["profiles"]) != ARM_NAMES
        or summary["profiles"] != protocol["profiles"] or summary["software"] != published["platform"]
        or protocol["tracks"] != [TRACK] or protocol["calibration_or_final_access"] is not False
    ):
        raise ValueError("Completed public v110 evidence identities disagree")
    result = timing(summary, replay, watchdog, ARM_NAMES, protocol["pilot_case_ids"], set(ARM_NAMES[2:]))
    return protocol, summary["software"], cohort, result


def read_direct(root, receipt_path, arm_protocol, arm_software):
    """Read a future independently signed and replayed eight-case Direct witness.

    The manifest must come from the verified artifact workflow. This metadata
    tool verifies its file/row/charge bindings and never signs or dispatches.
    """
    receipt = read(receipt_path)
    names = {"protocol", "summary", "watchdog", "completion", "replay", "hardware", "controls"}
    if set(receipt) != {"schema", "files"} or receipt["schema"] != "solar_direct_arm_qualification_v113":
        raise ValueError("Reviewed v113 Direct receipt schema required")
    files = receipt["files"]
    if set(files) != names or any(set(item) != {"path", "sha256"} for item in files.values()):
        raise ValueError("All original Direct worker evidence files must be pinned")
    evidence_root = Path(receipt_path).resolve().parent
    evidence = {
        key: pinned(evidence_root, (value["path"], value["sha256"]))
        for key, value in files.items()
    }
    protocol, summary, watchdog = (evidence[k] for k in ("protocol", "summary", "watchdog"))
    complete, replay = evidence["completion"], evidence["replay"]
    hardware, controls = evidence["hardware"], evidence["controls"]
    expected = pinned(root, DIRECT_HISTORICAL_PROTOCOL)["profiles"]
    run_id = watchdog.get("workflow_run_id")
    protocol_digest, summary_digest = files["protocol"]["sha256"], files["summary"]["sha256"]
    if (
        protocol.get("stage_mode") != "matched_direct_arm_cost_pilot_v113"
        or protocol.get("purpose") != "cost_pilot" or protocol.get("pilot_case_ids") != arm_protocol["pilot_case_ids"]
        or protocol.get("manifest") != arm_protocol["manifest"] or protocol.get("geometry_audit") != arm_protocol["geometry_audit"]
        or protocol.get("case_ids") != arm_protocol["case_ids"] or protocol.get("tracks") != [TRACK]
        or protocol.get("calibration_or_final_access") is not False or protocol.get("accepted_results") is not False
        or protocol.get("cpu_threads") != 2 or protocol.get("profiles") != expected
        or not 0 < protocol.get("max_calendar_seconds", 0) <= DIRECT_QUALIFICATION_CAP
        or summary.get("profiles") != expected or summary.get("case_ids") != protocol["pilot_case_ids"]
        or summary.get("protocol_sha256") != protocol_digest
        or summary.get("manifest_sha256") != protocol["input_sha256"][protocol["manifest"]]
        or protocol["input_sha256"][protocol["manifest"]] != arm_protocol["input_sha256"][arm_protocol["manifest"]]
        or watchdog.get("protocol_sha256") != protocol_digest or watchdog.get("run_attempt") != 1
        or watchdog.get("complete") is not True or watchdog.get("child_returncode") != 0
        or watchdog.get("failure") is not None or type(run_id) is not int or run_id <= 0
        or run_id in {41926059385, 41928100316, 41930272861}
        or watchdog.get("public_dispatch_evidence", {}).get("verified_commit") is not True
        or complete.get("complete") is not True or complete.get("external_watchdog_passed") is not True
        or complete.get("protocol_sha256") != protocol_digest or complete.get("summary_sha256") != summary_digest
        or replay.get("summary_sha256") != summary_digest
        or hardware.get("machine") != "aarch64" or hardware.get("native_bf16") is not True
        or controls.get("passed") is not True or controls.get("profile") != DIRECT_NAME
        or controls.get("protocol_sha256") != protocol_digest or controls.get("fresh_actual_arm") is not True
    ):
        raise ValueError("The separate fresh Direct run does not match v110 geometry, cases and qualification policy")
    runtime = summary["software"]
    if (
        any(runtime.get(key) != arm_software[key] for key in ("machine", "torch", "opencv", "numpy", "cpu_threads"))
        or not runtime.get("platform", "").startswith("Linux-") or "aarch64" not in runtime["platform"]
        or not runtime["platform"].endswith("with-glibc2.39")
    ):
        raise ValueError("Direct timing is not from the qualified Linux ARM64 worker stratum")
    reservation = watchdog["original_reservation"]["reservation"]
    journal = read(Path(root) / arm_protocol["reservation_journal"])
    if (
        journal["ledger_sha256"] != arm_protocol["compute_ledger_sha256"]
        or sum(entry == reservation for entry in journal["reservations"]) != 1
        or reservation["job_id"] != protocol_digest + "/evaluation" or reservation["category"] != "baseline"
        or reservation["seconds"] != protocol["max_calendar_seconds"] or reservation["refund_on_failure"] is not False
    ):
        raise ValueError("The fresh Direct qualification is not recorded in the same canonical budget")
    return timing(summary, replay, watchdog, (DIRECT_NAME,), protocol["pilot_case_ids"], {DIRECT_NAME})


def cost_plan(arm, remaining, v112_unreserved=V112_PROTECTED, direct=None):
    for value, name in ((remaining, "remaining"), (v112_unreserved, "unreserved v112")):
        _seconds(value, name, allow_zero=True)
    if v112_unreserved > V112_PROTECTED:
        raise ValueError("Unreserved v112 commitment exceeds its original 8400-second envelope")
    direct_pending = DIRECT_QUALIFICATION_CAP if direct is None else 0
    cumulative = max(0.0, remaining - v112_unreserved - direct_pending)
    effective = min(RUN_CAP, cumulative)
    baseline_fixed = sum(a.model_load_seconds + a.fresh_controls_seconds for a in arm.arms)
    baseline_fixed += arm.watchdog_unattributed_seconds
    baseline_peak = sum(a.case_peak_seconds for a in arm.arms)
    future_direct_fixed = future_direct_peak = None
    if direct is not None:
        sole, = direct.arms
        # The Direct-only sentry has a second setup envelope. Add it in full
        # rather than assuming the two independently observed setups amortize.
        future_direct_fixed = (
            sole.model_load_seconds + sole.fresh_controls_seconds + direct.watchdog_unattributed_seconds
            + direct.workflow_before_watchdog_seconds
        )
        future_direct_peak = sole.case_peak_seconds
    rows = {}
    for count in (96, 64, 48):
        parent_raw = baseline_fixed + count * baseline_peak
        inclusive_raw = parent_raw + arm.workflow_before_watchdog_seconds
        direct_budget = effective / MARGIN - inclusive_raw
        projection = None
        if direct is not None:
            projection = MARGIN * (inclusive_raw + future_direct_fixed + count * future_direct_peak)
        rows[str(count)] = {
            "cases_per_method": count, "five_arm_cells": 5 * count,
            "four_arm_parent_seconds": MARGIN * parent_raw,
            "four_arm_including_pre_watchdog_seconds": MARGIN * inclusive_raw,
            "raw_headroom_for_direct_fixed_and_cases": direct_budget,
            "direct_per_case_ceiling_if_fixed_zero": direct_budget / count,
            "five_arm_conservative_seconds": projection,
            "fits_measured_cost": None if projection is None else projection <= effective,
        }
    largest = next((int(count) for count, item in rows.items() if item["fits_measured_cost"] is True), None)
    return {
        "canonical_remaining_seconds": remaining,
        "unreserved_v112_protected_seconds": v112_unreserved,
        "unreserved_direct_qualification_seconds": direct_pending,
        "cumulative_headroom_after_unreserved_commitments_seconds": cumulative,
        "original_run_cap_seconds": RUN_CAP, "effective_larger_run_cap_seconds": effective,
        "v110_case_peak_sum_seconds": baseline_peak,
        "v110_fresh_controls_and_model_load_seconds": baseline_fixed - arm.watchdog_unattributed_seconds,
        "v110_watchdog_unattributed_seconds": arm.watchdog_unattributed_seconds,
        "v110_pre_watchdog_setup_seconds": arm.workflow_before_watchdog_seconds,
        "direct_fixed_seconds": future_direct_fixed, "direct_peak_seconds": future_direct_peak,
        "largest_measured_budget_fit": largest, "cohorts": rows,
        "scope": "Forecast only. Use the same actual worker for all five arms of any later comparison.",
        "execution_authorized": False, "scientific_execution": False, "reservation_written": False,
    }


def outstanding_v112(root, journal):
    """Retain both fixed charges while releasing work cancelled by the fixed stop."""
    entries = [r for r in journal["reservations"] if r["job_id"].endswith("/secondary-v112")]
    if (
        len({r["job_id"] for r in entries}) != len(entries)
        or any(r["category"] != "baseline" or r["refund_on_failure"] is not False for r in entries)
    ):
        raise ValueError("Unreconciled v112 commitment entries")
    spent = sum(r["seconds"] for r in entries)
    if not 0 <= spent <= V112_PROTECTED:
        raise ValueError("Unexpected v112 charges above the original envelope")
    stop_file = Path(root) / V112_STOP[0]
    if not stop_file.exists():
        return V112_PROTECTED - spent
    stop = pinned(root, V112_STOP)
    positives = [row for row in stop["rows"] if row["positive"] is True]
    if (
        {r["job_id"]: r["seconds"] for r in entries} != V112_CHARGES
        or stop.get("schema") != "v112_independent_blind_sentry_replay"
        or stop.get("passed") is not True or stop.get("coverage_gate_pass") is not False
        or len(stop.get("positive_gates", [])) != 2 or len(positives) != 2
        or any(row.get("coverage_pass") is not False for row in positives)
    ):
        raise ValueError("Cannot release v112 protection without the exact preregistered stop evidence")
    return 0


def read_current_budget(protocol):
    from solar_learning_pipeline_v73 import load_contract, remaining
    return remaining(load_contract(protocol["learning_contract"]))["baseline"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v112-unreserved-seconds", type=int)
    parser.add_argument("--direct-receipt", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    protocol, software, cohort, arm = read_v110(root)
    direct = None if args.direct_receipt is None else read_direct(root, args.direct_receipt, protocol, software)
    protected = outstanding_v112(root, read(root / protocol["reservation_journal"]))
    if args.v112_unreserved_seconds is not None and args.v112_unreserved_seconds != protected:
        raise ValueError("Explicit v112 commitment must agree with canonical charges and stop evidence")
    result = cost_plan(arm, read_current_budget(protocol), protected, direct)
    result["arm_v110_exact_source_sha256"] = dict((path, digest) for path, digest in (
        ARM_PUBLISHED, ARM_PROTOCOL, ARM_SUMMARY, ARM_WATCHDOG, ARM_REPLAY, COHORT, DIRECT_TRANSPORT_PREFLIGHT,
    ))
    result["matched_case_counts"] = {key: len(value["case_ids"]) for key, value in cohort["subsets"].items()}
    transport = pinned(root, DIRECT_TRANSPORT_PREFLIGHT)
    result["direct_transport_preflight"] = {
        key: transport[key] for key in (
            "additional_direct_bytes", "combined_logical_bytes", "remaining_logical_headroom_bytes",
        )
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
