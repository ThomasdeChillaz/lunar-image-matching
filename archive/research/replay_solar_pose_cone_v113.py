"""Independently score the four sealed cone banks against the exact v112 sentry."""

import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

import replay_solar_secondary_sentry_v112 as previous


ROOT = Path("runs/solar-pose-cone-v113")
SEAL = ROOT / "COMPLETE.json"
DISPATCH = "reports/solar-pose-cone-dispatch-v113.json"
V112_REPLAY = "reports/solar-secondary-sentry-replay-v112.json"
V112_REPLAY_SHA = "66b76e4e38ac9c1b9b13d11cea2a45d7273491d92de33e3f014d94e1f67aeeff"
V112_DISPATCH = "reports/solar-secondary-sentry-dispatch-v112.json"
PREPARED = previous.BLIND / "PREPARED.json"
REQUEST = previous.BLIND / "reference-only-request.json"
REFERENCE_TABLE = previous.BLIND / "reference-table.json"
CONE_DEGREES = 52.
MINIMUM_DOWN_DOT = math.cos(math.radians(CONE_DEGREES)) - 16 * np.finfo(float).eps
SLOT_FIELDS = {"id", "blind_bank", "compact_final_pose", "compact_final_reason", "diagnostics", "elapsed_seconds"}
CANDIDATE_FIELDS = {"source_beam_index", "pose", "sun_reference_enu", "score", "sun_index"}
INITIAL_COUNTERS = {"calls", "returned_poses", "iterations_requested", "minimal_sets_executed",
                    "roots_evaluated", "roots_rejected_off_nadir", "lm_refits_rejected_off_nadir"}
NATIVE_COUNTERS = {"calls", "returned_poses", "off_nadir_poses", "invalid_poses"}


def checked_counters(values, keys):
    if set(values) != keys or any(type(value) is not int or value < 0 for value in values.values()):
        raise ValueError("Stage diagnostics must preserve the frozen nonnegative counters")


def verify_diagnostics(record, poses, final):
    expected_fields = {"schema", "cone_degrees", "initial_v73", "native_v84", "pre_final_beam_poses",
                       "pre_final_off_nadir_poses", "ordinary_final_off_nadir", "ordinary_final_iterations_reported"}
    if (set(record) != expected_fields or record["schema"] != "solar_blind_cone_diagnostics_v113"
            or record["cone_degrees"] != CONE_DEGREES or record["pre_final_beam_poses"] != len(poses)
            or record["pre_final_off_nadir_poses"] != sum(not feasibility(p)["within_simulator_cone"] for p in poses)
            or record["ordinary_final_off_nadir"] is not (
                None if final is None else not feasibility(final)["within_simulator_cone"])):
        raise ValueError("Reported cone-stage outcomes differ from the sealed poses")
    initial = record["initial_v73"]
    checked_counters(initial, INITIAL_COUNTERS)
    if (initial["roots_rejected_off_nadir"] > initial["roots_evaluated"]
            or initial["minimal_sets_executed"] > initial["iterations_requested"]
            or initial["returned_poses"] > initial["calls"]):
        raise ValueError("Initial root accounting exceeds the attempted budget")
    native = record["native_v84"]
    if set(native) != {"native_ap3p", "native_lm"}:
        raise ValueError("Native AP3P and LM observations must remain separate")
    for values in native.values():
        checked_counters(values, NATIVE_COUNTERS)
        if (values["off_nadir_poses"] + values["invalid_poses"] > values["returned_poses"]
                or values["returned_poses"] > values["calls"]):
            raise ValueError("Native pose stage counters are internally inconsistent")


def verify_slot(sealed, slot):
    record = json.loads(previous.asset_bytes(sealed, ROOT / "banks" / f"{slot}.json"))
    if (set(record) != SLOT_FIELDS or record["id"] != slot or not isinstance(record["diagnostics"], dict)
            or type(record["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0
            or (record["compact_final_reason"] is not None and not isinstance(record["compact_final_reason"], str))):
        raise ValueError("Unexpected frozen cone bank receipt")
    bank = record["blind_bank"]
    if (set(bank) != {"schema", "track", "pose_frame", "sun_frame", "capture_reached", "reason", "candidates"}
            or bank["schema"] != "solar_blind_beams_v112" or bank["track"] != "estimated_sun"
            or bank["pose_frame"] != "camera_to_reference_scene_enu" or bank["sun_frame"] != "reference_scene_enu"
            or type(bank["capture_reached"]) is not bool):
        raise ValueError("Expected the unchanged blind pose and Sun bank schema")
    candidates = bank["candidates"]
    if (not isinstance(candidates, list) or len(candidates) > 8
            or (candidates and (not bank["capture_reached"] or bank["reason"] is not None))
            or (not candidates and not isinstance(bank["reason"], str))):
        raise ValueError("Invalid bank inventory or explicit no-candidate outcome")
    indices, scores, poses = [], [], []
    for row in candidates:
        if set(row) != CANDIDATE_FIELDS:
            raise ValueError("The blind beam receipt cannot carry scoring labels")
        if (type(row["source_beam_index"]) is not int or not 0 <= row["source_beam_index"] < 8
                or type(row["sun_index"]) is not int or row["sun_index"] < 0
                or type(row["score"]) not in (int, float) or not math.isfinite(row["score"])):
            raise ValueError("Unexpected frozen beam identity or score")
        vector = np.asarray(row["sun_reference_enu"], dtype=float)
        if (vector.shape != (3,) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1., rtol=0., atol=1e-6)):
            raise ValueError("The frozen beam needs a finite estimated unit Sun")
        indices.append(row["source_beam_index"])
        scores.append(row["score"])
        poses.append(previous.validated_pose(row["pose"]))
    if len(set(indices)) != len(indices) or any(a < b for a, b in zip(scores, scores[1:])):
        raise ValueError("Cone candidates must preserve unique original beam indices and score order")
    final = record["compact_final_pose"]
    if final is not None:
        final = previous.validated_pose(final)
    verify_diagnostics(record["diagnostics"], poses, final)
    return dict(id=slot, poses=poses, final=final, final_reason=record["compact_final_reason"],
                empty_reason=bank["reason"], diagnostics=record["diagnostics"],
                elapsed_seconds=record["elapsed_seconds"], sha256=sealed["sha256"])


def verify_cone(seal_sha):
    seal = previous.sealed_json(SEAL, seal_sha)
    if (seal["schema"] != "v113_sealed_blind_cone" or seal["complete"] is not True
            or seal["query_truth_opened"] is not False or len(seal["banks"]) != len(previous.SLOTS)):
        raise ValueError("Require the exact four-slot, pre-truth cone seal")
    return seal, [verify_slot(item, slot) for item, slot in zip(seal["banks"], previous.SLOTS)]


def verify_dispatch(seal):
    data = previous.sealed_bytes(DISPATCH, seal["dispatch_sha256"])
    commit = seal["publication_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected the original complete v113 publication commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    if subprocess.check_output(["git", "show", f"{commit}:{DISPATCH}"], timeout=5) != data:
        raise ValueError("Actual cone dispatch differs from its frozen publication")
    protocol = json.loads(data)
    cap = protocol["cap_seconds"]
    if (protocol["schema"] != 113 or protocol["phase"] != "paired_simulator_attitude_cone"
            or protocol["split"] != "model_selection" or protocol["root"] != str(ROOT)
            or protocol["cohort"] != list(previous.SLOTS) or cap != 180 or protocol["cpu_threads"] != 2
            or protocol["calibration_or_final_access"] is not False
            or protocol["score_after_sealing"] is not True or protocol["poses_are_raw"] is not True
            or protocol["prior"] != "reference_ENU_52deg_off_nadir"
            or protocol["metadata_sha256"][V112_REPLAY] != V112_REPLAY_SHA
            or protocol["source_sha256"][Path(__file__).name] != previous.digest(Path(__file__).read_bytes())):
        raise ValueError("The frozen cone scoring contract or scorer source differs")
    allocation = seal["reservation"]
    reserved = datetime.fromisoformat(allocation["reserved_utc"])
    deadline = datetime.fromisoformat(seal["absolute_deadline_utc"])
    if (allocation["job_id"] != seal["dispatch_sha256"] + "/pose-cone-v113"
            or allocation["category"] != "baseline" or allocation["seconds"] != cap
            or allocation["refund_on_failure"] is not False or reserved.utcoffset() is None
            or deadline != reserved + timedelta(seconds=cap)
            or not datetime.fromisoformat(protocol["created_utc"]) <= reserved
            <= datetime.fromisoformat(protocol["latest_start_utc"])):
        raise ValueError("Cone seal did not use the original one-use reservation and deadline")
    return protocol


def verify_opaque_lineage(pins):
    prepared = previous.sealed_json(PREPARED, pins[str(PREPARED)])
    if (prepared["query_labels_in_request"] is not False or prepared["observation_files"] != 6
            or prepared["request"]["sha256"] != pins[str(REQUEST)]):
        raise ValueError("Cone input handoff differs from the original opaque v112 request")
    request = json.loads(previous.asset_bytes(prepared["request"], REQUEST))
    if (set(request) != {"reference", "slots", "query_intrinsics", "profile"}
            or [row["id"] for row in request["slots"]] != list(previous.SLOTS)):
        raise ValueError("Require the identical four opaque observations and original compact profile")
    assets = {}
    for row in request["slots"]:
        if set(row) != {"id", "reference_exr", "query_exr"}:
            raise ValueError("Opaque request cannot expose a case label")
        for field in ("reference_exr", "query_exr"):
            item = row[field]
            name, checksum = item["path"], item["sha256"]
            if name in assets and assets[name] != checksum:
                raise ValueError("Shared observation changed within the slot manifest")
            assets[name] = checksum
    expected = {str(previous.BLIND / "inputs" / f"o{i:02d}.exr") for i in range(6)}
    if set(assets) != expected:
        raise ValueError("Expected the same six immutable opaque observation files")
    receipt = previous.sealed_json(REFERENCE_TABLE, pins[str(REFERENCE_TABLE)])
    table = str(previous.BLIND / "reference-table.npz")
    profile = request["profile"]
    checkpoint = profile["checkpoint"]
    expected = {**assets, table: receipt["archive"]["sha256"],
                checkpoint: profile["configuration"]["checkpoint_sha256"],
                str(PREPARED): pins[str(PREPARED)], str(REQUEST): prepared["request"]["sha256"],
                str(REFERENCE_TABLE): pins[str(REFERENCE_TABLE)]}
    if receipt["archive"]["path"] != table or pins != expected:
        raise ValueError("Frozen opaque checkpoint, reference table or observation bindings differ")
    for name in (*assets, table, checkpoint):
        previous.sealed_bytes(name, pins[name])
    result = {"prepared_sha256": pins[str(PREPARED)], "request_sha256": prepared["request"]["sha256"],
              "observation_sha256": assets, "reference_table_sha256": pins[table],
              "checkpoint_sha256": pins[checkpoint]}
    return result


def old_and_truth(pins):
    """Open protected scoring data only after both new and old blind chains."""
    old = previous.sealed_json(V112_REPLAY, pins[V112_REPLAY])
    if (old["schema"] != "v112_independent_blind_sentry_replay" or old["passed"] is not True
            or old["track"] != "estimated_sun" or old["calibration_or_final_access"] is not False
            or old["dispatch_sha256"] != pins[V112_DISPATCH]
            or old["matcher_calls"] != 0 or old["renderer_calls"] != 0 or old["fresh_pose_fits"] != 0):
        raise ValueError("Require the published independent v112 sentry baseline")
    old_blind = previous.verify_blind(old["rank_seal_sha256"])
    labels, pose, distance, gates, truth_hashes = previous.open_postseal_truth(
        old_blind, old["rank_seal_sha256"], old["blind_complete_sha256"], old["watchdog_sha256"])
    exact = previous.score(old_blind, labels, pose, distance)
    if (exact != old["rows"] or previous.summarize(exact) != old["summary"]
            or distance != old["median_reference_range_m"] or gates != old["positive_gates"]
            or any(old.get(key) != value for key, value in truth_hashes.items())):
        raise ValueError("The published v112 metrics did not exactly replay")
    baseline_final_reasons = {}
    for row in old_blind:
        slot = row["id"]
        bank = previous.sealed_json(previous.BLIND / "banks" / f"{slot}.json", row["bank_sha256"])
        baseline_final_reasons[slot] = bank["compact_final_reason"]
    return old_blind, baseline_final_reasons, labels, pose, distance, truth_hashes


def feasibility(pose):
    down_dot = float(-pose[2, 2])
    return dict(off_nadir_deg=float(np.degrees(np.arccos(np.clip(down_dot, -1., 1.)))),
                within_simulator_cone=bool(down_dot >= MINIMUM_DOWN_DOT))


def measure_slot(poses, final, final_reason, empty_reason, positive, truth, distance):
    def score_pose(pose):
        metrics = previous.pose_score(pose, positive, truth, distance)
        if pose is not None:
            metrics.update(feasibility(pose))
        else:
            metrics.update(off_nadir_deg=None, within_simulator_cone=False)
        return metrics

    candidates = [score_pose(pose) for pose in poses]
    eligible = [row for row in candidates if row["within_simulator_cone"]]
    return dict(
        candidate_count=len(poses), distinct_poses=previous.distinct_poses(poses),
        cone_feasible_count=len(eligible), cone_feasible_distinct=previous.distinct_poses(
            [pose for pose, row in zip(poses, candidates) if row["within_simulator_cone"]]),
        oracle_correct=any(row["correct"] for row in candidates),
        physical_oracle_correct=any(row["correct"] for row in eligible),
        candidate_metrics=candidates, empty_bank_reason=empty_reason,
        raw_top1=candidates[0] if candidates else score_pose(None),
        compact_final=score_pose(final), compact_final_reason=final_reason,
    )


def matched_rows(new, old, reasons, labels, pose, distance):
    rows = []
    for cone, baseline, identity in zip(new, old, labels, strict=True):
        if cone["id"] != baseline["id"] or cone["id"] != identity["id"]:
            raise ValueError("Sentry slot changed identity")
        positive = identity["case_id"] == previous.POSITIVE
        original = measure_slot(baseline["poses"], baseline["final"], reasons[baseline["id"]],
                                None, positive, pose, distance)
        proposed = measure_slot(cone["poses"], cone["final"], cone["final_reason"], cone["empty_reason"],
                                positive, pose, distance)
        rows.append(dict(
            **identity, positive=positive, v112=original, cone_v113=proposed,
            cone_bank_sha256=cone["sha256"], v112_bank_sha256=baseline["bank_sha256"],
            cone_diagnostics=cone["diagnostics"], cone_elapsed_seconds=cone["elapsed_seconds"],
        ))
    return rows


def summarize(rows):
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    if len(positives) != 2 or len(negatives) != 2:
        raise ValueError("Count both renderer replicates of the original positive and matched nonoverlap")
    totals = {}
    for name in ("v112", "cone_v113"):
        p, n = ([row[name] for row in group] for group in (positives, negatives))
        modes = {}
        for mode in ("raw_top1", "compact_final"):
            got, rejects = ([row[mode] for row in group] for group in (p, n))
            modes[mode] = dict(
                correct_top1=sum(row["correct"] for row in got),
                positive_false_negatives=sum(not row["correct"] for row in got),
                wrong_positive_poses=sum(row["pose_returned"] and not row["correct"] for row in got),
                no_positive_pose=sum(not row["pose_returned"] for row in got),
                raw_nonoverlap_outputs=sum(row["nonoverlap_output"] for row in rejects),
            )
        totals[name] = dict(
            candidate_oracle_recall=sum(row["oracle_correct"] for row in p),
            physical_candidate_oracle_recall=sum(row["physical_oracle_correct"] for row in p),
            positive_candidate_count=sum(row["candidate_count"] for row in p),
            positive_cone_feasible_candidates=sum(row["cone_feasible_count"] for row in p),
            nonoverlap_candidate_count=sum(row["candidate_count"] for row in n),
            nonoverlap_cone_feasible_candidates=sum(row["cone_feasible_count"] for row in n),
            **modes,
        )
    paired = {}
    for name in ("oracle_correct", "physical_oracle_correct", "raw_top1", "compact_final"):
        old, new = ([row[arm][name] for row in positives] for arm in ("v112", "cone_v113"))
        if name in ("raw_top1", "compact_final"):
            old, new = ([row["correct"] for row in values] for values in (old, new))
        paired[name] = dict(gains=sum(b and not a for a, b in zip(old, new)),
                            losses=sum(a and not b for a, b in zip(old, new)))
    old, new = (totals[arm] for arm in ("v112", "cone_v113"))
    gate = dict(
        both_positive_banks_contain_correct_pose=new["candidate_oracle_recall"] == len(positives),
        raw_nonoverlap_outputs_did_not_increase=(
            new["raw_top1"]["raw_nonoverlap_outputs"] <= old["raw_top1"]["raw_nonoverlap_outputs"]),
        compact_final_nonoverlap_outputs_did_not_increase=(
            new["compact_final"]["raw_nonoverlap_outputs"] <= old["compact_final"]["raw_nonoverlap_outputs"]),
    )
    gate["passed"] = all(gate.values())
    return dict(positive_renderer_replicates=2, negative_renderer_replicates=2,
                independent_positive_camera_pairs=1, independent_negative_camera_pairs=1,
                arms=totals, paired_positive_changes=paired, gate=gate)


def replay(seal_sha):
    seal, cone = verify_cone(seal_sha)
    protocol = verify_dispatch(seal)
    pins = protocol["metadata_sha256"]
    lineage = verify_opaque_lineage(protocol["opaque_input_sha256"])
    old, reasons, labels, pose, distance, truth_hashes = old_and_truth(pins)
    rows = matched_rows(cone, old, reasons, labels, pose, distance)
    return dict(
        schema="v113_independent_sealed_cone_replay", passed=True, track="estimated_sun",
        calibration_or_final_access=False, matcher_calls=0, renderer_calls=0, fresh_pose_fits=0,
        cone_degrees=CONE_DEGREES, rotation_threshold_deg=2., translation_fraction_threshold=.01,
        median_reference_range_m=distance, cone_seal_sha256=seal_sha,
        cone_dispatch_sha256=seal["dispatch_sha256"], cone_publication_commit=seal["publication_commit"],
        v112_replay_sha256=pins[V112_REPLAY], opaque_lineage=lineage, historical_truth_hashes=truth_hashes,
        summary=summarize(rows), rows=rows, replay_source_sha256=previous.digest(Path(__file__).read_bytes()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous cone scoring replay")
    result = replay(args.seal_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
