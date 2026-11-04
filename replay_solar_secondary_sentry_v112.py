"""Model-free scoring and coverage replay for the sealed v112 blind sentry."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path("runs/solar-secondary-acquisition-v112")
BLIND = ROOT / "blind-sentry"
RANK_SEAL = BLIND / "ranks/COMPLETE.json"
SENTRY_COMPLETE = BLIND / "COMPLETE.json"
WATCHDOG = ROOT / "watchdog-sentry.json"
DISPATCH = "reports/solar-secondary-sentry-dispatch-v112.json"
ROSTER = "reports/solar-secondary-ranking-roster-v112.json"
ROSTER_SHA = "59ea302a5bd9f924dd03366ebe66b92b5458ea29a4abb47371ae205ae6e3826f"
SLOTS = tuple(f"r{index:02d}" for index in range(4))
METHODS = ("original_beam", "combined", "direct", "combined_residual", "direct_residual", "compact_final")
POSITIVE = "v112-np31-r0-c7-l08-23-positive"
NEGATIVE = "v112-np31-r0-c7-l08-23-to-np31-r3-c2-nonoverlap"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sealed_bytes(name, sha256):
    path = Path(name)
    if (not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or path.is_absolute() or ".." in path.parts or str(path) != str(name)
            or path.resolve() != Path.cwd() / path):
        raise ValueError("A canonical repository-relative path and full artifact SHA are required")
    data = path.read_bytes()
    if digest(data) != sha256:
        raise ValueError("Sealed artifact changed: " + str(path))
    return data


def sealed_json(path, sha256):
    return json.loads(sealed_bytes(path, sha256))


def asset_bytes(asset, expected):
    if set(asset) != {"path", "sha256"} or asset["path"] != str(expected):
        raise ValueError("Unexpected sealed artifact inventory: " + str(expected))
    return sealed_bytes(asset["path"], asset["sha256"])


def validated_pose(values):
    pose = np.asarray(values, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0., 0., 0., 1.], rtol=0., atol=1e-8)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0., atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., rtol=0., atol=1e-5)):
        raise ValueError("Expected a finite camera-to-world rigid transform")
    return pose


def distinct_poses(poses):
    return len({tuple(pose.ravel()) for pose in poses})


def support_statistics(masks):
    """Independently count the all-beam, both-atlas-arm native support."""
    if (masks.ndim != 3 or not len(masks) or masks.shape[1:] != (512, 512)
            or masks.dtype != np.bool_):
        raise ValueError("Expected one native Boolean support raster per eligible beam")
    common = np.logical_and.reduce(masks, axis=0)
    counts = np.array([[np.count_nonzero(common[y:y + 32, x:x + 32]) for x in range(0, 512, 32)]
                       for y in range(0, 512, 32)])
    qualifying = counts >= 512
    quadrants = (qualifying[:8, :8], qualifying[:8, 8:], qualifying[8:, :8], qualifying[8:, 8:])
    quadrant_blocks = [int(np.count_nonzero(part)) for part in quadrants]
    active = [i for i, value in enumerate(quadrant_blocks) if value >= 2]
    retained = qualifying.copy()
    for i, slices in enumerate((np.s_[:8, :8], np.s_[:8, 8:], np.s_[8:, :8], np.s_[8:, 8:])):
        if i not in active:
            retained[slices] = False
    return dict(common_pixels=int(np.count_nonzero(common)), retained_blocks=int(np.count_nonzero(retained)),
                retained_pixels=int(counts[retained].sum()), quadrant_blocks=quadrant_blocks,
                active_quadrants=active)


def ordered_mode(mode, expected_order, force_fallback):
    order = mode["order"]
    if (type(order) is not list or any(type(index) is not int for index in order)
            or sorted(order) != expected_order):
        raise ValueError("Ranked indices must be a permutation of the sealed bank")
    if force_fallback and mode["reason"] != "insufficient_shared_quadrant_support":
        raise ValueError("A coverage failure must have the original-order fallback")
    if mode["reason"] is not None:
        if order != expected_order or mode["scores"] is not None or mode["quadrant_losses"] is not None:
            raise ValueError("A rank fallback cannot change the original ordering")
    else:
        scores = np.asarray(mode["scores"], dtype=float)
        if scores.shape != (len(order),) or not np.isfinite(scores).all():
            raise ValueError("Ranked mode is missing sealed candidate scores")
    return order


def verify_slot(seal_asset, slot):
    rank_path = BLIND / "ranks" / f"{slot}.json"
    bank_path = BLIND / "banks" / f"{slot}.json"
    prediction_path = BLIND / "predictions" / f"{slot}.json"
    rank = json.loads(asset_bytes(seal_asset, rank_path))
    prediction = json.loads(asset_bytes(rank["prediction"], prediction_path))
    if rank["bank"] != prediction["bank"]:
        raise ValueError("Ranking and prediction used different blind banks")
    bank = json.loads(asset_bytes(rank["bank"], bank_path))
    if rank["id"] != slot or bank["id"] != slot:
        raise ValueError("Opaque bank and ranking identities differ")
    asset_bytes(bank["query_luminance"], BLIND / "banks" / f"{slot}-luminance.npz")
    blind = bank["blind_bank"]
    if (blind["schema"] != "solar_blind_beams_v112" or blind["track"] != "estimated_sun"
            or blind["pose_frame"] != "camera_to_reference_scene_enu" or blind["sun_frame"] != "reference_scene_enu"):
        raise ValueError("Unrecognized blind pose and Sun bank convention")
    candidates = blind["candidates"]
    poses = [validated_pose(row["pose"]) for row in candidates]
    original = list(range(len(poses)))
    beam_ids = [row["source_beam_index"] for row in candidates]
    beam_scores = np.array([row["score"] for row in candidates], dtype=float)
    distinct = distinct_poses(poses)
    if (len(poses) > 8 or rank["original_order"] != original or rank["distinct_poses"] != distinct
            or not np.isfinite(beam_scores).all() or np.any(np.diff(beam_scores) > 0)
            or len(set(beam_ids)) != len(beam_ids)
            or any(type(index) is not int or not 0 <= index < 8 for index in beam_ids)
            or (poses and (blind["capture_reached"] is not True or blind["reason"] is not None))):
        raise ValueError("Unexpected order, diversity or inventory in the frozen blind bank")
    suns = np.asarray([row["sun_reference_enu"] for row in candidates], dtype=float)
    if len(poses) and (suns.shape != (len(poses), 3) or not np.isfinite(suns).all()
                       or not np.allclose(np.linalg.norm(suns, axis=1), 1., rtol=0., atol=1e-6)):
        raise ValueError("Require finite estimated unit Sun directions")
    unavailable = ("no_eligible_blind_beams" if not poses else
                   "below_horizon_candidate_sun" if np.any(suns[:, 2] <= 0) else None)
    if prediction["unavailable"] != unavailable:
        raise ValueError("Prediction availability does not match the sealed blind inputs")
    orders = {method: original for method in METHODS if method != "compact_final"}
    support = None
    passed = False
    if unavailable is not None:
        reason = unavailable
        if prediction["prediction"] is not None or rank["rankings"] is not None or "support" in rank:
            raise ValueError("Unavailable atlas cannot supply ranked results")
    else:
        data = asset_bytes(prediction["prediction"], BLIND / "predictions" / f"{slot}.npz")
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if set(arrays.files) != {"reference_valid", "direct", "combined"}:
                raise ValueError("Unexpected reference prediction arrays")
            masks = arrays["reference_valid"]
            if len(masks) != len(poses):
                raise ValueError("Atlas mask count differs from the blind bank")
            support = support_statistics(masks)
            for arm in ("direct", "combined"):
                image = arrays[arm]
                if image.shape != masks.shape or not np.isfinite(image[masks]).all():
                    raise ValueError("A supported atlas pixel must be finite")
        if rank["support"] != support:
            raise ValueError("Independent reference-only coverage does not match the sealed decision")
        covered = len(support["active_quadrants"]) >= 3
        passed = distinct >= 2 and covered
        reason = "candidate_starved" if distinct < 2 else None if covered else "insufficient_shared_quadrant_support"
        if set(rank["rankings"]) != {"combined", "direct"}:
            raise ValueError("Both preregistered atlas rankings are required")
        for arm in ("combined", "direct"):
            replay = rank["rankings"][arm]
            if any(replay[key] != value for key, value in support.items()):
                raise ValueError("Ranked mode used different native support")
            orders[arm] = ordered_mode(replay["primary"], original, not covered)
            orders[arm + "_residual"] = ordered_mode(replay["residual_after_plane"], original, not covered)
    if rank["coverage_pass"] is not passed or rank["reason"] != reason:
        raise ValueError("Independent sentry coverage decision differs")
    final = bank["compact_final_pose"]
    if final is not None:
        final = validated_pose(final)
    return dict(
        id=slot, poses=poses, orders=orders, final=final, coverage_pass=passed, reason=reason,
        distinct_poses=distinct, support=support,
        rank_sha256=seal_asset["sha256"], bank_sha256=rank["bank"]["sha256"],
        prediction_receipt_sha256=rank["prediction"]["sha256"],
        prediction_sha256=None if unavailable else prediction["prediction"]["sha256"],
        luminance_sha256=bank["query_luminance"]["sha256"],
    )


def verify_blind(rank_seal_sha):
    """Finish the blind content-graph checks before any scoring metadata opens."""
    seal = sealed_json(RANK_SEAL, rank_seal_sha)
    if (set(seal) != {"ranks", "scoring_truth_opened"} or seal["scoring_truth_opened"] is not False
            or len(seal["ranks"]) != len(SLOTS)):
        raise ValueError("Require the complete pre-truth blind rank seal")
    return [verify_slot(item, slot) for item, slot in zip(seal["ranks"], SLOTS)]


def reference_range(depth, intrinsics):
    if depth.shape != (512, 512) or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("Geometry gate requires complete finite positive axial depth")
    k = np.asarray(intrinsics, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Expected finite native reference intrinsics")
    y, x = np.indices(depth.shape)
    pixels = np.stack((x.ravel(), y.ravel(), np.ones(depth.size)))
    rays = np.linalg.solve(k, pixels)
    return float(np.median(depth.ravel() * np.sqrt(np.sum(rays**2, axis=0))))


def open_postseal_truth(verified, rank_seal_sha, complete_sha, watchdog_sha):
    """Open hashed scoring assets only after the blind content graph was verified."""
    complete = sealed_json(SENTRY_COMPLETE, complete_sha)
    if (complete["complete"] is not True or complete["query_pose_scored"] is not False
            or complete["calibration_or_final_access"] is not False
            or complete["gate_uses_only_blind_candidate_diversity_and_reference_masks"] is not True
            or complete["rank_seal"] != {"path": str(RANK_SEAL), "sha256": rank_seal_sha}):
        raise ValueError("Sentry completion does not bind the independently verified blind seal")
    watchdog = sealed_json(WATCHDOG, watchdog_sha)
    if (watchdog["complete"] is not True or watchdog["phase"] != "sentry" or watchdog["error"] is not None
            or watchdog["finished_steps"] != ["prepare", "reference", "blind", "project", "seal"]):
        raise ValueError("Require the terminal published sentry controller")
    plan_bytes = sealed_bytes(DISPATCH, watchdog["protocol_sha256"])
    commit = watchdog["publication_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected the original published full commit")
    if subprocess.check_output(["git", "show", f"{commit}:{DISPATCH}"], timeout=5) != plan_bytes:
        raise ValueError("Actual sentry dispatch differs from the published commit")
    plan = json.loads(plan_bytes)
    if (plan["schema"] != 112 or plan["phase"] != "sentry" or plan["root"] != str(ROOT)
            or plan["split"] != "model_selection" or plan["calibration_or_final_access"] is not False
            or plan["metadata_sha256"][ROSTER] != ROSTER_SHA):
        raise ValueError("Require the fixed development-only sentry dispatch")

    # These contain or join query labels. Hash every upstream receipt first.
    acquisition = ROOT / "watchdog-acquire.json"
    gate_path = ROOT / "gate.json"
    geometry_path = ROOT / "geometry/COMPLETE.json"
    render_path = ROOT / "sentry/COMPLETE.json"
    upstream = {str(path) for path in (acquisition, gate_path, geometry_path, render_path)}
    if set(plan["predecessor_sha256"]) != upstream:
        raise ValueError("Frozen acquisition predecessor inventory differs")
    frozen = {name: sealed_bytes(name, sha) for name, sha in plan["predecessor_sha256"].items()}
    if (json.loads(frozen[str(acquisition)])["complete"] is not True
            or json.loads(frozen[str(gate_path)])["passed"] is not True
            or json.loads(frozen[str(render_path)])["complete"] is not True):
        raise ValueError("Acquisition predecessor or geometry gate failed")
    geometry = json.loads(frozen[str(geometry_path)])
    if geometry["complete"] is not True or geometry["calibration_or_final_access"] is not False:
        raise ValueError("Frozen development camera geometry incomplete")

    prepared = json.loads((BLIND / "PREPARED.json").read_bytes())
    if prepared["query_labels_in_request"] is not False or prepared["observation_files"] != 6:
        raise ValueError("Expected opaque sanitized observation handoff")
    assignments = json.loads(asset_bytes(prepared["scoring_index"], BLIND / "scoring-index.json"))
    expected = [dict(id=slot, case_id=case, replicate=replicate)
                for slot, (case, replicate) in zip(SLOTS, [(case, replicate) for case in (POSITIVE, NEGATIVE)
                                                         for replicate in (0, 1)])]
    if assignments != expected:
        raise ValueError("Scoring index differs from the frozen case and renderer-replicate order")
    roster = sealed_json(ROSTER, ROSTER_SHA)
    cases = {row["case_id"]: row for row in roster["cases"]}
    endpoints = {row["endpoint"]: row for row in roster["endpoints"]}
    if (roster["calibration_or_final_access"] is not False
            or roster["early_coverage_stop"]["fixed_case_ids"] != [POSITIVE, NEGATIVE]
            or roster["early_coverage_stop"]["positive_gate_case_id"] != POSITIVE
            or cases[POSITIVE]["positive"] is not True or cases[NEGATIVE]["positive"] is not False
            or any(cases[name]["split"] != "model_selection" for name in (POSITIVE, NEGATIVE))):
        raise ValueError("Invalid protected roster split or sentry case identity")
    cameras = {(row["endpoint"], row["role"]): row for row in geometry["cameras"]}
    reference = cases[POSITIVE]["reference_endpoint"]
    query = cases[POSITIVE]["query_endpoint"]
    if reference != query or reference != cases[NEGATIVE]["reference_endpoint"]:
        raise ValueError("The positive pair must share the matched negative reference endpoint")
    ref_row, query_row = (cameras[(reference, "reference")], cameras[(query, "query")])
    with np.load(io.BytesIO(sealed_bytes(ref_row["path"], ref_row["sha256"])), allow_pickle=False) as ref:
        if (not np.array_equal(ref["cam2world"], endpoints[reference]["camera_pair"]["reference"]["cam2world"])
                or not np.array_equal(ref["intrinsics"], endpoints[reference]["camera_pair"]["reference"]["intrinsics"])):
            raise ValueError("Sealed reference camera differs from the frozen roster")
        median_range = reference_range(ref["axial_depth"], ref["intrinsics"])
    with np.load(io.BytesIO(sealed_bytes(query_row["path"], query_row["sha256"])), allow_pickle=False) as query_file:
        query_pose = validated_pose(query_file["cam2world"])
        if not np.array_equal(query_pose, endpoints[query]["camera_pair"]["query"]["cam2world"]):
            raise ValueError("Sealed query truth differs from the frozen roster")
    gates = [dict(**item, coverage_pass=record["coverage_pass"], reason=record["reason"])
             for item, record in zip(assignments, verified) if item["case_id"] == POSITIVE]
    if complete["positive_gates"] != gates or complete["sentry_pass"] is not all(g["coverage_pass"] for g in gates):
        raise ValueError("Positive-only coverage gate differs from the sealed worker verdict")
    return assignments, query_pose, median_range, gates, {
        "publication_commit": commit, "dispatch_sha256": watchdog["protocol_sha256"],
        "rank_seal_sha256": rank_seal_sha, "blind_complete_sha256": complete_sha, "watchdog_sha256": watchdog_sha,
        "scoring_index_sha256": prepared["scoring_index"]["sha256"], "roster_sha256": ROSTER_SHA,
        "reference_geometry_sha256": ref_row["sha256"], "query_geometry_sha256": query_row["sha256"],
    }


def pose_score(pose, positive, truth, median_range):
    rotation = translation = None
    if pose is not None and positive:
        cos_angle = (np.sum(pose[:3, :3] * truth[:3, :3]) - 1.) / 2.
        rotation = float(np.degrees(np.arccos(np.clip(cos_angle, -1., 1.))))
        translation = float(np.linalg.norm(pose[:3, 3] - truth[:3, 3]))
    correct = bool(rotation is not None and rotation <= 2. and translation <= .01 * median_range)
    return dict(pose_returned=pose is not None, correct=correct,
                nonoverlap_output=bool(not positive and pose is not None),
                rotation_error_deg=rotation, translation_error_m=translation)


def score(verified, assignments, truth, median_range):
    rows = []
    for bank, identity in zip(verified, assignments, strict=True):
        if bank["id"] != identity["id"]:
            raise ValueError("Protected join changed opaque slot identity")
        positive = identity["case_id"] == POSITIVE
        candidates = [pose_score(pose, positive, truth, median_range) for pose in bank["poses"]]
        results = {method: dict(candidate_index=(order[0] if order else None),
                                **(candidates[order[0]] if order else pose_score(None, positive, truth, median_range)))
                   for method, order in bank["orders"].items()}
        results["compact_final"] = dict(candidate_index=None,
                                        **pose_score(bank["final"], positive, truth, median_range))
        provenance = {k: v for k, v in bank.items() if k.endswith("_sha256")}
        rows.append(dict(**identity, positive=positive, candidate_count=len(candidates),
                         candidate_oracle_correct=any(row["correct"] for row in candidates),
                         coverage_pass=bank["coverage_pass"], gate_reason=bank["reason"],
                         distinct_poses=bank["distinct_poses"], support=bank["support"],
                         candidates=candidates, methods=results, blind_sha256=provenance))
    return rows


def summarize(rows):
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    if len(positives) != 2 or len(negatives) != 2:
        raise ValueError("Keep both positive and both nonoverlap renderer replicates")
    results = {}
    for method in METHODS:
        p = [row["methods"][method] for row in positives]
        n = [row["methods"][method] for row in negatives]
        original = [row["methods"]["original_beam"] for row in positives]
        results[method] = dict(
            correct_top1=sum(row["correct"] for row in p), positive_total=len(p),
            wrong_positive_poses=sum(row["pose_returned"] and not row["correct"] for row in p),
            no_positive_pose=sum(not row["pose_returned"] for row in p),
            nonoverlap_outputs=sum(row["nonoverlap_output"] for row in n), nonoverlap_total=len(n),
            gains_vs_original_beam=sum(a["correct"] and not b["correct"] for a, b in zip(p, original)),
            losses_vs_original_beam=sum(b["correct"] and not a["correct"] for a, b in zip(p, original)),
        )
    return dict(methods=results, candidate_oracle_correct=sum(row["candidate_oracle_correct"] for row in positives),
                positive_renderer_replicates=len(positives), nonoverlap_renderer_replicates=len(negatives),
                positive_case_ids=len({row["case_id"] for row in positives}),
                negative_case_ids=len({row["case_id"] for row in negatives}))


def replay(rank_seal_sha, complete_sha, watchdog_sha):
    blind = verify_blind(rank_seal_sha)
    assignments, truth, distance, gates, provenance = open_postseal_truth(
        blind, rank_seal_sha, complete_sha, watchdog_sha)
    rows = score(blind, assignments, truth, distance)
    return dict(
        schema="v112_independent_blind_sentry_replay", passed=True,
        calibration_or_final_access=False, matcher_calls=0, renderer_calls=0, fresh_pose_fits=0,
        track="estimated_sun", coverage_gate_pass=all(row["coverage_pass"] for row in gates),
        positive_gates=gates, rotation_threshold_deg=2., translation_fraction_threshold=.01,
        median_reference_range_m=distance, summary=summarize(rows), rows=rows, **provenance,
        replay_source_sha256=digest(Path(__file__).read_bytes()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-seal-sha256", required=True)
    parser.add_argument("--blind-complete-sha256", required=True)
    parser.add_argument("--watchdog-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior replay output")
    result = replay(args.rank_seal_sha256, args.blind_complete_sha256, args.watchdog_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(coverage_gate_pass=result["coverage_gate_pass"],
                          positive_gates=result["positive_gates"], **result["summary"]), indent=2))


if __name__ == "__main__":
    main()
