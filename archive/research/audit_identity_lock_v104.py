"""Audit identity-lock in the sealed v101 radiometric trial without matching."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from lunar_geometry_v5 import map_points_z, project_z
from run_solar_full_native_baselines_v93 import seal_rows
from solar_aligned_scoring_v91 import load_aligned_scoring


ROOT = Path("runs/solar-radiometric-pilot-v101/pilot")
PROTOCOL = Path("reports/solar-radiometric-pilot-dispatch-v101.json")
REPLAY = Path("reports/solar_radiometric_pilot_replay_v101.json")
MANIFEST = Path("runs/solar-reference-v80-full/case-assets/manifest.json")
PINS = {
    PROTOCOL: "b7e821211828566d7d60d5250fb230d02e1e73d6092a96418ce5435ab3306b28",
    REPLAY: "4cb8d85c39cabcf84aff8f805bc9160406a268bf47a130acfe346c4f6da258f9",
    MANIFEST: "750919f111d69cba13b1560409955f7af2c4369ca6cc526b93bc01579993c3e2",
    ROOT / "COMPLETE.json": "51573531e76a6b4cac2c0efe30d9ea07a7f55bda3eb4789f217e81f75b3e2e73",
    ROOT / "summary.json": "846e49015894f481faa233f2bf3272818e7f35898ebfe913a5536e00077504ff",
    ROOT / "watchdog.json": "71a1882b1887b933caf1cd8931bb62c5819574d06831ed55618c2b30e31a803e",
    ROOT / "execution.json": "9aff56411e70be2500afa9afda969cd47a2881bab1118d6bfbfe1077a2401110",
    ROOT / "reservation.json": "96a44cc6e75e0de03c643e612476adf9bc695cf0c4f30e029f34f99c8c3122ab",
}
FOCUS = ("u025-l00-00-positive", "u027-l08-01-positive")
THRESHOLDS = (0.25, 1, 2, 3, 4, 8, 12)
ARM = ("reference_camera_render_match", "reference_camera_render_match_p995_exposure")
OUTCOMES = ("correct", "strict", "pose_returned", "wrong_positive", "raw_nonoverlap")
ARRAYS = ("reference_xy", "query_xy", "pose", "scoring_query_xy", "scoring_visible", "scoring_query_cam2world")


def checked_bytes(path, expected):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Frozen bytes differ at {path}")
    return data


@lru_cache(maxsize=72)
def assert_geometry_pin(path, expected):
    checked_bytes(path, expected)


@lru_cache(maxsize=32)
def reference_geometry(path, expected):
    with np.load(io.BytesIO(checked_bytes(path, expected)), allow_pickle=False) as z:
        return z["depth"], z["intrinsics"], z["cam2world"]


def errors(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Only finite selected residuals enter statistics")
    return dict(
        count=len(values),
        zero=int(np.count_nonzero(values == 0.0)),
        within={str(t): int(np.count_nonzero(values <= t)) for t in THRESHOLDS},
        median_px=float(np.median(values)) if len(values) else None,
        p95_px=float(np.quantile(values, .95)) if len(values) else None,
    )


def pose_delta(a, b):
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.) / 2.
    return dict(rotation_deg=float(np.degrees(np.arccos(np.clip(cos, -1., 1.)))),
                translation_m=float(np.linalg.norm(a[:3, 3] - b[:3, 3])))


def diagnose(row, case):
    with np.load(io.BytesIO(checked_bytes(row["saved_arrays"], row["saved_arrays_sha256"])), allow_pickle=False) as z:
        p, q, pose, target, visible, query_pose = [z[key] for key in ARRAYS]
    if (p.shape != q.shape or p.shape != (len(p), 2)
            or not np.isfinite(p).all() or not np.isfinite(q).all()
            or not ((p >= 0) & (p <= 511)).all() or not ((q >= 0) & (q <= 511)).all()
            or pose.shape not in ((4, 4), (0, 4))):
        raise ValueError("Invalid native pixel or pose array")
    metrics = row["metrics"]
    has_pose = pose.shape == (4, 4)
    if has_pose != metrics["pose_returned"]:
        raise ValueError("Saved pose presence differs from the independent row")

    paths = case["input_sha256"]
    ref_path = case["reference_geometry"]
    depth, k_ref, ref_pose = reference_geometry(ref_path, paths[ref_path])
    k_query = np.asarray(case["query_intrinsics"], float)
    world, mapped = map_points_z(p, depth, k_ref, ref_pose)
    q_mapped = q[mapped]

    def reprojection(at):
        uv, _, in_front = project_z(world, k_query, at)
        valid = in_front & np.isfinite(uv).all(axis=1)
        return errors(np.linalg.norm(q_mapped[valid] - uv[valid], axis=1))

    displacement = errors(np.linalg.norm(q - p, axis=1))
    valid_truth = case["positive"]
    if valid_truth:
        for key in ("scoring_labels", "scoring_geometry"):
            path = case[key]
            assert_geometry_pin(path, paths[path])
        truth, method = load_aligned_scoring(case, p)
        if (method != row["label_method"]
                or not np.array_equal(target, truth["query_xy"], equal_nan=True)
                or not np.array_equal(visible, truth["visible"])
                or not np.array_equal(query_pose, truth["query_cam2world"])):
            raise ValueError("Independent positive scoring geometry differs")
        true_error = errors(np.linalg.norm(q[visible] - target[visible], axis=1))
        true_motion = errors(np.linalg.norm(p[visible] - target[visible], axis=1))
        for threshold, details in metrics["correspondence"].items():
            if details != dict(correct=true_error["within"][threshold], co_visible=int(visible.sum()), returned=len(p)):
                raise ValueError("Recomputed true correspondence counts differ")
    else:
        if len(target) or len(visible) or len(query_pose):
            raise ValueError("Nonoverlap must not have positive geometry labels")
        true_error = true_motion = None
    error = pose_delta(pose, query_pose) if has_pose and valid_truth else None
    if error is not None and any(abs(error[k] - v) > 1e-9 for k, v in metrics["pose_error"].items()):
        raise ValueError("Recomputed true pose error differs")

    render = row["render_search"]
    radiometry = render["radiometry"]
    selected = render["selected_light_index"]
    final_reference = next(entry for entry in radiometry["reference_renders"] if entry["light_index"] == selected)
    return dict(
        case_id=row["case_id"], profile=row["profile"], terrain=case["component"], query_terrain=case["query_component"],
        positive=valid_truth, light_a=case["light_a"], light_b=case["light_b"], selected_reference_light=selected,
        arrays_sha256=row["saved_arrays_sha256"], reason=row["reason"], **{k: metrics[k] for k in OUTCOMES},
        query_radiometry=radiometry["query"], selected_reference_radiometry=final_reference,
        final_estimator_inliers=row["final_estimator"]["inliers"],
        reference_depth_support=int(mapped.sum()), native_displacement=displacement,
        reference_pose_reprojection_using_query_intrinsics=reprojection(ref_pose),
        fitted_pose_reprojection=reprojection(pose) if has_pose else None,
        true_correspondence_error=true_error, true_native_displacement=true_motion,
        fitted_pose_delta_from_reference=pose_delta(pose, ref_pose) if has_pose else None,
        true_pose_delta_from_reference=pose_delta(query_pose, ref_pose) if valid_truth else None,
        fitted_pose_error_from_truth=error,
        identity_concentrated=(len(p) >= 6 and displacement["within"]["1"] >= .9 * len(p)),
    )


def audit():
    pinned = {p: json.loads(checked_bytes(p, digest)) for p, digest in PINS.items()}
    protocol, replay, manifest = (pinned[path] for path in (PROTOCOL, REPLAY, MANIFEST))
    seal, summary, watchdog, execution, reservation = (pinned[ROOT / name] for name in (
        "COMPLETE.json", "summary.json", "watchdog.json", "execution.json", "reservation.json"))
    if (protocol["split"] != "model_selection" or protocol["tracks"] != ["estimated_sun"]
            or protocol["calibration_or_final_access"] or protocol["prepared_only"]
            or protocol["manifest"] != str(MANIFEST) or protocol["output"] != str(ROOT)
            or tuple(profile["name"] for profile in protocol["profiles"]) != ARM
            or len(protocol["case_ids"]) != 48 or manifest["calibration_or_final_access"]
            or not (seal["external_watchdog_passed"] and summary["complete"] and watchdog["complete"] and replay["passed"])
            or replay["accepted_results"] or replay["replayed_rows"] != 96
            or watchdog["child_returncode"] != 0 or watchdog["compute_reservation"] != reservation
            or any(value["protocol_sha256"] != PINS[PROTOCOL] for value in (seal, summary, watchdog, execution, replay))
            or any(value["summary_sha256"] != PINS[ROOT / "summary.json"] for value in (seal, replay))
            or any(value["watchdog_sha256"] != PINS[ROOT / "watchdog.json"] for value in (seal, replay))):
        raise ValueError("Require the exact completed, independently replayed, primary-development v101 evidence")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows, sealed_pins = seal_rows(ROOT, protocol, protocol["case_ids"])
    if rows != summary["rows"] or len(rows) != len(replay["rows"]) or len(sealed_pins) != 3*len(rows):
        raise ValueError("Terminal evidence or one-cell-per-case seal differs")
    output = []
    for row, verified in zip(rows, replay["rows"], strict=True):
        if (any(row[field] != verified[field] for field in ("case_id", "profile"))
                or any(row["metrics"][field] != verified[field] for field in (*OUTCOMES, "correspondence"))
                or row["inference_failure"] is not None):
            raise ValueError("Sealed row differs from the verified metrics")
        case = cases[row["case_id"]]
        if any(row[k] != case[k] for k in ("component", "query_component", "positive", "light_a", "light_b")):
            raise ValueError("Original case identity differs")
        output.append(diagnose(row, case))

    arms = {}
    for name in ARM:
        arm = [row for row in output if row["profile"] == name]
        groups = Counter(
            "correct_positive" if row["correct"] else
            "wrong_positive" if row["wrong_positive"] else
            "nonoverlap_pose" if row["raw_nonoverlap"] else
            "no_pose_positive" if row["positive"] else "no_pose_nonoverlap"
            for row in arm if row["identity_concentrated"]
        )
        arms[name] = dict(
            cases=len(arm), eligible_positive=sum(row["positive"] for row in arm),
            identity_concentrated_by_outcome=dict(groups),
            identity_concentrated_cases=[row["case_id"] for row in arm if row["identity_concentrated"]],
        )
    return dict(
        scope="Exact sealed v101 development outputs only. No image files, inference, pose fitting, acceptance or held-out data.",
        interpretation="Identity concentration is a post hoc descriptive pattern, not a validated rejection threshold or cause.",
        descriptive_criterion="At least six correspondences and at least 90% within 1 native pixel of zero displacement.",
        p95_convention="Linear quantile of finite retained residuals; null for empty rows.",
        terminal_sha256={str(k): v for k, v in PINS.items()}, sealed_cell_count=len(output),
        arm_summary=arms, focus=[row for row in output if row["case_id"] in FOCUS], rows=output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
