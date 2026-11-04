"""Saved v97 correspondence, reprojection and local-deformation diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lunar_geometry_v5 import map_points_z, project_z


ROOT = Path("runs/solar-direct-additive-windows-v97/pilot")
ARMS = ("uniform", "translation_information")
SCORING = "reports/solar_direct_additive_windows_pilot_replay_v97.json"
PROTOCOL = "reports/solar-direct-additive-windows-pilot-v97.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return None if not len(a) else dict(median=float(np.median(a)), p90=float(np.quantile(a, .9)))


def reprojection(p, q, depth, k, ref, query_k, pose):
    world, usable = map_points_z(p, depth, k, ref)
    uv, _, front = project_z(world, query_k, pose)
    residuals = np.full(len(p), np.inf)
    residuals[np.flatnonzero(usable)[front]] = np.linalg.norm(uv[front] - q[usable][front], axis=1)
    return residuals


def local_deformation(points, depth, k, ref, query_k, pose):
    """RMS relative warp vs the translation-only template assumed by v96."""
    dy, dx = np.mgrid[-12:13:4, -12:13:4]
    offsets = np.column_stack((dx.ravel(), dy.ravel()))
    p = (points[:, None] + offsets).reshape(-1, 2)
    world, valid = map_points_z(p, depth, k, ref)
    uv, _, front = project_z(world, query_k, pose)
    target = np.full_like(p, np.nan)
    target[np.flatnonzero(valid)[front]] = uv[front]
    target = target.reshape(len(points), len(offsets), 2)
    centre = np.flatnonzero((offsets == 0).all(1)).item()
    deformation = target - target[:, [centre]] - offsets
    return np.sqrt((deformation ** 2).sum(2).mean(1))


def audit():
    protocol = read(PROTOCOL)
    summary, seal, replay = read(ROOT / "summary.json"), read(ROOT / "COMPLETE.json"), read(SCORING)
    if (seal["summary_sha256"] != sha(ROOT / "summary.json") or seal["protocol_sha256"] != sha(PROTOCOL)
            or replay["passed"] is not True or replay["replayed_rows"] != 24
            or replay["summary_sha256"] != seal["summary_sha256"]):
        raise ValueError("Exact completed v97 run and independent replay required")
    cases = {c["case_id"]: c for c in read(protocol["manifest"])["cases"]}
    rows = {(r["profile"], r["case_id"]): r for r in summary["rows"]}
    result = []
    for cid in protocol["case_ids"]:
        case = cases[cid]
        seed = rows["frozen_direct", cid]
        if not case["positive"] or not seed["metrics"]["pose_returned"]:
            continue
        if sha(seed["saved_arrays"]) != seed["saved_arrays_sha256"]:
            raise ValueError("Saved seed changed")
        if sha(case["reference_geometry"]) != case["input_sha256"][case["reference_geometry"]]:
            raise ValueError("Saved geometry changed")
        with np.load(seed["saved_arrays"], allow_pickle=False) as a:
            original = {k: a[k] for k in ("pose", "reference_xy", "query_xy")}
        with np.load(case["reference_geometry"], allow_pickle=False) as geometry:
            depth, k, ref = (geometry[key] for key in ("depth", "intrinsics", "cam2world"))
        query_k = np.asarray(case["query_intrinsics"])
        count = len(original["reference_xy"])
        for profile in ARMS:
            row = rows[profile, cid]
            if sha(row["saved_arrays"]) != row["saved_arrays_sha256"]:
                raise ValueError("Saved candidate changed")
            with np.load(row["saved_arrays"], allow_pickle=False) as a:
                p, q, pose = (a[k] for k in ("reference_xy", "query_xy", "pose"))
                visible, truth = a["scoring_visible"], a["scoring_query_xy"]
            if any(not np.array_equal(a, original[key]) for a, key in (
                (p[:count], "reference_xy"), (q[:count], "query_xy"),
            )):
                raise ValueError("Additive seed prefix changed")
            before = reprojection(p, q, depth, k, ref, query_k, original["pose"])
            after = reprojection(p, q, depth, k, ref, query_k, pose)
            truth_error = np.full(len(p), np.nan)
            truth_error[visible] = np.linalg.norm(q[visible] - truth[visible], axis=1)
            deformation = local_deformation(p[count:], depth, k, ref, query_k, original["pose"])
            after_new = after[count:] <= 3
            result.append(dict(
                case_id=cid, profile=profile, selected_light=seed["render_search"]["selected_light_index"],
                scoring_query_light=case["light_b"], added=len(p) - count, seed_matches=count,
                original_reprojection_inliers_at_seed=int((before[:count] <= 3).sum()),
                original_reprojection_inliers_at_refit=int((after[:count] <= 3).sum()),
                original_reprojection_error_px_at_seed=stats(before[:count]),
                original_reprojection_error_px_at_refit=stats(after[:count]),
                added_reprojection_inliers_at_seed=int((before[count:] <= 3).sum()),
                added_reprojection_inliers_at_refit=int(after_new.sum()),
                added_within_2px_truth=int((truth_error[count:] <= 2).sum()),
                added_refit_inliers_within_2px_truth=int((after_new & (truth_error[count:] <= 2)).sum()),
                added_true_error_px=stats(truth_error[count:]),
                added_reprojection_error_px_at_seed=stats(before[count:]),
                added_reprojection_error_px_at_refit=stats(after[count:]),
                added_reference_patch_translation_only_warp_rms_px=stats(deformation),
                pose_error_m_before=seed["metrics"]["pose_error"]["translation_m"],
                pose_error_m_after=row["metrics"]["pose_error"]["translation_m"],
            ))
    return dict(rows=result, scope="Saved development outputs only. Warp evaluated from the saved seed and reference map. "
                "Query truth is used only for retrospective error. Deformation and bad matches are associated, not experimentally isolated.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    with args.output.open("x") as stream:
        json.dump(audit(), stream, indent=2, allow_nan=False)
        stream.write("\n")
