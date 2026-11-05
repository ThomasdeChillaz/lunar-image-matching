"""Post-hoc geometry-only audit of the sealed native search windows; no new matching."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "solar-local-v127"))
import pipeline
from lunar_geometry_v5 import correspondences_z, map_points_z, project_z


def local_grid_reachable(seed, target, visible, tolerance):
    """Could any integer offset in the declared square meet this pixel tolerance?"""
    delta = target-seed
    closest = np.clip(np.rint(delta), -16., 16.)
    return visible & np.isfinite(seed).all(axis=1) & (np.linalg.norm(delta-closest, axis=1) <= tolerance)


def audit():
    spec = spec_from_file_location("audit_score_v127", HERE.parent / "solar-local-v127/experiment.py")
    metric = module_from_spec(spec); spec.loader.exec_module(metric)
    home, run = Path("reports/solar-local-v130"), Path("runs/solar-local-v130/comparison")
    p = metric.read(home / "protocol.json")
    complete = metric.read(run / "COMPLETE.json")
    seal = metric.read(run / "PREDICTIONS_SEALED.json")
    raw = metric.read(run / "predictions.json")
    metric.check(complete["complete"] and seal["sha256"] == metric.sha(run / "predictions.json")
                 and raw["query_truth_used"] is False and [r["case"] for r in raw["rows"]] == p["cases"],
                 "Only complete sealed predictions may be diagnosed")
    anchors = pipeline.previous.ANCHORS
    geometry, rows = {}, []
    for index, item in enumerate(p["inputs"]):
        label = p["labels"][index]["joint_scoring_geometry"]
        for record in (item["reference_geometry"], label):
            metric.check(metric.sha(record["path"]) == record["sha256"], "Changed scoring geometry")
        with np.load(item["reference_geometry"]["path"], allow_pickle=False) as z:
            depth, k, pose = [z[key].copy() for key in ("axial_depth", "intrinsics", "cam2world")]
        with np.load(label["path"], allow_pickle=False) as z:
            fields = [z[key].copy() for key in ("reference_depth", "query_depth", "reference_intrinsics",
                       "reference_cam2world", "query_intrinsics", "query_cam2world")]
        _, target, _, visible = correspondences_z(*fields, points=anchors)
        truth = np.full((len(anchors), 2), np.nan); truth[visible] = target
        world, use = map_points_z(anchors, depth, k, pose)
        all_world = np.full((len(anchors), 3), np.nan); all_world[use] = world
        geometry[index] = all_world, truth, visible, pipeline.coarse.reference_pose_bank(pose)
    for record in raw["rows"]:
        case = record["case"]
        if not case["positive"]:
            continue
        world, target, visible, bank = geometry[case["reference_index"]]
        indices = record["initialization"]["indices"]
        predicted = np.full_like(target, np.nan)
        if indices:
            predicted, _, front = project_z(world, np.asarray(p["inputs"][case["query_index"]]["query_intrinsics"]), bank[indices[0]])
            predicted[~front] = np.nan
        masks = {str(t):local_grid_reachable(predicted, target, visible, t) for t in (1, 2, 3, 4)}
        row = dict(case=case, first_seed_index=indices[0] if indices else None, methods={})
        for name in ("legacy", "halo", "pooled_uniform", "pooled_parallax"):
            selected = record["result"]["methods"][name]["selected_anchors"]
            row["methods"][name] = dict(selected_anchors=len(selected), visible_selected=int(visible[selected].sum()),
                possible_correct_native_grid_matches={key:int(value[selected].sum()) for key,value in masks.items()},
                observed_retained_matches=len(record["result"]["methods"][name]["reference_xy"]))
        rows.append(row)
    return dict(schema="posthoc_sealed_window_capture_v130", predictions_sha256=seal["sha256"],
                posthoc=True, new_image_matching=False, new_image_decoding=False,
                calibration_or_final_access=False, rows=rows,
                scope="Geometry-only search-window potential, using query labels after inference. It is not observed matching or a guaranteed pose-recovery bound.")


if __name__ == "__main__":
    if Path.cwd() != ROOT:
        raise ValueError("Run from repository root")
    print(json.dumps(audit(), indent=2, allow_nan=False))
