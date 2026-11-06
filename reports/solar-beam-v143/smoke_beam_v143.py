"""Generated branch-switch control, not new lunar or physical-sunlight evidence."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beam_matcher_v143 as beam
from procedural_fixture_v138 import fixture


def control():
    cv2.setNumThreads(1)
    catalogue, query, depth, k, poses = fixture()
    # A controlled proposal bank deliberately puts the wrong light/camera first.
    # The known transform is used only for this generated software-control setup
    # and subsequent scoring, never supplied to the real image-only initializer.
    hypotheses = [dict(branch_id=i, light_slot=slot, light_id=catalogue.light_ids[slot],
        catalogue_sun_world=catalogue.sun_world[slot].tolist(), pose=poses[pose])
        for i, (slot, pose) in enumerate(((0, 0), (1, 1)))]
    mask = np.ones_like(depth, bool)
    observed, valid = beam.native.prepare(query)
    branches = beam.prepare_branches(catalogue, hypotheses, depth, k, np.eye(4), k, mask, observed, valid)
    rows, saved = {}, {}
    for mode in ("single_uniform", "beam_uniform", "beam_active"):
        result = beam.run_policy(branches, observed, valid, depth, k, np.eye(4), k,
                                 np.array([0., 0., -1.]), mode, 64)
        predicted = None if result["pose"] is None else np.asarray(result["pose"])
        error = None if predicted is None else float(np.linalg.norm(predicted[:3, 3]-poses[1, :3, 3]))
        rows[mode] = dict(correct=error is not None and error < 2., translation_error_m=error,
            chosen_light_id=result["light_id"], attempted_native_fields=result["acquisition"]["attempted_native_fields"],
            retained_branch_ids=result["acquisition"]["retained_branch_ids"])
        saved[mode] = result
    passed = (not rows["single_uniform"]["correct"]
        and rows["beam_uniform"]["correct"] and rows["beam_active"]["correct"]
        and all(row["attempted_native_fields"] == 64 for row in rows.values())
        and all(rows[n]["chosen_light_id"] == 8 for n in ("beam_uniform", "beam_active")))
    return dict(schema="generated_persistent_branch_switch_v143", passed=passed, rows=rows,
        generated_only=True, physical_sunlight_simulation=False, lunar_images_opened=False,
        deliberately_injected_control_hypotheses=True, actual_native_correlations_and_final_ap3p=True,
        raw_research=saved, scoring_only_camera=poses[1].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = control()
    with parser.parse_args().output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_research"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
