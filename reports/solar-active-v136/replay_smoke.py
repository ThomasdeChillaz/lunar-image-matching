"""Independent saved arithmetic for generated acquisition controls; no matcher."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def replay(path):
    path = Path(path); raw = json.loads(path.read_text())
    if (raw["procedural_only"] is not True or raw["lunar_images_opened"] is not False
            or raw["fixed_extra_patch_budget"] != 2 or len(raw["rows"]) != 2):
        raise ValueError("Require the complete small software control, not a lunar study")
    for source, digest in raw["source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError("Control source changed: "+source)
    count = 0
    for row in raw["rows"]:
        exposed = row["result"]
        if exposed["pose"] is not None or exposed["acceptance_status"] != "uncalibrated":
            raise ValueError("Uncalibrated public output exposed a pose")
        truth = np.array(row["scoring_only_truth"]["cam2world"])
        names = {"joint_no_extra", "uniform", "appearance", "geometry", "active"}
        if set(exposed["raw_research"]["methods"]) != names:
            raise ValueError("Missing acquisition controls")
        schedules = []
        for name, method in exposed["raw_research"]["methods"].items():
            p, q = (np.asarray(method[k], float).reshape(-1, 2) for k in ("reference_xy", "query_xy"))
            if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
                raise ValueError("Invalid measured image coordinates")
            expected = p.copy()
            for band in row["scoring_only_truth"]["depth_strips"]:
                selected = (p[:, 1] >= band["start"]) & (p[:, 1] < band["stop"])
                expected[selected, 0] -= band["shift"]
            error = None if not len(p) else float(np.median(np.linalg.norm(q-expected, axis=1)))
            pose = method["pose"]
            translation = None if pose is None else float(np.linalg.norm(np.array(pose)[:3, 3]-truth[:3, 3]))
            metrics = row["metrics"][name]
            for actual, saved in ((error, metrics["median_native_error_px"]), (translation, metrics["translation_error_m"])):
                if (actual is None) != (saved is None) or (actual is not None and not np.isclose(actual, saved, rtol=0, atol=1e-10)):
                    raise ValueError("Saved control metric differs from independent arithmetic")
            passed = translation is not None and translation < 2. and error is not None and error < .2
            if name != "joint_no_extra":
                acquisition = method["acquisition"]; selected = acquisition["selected_indices"]
                if len(selected) != len(set(selected)) or len(selected) != acquisition["attempted_extra_patches"]:
                    raise ValueError("Missing/duplicate/uncharged extra acquisition")
                passed &= len(selected) == 2
                schedules.append(acquisition["cell_schedule"])
            if bool(passed) != metrics["passed"] or not passed:
                raise ValueError("Generated known-transform control failed")
            count += 1
        if len({tuple(s) for s in schedules}) != 1:
            raise ValueError("Acquisition controls used different spatial slots")
    return dict(passed=True, scope="Generated-image arithmetic only; no lunar performance claim.",
                saved_method_case_cells=count, source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                new_matcher_inference=False, new_pose_fitting=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps(result, indent=2))
