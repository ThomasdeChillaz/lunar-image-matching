"""Small generated-image wiring control, not a lunar experiment or cost pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from matcher import POLICIES, public_result, run
from smoke_v126 import generated

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = (HERE/"policy.py", HERE/"matcher.py", HERE/"smoke.py", HERE/"replay_smoke.py",
           HERE.parent/"solar-parallax-v126/smoke_v126.py")


def control():
    cv2.setNumThreads(1)
    rows = []
    for multidepth in (False, True):
        args, mask, truth = generated(multidepth)
        ref, depth, k, reference_pose, query, query_k, _ = args
        result = run(ref, query, depth, k, reference_pose, query_k, reference_pose,
                     reference_valid=np.ones_like(mask), height_direction_reference=[0., 0., -1.], budget=2)
        scored = {}
        for name, method in result["methods"].items():
            pose = None if method["pose"] is None else np.array(method["pose"])
            p = np.array(method["reference_xy"]).reshape(-1, 2)
            q = np.array(method["query_xy"]).reshape(-1, 2)
            expected = p.copy()
            if len(p):
                expected[:, 0] -= k[0, 0]*20./depth[p[:, 1].astype(int), p[:, 0].astype(int)]
            error = None if not len(p) else float(np.median(np.linalg.norm(q-expected, axis=1)))
            translation = None if pose is None else float(np.linalg.norm(pose[:3, 3]-truth[:3, 3]))
            searches = 0 if name == "joint_no_extra" else method["acquisition"]["attempted_extra_patches"]
            passed = translation is not None and translation < 2. and error is not None and error < .2
            if name in POLICIES:
                passed &= searches == 2
            scored[name] = dict(passed=bool(passed), translation_error_m=translation,
                               median_native_error_px=error, additional_patch_searches=searches)
        rows.append(dict(multidepth=multidepth, metrics=scored,
            result=public_result(result, research_output=True),
            scoring_only_truth=dict(cam2world=truth.tolist(),
                depth_strips=[dict(start=0, stop=180, shift=10), dict(start=180, stop=333, shift=8),
                              dict(start=333, stop=512, shift=6)] if multidepth else [dict(start=0, stop=512, shift=8)],
                focal_pixels=float(k[0, 0]))))
    return dict(schema="sequential_acquisition_procedural_v136", procedural_only=True,
        lunar_images_opened=False, calibration_or_final_access=False, training=False,
        fixed_extra_patch_budget=2, full32_patch_cost_pilot=False,
        passed=all(m["passed"] for r in rows for m in r["metrics"].values()), rows=rows,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
        scope="Generated native512 imagery; wiring only. No lunar improvement, runtime advantage, calibrated information gain or novelty established.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = control()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
