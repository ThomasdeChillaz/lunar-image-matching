"""Explain the frozen pilot using saved evidence only; never fit another pose."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path("runs/solar-joint-v135/comparison")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose():
    raw = json.loads((RUN/"predictions.json").read_text())
    seal = json.loads((RUN/"PREDICTIONS_SEALED.json").read_text())
    scores = json.loads((RUN/"scores.json").read_text())
    assert digest(RUN/"predictions.json") == seal["sha256"]
    assert len(raw["rows"]) == len(scores) == seal["rows"] == 8
    rows = []
    names = ("hard", "joint", "joint_camera_proposal", "cached_camera_proposal", "cached_framed_direct")
    for row, score in zip(raw["rows"], scores, strict=True):
        assert row["case"] == score["case"]
        if not row["case"]["positive"]:
            continue
        methods = {}
        for name in names:
            method, values = row["result"]["methods"][name], score["methods"][name]
            methods[name] = dict(returned_matches=len(method["reference_xy"]),
                native_one_pixel_matches=values["correspondence"]["1"]["correct"],
                native_two_pixel_matches=values["correspondence"]["2"]["correct"],
                correct=values["correct"], pose_returned=values["pose_returned"],
                pose_error=values["pose_error"], estimator=method.get("estimator"),
                rejection_reasons=dict(Counter(item["reason"] for item in method.get("failures", []))))
        rows.append(dict(reference_index=row["case"]["reference_index"],
                         component=row["case"]["component"], methods=methods,
                         camera_fit=row["result"]["fit"]))
    assert len(rows) == 4
    return dict(schema="saved_only_joint_pilot_diagnostic_v135", post_hoc=True,
        predictions_sha256=seal["sha256"], scores_sha256=digest(RUN/"scores.json"),
        new_matcher_inference=False, new_pose_fitting=False, threshold_changes=False,
        independent_landmark_validation=False, rows=rows,
        interpretation="The soft camera is a proposal. The primary result uses unchanged AP3P and includes insufficient-match failures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        json.dump(diagnose(), handle, indent=2, allow_nan=False)
        handle.write("\n")
