"""Post hoc proposal diagnosis using sealed outputs and scoring-only query truth.

No inference, alternative candidate search, model selection or policy changes.
"""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np

from analyze_saved import ROOT, RUN, OUT, digest, read


def pose_error(camera, truth, reference_range):
    camera, truth = np.asarray(camera), np.asarray(truth)
    relative = camera[:3, :3].T @ truth[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
    shift = float(np.linalg.norm(camera[:3, 3]-truth[:3, 3]))
    return dict(rotation_deg=angle, translation_m=shift,
                correct=angle <= 2 and shift <= .01 * reference_range)


def main():
    raw = read(RUN/"predictions.json")
    if read(RUN/"PREDICTIONS_SEALED.json") != dict(rows=8, sha256=digest(RUN/"predictions.json")):
        raise ValueError("Complete sealed predictions required before post hoc diagnosis")
    protocol = read(ROOT/"reports/solar-blind-v140/protocol.json")
    inputs, labels = read(ROOT/protocol["inference_manifest"]), read(ROOT/protocol["scoring_manifest"])
    path = ROOT/"reports/solar-active-v137/replay_saved.py"
    spec = spec_from_file_location("diagnostic_range_geometry_v142", path)
    geometry = module_from_spec(spec); spec.loader.exec_module(geometry)
    rows = []
    for record, inp, label in zip(raw["rows"], inputs, labels, strict=True):
        if not record["id"] == inp["id"] == label["id"]:
            raise ValueError("Case order differs")
        if not label["positive"]:
            continue
        with np.load(ROOT/inp["reference_geometry"]["path"], allow_pickle=False) as a:
            distance = geometry.reference_range(a["axial_depth"], a["intrinsics"])
        with np.load(ROOT/label["query_geometry"]["path"], allow_pickle=False) as b:
            truth = b["cam2world"].copy()
        suns = np.array([c["sun_world"] for c in inp["catalogue"]])
        target = np.array(label["query_sun_scoring_only"])
        sun_errors = np.degrees(np.arctan2(np.linalg.norm(np.cross(suns, target), axis=1), suns @ target))
        nearest = int(sun_errors.argmin())
        initialization = record["result"]["initialization"]
        per_light = [min((v for v in row if v is not None), default=float("inf"))
                     for row in initialization["scores"]]
        light_order = sorted(range(len(per_light)), key=lambda i: (per_light[i], i))
        hypotheses = [dict(light_id=h["light_id"],
            sun_error_deg=float(sun_errors[h["light_slot"]]),
            **pose_error(h["pose"], truth, distance))
            for h in initialization["photometric_hypotheses"]]
        rows.append(dict(id=record["id"], source_case=label["source_case"],
            nearest_true_catalogue_light_id=inp["catalogue"][nearest]["light_id"],
            nearest_true_catalogue_error_deg=float(sun_errors[nearest]),
            nearest_light_rank_by_min_coarse_loss=light_order.index(nearest)+1,
            retained_coarse_light_ids=sorted({s["light_id"] for s in initialization["seeds"]}),
            photometric_hypotheses=hypotheses,
            any_correct_retained_camera=any(h["correct"] for h in hypotheses),
            translation_limit_m=.01*distance))
    output = dict(schema="posthoc_sealed_proposal_diagnosis_v142", post_hoc=True,
        query_truth_scoring_only=True, inference_rerun=False, metric_or_policy_changes=False,
        initial_correct_pose_cases=sum(r["any_correct_retained_camera"] for r in rows),
        eligible_positive_cases=len(rows), cases=rows,
        claim_limit="Describes saved proposal retention; an absent good proposal does not establish whether untested alternatives would succeed")
    (OUT/"proposal-diagnosis.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({r["id"]: {k: r[k] for k in ("nearest_light_rank_by_min_coarse_loss",
        "retained_coarse_light_ids", "any_correct_retained_camera")} for r in rows}, indent=2))


if __name__ == "__main__":
    main()
