"""Post hoc diagnosis of saved branches; query truth is scoring-only.

No search, inference, policy adjustment, calibration or model selection occurs.
An oracle branch statistic describes a missed opportunity, never a measured output.
"""
import json
import numpy as np
from analyze_saved import ROOT, RUN, OUT, METHODS, digest, load_module, read, require


def pose_error(camera, truth, reference_range):
    if camera is None:
        return None
    camera, truth = np.asarray(camera), np.asarray(truth)
    relative = camera[:3, :3].T @ truth[:3, :3]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
    shift = float(np.linalg.norm(camera[:3, 3]-truth[:3, 3]))
    return dict(rotation_deg=angle, translation_m=shift,
                correct=angle <= 2 and shift <= .01 * reference_range)


def diagnose():
    require(read(RUN / "PREDICTIONS_SEALED.json") == dict(rows=8, sha256=digest(RUN / "predictions.json"))
            and read(RUN / "replay.json")["passed"] is True,
            "Complete sealed and replayed outputs required before post hoc diagnosis")
    protocol = read(ROOT / "reports/solar-beam-v143/protocol.json")
    inputs, labels = read(ROOT / protocol["inference_manifest"]), read(ROOT / protocol["scoring_manifest"])
    geometry = load_module("diagnostic_geometry_v144", "reports/solar-active-v137/replay_saved.py")
    rows = []
    for record, inp, label in zip(read(RUN / "predictions.json")["rows"], inputs, labels, strict=True):
        require(record["id"] == inp["id"] == label["id"], "Case order differs")
        if not label["positive"]:
            continue
        with np.load(ROOT / inp["reference_geometry"]["path"], allow_pickle=False) as a:
            da, ka, ca = (a[name].copy() for name in ("axial_depth", "intrinsics", "cam2world"))
        with np.load(ROOT / label["query_geometry"]["path"], allow_pickle=False) as b:
            db, kb, truth = (b[name].copy() for name in ("axial_depth", "intrinsics", "cam2world"))
        distance = geometry.reference_range(da, ka)
        fields = da, db, ka, ca, kb, truth
        suns = np.asarray([c["sun_world"] for c in inp["catalogue"]])
        target = np.asarray(label["query_sun_scoring_only"])
        errors = np.degrees(np.arctan2(np.linalg.norm(np.cross(suns, target), axis=1), suns @ target))
        nearest = int(errors.argmin())
        initialization = record["result"]["initialization"]
        per_light = [min((v for v in row if v is not None), default=float("inf"))
                     for row in initialization["coarse_scores"]]
        light_order = sorted(range(len(per_light)), key=lambda i: (per_light[i], i))
        hypotheses = [dict(branch_id=h["branch_id"], light_id=h["light_id"],
            sun_error_deg=float(errors[h["light_slot"]]), **pose_error(h["pose"], truth, distance))
            for h in initialization["photometric_hypotheses"]]
        policies = {}
        for name in METHODS[3:]:
            method = record["result"]["methods"][name]
            candidates = []
            for candidate in method.get("branch_candidates", []):
                candidates.append(dict(branch_id=candidate["branch_id"], light_id=candidate["light_id"],
                    selected=candidate["branch_id"] == method["branch_id"],
                    attempted_fields=candidate["attempted_fields"],
                    soft_camera_error=pose_error(candidate["soft_camera_proposal"], truth, distance),
                    final_metrics=geometry.arithmetic(candidate, True, fields)))
            trace = []
            for step in method["acquisition"].get("trace", []):
                branch_errors = [dict(branch_id=b["branch_id"], observation_present=b["observation_present"],
                    **pose_error(b["retained_camera"], truth, distance)) for b in step["branches"]]
                trace.append(dict(index=step["index"], branch_errors=branch_errors,
                                  any_correct_soft_camera=any(b["correct"] for b in branch_errors)))
            policies[name] = dict(candidates=candidates, acquisition_trace=trace,
                any_correct_final_branch=any(c["final_metrics"]["correct"] for c in candidates),
                any_correct_soft_camera_during_acquisition=any(t["any_correct_soft_camera"] for t in trace),
                selected_correct=geometry.arithmetic(method, True, fields)["correct"])
        rows.append(dict(id=record["id"], source_case=label["source_case"],
            nearest_true_catalogue_light_id=inp["catalogue"][nearest]["light_id"],
            nearest_light_rank_by_min_coarse_loss=light_order.index(nearest)+1,
            retained_coarse_light_ids=sorted({s["light_id"] for s in initialization["seeds"]}),
            unavailable_seeds=len(initialization["unavailable_before_acquisition"]),
            photometric_hypotheses=hypotheses,
            any_correct_retained_camera=any(h["correct"] for h in hypotheses),
            translation_limit_m=.01 * distance, policies=policies))
    return dict(schema="posthoc_persistent_beam_diagnosis_v144", post_hoc=True,
        query_truth_scoring_only=True, inference_rerun=False, metric_or_policy_changes=False,
        initial_correct_pose_cases=sum(r["any_correct_retained_camera"] for r in rows),
        eligible_positive_cases=len(rows), cases=rows,
        claim_limit="Oracle retained-branch results are diagnostic only; the selected output determines recovery")


if __name__ == "__main__":
    result = diagnose()
    (OUT / "branch-diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({r["id"]: dict(initial_correct=r["any_correct_retained_camera"],
        policies={name: {key: row[key] for key in ("any_correct_final_branch", "selected_correct")}
                  for name, row in r["policies"].items()}) for r in result["cases"]}, indent=2))
