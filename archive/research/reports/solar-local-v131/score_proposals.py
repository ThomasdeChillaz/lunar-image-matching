"""Post-hoc scoring of frozen photometric proposals, separately from measured-match poses."""

from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=spec_from_file_location("frozen_proposal_metric",HERE.parent/"solar-local-v127/experiment.py")
metric=module_from_spec(spec);spec.loader.exec_module(metric)


def audit():
    home=Path("reports/solar-local-v131");run=Path("runs/solar-local-v131/comparison")
    p=metric.read(home/"protocol.json");raw=metric.read(run/"predictions.json")
    complete=metric.read(run/"COMPLETE.json");seal=metric.read(run/"PREDICTIONS_SEALED.json")
    metric.check(complete["complete"] and seal["sha256"]==metric.sha(run/"predictions.json")
                 and [r["case"] for r in raw["rows"]]==p["cases"] and not raw["query_truth_used"],"Completed sealed predictions required")
    counts=dict(first_correct=0,first_strict=0,wrong_positive_proposals=0,nonoverlap_proposals=0,
                any_correct_hindsight=0,positives=16,negatives=16)
    rows=[]
    for row in raw["rows"]:
        hypotheses=row["result"]["photometric_initialization"]
        # The existing frozen scorer supplies identical pose arithmetic. Empty
        # coordinate lists do not pretend projected positions were observations.
        probe=dict(methods={str(i):dict(reference_xy=[],query_xy=[],pose=h["pose"]) for i,h in enumerate(hypotheses)})
        scores=metric.score_result(probe,row["case"],p["inputs"],p["labels"])
        first=scores.get("0",{})
        counts["first_correct"]+=bool(first.get("correct"))
        counts["first_strict"]+=bool(first.get("strict"))
        counts["wrong_positive_proposals"]+=bool(first.get("wrong_positive"))
        counts["nonoverlap_proposals"]+=bool(first.get("raw_nonoverlap"))
        counts["any_correct_hindsight"]+=any(s["correct"] for s in scores.values())
        rows.append(dict(case=row["case"],hypotheses=len(hypotheses),first=first))
    return dict(posthoc=True,new_inference=False,accepted_pose_result=False,
                scope="Diagnostic camera proposals before image-correspondence verification; primary seven-method results remain unchanged.",
                predictions_sha256=seal["sha256"],counts=counts,rows=rows)


if __name__=="__main__":print(json.dumps(audit(),indent=2,allow_nan=False))
