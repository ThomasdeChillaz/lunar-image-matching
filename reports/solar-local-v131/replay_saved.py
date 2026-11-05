"""Replay all seven methods and require exact repetition of the five frozen controls."""

import argparse
from importlib.util import module_from_spec,spec_from_file_location
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=spec_from_file_location("frozen_sealed_summary",HERE.parent/"solar-local-v129/replay_saved.py")
saved=module_from_spec(spec);spec.loader.exec_module(saved)
saved.HOME=Path("reports/solar-local-v131")
saved.RUN=Path("runs/solar-local-v131/comparison")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=saved.summarize()
    old=saved.metric.read("runs/solar-local-v130/comparison/predictions.json")
    new=saved.metric.read(saved.RUN/"predictions.json")
    for before,after in zip(old["rows"],new["rows"],strict=True):
        saved.metric.check(before["case"]==after["case"],"Changed paired case")
        for name in ("legacy","halo","pooled_uniform","pooled_parallax","direct"):
            for key in ("reference_xy","query_xy","pose","estimator"):
                saved.metric.check(before["result"]["methods"][name][key]==after["result"]["methods"][name][key],"Frozen control prediction changed")
    direct=result["methods"]["direct"]
    for name,row in result["methods"].items():
        if name.startswith("shared_camera_"):
            row["pilot_improvement_rule_passed"]=(row["correct"]>=direct["correct"]+2
                and row["wrong_positive"]<=direct["wrong_positive"] and row["raw_nonoverlap"]<=direct["raw_nonoverlap"])
    result["schema"]="shared_camera_photometric_consensus_results_v131"
    result["frozen_control_predictions_exact"]=True
    if args.output:saved.metric.put(args.output,result)
    else:print(json.dumps(result,indent=2,allow_nan=False))
