"""Independently replay native radiance results and seven historical controls."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("sealed_v129_summary_for_v132", HERE.parent/"solar-local-v129/replay_saved.py")
saved = module_from_spec(spec); spec.loader.exec_module(saved)
saved.HOME = Path("reports/solar-local-v132")
saved.RUN = Path("runs/solar-local-v132/comparison")


def summarize():
    result = saved.summarize()
    old = saved.metric.read("runs/solar-local-v131/comparison/predictions.json")
    new = saved.metric.read(saved.RUN/"predictions.json")
    for before, after in zip(old["rows"], new["rows"], strict=True):
        saved.metric.check(before["case"] == after["case"], "Changed paired case")
        for name in before["result"]["methods"]:
            for key in ("reference_xy", "query_xy", "pose", "estimator"):
                saved.metric.check(before["result"]["methods"][name][key] == after["result"]["methods"][name][key], "Historical control changed")
    direct = result["methods"]["direct"]
    for name, row in result["methods"].items():
        if name.startswith("native_"):
            row["pilot_improvement_rule_passed"] = (row["correct"] >= direct["correct"]+2
                and row["wrong_positive"] <= direct["wrong_positive"]
                and row["raw_nonoverlap"] <= direct["raw_nonoverlap"])
    result.update(schema="native_radiance_subpixel_terrain_results_v132", historical_seven_controls_exact=True)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__); cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = summarize()
    if args.output:
        saved.metric.put(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
