"""Replay twenty paired methods on32 development image pairs."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("sealed_v129_summary_for_v133", HERE.parent/"solar-local-v129/replay_saved.py")
saved = module_from_spec(spec); spec.loader.exec_module(saved)
saved.HOME = Path("reports/solar-local-v133")
saved.RUN = Path("runs/solar-local-v133/comparison")
HISTORICAL = ("legacy", "halo", "pooled_uniform", "pooled_parallax", "direct",
              "shared_camera_pooled_uniform", "shared_camera_pooled_parallax")


def summarize():
    result = saved.summarize()
    old = saved.metric.read("runs/solar-local-v131/comparison/predictions.json")
    new = saved.metric.read(saved.RUN/"predictions.json")
    for before, after in zip(old["rows"], new["rows"], strict=True):
        saved.metric.check(before["case"] == after["case"], "Changed paired case")
        for name in HISTORICAL:
            for key in ("reference_xy", "query_xy", "pose", "estimator"):
                saved.metric.check(before["result"]["methods"][name][key] == after["result"]["methods"][name][key], "Historical control changed")
        for name in after["result"]["methods"]:
            if name.startswith("framed_"):
                original = after["result"]["methods"][name.removeprefix("framed_")]
                saved.metric.check(all(after["result"]["methods"][name][k] == original[k] for k in ("reference_xy", "query_xy")),
                                   "Rigid frame control changed image measurements")
    direct = result["methods"]["framed_direct"]
    strongest = max(result["methods"][prefix+name]["correct"] for prefix in ("", "framed_") for name in HISTORICAL)
    for name, row in result["methods"].items():
        row["pilot_improvement_rule_passed"] = (name.startswith("framed_native_")
            and row["correct"] >= strongest+2 and row["wrong_positive"] <= direct["wrong_positive"]
            and row["raw_nonoverlap"] <= direct["raw_nonoverlap"])
    result.update(schema="reference_frame_native_radiance_results_v133", historical_seven_controls_exact=True,
                  coordinate_controls_preserve_all_image_endpoints=True,
                  strongest_historical_control_correct=strongest,
                  principal_method="framed_native_terrain_subpixel", principal_baseline="framed_direct")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__); cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = summarize()
    if args.output:
        saved.metric.put(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
