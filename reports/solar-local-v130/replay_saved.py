"""Use unchanged v129 saved-score arithmetic for the prospectively admitted run."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("frozen_summary_v129", HERE.parent / "solar-local-v129/replay_saved.py")
saved = module_from_spec(spec); spec.loader.exec_module(saved)
saved.HOME = Path("reports/solar-local-v130")
saved.RUN = Path("runs/solar-local-v130/comparison")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    saved.metric.check(Path.cwd() == saved.ROOT, "Replay from repository root")
    result = saved.summarize()
    result["schema"] = "sealed_native_measurement_repair_results_v130"
    if args.output:
        saved.metric.put(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
