"""Independent metric and historical-control replay after diagnostic serialization repair."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = spec_from_file_location("native_v133_replay_for_v134", HERE.parent/"solar-local-v133/replay_saved.py")
old = module_from_spec(spec); spec.loader.exec_module(old)
old.saved.HOME = Path("reports/solar-local-v134")
old.saved.RUN = Path("runs/solar-local-v134/comparison")
saved = old.saved


def summarize():
    result = old.summarize()
    result["schema"] = "finite_rejected_diagnostics_native_results_v134"
    result["v133_invalid_attempt_preserved"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved.metric.put(args.output, summarize())
