"""Private process boundary for the byte-preserved experimental backend."""
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import subprocess
import sys

import cv2

from .paths import research_root


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(request):
    root = research_root()
    cv2.setNumThreads(1)
    sys.path.insert(0, str(root))
    home = root / "reports/solar-beam-v143"
    if request["action"] == "replay":
        result = subprocess.run([sys.executable, str(root / "reports/solar-beam-v144/replay_saved.py")],
                                cwd=root, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)
    sys.path.insert(0, str(home))
    if request["action"] == "smoke":
        result = load("frozen_beam_smoke", home / "smoke_beam_v143.py").control()
        return {key: value for key, value in result.items()
                if key not in {"raw_research", "scoring_only_camera"}}
    if request["action"] != "localize":
        raise ValueError("Unsupported worker operation.")
    module = load("frozen_beam_command", home / "localize_beam.py")
    catalogue, query, depth, k, pose, qk, valid = module.load_case(request["case"])
    return module.run(catalogue, query, depth, k, pose, qk, reference_valid=valid,
                      height_direction_reference=pose[:3, :3].T @ module.np.array([0., 0., 1.]),
                      policies=tuple(request["policies"]), field_budget=request["field_budget"],
                      research_output=request["research_output"])


def main():
    request = json.load(sys.stdin)
    # Scientific logging must not corrupt the machine-readable response.
    with redirect_stdout(io.StringIO()):
        result = execute(request)
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
