"""Keep historical module names and mutable import paths out of the public process."""
import json
import os
import subprocess
import sys

from .paths import repository_root, research_root


def run_backend(action: str, **parameters) -> dict:
    if action not in {"localize", "smoke", "replay"}:
        raise ValueError("Unsupported research operation.")
    env = dict(os.environ, PYTHONPATH=str(repository_root() / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               VECLIB_MAXIMUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-m", "lunar_matching._worker"],
                            input=json.dumps({"action": action, **parameters}),
                            text=True, capture_output=True, cwd=research_root(), env=env)
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "The isolated research operation failed.")
    return json.loads(result.stdout)
