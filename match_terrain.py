"""Compatibility entry point; the historical implementation remains byte-preserved."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lunar_matching.compat import legacy_main

if __name__ == "__main__":
    raise SystemExit(legacy_main("match_terrain.py"))
