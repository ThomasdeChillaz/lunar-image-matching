"""Explicit repository paths; the historical workspace has its own working root."""
from pathlib import Path


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "archive/research/reports/solar-beam-v143/protocol.json").is_file():
        raise RuntimeError("Use an editable installation from the complete research repository.")
    return root


def research_root() -> Path:
    return repository_root() / "archive/research"
