"""Read and cross-check measured results; never infer scientific success from tests."""
import math
from pathlib import Path

from .artifacts import read_json, sha256
from .paths import repository_root, research_root

METHODS = ("direct", "legacy_no_extra", "legacy_active32", "single_uniform128",
           "beam_uniform128", "beam_geometry128", "beam_active128")
LABELS = ("Direct", "Single, no extra", "Previous active +32", "Single, full field cap",
          "Beam, uniform", "Beam, geometry", "Beam, active")


def load_evidence(path: Path | None = None) -> dict:
    source = path or research_root() / "reports/solar-beam-v144/results.json"
    if path is None:
        expected = read_json(repository_root() / "docs/frozen-files.json")["files"][
            "reports/solar-beam-v144/results.json"]["sha256"]
        if sha256(source) != expected:
            raise ValueError("The measured result differs from the frozen archive.")
    result = read_json(source)
    if (result.get("schema") != "sealed_persistent_beam_summary_v144"
            or result.get("accepted_pose_study") is not False
            or tuple(result.get("methods", {})) != METHODS):
        raise ValueError("A complete frozen development result is required.")
    cases = result["case_results"]
    if (len(cases) != 8 or len({row["id"] for row in cases}) != 8
            or any(type(row.get("positive")) is not bool for row in cases)
            or sum(row["positive"] for row in cases) != 4):
        raise ValueError("All four positives and four nonoverlaps must remain in the report.")
    for name in METHODS:
        record = result["methods"][name]
        for metric in ("correct", "strict", "wrong_positive", "raw_nonoverlap"):
            flags = [row["methods"][name][metric] for row in cases]
            if any(type(flag) is not bool for flag in flags) or sum(flags) != record[metric]:
                raise ValueError(f"Saved case counts disagree: {name}/{metric}")
        if (record["positive_pairs"] != 4 or record["nonoverlap_pairs"] != 4
                or not math.isfinite(record["median_positive_online_seconds"])
                or record["median_positive_online_seconds"] <= 0):
            raise ValueError("Fixed denominators and finite measured latency are required.")
    return result


def summary(result: dict) -> dict:
    return {"evidence": "exposed development, raw poses; not calibrated acceptance",
            "positive_pairs": 4, "nonoverlap_pairs": 4,
            "positive_terrain_components": len(result["positive_terrain_components"]),
            "new_trainable_parameters": 0,
            "methods": {name: {key: result["methods"][name][key] for key in
                ("correct", "strict", "wrong_positive", "raw_nonoverlap", "median_positive_online_seconds")}
                for name in METHODS},
            "conclusion": "No recovery advantage; no candidate promoted. Final terrain remains unused."}


def format_table(result: dict) -> str:
    lines = ["Measured pilot: four positive pairs, four nonoverlaps, three exposed terrains.",
             "Raw poses only; acceptance is uncalibrated.", "",
             f"{'Method':<27} {'Correct':>7} {'Wrong +':>8} {'Nonoverlap':>11} {'Seconds':>8}"]
    for name, label in zip(METHODS, LABELS, strict=True):
        row = result["methods"][name]
        lines.append(f"{label:<27} {row['correct']:>5}/4 {row['wrong_positive']:>8} "
                     f"{row['raw_nonoverlap']:>9}/4 {row['median_positive_online_seconds']:>8.2f}")
    lines += ["", "All methods recover the same positive. No superiority claim."]
    return "\n".join(lines)
