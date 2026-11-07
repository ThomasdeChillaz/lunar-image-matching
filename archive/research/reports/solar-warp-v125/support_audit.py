"""Compare coarse pose ranking with a detector-support control after replay."""

import argparse
import json
from pathlib import Path

import numpy as np

import verify_warp_archive_v117 as frozen


def support_order(indices, counts):
    """Prefer coverage in the least supported cell, then total coverage."""
    if not len(indices):
        return []
    order = np.lexsort((indices, -counts.sum(axis=1), -counts.min(axis=1)))
    return indices[order].tolist()


def saved_support(directory, prediction):
    """Freeze the control's rankings before opening the scoring inputs."""
    rows = prediction["rows"]
    keys = ("tag", "reference_seed", "query_seed", "cell_pixels")
    if [tuple(row[key] for key in keys) for row in rows] != frozen.GRID:
        raise ValueError("Require the complete ordered inference grid")
    ranks = []
    for row in rows:
        data = frozen.read_cells(directory, row)
        indices, counts = data["indices"], data["counts"]
        if (indices.ndim != 1 or indices.dtype.kind not in "iu"
                or len(indices) != row["common_supported_candidates"]
                or np.any(indices < 0) or np.any(indices >= row["candidate_count"])
                or np.any(indices[1:] <= indices[:-1])
                or counts.dtype != np.int16 or counts.shape != (len(indices), row["fixed_cells"])
                or counts.shape[1] == 0 or np.any(counts <= 0)):
            raise ValueError("Unexpected common detector support")
        arm_indices = [set(row["orders"][arm]) for arm in frozen.ARMS]
        jointly_scored = len(set.intersection(*arm_indices))
        ranks.append((support_order(indices, counts), jointly_scored))
    return ranks


def attribution_rows(prediction, rankings, basin):
    rows = []
    for saved, (ranked, jointly_scored) in zip(prediction["rows"], rankings, strict=True):
        case, near, _ = basin[saved["tag"]]
        rows.append({
            "case_id": case, "reference_seed": saved["reference_seed"], "query_seed": saved["query_seed"],
            "cell_pixels": saved["cell_pixels"], "common_supported_candidates": len(ranked),
            "all_photometric_arms_scoreable_candidates": jointly_scored,
            "support_only_first_hit": frozen.top_hit(ranked, near),
            "reference_order_first_hit": frozen.top_hit(sorted(ranked), near),
            "geometry_3d_first_hit": frozen.top_hit(saved["orders"]["geometry_3d"], near),
            "planar_first_hit": frozen.top_hit(saved["orders"]["planar"], near),
            "gradient_first_hit": frozen.top_hit(saved["orders"]["gradient"], near),
        })
    return rows


def audit(root, directory, prediction_sha, replay_sha, inputs_path, labels_path):
    prediction = frozen.pinned_json(directory / "predictions.json", prediction_sha, 32 * 1024**2)
    rankings = saved_support(directory, prediction)
    replay = frozen.pinned_json(directory / "independent-replay.json", replay_sha, 512 * 1024)
    expected_cells = {row["cell_evidence"]["path"]: row["cell_evidence"]["sha256"]
                      for row in prediction["rows"]}
    if (replay.get("schema") != "solar_archived_warp_candidate_independent_replay_v117"
            or replay.get("passed") is not True or replay.get("replayed_rows") != len(frozen.GRID)
            or replay.get("verification_source_sha256") != frozen.sha(frozen.__file__)
            or replay.get("prediction_sha256") != prediction_sha
            or replay.get("frozen_cell_sha256") != expected_cells
            or replay.get("acceptance_or_final_access") is not False):
        raise ValueError("Require the exact independent replay of the same sealed prediction and cells")
    inputs = frozen.inference_inputs(frozen.pinned_json(inputs_path, replay["inputs_sha256"], 256 * 1024))
    labels = frozen.scoring_inputs(frozen.pinned_json(labels_path, replay["labels_sha256"], 256 * 1024), inputs)
    banks = {tag: frozen.proper_poses(prediction["banks"][tag]) for tag in frozen.TAGS}
    rows = attribution_rows(prediction, rankings, frozen.prepare_basin(root, inputs, labels, banks))
    for row, original in zip(rows, replay["rows"], strict=True):
        if (row["case_id"] != original["case_id"]
                or row["geometry_3d_first_hit"] != original["top_eight_first_hit"]["geometry_3d"]
                or row["reference_order_first_hit"] != original["prior_first_top_eight_hit"]):
            raise ValueError("Coarse hit and reference-order check disagree with the original replay")
    return {
        "schema": "solar_warp_support_only_diagnostic_v125", "prediction_sha256": prediction_sha,
        "independent_replay_sha256": replay_sha, "source_sha256": frozen.sha(__file__),
        "rows": rows, "calibration_or_final_access": False,
        "scope": "Top-eight coarse-basin recall on sealed known-light development evidence. "
                 "The support control uses eligible detector counts without radiance scores. "
                 "This diagnostic supplements the original gate and performs no image inference.",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("--predictions-sha256", required=True)
    cli.add_argument("--replay-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    roster = Path(__file__).resolve().parents[1] / "solar-warp-v117"
    report = audit(Path.cwd().resolve(), args.archive, args.predictions_sha256, args.replay_sha256,
                   roster / "inference-inputs.json", roster / "scoring-inputs.json")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
