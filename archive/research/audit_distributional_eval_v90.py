"""Recompute raw pose, candidate-window recall and paid acquisition from saved rows."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import verify_solar_evaluation_v70 as legacy


BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def replay(run, protocol_path, output):
    run = Path(run)
    protocol_path = Path(protocol_path)
    output = Path(output)
    summary_path = run / "summary.json"
    summary = read(summary_path)
    protocol = read(protocol_path)
    marker = read(run / "COMPLETE.json")
    watchdog_path = run / "watchdog.json"
    watchdog = read(watchdog_path)
    if (not marker["external_watchdog_passed"] or not watchdog["complete"]
            or marker["protocol_sha256"] != sha(protocol_path)
            or marker["watchdog_sha256"] != sha(watchdog_path)
            or marker["summary_sha256"] != sha(summary_path)):
        raise ValueError("Original completed controller receipts differ")
    if protocol["metadata_sha256"][BASIS] != sha(BASIS):
        raise ValueError("Original published case basis differs")
    basis = read(BASIS)
    manifest_path = protocol.get("manifest", basis["manifest"])
    if manifest_path != basis["manifest"]:
        raise ValueError("Independent replay requires the original v86 case manifest")
    manifest_digest = sha(manifest_path)
    input_hashes = protocol.get("local_input_sha256", protocol.get("input_sha256", {}))
    if input_hashes.get(manifest_path) != manifest_digest:
        raise ValueError("Original frozen case manifest differs")
    # The older arithmetic verifier expects these two schema fields.
    # Adapt only in memory after checking their pinned source bytes. It still
    # reads and checks every original row, array, source label and seal.
    adapted = {
        protocol_path.resolve(): dict(protocol, manifest=manifest_path),
        summary_path.resolve(): dict(summary, manifest_sha256=manifest_digest),
    }

    def schema_view(path):
        resolved = Path(path).resolve()
        return adapted[resolved] if resolved in adapted else read(path)

    with patch.object(legacy, "read", schema_view):
        arithmetic = legacy.replay(run, protocol_path, output.with_suffix(".pose-replay.json"))
    profiles = {profile["name"]: profile for profile in protocol["profiles"]}
    results = []
    for row in summary["rows"]:
        array_path = Path(row["saved_arrays"])
        row_path = array_path.with_suffix(".json")
        seal = read(array_path.with_suffix(".COMPLETE.json"))
        if (not seal["complete"] or seal["arrays_path"] != str(array_path)
                or seal["arrays_sha256"] != sha(array_path)
                or seal["row_path"] != str(row_path) or seal["row_sha256"] != sha(row_path)
                or row != read(row_path)):
            raise ValueError("Case-level saved evidence differs")
        record = {key: row[key] for key in ("profile", "case_id", "positive", "light_a", "light_b")}
        record["correct"] = row["metrics"]["correct"]
        record["wrong_positive"] = row["metrics"]["wrong_positive"]
        record["raw_nonoverlap"] = row["metrics"]["raw_nonoverlap"]
        record["inference_seconds"] = row["inference_seconds"]
        trace = row["raw_inference_result"]
        if trace is not None:
            record["coarse_response"] = trace.get("coarse_response")
        if row["positive"]:
            with np.load(array_path, allow_pickle=False) as a:
                if "coarse_reference_xy" in a.files:
                    p = a["coarse_reference_xy"]
                    q = a["coarse_query_candidates"]
                    rotations = a["coarse_rotation_indices"]
                    truth = a["scoring_coarse_query_xy"]
                    visible = a["scoring_coarse_visible"].astype(bool)
                    if truth.shape != (len(p), 2) or visible.shape != (len(p),):
                        raise ValueError("Coordinate-joined coarse query labels missing")
                    valid = visible & np.isfinite(truth).all(axis=1)
                    radial = np.full(q.shape[:2], np.inf)
                    radial[valid] = np.linalg.norm(q[valid] - truth[valid, None], axis=-1)
                    record["retained_anchors"] = len(p)
                    record["co_visible_anchors"] = int(valid.sum())
                    record["recalled_at_px"] = {str(t): int((radial.min(axis=1) <= t).sum()) for t in (1, 2, 4, 8, 12, 16)}
                    # v90 fixed quarter-turn bank. Its integer-aligned native
                    # 33×33 search is centred on round(rotated q), with +/-16
                    # per coordinate. This exact window differs from radius16.
                    if profiles[row["profile"]]["configuration"]["proposal_angles"] != [0., 90., 180., 270.]:
                        raise ValueError("Window arithmetic only implemented for the declared quarter-turn bank")

                    def bank_xy(points, turns):
                        x, y = points[..., 0], points[..., 1]
                        return np.stack((
                            np.choose(turns, (x, y, 511 - x, 511 - y)),
                            np.choose(turns, (y, 511 - x, 511 - y, x)),
                        ), axis=-1)

                    bank_centres = np.rint(bank_xy(q, rotations))
                    truth_rotated = bank_xy(np.broadcast_to(truth[:, None], q.shape), rotations)
                    in_window = valid[:, None] & (np.abs(truth_rotated - bank_centres) <= 16).all(axis=-1)
                    record["oracle_native_windows"] = int(in_window.any(axis=1).sum())
                    acquired = [] if trace is None else trace.get("measurements", [])
                    if not isinstance(acquired, list):
                        raise ValueError("Native paid acquisitions must retain explicit ordered records")
                    chosen = [(item["anchor"], item["candidate"]) for item in acquired]
                    if len(chosen) != len(set(chosen)) or any(not 0 <= i < len(p) or not 0 <= c < 8 for i, c in chosen):
                        raise ValueError("Invalid duplicate or outside acquired candidate")
                    record["paid_windows"] = len(chosen)
                    record["paid_windows_with_truth"] = sum(bool(in_window[i, c]) for i, c in chosen)
                    record["paid_anchors_with_truth"] = len({i for i, c in chosen if in_window[i, c]})
        results.append(record)
    result = dict(
        passed=arithmetic["passed"], original_summary_sha256=sha(summary_path),
        original_protocol_sha256=sha(protocol_path), replayed_rows=arithmetic["replayed_rows"],
        rows=results,
        scope="Saved arrays, coordinate-joined original truth and paid-window traces only. No matcher, rendering or inference rerun.",
    )
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replay(args.run, args.protocol, args.output)
    print("Replayed", result["replayed_rows"], "rows")
