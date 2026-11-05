"""Numerically compare exact saved v115 outputs across platforms. No inference."""
import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from zipfile import ZipFile

import numpy as np


RUN_ID = 41941904269
ARTIFACT_ID = 13386000111
OUTER_SHA = "813f2c9e6c5986ae711ab885d3a6c25a146cb87fead8261f99984e689f5e538b"
OUTER_BYTES = 13164199
NONCE = "f4185e74104c0a62d3b14d64e61cc6b3ef622feec4b28c3191ceba871855dd11"
REMOTE = f"runs/solar-remote-five-arm-v115/{NONCE}"
OUTPUT = "runs/solar-five-arm-v115/comparison"
PROTOCOL = "reports/solar-five-arm-v115/comparison-protocol.json"
COHORT = "reports/solar_matched_arm_cohorts_v103.json"
EXACT_FAILURE = "ValueError: Saved labels differ from independently reopened source labels"
FROZEN_FILES = {
    "github_v115_metadata.json": "2d3bbe4e9cdc63b7f49fe80390c6410a2c6aad1810ce0228b958bec425235e46",
    "nested_cell_inventory_v115.json": "6e14ec6770a1699fa26438464a73cde5721d0143a7ba348077d205f5d9c95a30",
    PROTOCOL: "c43b7a0a5236b27055a7179caf1cfa1a274b18743a9fd868d0054fa96d23878c",
    f"{OUTPUT}/summary.json": "3f65b164472e2300d6b5aeb1611a2988d0f5d0155fe850c6a5f229eb567a38bc",
    f"{OUTPUT}/COMPLETE.json": "dd108e49fa87be8ac49308a378405c263ede1cc5ed9d41a4b952823a0c0c2c50",
    f"{REMOTE}/remote_watchdog.json": "ccdb2ae888b870d2cfd856819a056c85b503250875ec166b95073562e9e96430",
    f"{REMOTE}/scientific_results.zip": "3b08451d43580c598bb9e6176f7fa96e53cb09714eb4e9fbdc0595ca34f010f2",
}
LOG_SHA = "1fd4bedff2aae2a4291453079dcd65cae81be5033838ed491aad93bd03f2c84f"
THRESHOLDS = (1, 2, 4, 8, 12)
ABS_COORD_TOLERANCE = 1e-9
COORD_ULPS = 4


def need(condition, why):
    if not condition:
        raise ValueError(why)


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def file(root, name):
    rel = PurePosixPath(name)
    need(rel.parts and not rel.is_absolute() and str(rel) == name and ".." not in rel.parts,
         "Invalid frozen relative path")
    path = root / rel
    need(path.is_file() and path.resolve() == path, "Expected regular original audit file: " + name)
    return path


def check_zip(archive, names):
    with ZipFile(archive) as zip_file:
        for name in names:
            need(zip_file.namelist().count(name) == 1, "Ambiguous original member: " + name)
            with zip_file.open(name) as stream:
                need(file_digest(stream, "sha256").hexdigest() == FROZEN_FILES[name], "Original member differs: " + name)


def verify_scope(root, outer, original_log):
    need(outer.stat().st_size == OUTER_BYTES and sha(outer) == OUTER_SHA, "Original official outer ZIP differs")
    need(sha(original_log) == LOG_SHA and EXACT_FAILURE in original_log.read_text(),
         "Preserve the original exact replay failure")
    for name, digest in FROZEN_FILES.items():
        need(sha(file(root, name)) == digest, "Original audit input hash differs: " + name)
    meta = read(root / "github_v115_metadata.json")
    artifact = meta["artifacts"]["outer"]
    need(meta["run_id"] == RUN_ID and meta["conclusion"] == "success" and artifact["id"] == ARTIFACT_ID
         and artifact["size_in_bytes"] == OUTER_BYTES and artifact["digest"] == "sha256:" + OUTER_SHA,
         "Official run and artifact disagree")
    watch = read(root / REMOTE / "remote_watchdog.json")
    inventory = read(root / "nested_cell_inventory_v115.json")
    protocol = read(root / PROTOCOL)
    summary = read(root / OUTPUT / "summary.json")
    completion = read(root / OUTPUT / "COMPLETE.json")
    need(watch["complete"] is True and watch["child_returncode"] == 0 and watch["workflow_run_id"] == RUN_ID
         and watch["run_attempt"] == 1 and watch["result_artifact"]["sha256"] == FROZEN_FILES[f"{REMOTE}/scientific_results.zip"]
         and completion["complete"] is True and completion["summary_sha256"] == FROZEN_FILES[f"{OUTPUT}/summary.json"]
         and inventory["expected_cells"] == inventory["sealed_cells"] == len(summary["rows"]) == 480
         and inventory["invalid_markers"] == [] and inventory["protocol_sha256"] == FROZEN_FILES[PROTOCOL]
         and summary["calibration_or_final_access"] is False and summary["accepted_results"] is False,
         "Original signed archive, completion and 480 cell seals disagree")
    check_zip(outer, (f"{REMOTE}/remote_watchdog.json", f"{REMOTE}/scientific_results.zip"))
    check_zip(root / REMOTE / "scientific_results.zip", (PROTOCOL, f"{OUTPUT}/summary.json", f"{OUTPUT}/COMPLETE.json"))
    for name, digest in protocol["source_sha256"].items():
        need(sha(file(root / "frozen_source", name)) == digest, "Frozen source changed: " + name)
    for name in (COHORT, protocol["manifest"]):
        need(sha(file(root, name)) == protocol["input_sha256"][name], "Original roster changed: " + name)
    manifest = read(root / protocol["manifest"])
    cases = {row["case_id"]: row for row in manifest["cases"]}
    checked = set()
    for case_id in protocol["comparison_case_ids"]:
        for name, digest in cases[case_id]["input_sha256"].items():
            need(digest == protocol["input_sha256"][name], "Frozen scoring input digests differ")
            if name not in checked:
                need(sha(file(root, name)) == digest, "Original selected input changed: " + name)
                checked.add(name)
    return protocol, summary, manifest, len(checked)


def compare_xy(stored, recomputed, row, name, changes):
    need(stored.shape == recomputed.shape and stored.dtype == recomputed.dtype == np.dtype("float64")
         and not np.isinf(stored).any() and not np.isinf(recomputed).any()
         and np.array_equal(np.isnan(stored), np.isnan(recomputed)),
         "Coordinate shape, precision, or exact NaN pattern differs for " + name)
    valid = np.isfinite(stored)
    for index, axis in zip(*np.nonzero(valid & (stored != recomputed)), strict=True):
        before, after = float(stored[index, axis]), float(recomputed[index, axis])
        distance = abs(before - after)
        ulps = distance / max(math.ulp(before), math.ulp(after))
        need(distance <= ABS_COORD_TOLERANCE and ulps <= COORD_ULPS, "Projected coordinate mismatch exceeds portability bound")
        changes.append(dict(profile=row["profile"], case_id=row["case_id"], array=name,
                            point_index=int(index), axis=int(axis), stored=before, reopened=after,
                            stored_hex=before.hex(), reopened_hex=after.hex(), abs_native_px=distance, ulps=ulps))


def threshold_error(query, xy, visible):
    errors = np.full(len(query), np.inf)
    delta = query[visible] - xy[visible]
    errors[visible] = np.hypot(delta[:, 0], delta[:, 1])
    return errors


def correspondence_recheck(query, stored, recomputed, counts, label, margins):
    a = threshold_error(query, *stored)
    b = threshold_error(query, *recomputed)
    for threshold in THRESHOLDS:
        before, after = a <= threshold, b <= threshold
        need(np.array_equal(before, after), "A native correspondence crossed the saved threshold")
        key = f"{label}_{threshold}px"
        margin = float(np.min(np.abs(a[np.isfinite(a)] - threshold), initial=np.inf))
        margins[key] = min(margins[key], margin)
        if counts is not None:
            expected = dict(correct=int(after.sum()), co_visible=int(recomputed[1].sum()), returned=len(query))
            need(counts[str(threshold)] == expected, "A native correspondence count differs")


def median_range(path):
    with np.load(path, allow_pickle=False) as z:
        depth, intrinsics = z["depth"], z["intrinsics"]
    y, x = np.indices(depth.shape)
    directions = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)]) @ np.linalg.inv(intrinsics).T
    z = depth.ravel()
    valid = np.isfinite(z) & (z > 0)
    d = directions[valid]
    return float(np.median(z[valid] * np.hypot(np.hypot(d[:, 0], d[:, 1]), d[:, 2])))


def pose_recheck(row, case, arrays, truth, distance, margins):
    pose = arrays["pose"]
    has = pose.shape == (4, 4)
    need(has or pose.shape == (0, 4), "Changed explicit no-pose sentinel")
    pos = case["positive"]
    metrics = row["metrics"]
    correct = strict = False
    if has and pos:
        target = truth["query_cam2world"]
        relative = pose[:3, :3].T @ target[:3, :3]
        skew = np.array([relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0],
                         relative[1, 0]-relative[0, 1]])
        angle = float(np.rad2deg(np.arctan2(np.linalg.norm(skew), np.trace(relative)-1.)))
        translation = float(np.linalg.norm(pose[:3, 3]-target[:3, 3]))
        correct, strict = angle <= 2. and translation <= .01*distance, angle <= .5 and translation <= .001*distance
        for limit in (.5, 2.):
            key = f"rotation_deg_{limit}"
            margins[key] = min(margins[key], abs(angle-limit))
        for limit in (.001, .01):
            key = f"translation_fraction_{limit}"
            margins[key] = min(margins[key], abs(translation/distance-limit))
    else:
        need(metrics["pose_error"] is None, "Unexpected scored error on a negative or null pose")
    actual = dict(correct=correct, strict=strict, pose_returned=has, wrong_positive=pos and has and not correct,
                  raw_nonoverlap=not pos and has)
    need(all(metrics[key] is value for key, value in actual.items()), "A positive/negative/native/strict pose bucket differs")
    return dict(profile=row["profile"], case_id=row["case_id"], track=row["track"], **actual)


def recheck(root, protocol, summary, manifest):
    sys.path.insert(0, str(root / "frozen_source"))
    from solar_aligned_scoring_v91 import load_aligned_scoring as load_scoring
    from summarize_solar_five_arm_v115 import analyze, cohort_cases
    import cv2

    counts = Counter()
    margins = defaultdict(lambda: float("inf"))
    changes, checked_rows = [], []
    ranges = {}
    cohort, groups = cohort_cases(read(root / COHORT), manifest)
    cases = {row["case_id"]: row for row in cohort}
    for row in summary["rows"]:
        case = cases[row["case_id"]]
        name = row["saved_arrays"]
        need(name.startswith(OUTPUT + "/") and name.endswith(".npz"), "Out-of-scope prediction array")
        path = file(root, name)
        need(sha(path) == row["saved_arrays_sha256"], "Original prediction array hash changed")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        truth, method = load_scoring(case, arrays["reference_xy"])
        need(method == row["label_method"], "Changed frozen scoring projection method")
        reference = case["reference_geometry"]
        if reference not in ranges:
            ranges[reference] = median_range(file(root, reference))
        distance = ranges[reference]
        difference = abs(distance - row["metrics"]["reference_range_m"])
        need(difference <= 1e-9, "Native median reference range mismatch")
        counts["exact_reference_range_rows"] += difference == 0.
        if case["positive"]:
            counts["positive_rows"] += 1
            need(np.array_equal(arrays["scoring_visible"], truth["visible"])
                 and np.array_equal(arrays["scoring_query_cam2world"], truth["query_cam2world"]),
                 "Exact original visibility or camera pose label differs")
            compare_xy(arrays["scoring_query_xy"], truth["query_xy"], row, "scoring_query_xy", changes)
            correspondence_recheck(arrays["query_xy"], (arrays["scoring_query_xy"], arrays["scoring_visible"]),
                (truth["query_xy"], truth["visible"]), row["metrics"]["correspondence"], "main", margins)
            if "coarse_reference_xy" in arrays:
                counts["coarse_rows"] += 1
                coarse, _ = load_scoring(case, arrays["coarse_reference_xy"])
                need(np.array_equal(arrays["scoring_coarse_visible"], coarse["visible"]),
                     "Exact original coarse visibility differs")
                compare_xy(arrays["scoring_coarse_query_xy"], coarse["query_xy"], row, "scoring_coarse_query_xy", changes)
                candidates = arrays["coarse_query_candidates"]
                repeat = lambda pair: tuple(np.repeat(arr, 8, axis=0) for arr in pair)
                correspondence_recheck(candidates.reshape(-1, 2),
                    repeat((arrays["scoring_coarse_query_xy"], arrays["scoring_coarse_visible"])),
                    repeat((coarse["query_xy"], coarse["visible"])), None, "coarse", margins)
        else:
            counts["negative_rows"] += 1
            need(not row["metrics"]["correspondence"], "Unexpected nonoverlap correspondence metrics")
        checked_rows.append(pose_recheck(row, case, arrays, truth, distance, margins))
    report = analyze(summary, dict(passed=True, replayed_rows=len(checked_rows), rows=checked_rows), cohort, groups)
    return dict(
        rows=len(checked_rows), exact_positive_visibility_and_camera_labels=counts["positive_rows"],
        exact_coarse_visibility_rows=counts["coarse_rows"], matched_pose_buckets=len(checked_rows),
        exact_reference_range_rows=counts["exact_reference_range_rows"], unique_reference_geometries=len(ranges),
        matched_full_correspondence_threshold_rows=counts["positive_rows"],
        matched_coarse_correspondence_threshold_rows=counts["coarse_rows"],
        checked_negative_rows=counts["negative_rows"], thresholds_native_px=list(THRESHOLDS),
        absolute_coordinate_tolerance_native_px=ABS_COORD_TOLERANCE, max_coordinate_ulps_allowed=COORD_ULPS,
        mismatching_projected_coordinate_scalars=changes,
        maximum_coordinate_difference_native_px=max((row["abs_native_px"] for row in changes), default=0.),
        minimum_observed_distance_to_threshold={key: val for key, val in margins.items() if math.isfinite(val)},
        local_software=dict(machine=platform.machine(), platform=platform.system(), numpy=np.__version__, opencv=cv2.__version__),
        original_worker=report["worker"],
        provisional_matched_comparison=report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--outer-zip", required=True, type=Path)
    parser.add_argument("--original-frozen-replay-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, destination = args.audit_dir.resolve(), args.output.resolve()
    need(not destination.exists() and not destination.is_relative_to(root), "Use a fresh independent diagnostic path")
    protocol, summary, manifest, inputs = verify_scope(root, args.outer_zip, args.original_frozen_replay_log)
    # Frozen scoring helpers resolve only the original case input paths.
    os.chdir(root)
    observed = recheck(root, protocol, summary, manifest)
    result = dict(schema="solar_v115_saved_array_portability_diagnostic_v123",
                  scope="Saved-only independent numerical cross-platform diagnostic. The bitwise frozen replay failure is preserved. "
                        "No model inference, original archive mutation, calibration, final access, exact replay certification, or admission authority.",
                  workflow_run_id=RUN_ID, github_artifact_id=ARTIFACT_ID, github_outer_sha256=OUTER_SHA,
                  github_outer_bytes=OUTER_BYTES, frozen_replay_failure_sha256=LOG_SHA,
                  frozen_exact_replay_passed=False, new_inference=False, audited_input_hash_count=inputs,
                  compared_original_file_sha256=FROZEN_FILES, implementation_sha256=sha(Path(__file__).resolve()), **observed)
    with destination.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
