"""Independently replay the v117 saved candidate-bank, detector-cell and pose arithmetic."""
import argparse
from hashlib import file_digest
import json
import math
from pathlib import Path, PurePosixPath
import re
from zipfile import ZIP_STORED, ZipFile

import numpy as np


ARMS = ("geometry_3d", "planar", "gradient")
CASES = ("u061-l00-17-positive", "u013-l00-00-positive", "u025-l00-00-positive", "u027-l08-01-positive")
TAGS = tuple(f"exposure-{index}" for index in range(4))
SCALES = (32, 16)
SEEDS = (0, 1)
BANK_COUNT = 2048
OBSERVATION_FLOOR = 1e-9
ROTATION = 10.0
TRANSLATION = 0.10
TOP = 8
SOURCE = "reports/solar-raw-crossview-protocol-v110.json"
SOURCE_SHA = "478d90ff7f50ff69d1f030c4875b3a7378b186bf61e4a1471c0bdf7990b5b932"
MANIFEST_SHA = "750919f111d69cba13b1560409955f7af2c4369ca6cc526b93bc01579993c3e2"
GRID = [(tag, r, q, size) for tag in TAGS for r in SEEDS for q in SEEDS for size in SCALES]
HEX = re.compile(r"[a-f0-9]{64}")


def need(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pinned_json(path, expected, limit):
    path = Path(path)
    need(HEX.fullmatch(expected) and path.stat().st_size <= limit and sha(path) == expected,
         "Expected frozen JSON digest and byte bound: " + path.name)
    return json.loads(path.read_bytes())


def relative(name):
    rel = PurePosixPath(name)
    need(rel.parts and str(rel) == name and not rel.is_absolute() and ".." not in rel.parts
         and "\\" not in name and all(ord(c) >= 32 for c in name), "Unexpected relative path")
    return rel


def asset_record(record):
    need(set(record) == {"path", "sha256", "bytes"} and HEX.fullmatch(record["sha256"])
         and type(record["bytes"]) is int and 0 < record["bytes"] <= 32 * 1024**2,
         "Invalid authorized asset identity")
    return relative(record["path"])


def verify_source(root, record):
    path = root / asset_record(record)
    need(path.is_file() and path.resolve() == path and path.stat().st_size == record["bytes"]
         and sha(path) == record["sha256"], "Original geometry asset differs: " + path.name)
    return path


def validate_roster(document, role):
    fixed = dict(source_path=SOURCE, source_sha256=SOURCE_SHA, authority_sha256=MANIFEST_SHA,
                 native_image_shape=[512, 512], split="model_selection", calibration_or_final_access=False,
                 score_after_prediction_seal=True, role=role)
    need(all(document.get(key) == value for key, value in fixed.items())
         and tuple(row["tag"] for row in document["exposures"]) == TAGS,
         "Frozen, ordered development roster required")
    return document["exposures"]


def inference_inputs(document):
    inputs = validate_roster(document, "inference")
    for item in inputs:
        need(set(item) == {"tag", "reference_geometry", "reference_radiance", "query_radiance", "query_intrinsics"},
             "Inference roster is allowed only reference geometry, radiance and query intrinsics")
        geometry = asset_record(item["reference_geometry"])
        need(geometry.parent.name == "reference" and geometry.suffix == ".npz",
             "Inference geometry must identify the reference view")
        intrinsics = np.asarray(item["query_intrinsics"])
        need(intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all()
             and min(intrinsics[0, 0], intrinsics[1, 1]) > 0. and np.array_equal(intrinsics[2], [0, 0, 1]),
             "Require only valid query camera intrinsics")
        names = []
        for view in ("reference", "query"):
            assets = item[view + "_radiance"]
            need(len(assets) == 2, "Both independent render seeds per view are required")
            records = [asset_record(entry) for entry in assets]
            need(all(view in entry.parts and entry.suffix == ".exr" for entry in records),
                 "Inference is limited to authorized scene-linear imagery")
            names.extend(records)
        need(len(set(names)) == 4 and len({p.parts[-4] for p in names}) == 1,
             "Known-light archive requires four distinct assets at one explicitly matched light")
    return inputs


def scoring_inputs(document, inputs):
    labels = validate_roster(document, "scoring_only")
    need(tuple(item["case_id"] for item in labels) == CASES, "Exact four cases and control roles required")
    purposes = ("sparse_tail_primary", "bright_control", "near_uniform_control", "near_uniform_control")
    need(tuple(item["purpose"] for item in labels) == purposes, "Unexpected study control labels")
    inference = [asset for item in inputs for asset in
                 [item["reference_geometry"], *item["reference_radiance"], *item["query_radiance"]]]
    infer_paths = {asset["path"] for asset in inference}
    infer_digests = {asset["sha256"] for asset in inference}
    for label in labels:
        need(set(label) == {"tag", "case_id", "purpose", "query_geometry", "joint_scoring_geometry", "samples_per_pixel"},
             "Unexpected scoring roster fields")
        for key in ("query_geometry", "joint_scoring_geometry"):
            record = label[key]
            path = asset_record(record)
            need(record["path"] not in infer_paths and record["sha256"] not in infer_digests and path.suffix == ".npz",
                 "Scoring geometry overlaps a prediction input")
            need(key != "query_geometry" or path.parent.name == "query", "Scoring query pose has the wrong view")
    return labels


def candidate_basename(tag, r, q, size):
    return f"cells-{tag}-r{r}-q{q}-s{size}.npz"


def read_cells(directory, row):
    tag, r, q, size = (row[key] for key in ("tag", "reference_seed", "query_seed", "cell_pixels"))
    cell = row["cell_evidence"]
    need(set(cell) == {"path", "sha256"} and cell["path"] == candidate_basename(tag, r, q, size)
         and HEX.fullmatch(cell["sha256"]), "Cell attachment is outside the frozen prediction row")
    path = directory / cell["path"]
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 16 * 1024**2
         and sha(path) == cell["sha256"], "Cell evidence hash, size, or path differs")
    names = {"query_fixed", "observed", "indices", "geometry_3d", "planar", "counts"}
    with ZipFile(path) as zip_file:
        entries = zip_file.infolist()
        need(len(entries) == len(names) and {entry.filename for entry in entries} == {name + ".npy" for name in names}
             and all(entry.compress_type == ZIP_STORED for entry in entries),
             "Cell attachment must be the bounded original uncompressed six-array format")
    with np.load(path, allow_pickle=False) as archive:
        need(set(archive.files) == names and len(archive.files) == len(names), "Unexpected cell attachment arrays")
        return {name: archive[name] for name in names}


def centered_unit(values):
    """Remove offset, estimate RMS, and normalize each candidate independently."""
    array = np.asarray(values, float)
    offset = array[..., :1]
    stabilized = array - offset
    centered = stabilized - stabilized.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=-1))
    minimum = np.maximum(OBSERVATION_FLOOR, 64 * np.finfo(float).eps * np.max(np.abs(array), axis=-1))
    usable = rms > minimum
    return centered / np.expand_dims(np.where(usable, rms, 1.), axis=-1), usable


def trend_projection(terrain, size):
    """Independent ordinary least squares on an equally weighted Cartesian grid."""
    fixed_side = 512 // size // 2
    y, x = np.indices((fixed_side, fixed_side))
    x = x.ravel() - (fixed_side - 1) / 2
    y = y.ravel() - (fixed_side - 1) / 2
    # Constant, x and y are orthogonal here. Derive the coefficients without a pseudoinverse.
    centred = terrain - terrain[..., :1]
    slopes_x = np.sum(centred * x, axis=-1, keepdims=True) / np.sum(x * x)
    slopes_y = np.sum(centred * y, axis=-1, keepdims=True) / np.sum(y * y)
    return terrain[..., :1] + centred.mean(axis=-1, keepdims=True) + slopes_x * x + slopes_y * y


def photometric_loss(observed, template, observation_usable):
    template, usable = centered_unit(template)
    gain = np.mean(observed * template, axis=-1)
    residual = np.abs(observed - gain[:, None] * template)
    quadratic = np.minimum(residual, 1.)
    penalty = 0.5 * quadratic**2 + residual - quadratic
    losses = np.mean(penalty, axis=-1)
    return [float(loss) if passed else None for passed, loss in zip(usable & observation_usable & (gain > 0), losses)]


def reconstruct_row(row, raw):
    size, candidate_count = row["cell_pixels"], row["candidate_count"]
    fixed_count = (512 // size // 2)**2
    query, indices = raw["query_fixed"], raw["indices"]
    count = len(indices) if indices.ndim == 1 else -1
    maximum = (size // 4)**2
    minimum = math.ceil(0.125 * maximum)
    need(type(candidate_count) is int and 1 <= candidate_count <= BANK_COUNT
         and row["fixed_cells"] == fixed_count and type(row["common_supported_candidates"]) is int
         and row["common_supported_candidates"] == count and indices.dtype.kind in ("i", "u")
         and 0 <= count <= candidate_count and np.all(indices < candidate_count) and np.all(indices >= 0)
         and np.all(indices[1:] > indices[:-1]) and query.shape == (fixed_count,)
         and query.dtype == np.dtype("float64") and not np.isinf(query).any()
         and not np.any(query < 0), "Candidate order, fixed cells, or shared support counts differ")
    counts = raw["counts"]
    need(counts.dtype == np.dtype("int16") and counts.shape == (count, fixed_count)
         and np.all(counts >= minimum) and np.all(counts <= maximum), "Shared detector-cell occupancy is outside the frozen bound")
    for name in ("observed", "geometry_3d", "planar"):
        values = raw[name]
        need(values.dtype == np.dtype("float64") and values.shape == (count, fixed_count)
             and np.isfinite(values).all() and not np.any(values < 0), "Invalid raw common-support cell means: " + name)
    independent = {name: [None] * candidate_count for name in ARMS}
    if not np.isfinite(query).all():
        reason = "invalid_observation_on_fixed_cells"
        need(count == 0, "Invalid whole-cell observations cannot admit candidates")
    elif not centered_unit(query)[1]:
        reason = "uniform_observation_on_fixed_cells"
        need(count == 0, "Uniform whole-cell observations cannot admit candidates")
    elif count == 0:
        reason = "insufficient_common_candidate_support"
    else:
        observed, usable = centered_unit(raw["observed"])
        templates = (raw["geometry_3d"], raw["planar"], trend_projection(raw["geometry_3d"], size))
        for name, template in zip(ARMS, templates, strict=True):
            for index, loss in zip(indices, photometric_loss(observed, template, usable), strict=True):
                independent[name][index] = loss
        reason = None if any(value is not None for value in independent["geometry_3d"]) else "no_informative_3d_template"
    need(row["reason"] == reason and set(row["scores"]) == set(ARMS) and set(row["orders"]) == set(ARMS),
         "Worker reason or control-arm keys disagree with independent replay")
    orders, distance = {}, 0.
    for name in ARMS:
        saved, expected = row["scores"][name], independent[name]
        need(len(saved) == candidate_count, "Saved score vector dropped candidate cells")
        for found, wanted in zip(saved, expected, strict=True):
            if found is None or wanted is None:
                need(found is wanted, "Independent control loss mask differs for " + name)
            else:
                need(type(found) in (int, float) and np.isfinite(found) and found >= 0.,
                     "Invalid saved loss for " + name)
                distance = max(distance, abs(found - wanted))
                need(abs(found - wanted) <= 2e-12, "Independent control loss differs for " + name)
        saved_order = sorted((index for index, value in enumerate(saved) if value is not None),
                             key=lambda index: (saved[index], index))
        need(saved_order == row["orders"][name], "Archived control order does not sort its complete scores")
        orders[name] = sorted(saved_order, key=lambda index: (expected[index], index))
        # Distinct implementations can round mathematically tied scores differently.
        # A consequential ordering change is checked against the sealed truth below.
        for first, second in zip(saved_order, saved_order[1:]):
            need(expected[first] <= expected[second] + 2e-12, "Independent ranking contradicts saved ordering")
    return {"orders": orders, "support_indices": indices.tolist(), "loss_max_absolute_difference": distance}


def proper_poses(bank):
    values = np.asarray(bank)
    need(values.ndim == 3 and values.shape[1:] == (4, 4) and np.isfinite(values).all()
         and np.allclose(values[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0)
         and np.allclose(np.swapaxes(values[:, :3, :3], 1, 2) @ values[:, :3, :3], np.eye(3), atol=1e-5, rtol=0)
         and np.allclose(np.linalg.det(values[:, :3, :3]), 1, atol=1e-5, rtol=0),
         "Camera bank requires finite proper camera-to-world matrices")
    return values


def radical_inverse(index, base):
    result, denominator = 0., base
    while index:
        index, residue = divmod(index, base)
        result += residue / denominator
        denominator *= base
    return result


def look_basis(centre, target):
    forward = target - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(forward, right), forward))


def reference_bank(reference):
    """Recreate the frozen Halton bank without importing any inference helpers."""
    centre, forward = reference[:3, 3], reference[:3, 2]
    cosine = math.cos(math.radians(52.))
    need(centre[2] > 0 and forward[2] <= -cosine, "Frozen datum does not support this reference camera")
    target = centre - (centre[2] / forward[2]) * forward
    base = look_basis(centre, target)
    roll = math.atan2(np.dot(reference[:3, 0], base[:, 1]), np.dot(reference[:3, 0], base[:, 0]))
    poses = [reference]
    spans = np.array([500., 500., 500., 250., 250., 55.])
    for index in range(1, BANK_COUNT):
        halton = np.array([radical_inverse(index, base) for base in (2, 3, 5, 7, 11, 13)])
        delta = (2 * halton - 1.) * spans
        location, look = centre + delta[:3], target + [delta[3], delta[4], 0.]
        if location[2] <= 0 or location[2] < cosine * np.linalg.norm(location - look):
            continue
        angle = roll + math.radians(delta[5])
        cosine_roll, sine_roll = math.cos(angle), math.sin(angle)
        offset = np.array([[cosine_roll, -sine_roll, 0.], [sine_roll, cosine_roll, 0.], [0., 0., 1.]])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = look_basis(location, look) @ offset, location
        poses.append(pose)
    return np.asarray(poses)


def read_bank(directory, predictions):
    need(predictions["calibration_or_final_access"] is False and set(predictions["banks"]) == set(TAGS),
         "Missing complete primary development bank")
    rows = predictions["rows"]
    identity_fields = ("tag", "reference_seed", "query_seed", "cell_pixels")
    need([tuple(row[key] for key in identity_fields) for row in rows] == GRID, "Missing, duplicate, or reordered cell")
    banks = {tag: proper_poses(predictions["banks"][tag]) for tag in TAGS}
    row_evidence, evidence_bytes = [], 0
    for row in rows:
        need(row["candidate_count"] == len(banks[row["tag"]]), "Ranker used a different candidate bank")
        row_evidence.append(reconstruct_row(row, read_cells(directory, row)))
        evidence_bytes += (directory / row["cell_evidence"]["path"]).stat().st_size
    need(type(predictions["evidence_bytes"]) is int and predictions["evidence_bytes"] == evidence_bytes
         and 0 < evidence_bytes <= 512 * 1024**2, "Complete original evidence byte accounting differs")
    return banks, row_evidence


def native_reference(path):
    with np.load(path, allow_pickle=False) as raw:
        pose, valid, world = raw["cam2world"], raw["valid"], raw["position"]
    pose = proper_poses(pose[None])[0]
    need(valid.shape == (512, 512) and valid.dtype.kind == "b"
         and world.shape == (512, 512, 3) and np.any(valid), "Native reference positions are invalid")
    delta = world[valid] - pose[:3, 3]
    radius = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
    need(np.isfinite(radius).all() and np.all(radius > 0), "Reference range must be finite and positive")
    return pose, float(np.median(radius))


def basin_mask(bank, target, distance):
    proper_poses(target[None])
    # atan2 of skew and trace gives the same geodesic angle without relying on arccos.
    relative_rotation = np.swapaxes(bank[:, :3, :3], 1, 2) @ target[:3, :3]
    skew = np.column_stack([relative_rotation[:, k, j] - relative_rotation[:, j, k]
                            for j, k in ((1, 2), (2, 0), (0, 1))])
    angles = np.degrees(np.arctan2(np.sqrt(np.sum(skew * skew, axis=1)),
                                   np.trace(relative_rotation, axis1=1, axis2=2) - 1.))
    xyz = bank[:, :3, 3] - target[:3, 3]
    translation = np.hypot(np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2])
    return (angles <= ROTATION) & (translation <= distance * TRANSLATION)


def prepare_basin(root, inputs, labels, banks):
    basin = {}
    for inputs_row, labels_row in zip(inputs, labels, strict=True):
        tag = inputs_row["tag"]
        reference, distance = native_reference(verify_source(root, inputs_row["reference_geometry"]))
        expected = reference_bank(reference)
        actual = banks[tag]
        need(actual.shape == expected.shape and np.allclose(actual, expected, atol=2e-9, rtol=0),
             "Saved camera bank differs from independently reconstructed reference-only bank: " + tag)
        path = verify_source(root, labels_row["query_geometry"])
        with np.load(path, allow_pickle=False) as truth:
            target = truth["cam2world"]
        basin[tag] = (labels_row["case_id"], basin_mask(actual, target, distance), distance)
    return basin


def top_hit(indices, basin):
    return next((rank for rank, index in enumerate(indices[:TOP], 1) if basin[index]), None)


def comparison_rows(predictions, rows, basin):
    expected, diagnostics = [], []
    for saved, result in zip(predictions["rows"], rows, strict=True):
        tag = saved["tag"]
        case_id, close, distance = basin[tag]
        hits = {arm: top_hit(saved["orders"][arm], close) for arm in ARMS}
        independent_hits = {arm: top_hit(result["orders"][arm], close) for arm in ARMS}
        need(all((hits[arm] is None) == (independent_hits[arm] is None) for arm in ARMS),
             "Numerically ambiguous score order changes top-eight basin membership")
        supported = result["support_indices"]
        support_count = int(np.count_nonzero(close[supported]))
        expected.append(dict(case_id=case_id, reference_seed=saved["reference_seed"], query_seed=saved["query_seed"],
                             cell_pixels=saved["cell_pixels"], bank_contains_coarse_basin=bool(np.any(close)),
                             bank_coarse_basin_count=int(np.count_nonzero(close)), top_eight_first_hit=hits,
                             common_supported_candidates=len(supported), supported_coarse_basin_count=support_count,
                             prior_first_top_eight_hit=top_hit(supported, close), reason=saved["reason"]))
        diagnostics.append(dict(tag=tag, support_contains_coarse_basin=bool(support_count),
                                independent_top_eight_first_hit=independent_hits,
                                median_reference_range_m=distance,
                                loss_max_absolute_difference=result["loss_max_absolute_difference"]))
    return expected, diagnostics


def stage_gate(rows):
    outcomes = {}
    for size in SCALES:
        by_case = {}
        for identifier in CASES:
            matching = [row for row in rows if row["cell_pixels"] == size and row["case_id"] == identifier]
            need([(r["reference_seed"], r["query_seed"]) for r in matching] == [(r, q) for r in SEEDS for q in SEEDS],
                 "Stage gate requires every ordered renderer seed combination")
            arms = {arm: sum(row["top_eight_first_hit"][arm] is not None for row in matching) for arm in ARMS}
            by_case[identifier] = dict(
                pairings=len(matching), bank_available=sum(row["bank_contains_coarse_basin"] for row in matching),
                support_available=sum(row["support_contains_coarse_basin"] for row in matching),
                reference_order_top_eight=sum(row["prior_first_top_eight_hit"] is not None for row in matching),
                top_eight_counts=arms,
            )
        outcomes[str(size)] = by_case

    primary = outcomes["32"]
    tail, bright = primary[CASES[0]], primary[CASES[1]]
    enough_bank = tail["bank_available"] == bright["bank_available"] == 4
    enough_support = tail["support_available"] == bright["support_available"] == 4
    tail_repeatable = tail["top_eight_counts"]["geometry_3d"] == 4
    bright_control = bright["top_eight_counts"]["geometry_3d"] == 4
    advantage = all(tail["top_eight_counts"]["geometry_3d"] > tail["top_eight_counts"][name]
                    for name in ("planar", "gradient"))
    checks = dict(available_coarse_bank=enough_bank, usable_common_support=enough_support,
                  all_tail_seed_pairs=tail_repeatable, bright_control_all_seed_pairs=bright_control,
                  beats_both_photometric_controls=advantage)
    return dict(primary_cell_pixels=32, sensitivity_cell_pixels=16, passed=all(checks.values()),
                checks=checks, outcomes=outcomes,
                caveat="Primary criterion uses raw top-eight candidate basin recall, known light, and exposed development cases.")


def verify(root, inputs_path, inputs_sha, labels_path, labels_sha, predictions_path, predictions_sha, scores_path, scores_sha):
    inputs = inference_inputs(pinned_json(inputs_path, inputs_sha, 256 * 1024))
    # Consume the full sealed candidate and cell evidence before opening any scoring roster or geometry.
    predictions = pinned_json(predictions_path, predictions_sha, 32 * 1024**2)
    banks, recomputed = read_bank(predictions_path.parent, predictions)
    labels = scoring_inputs(pinned_json(labels_path, labels_sha, 256 * 1024), inputs)
    scores = pinned_json(scores_path, scores_sha, 512 * 1024)
    need(scores["prediction_sha256"] == predictions_sha and scores["threshold_rotation_deg"] == ROTATION
         and scores["threshold_range_fraction"] == TRANSLATION and scores["accepted_results"] is False,
         "Scorer must bind the sealed predictions and original raw-only basin threshold")
    basin = prepare_basin(root, inputs, labels, banks)
    original, extra = comparison_rows(predictions, recomputed, basin)
    # Match the frozen worker's fields exactly. Any new separately audited
    # fields are allowed only when they agree with this independent replay.
    need(len(scores["rows"]) == len(original), "Scorer omitted an eligible cell")
    rows = []
    for archived, expected, diagnostic in zip(scores["rows"], original, extra, strict=True):
        actual = expected | diagnostic
        need(all(archived.get(key) == value for key, value in expected.items())
             and all(key in actual and actual[key] == value for key, value in archived.items()),
             "Independent geometric comparison differs from the worker's scoring output")
        rows.append(actual)
    return dict(
        schema="solar_archived_warp_candidate_independent_replay_v117", passed=True,
        prediction_sha256=predictions_sha, inputs_sha256=inputs_sha, labels_sha256=labels_sha, scores_sha256=scores_sha,
        frozen_cell_sha256={row["cell_evidence"]["path"]: row["cell_evidence"]["sha256"] for row in predictions["rows"]},
        verification_source_sha256=sha(__file__), replayed_rows=len(rows),
        frozen_cell_evidence_bytes=predictions["evidence_bytes"],
        acceptance_or_final_access=False, gate=stage_gate(rows), rows=rows,
        verification_scope="Independent score/control/rank, reference-only bank and post-seal camera-label arithmetic. "
                           "The supplied evidence verifies shared sub-bin counts and cell means as stored. "
                           "It does not independently rerun the original image projection, occlusion, or intersection. "
                           "The scope covers named prediction/label artifacts and the frozen interface, without syscall-level isolation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for name in ("inputs", "labels", "predictions", "scores"):
        parser.add_argument("--" + name, required=True, type=Path)
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh replay receipt destination")
    report = verify(args.repo.resolve(), args.inputs, args.inputs_sha256, args.labels, args.labels_sha256,
                    args.predictions, args.predictions_sha256, args.scores, args.scores_sha256)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
