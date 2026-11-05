"""Audit native local-search reachability after the known-light v117 study is sealed."""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZIP_STORED, ZipFile

import numpy as np


NATIVE = (512, 512)
PIXELS = (80, 197, 314, 431)
ANCHORS = np.array([(x, y) for y in PIXELS for x in PIXELS])
RADII = (16, 32, 64, 128)
TOP = 8
SCALES = (32, 16)
TAGS = tuple(f"exposure-{index}" for index in range(4))
CASES = ("u061-l00-17-positive", "u013-l00-00-positive", "u025-l00-00-positive", "u027-l08-01-positive")
ARMS = ("geometry_3d", "planar", "gradient")
ROW_FIELDS = ("tag", "reference_seed", "query_seed", "cell_pixels")
GRID = [(tag, a, b, scale) for tag in TAGS for a in (0, 1) for b in (0, 1) for scale in SCALES]
HEX = re.compile(r"[a-f0-9]{64}")
SOURCE = "reports/solar-raw-crossview-protocol-v110.json"
SOURCE_SHA = "478d90ff7f50ff69d1f030c4875b3a7378b186bf61e4a1471c0bdf7990b5b932"
MANIFEST_SHA = "750919f111d69cba13b1560409955f7af2c4369ca6cc526b93bc01579993c3e2"
REPLAY = "reports/solar-warp-v117/verify_warp_archive_v117.py"
REF_KEYS = ("position", "valid", "axial_depth", "cam2world", "intrinsics", "body_origin", "enu_rows")


def need(condition, error):
    if not condition:
        raise ValueError(error)


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pinned(path, digest, limit):
    need(isinstance(digest, str) and HEX.fullmatch(digest) and path.is_file()
         and not path.is_symlink() and 0 < path.stat().st_size <= limit and sha(path) == digest,
         "Original artifact differs from its pinned bytes: " + str(path))
    return path


def read(path, digest, limit=1024**2):
    return json.loads(pinned(path, digest, limit).read_bytes())


def relative(name):
    rel = PurePosixPath(name)
    need(rel.parts and not rel.is_absolute() and str(rel) == name
         and ".." not in rel.parts and "\\" not in name, "Unsafe authorized path")
    return rel


def geometry_asset(root, record):
    need(set(record) == {"path", "sha256", "bytes"} and type(record["bytes"]) is int,
         "Invalid archived geometry record")
    path = root / relative(record["path"])
    need(path.resolve() == path and path.stat().st_size == record["bytes"] and path.suffix == ".npz",
         "Unexpected original geometry path")
    return pinned(path, record["sha256"], 32 * 1024**2)


def proper_poses(value):
    bank = np.asarray(value, dtype=float)
    need(bank.ndim in (2, 3) and bank.shape[-2:] == (4, 4), "Expected camera matrix shape")
    rotations = bank[..., :3, :3]
    need(np.isfinite(bank).all()
         and np.allclose(bank[..., 3, :], [0, 0, 0, 1], atol=1e-10, rtol=0)
         and np.allclose(np.swapaxes(rotations, -2, -1) @ rotations, np.eye(3), atol=1e-5, rtol=0)
         and np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0),
         "Expected proper camera-to-world poses")
    return bank


def validate_dispatch(completion, protocol):
    expected = {
        "schema": "solar_warp_archival_gate_v117", "phase": "archive", "split": "model_selection",
        "cpu_threads": 2, "cap_seconds": 180, "calibration_or_final_access": False, "prepared_only": False,
        "primary_cell_pixels": 32, "sensitivity_cell_pixels": 16, "known_reference_illumination": True,
        "query_pose_read_after_prediction_seal": True, "native_size": list(NATIVE),
        "expected_prediction_rows": len(GRID),
    }
    need(all(protocol.get(key) == value for key, value in expected.items())
         and completion.get("phase") == "archive" and completion.get("complete") is True
         and completion.get("calibration_or_final_access") is False
         and re.fullmatch(r"[a-f0-9]{40}", completion.get("source_commit", ""))
         and completion.get("steps") == ["predict", "score", "independent-replay"],
         "Expected the original complete one-use v117 known-light archive dispatch")


def roster(document, role):
    required = dict(source_path=SOURCE, source_sha256=SOURCE_SHA, authority_sha256=MANIFEST_SHA,
                    native_image_shape=list(NATIVE), split="model_selection", calibration_or_final_access=False,
                    score_after_prediction_seal=True, role=role)
    exposures = document["exposures"]
    need(all(document.get(key) == value for key, value in required.items())
         and tuple(item["tag"] for item in exposures) == TAGS, "Invalid v117 roster authority")
    return exposures


def rosters(inputs, labels):
    reference, truth = roster(inputs, "inference"), roster(labels, "scoring_only")
    need(tuple(item["case_id"] for item in truth) == CASES, "Original development case order changed")
    names = {"tag", "reference_geometry", "reference_radiance", "query_radiance", "query_intrinsics"}
    need(all(set(item) == names and PurePosixPath(item["reference_geometry"]["path"]).parent.name == "reference"
             for item in reference), "Query truth appears in inference roster")
    inference_assets = [asset for item in reference for asset in
                        [item["reference_geometry"], *item["reference_radiance"], *item["query_radiance"]]]
    paths = {item["path"] for item in inference_assets}
    hashes = {item["sha256"] for item in inference_assets}
    expected = {"tag", "case_id", "purpose", "query_geometry", "joint_scoring_geometry", "samples_per_pixel"}
    for item in truth:
        need(set(item) == expected and PurePosixPath(item["query_geometry"]["path"]).parent.name == "query",
             "Invalid scoring-only view identity")
        for key in ("query_geometry", "joint_scoring_geometry"):
            asset = item[key]
            need(asset["path"] not in paths and asset["sha256"] not in hashes,
                 "Scoring geometry overlaps the inference roster")
    return reference, truth


def evidence_indices(archive, row, expected, bank_size):
    tag, a, b, scale = (row[key] for key in ROW_FIELDS)
    name = f"cells-{tag}-r{a}-q{b}-s{scale}.npz"
    binding = row["cell_evidence"]
    need(binding == {"path": name, "sha256": expected[name]}, "Case attachment does not bind the replay")
    path = pinned(archive / name, expected[name], 16 * 1024**2)
    arrays = ("indices", "query_fixed", "observed", "geometry_3d", "planar", "counts")
    with ZipFile(path) as packed:
        entries = packed.infolist()
        need(len(entries) == len(arrays) and {item.filename for item in entries} == {n + ".npy" for n in arrays}
             and all(item.compress_type == ZIP_STORED for item in entries), "Changed v117 evidence array format")
    with np.load(path, allow_pickle=False) as evidence:
        indices = evidence["indices"]
    need(indices.ndim == 1 and indices.dtype.kind in "iu" and np.all(indices >= 0)
         and np.all(indices < bank_size) and np.all(indices[1:] > indices[:-1])
         and len(indices) == row["common_supported_candidates"], "Invalid original common-support list")
    return indices.tolist(), path.stat().st_size


def check_order(row, support, bank_size):
    need(set(row["scores"]) == set(row["orders"]) == set(ARMS), "Changed archive comparison arms")
    support = set(support)
    for arm in ARMS:
        scores = row["scores"][arm]
        need(len(scores) == bank_size, "Candidate score vector is incomplete")
        scored = []
        for index, score in enumerate(scores):
            if score is not None:
                need(index in support and type(score) in (int, float) and np.isfinite(score)
                     and score >= 0, "Invalid scored candidate")
                scored.append(index)
        need(row["orders"][arm] == sorted(scored, key=lambda index: (scores[index], index)),
             "Changed sealed proposal order")


def sealed_artifacts(archive, protocol_path, inputs_path, labels_path, completion_sha):
    completion = read(archive / "COMPLETE.json", completion_sha)
    need(not (archive / "FAILED.json").exists(), "Preserve partial runs without opening scoring geometry")
    protocol = read(protocol_path, completion["protocol_sha256"])
    validate_dispatch(completion, protocol)
    psha, ssha, rsha = (completion[name] for name in
                         ("predictions_sha256", "scores_sha256", "independent_replay_sha256"))
    predictions = read(archive / "predictions.json", psha, 32 * 1024**2)
    scores = read(archive / "scores.json", ssha)
    replay = read(archive / "independent-replay.json", rsha)
    sizes = predictions["evidence_bytes"]
    seal_path = archive / "PREDICTIONS_SEALED.json"
    need(not seal_path.is_symlink() and 0 < seal_path.stat().st_size <= 4096
         and json.loads(seal_path.read_bytes()) == {"rows": len(GRID), "sha256": psha, "evidence_bytes": sizes},
         "Original predictions must be sealed before opening scoring geometry")
    need(replay["schema"] == "solar_archived_warp_candidate_independent_replay_v117"
         and replay["passed"] is True and replay["replayed_rows"] == len(GRID)
         and replay["prediction_sha256"] == psha and replay["scores_sha256"] == ssha
         and replay["verification_source_sha256"] == protocol["source_sha256"][REPLAY]
         and replay["inputs_sha256"] == protocol["inputs"]["sha256"]
         and replay["labels_sha256"] == protocol["labels"]["sha256"]
         and replay["frozen_cell_evidence_bytes"] == sizes
         and replay["acceptance_or_final_access"] is False
         and replay["gate"] == completion["advancement_gate"]
         and scores["prediction_sha256"] == psha and scores["accepted_results"] is False
         and predictions["calibration_or_final_access"] is False,
         "Original independent saved-cell replay must pass before capture audit")
    rows = predictions["rows"]
    need([tuple(row[key] for key in ROW_FIELDS) for row in rows] == GRID
         and set(predictions["banks"]) == set(TAGS) and len(replay["rows"]) == len(scores["rows"]) == len(GRID),
         "Require all sealed cases, seeds and detector scales")
    expected = replay["frozen_cell_sha256"]
    need(len(expected) == len(GRID) and 0 < sizes <= 512 * 1024**2,
         "Incorrect archive sidecar count or size")
    banks = {tag: proper_poses(value) for tag, value in predictions["banks"].items()}
    evidence, total = [], 0
    for row, historical, verified in zip(rows, scores["rows"], replay["rows"], strict=True):
        tag, a, b, scale = (row[key] for key in ROW_FIELDS)
        bank = banks[tag]
        need(bank.ndim == 3 and 1 <= len(bank) <= 2048 and row["candidate_count"] == len(bank)
             and (historical["case_id"], historical["reference_seed"], historical["query_seed"], historical["cell_pixels"])
             == (CASES[TAGS.index(tag)], a, b, scale)
             and verified["tag"] == tag and all(verified[key] == value for key, value in historical.items()),
             "Saved score row, original candidate bank or replay identity changed")
        supported, size = evidence_indices(archive, row, expected, len(bank))
        check_order(row, supported, len(bank))
        evidence.append(supported)
        total += size
    need(total == sizes, "Full sealed cell storage differs")
    inputs = read(inputs_path, protocol["inputs"]["sha256"])
    labels = read(labels_path, protocol["labels"]["sha256"])
    # Geometry may open only after all provenance and cell checks have passed.
    reference, truth = rosters(inputs, labels)
    return completion, protocol, rows, banks, evidence, reference, truth


def project(points, intrinsics, pose):
    camera_xyz = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels_h = camera_xyz @ intrinsics.T
    depth = camera_xyz[:, 2]
    valid = np.isfinite(camera_xyz).all(axis=1) & (depth > 0.)
    result = np.full((len(points), 2), np.nan)
    result[valid] = pixels_h[valid, :2] / pixels_h[valid, 2, None]
    return result, depth


def in_bounds(pixels, shape, interior=False):
    maximum = np.asarray(shape[::-1]) - 1
    finite = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1)
    return finite & ((pixels < maximum) if interior else (pixels <= maximum)).all(axis=1)


def native_frame(geometry, anchors, shape=NATIVE):
    position, depth, valid = (geometry[name] for name in ("position", "axial_depth", "valid"))
    intrinsics, pose = geometry["intrinsics"], proper_poses(geometry["cam2world"])
    need(position.shape == (*shape, 3) and depth.shape == valid.shape == shape and valid.dtype.kind == "b"
         and intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all()
         and min(intrinsics[0, 0], intrinsics[1, 1]) > 0.
         and np.array_equal(intrinsics[2], [0., 0., 1.]), "Invalid original native camera or geometry arrays")
    ix, iy = anchors.T
    world = position[iy, ix]
    geometry_valid = valid[iy, ix]
    pixel, z = project(world, intrinsics, pose)
    need(np.isfinite(pixel[geometry_valid]).all() and (z[geometry_valid] > 0.).all()
         and np.allclose(pixel[geometry_valid], anchors[geometry_valid], atol=1e-3, rtol=0)
         and np.allclose(z[geometry_valid], depth[iy, ix][geometry_valid], atol=1e-5, rtol=0),
         "Original depth/position arrays do not reproject to the same native anchor centres")
    return world, geometry_valid


def open_geometry(root, inference, label):
    paths = [geometry_asset(root, record) for record in
             (inference["reference_geometry"], label["query_geometry"], label["joint_scoring_geometry"])]
    reference, query = {}, {}
    for path, result in zip(paths[:2], (reference, query), strict=True):
        with np.load(path, allow_pickle=False) as data:
            result.update({key: data[key] for key in REF_KEYS})
    need(np.array_equal(inference["query_intrinsics"], query["intrinsics"])
         and np.array_equal(reference["body_origin"], query["body_origin"])
         and np.array_equal(reference["enu_rows"], query["enu_rows"])
         and reference["body_origin"].shape == (3,) and reference["enu_rows"].shape == (3, 3)
         and np.isfinite(reference["body_origin"]).all() and np.isfinite(reference["enu_rows"]).all(),
         "Reference and scoring view do not share original calibration and ENU frame")
    with np.load(paths[2], allow_pickle=False) as joint:
        for role, geometry in (("reference", reference), ("query", query)):
            for original, common in (("axial_depth", "depth"), ("cam2world", "cam2world"), ("intrinsics", "intrinsics")):
                need(np.array_equal(geometry[original], joint[f"{role}_{common}"], equal_nan=True),
                     "Independent joint geometry changes an original view array")
    world, valid = native_frame(reference, ANCHORS)
    native_frame(query, ANCHORS)
    return world, valid, query


def query_visibility(world, reference_valid, query, shape=NATIVE):
    points, target_z = project(world, query["intrinsics"], query["cam2world"])
    reason = np.full(len(world), "invalid_reference", dtype=object)
    valid_forward = reference_valid & (target_z > 0) & np.isfinite(points).all(axis=1)
    reason[reference_valid & ~valid_forward] = "query_behind_or_nonfinite"
    inside = valid_forward & in_bounds(points, shape, interior=True)
    reason[valid_forward & ~inside] = "query_outside_bilinear_domain"
    for index in np.flatnonzero(inside):
        x, y = points[index]
        x0, y0 = int(x), int(y)
        alpha, beta = x - x0, y - y0
        stencil = np.s_[y0:y0 + 2, x0:x0 + 2]
        depth = query["axial_depth"][stencil]
        if not query["valid"][stencil].all() or not np.isfinite(depth).all() or (depth <= 0.).any():
            reason[index] = "query_depth_missing"
            continue
        tolerance = .05 + .01 * target_z[index]
        difference = depth - target_z[index]
        if difference.max() < -tolerance:
            reason[index] = "query_foreground_occludes"
        elif difference.min() > tolerance:
            reason[index] = "query_depth_farther_than_reference_point"
        elif np.ptp(depth) > tolerance:
            reason[index] = "query_depth_discontinuity_ambiguous"
        else:
            interpolated = (1. - beta) * ((1. - alpha) * depth[0, 0] + alpha * depth[0, 1]) \
                + beta * ((1. - alpha) * depth[1, 0] + alpha * depth[1, 1])
            reason[index] = ("co_visible_depth_consistent" if abs(interpolated - target_z[index]) <= tolerance
                             else "query_depth_disagreement")
    return points, reason


def quadrants(points, mask, shape=NATIVE):
    selected = points[mask]
    sides = selected >= np.asarray(shape[::-1]) / 2
    index = 2 * sides[:, 1] + sides[:, 0]
    return np.bincount(index, minlength=4).tolist()


def by_region(anchor_points, query_points, mask, co_visible, shape=NATIVE):
    count = int(mask.sum())
    ref_regions = quadrants(anchor_points, mask, shape)
    query_regions = quadrants(query_points, mask, shape)
    return dict(count=count, of_fixed_anchors=len(anchor_points), of_co_visible_anchors=co_visible,
                fraction_of_fixed=count / len(anchor_points),
                fraction_of_co_visible=count / co_visible if co_visible else None,
                reference_quadrant_counts=ref_regions, true_query_quadrant_counts=query_regions,
                distributed_six=(count >= 6 and sum(n > 0 for n in ref_regions) >= 3
                                 and sum(n > 0 for n in query_regions) >= 3))


def finite_xy(value):
    return value.tolist() if np.isfinite(value).all() else None


def proposal_capture(anchors, world, target, truth_reasons, query_intrinsics, pose, shape=NATIVE):
    centres, predicted_depth = project(world, query_intrinsics, pose)
    co_visible = truth_reasons == "co_visible_depth_consistent"
    front = np.isfinite(centres).all(axis=1) & (predicted_depth > 0.)
    eligible_center = front & in_bounds(centres, shape)
    usable = co_visible & eligible_center
    reason = np.full(len(anchors), "true_query_visibility_unavailable", dtype=object)
    reason[co_visible & ~front] = "proposal_behind_or_nonfinite"
    reason[co_visible & front & ~eligible_center] = "proposal_center_outside_native_image"
    reason[usable] = "proposal_center_inside_native_image"
    rounded = np.rint(centres)
    distance = np.linalg.norm(centres - target, axis=1)
    rounded_axis_error = np.max(np.abs(rounded - target), axis=1)
    domain = np.array(shape[::-1]) - 1
    metrics, safe = {}, {}
    for radius in RADII:
        euclidean = usable & (distance <= radius)
        reachable = usable & (rounded_axis_error <= radius)
        support = (rounded - radius >= 0).all(axis=1) & (rounded + radius <= domain).all(axis=1)
        safe[radius] = reachable & support
        summary = lambda mask: by_region(anchors, target, mask, int(co_visible.sum()), shape)
        metrics[str(radius)] = dict(euclidean_unrounded_center=summary(euclidean),
                                    rounded_center_square=summary(reachable),
                                    rounded_center_square_full_window=summary(safe[radius]),
                                    geometrically_reachable_but_window_clipped=int((reachable & ~support).sum()),
                                    usable_center_but_truth_outside_square=int((usable & ~reachable).sum()))
    known = np.flatnonzero(usable)
    quantiles = np.quantile(distance[known], [0, .5, .9, 1]).tolist() if len(known) else None
    points = [
        dict(anchor_index=i, projection_xy_native=finite_xy(centre), projection_state=reason[i],
             euclidean_residual_native_px=float(distance[i]) if usable[i] else None)
        for i, centre in enumerate(centres)
    ]
    return dict(projection_states=dict(Counter(reason)), residual_quantiles_0_50_90_100_px=quantiles,
                radii=metrics, projected_anchors=points), safe


def arm_capture(anchors, world, target, reasons, query_intrinsics, bank, order, shape=NATIVE):
    proposals, masks = [], []
    for rank, index in enumerate(order[:TOP], 1):
        measured, safe = proposal_capture(anchors, world, target, reasons, query_intrinsics, bank[index], shape)
        proposals.append(dict(rank=rank, bank_index=index, **measured))
        masks.append(safe)
    aggregate = {}
    for radius in RADII:
        label = str(radius)
        candidates = [proposal["radii"][label]["rounded_center_square_full_window"] for proposal in proposals]
        best = max(candidates, key=lambda value: value["count"], default=None)
        union = np.logical_or.reduce([item[radius] for item in masks]) if masks else np.zeros(len(anchors), bool)
        aggregate[label] = dict(
            max_full_window_anchors_single_proposal=best["count"] if best else 0,
            first_distributed_six_rank=next((rank for rank, result in enumerate(candidates, 1)
                                             if result["distributed_six"]), None),
            oracle_union_across_proposals=by_region(anchors, target, union,
                                                    int(np.count_nonzero(reasons == "co_visible_depth_consistent")), shape),
        )
    return dict(proposals=proposals, ranked_proposals=len(proposals), missing_top_eight_slots=TOP - len(proposals),
                aggregate=aggregate)


def audit(root, archive, protocol, inputs, labels, completion_sha):
    complete, dispatch, rows, banks, support, inference, truth = sealed_artifacts(
        archive, protocol, inputs, labels, completion_sha,
    )
    geometry = {}
    for item, label in zip(inference, truth, strict=True):
        world, valid, query = open_geometry(root, item, label)
        targets, reasons = query_visibility(world, valid, query)
        geometry[item["tag"]] = (world, targets, reasons, query["intrinsics"])
    output = []
    for row, supported in zip(rows, support, strict=True):
        tag = row["tag"]
        world, targets, reasons, intrinsics = geometry[tag]
        arms = {
            arm: arm_capture(ANCHORS, world, targets, reasons, intrinsics, banks[tag], order)
            for arm, order in (row["orders"] | {"reference_order_prior": supported}).items()
        }
        output.append(dict(
            {key: row[key] for key in ROW_FIELDS}, case_id=CASES[TAGS.index(tag)],
            fixed_anchor_denominator=len(ANCHORS),
            co_visible_denominator=int(np.count_nonzero(reasons == "co_visible_depth_consistent")),
            truth_visibility_states=dict(Counter(reasons)),
            truth_anchors=[dict(anchor_index=i, visibility=reasons[i], query_xy_native=finite_xy(value))
                           for i, value in enumerate(targets)],
            controls=arms,
        ))
    counts = {}
    for size in SCALES:
        for case in CASES:
            selected = [row for row in output if row["cell_pixels"] == size and row["case_id"] == case]
            key = f"{size}:{case}"
            counts[key] = {
                arm: {str(radius): sum(row["controls"][arm]["aggregate"][str(radius)]["first_distributed_six_rank"]
                                       is not None for row in selected) for radius in RADII}
                for arm in (*ARMS, "reference_order_prior")
            }
    return dict(
        schema="solar_warp_native_geometry_capture_v120", replay_source_sha256=sha(Path(__file__)),
        completion_sha256=completion_sha, protocol_sha256=complete["protocol_sha256"],
        predictions_sha256=complete["predictions_sha256"], scores_sha256=complete["scores_sha256"],
        independent_replay_sha256=complete["independent_replay_sha256"],
        roster_sha256={role: dispatch[role]["sha256"] for role in ("inputs", "labels")},
        anchor_source_geometry_sha256={item["tag"]: item["reference_geometry"]["sha256"] for item in inference},
        scoring_geometry_sha256={item["tag"]: {role: item[role]["sha256"]
                                 for role in ("query_geometry", "joint_scoring_geometry")} for item in truth},
        native_shape=list(NATIVE), anchors_xy_native_y_major=ANCHORS.tolist(), local_search_radii_pixels=list(RADII),
        primary_detector_cell_pixels=32, sensitivity_detector_cell_pixels=16, max_ranked_proposals=TOP,
        quadrants=["upper_left", "upper_right", "lower_left", "lower_right"],
        capture_definition="A full square window lies within native pixel centres 0 through 511, "
                           "and contains the true query projection using round-to-even proposed centres. "
                           "Euclidean residuals and clipped-square coverage are separate. This does not check patch "
                           "support outside the declared search square, image texture, descriptor or correspondence accuracy.",
        distribution_definition="At least six geometrically reachable anchors in at least three reference and "
                                "three true query quadrants in one proposal. This is a diagnostic for geometric "
                                "coverage only. An oracle union cannot qualify a jointly usable proposal.",
        visibility_definition="Conservative visibility diagnostic from all four adjacent positive finite axial depths, "
                              "depth tolerance 0.05 m plus 1% of target axial range, exact bilinear interpolation, "
                              "and rejection of discontinuities above this tolerance. Exact subpixel BVH labels were not rerendered.",
        scope="Post-seal exposed development geometry only. No image decoding, new matching, candidate reranking, "
              "calibration, final access, or native pose accuracy.",
        distributed_pairings_of_four_by_scale_case=counts,
        rows=output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "archive", "protocol", "inputs", "labels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    root = args.repo.resolve()
    archive = args.archive.resolve()
    if target.exists() or target.is_relative_to(root) or target.is_relative_to(archive):
        parser.error("Use a fresh audit receipt outside the scientific checkout and sealed archive")
    result = audit(root, archive, args.protocol, args.inputs, args.labels, args.completion_sha256)
    with target.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
