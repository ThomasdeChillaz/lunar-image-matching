"""Track image-only Scene 1 observations and score fixed geometry conventions."""

from hashlib import file_digest
import json
from pathlib import Path

import cv2
import numpy as np

from audit_lusnar_scene1_stereo_v117 import decode_pfm_storage, decode_semantic_png, decode_stereo_png
from lusnar_temporal_geometry_v119 import (
    CONVENTIONS, compare_conventions, exact_geometry_pose, interpret_depth, intrinsics, left_camera,
)

FREEZE_SHA = "7ae6bfd9fc88651effce3d43df1282668f5212a90dd90242953bb3daf2ea9d45"
EXTRACTION_SHA = "e6cd62430db4dda6c05d3209b59dac7758e0ffca281a3bc750d748597ffa2a30"
POSES_SHA = "023a9352b744e7739d465e720379b538bcdc10af6f7fd5e5424185cb0a470d6e"
LIMIT_PER_CELL = 48


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verified(path, expected):
    if path.is_symlink() or sha(path) != expected:
        raise ValueError("Frozen bytes differ for " + str(path))
    return path


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")


def triplet_index(frozen_path, root):
    frozen = json.loads(verified(frozen_path, FREEZE_SHA).read_bytes())
    proof = json.loads(verified(root / "COMPLETE.json", EXTRACTION_SHA).read_bytes())
    if (proof["scene"] != 1 or proof["reserved_scenes_opened"]
            or frozen["scene"] != 1 or frozen["reserve_access_allowed"]
            or proof["freeze_sha256"] != FREEZE_SHA or len(proof["entries"]) != 180):
        raise ValueError("Expected exact development-only temporal extraction receipt")
    expected = {(row["timestamp_ns"], role): asset for row in frozen["temporal_frames"]
                for role, asset in row["members"].items()}
    observed = {(row["timestamp_ns"], row["role"]): row for row in proof["entries"]}
    if len(observed) != 180 or observed.keys() != expected.keys():
        raise ValueError("Incomplete frame-role identity")
    for key, row in observed.items():
        asset = expected[key]
        if (row["path"] != asset["name"] or row["bytes"] != asset["bytes"]
                or row["local_zip_crc32"] != asset["crc32"]):
            raise ValueError("Receipt row differs from frozen archive identity")
    pairs = [dict(pair, group=group["group_id"], scope=group["scope"])
             for group in frozen["groups"] for pair in group["pairs"]]
    if len(pairs) != 30:
        raise ValueError("The protocol requires all 30 original development pairs")
    return observed, pairs, frozen["temporal_frames"]


def load_asset(root, row):
    path = root / row["path"]
    if not path.resolve().is_relative_to(root.resolve()) or path.resolve() != path.absolute():
        raise ValueError("Unexpected extracted local path")
    raw = verified(path, row["sha256"]).read_bytes()
    if len(raw) != row["bytes"]:
        raise ValueError("Unexpected original byte length")
    return raw


def track(left, right, left_valid, right_valid):
    """Observe forward/backward optical motion before pose, depth or label access."""
    h, w = left.shape
    if left.shape != right.shape or left.shape != left_valid.shape or right.shape != right_valid.shape:
        raise ValueError("Aligned monochrome and opaque masks are required")
    proposals = []
    for y, y1 in zip(np.linspace(0, h, 5, dtype=int)[:-1], np.linspace(0, h, 5, dtype=int)[1:]):
        for x, x1 in zip(np.linspace(0, w, 5, dtype=int)[:-1], np.linspace(0, w, 5, dtype=int)[1:]):
            crop = left[y:y1, x:x1]
            mask = np.uint8(left_valid[y:y1, x:x1]) * 255
            peaks = cv2.goodFeaturesToTrack(crop, LIMIT_PER_CELL, .01, 8., mask=mask, blockSize=5)
            if peaks is not None:
                proposals.extend(peaks.reshape(-1, 2) + [x, y])
    if not proposals:
        return np.empty((0, 2)), np.empty((0, 2))
    a = np.asarray(proposals, dtype=np.float32).reshape(-1, 1, 2)
    opts = dict(winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    b, forward, _ = cv2.calcOpticalFlowPyrLK(left, right, a, None, **opts)
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(right, left, b, None, **opts)
    a, b, returned = (p.reshape(-1, 2) for p in (a, b, returned))
    valid = forward.ravel().astype(bool) & backward.ravel().astype(bool)
    valid &= np.isfinite(b).all(axis=1) & np.isfinite(returned).all(axis=1)
    valid &= (b >= 8).all(axis=1) & (b < [w-8, h-8]).all(axis=1)
    valid &= (a >= 8).all(axis=1) & (a < [w-8, h-8]).all(axis=1)
    valid &= np.linalg.norm(a-returned, axis=1) <= .75
    a, b = a[valid], b[valid]
    bx, by = np.rint(b).astype(int).T
    on_right = right_valid[by, bx]
    return a[on_right], b[on_right]


def observe(frozen, root, output):
    cv2.setNumThreads(1)
    index, pairs, _ = triplet_index(frozen, root)
    observations = []
    for row in pairs:
        left, right = (decode_stereo_png(load_asset(root, index[(row[key], "left")]))
                       for key in ("reference_timestamp_ns", "query_timestamp_ns"))
        a, b = track(left[0], right[0], left[1], right[1])
        observations.append(dict(pair_id=row["pair_id"], reference_xy=a.tolist(), query_xy=b.tolist()))
    write(output, dict(schema="lusnar_scene1_temporal_observations_v119", scene=1,
                       freeze_sha256=FREEZE_SHA, extraction_sha256=EXTRACTION_SHA,
                       query_truth_opened=False, rows=observations))


def assess(rows):
    names = [c.name for c in CONVENTIONS]
    outcome = {}
    for scope in ("convention_selection", "convention_confirmation"):
        group = [row for row in rows if row["scope"] == scope]
        eligible = [row for row in group if row["status"] == "scored"]
        by_candidate = []
        for name in names:
            if eligible:
                values = [row["hypotheses"][name] for row in eligible]
                by_candidate.append(dict(name=name,
                    median_pair_pixel_error=float(np.median([v["median_pixel_error"] for v in values])),
                    median_pair_log_depth_error=float(np.median([v["median_absolute_log_depth_error"] for v in values]))))
        by_candidate.sort(key=lambda v: (v["median_pair_pixel_error"], v["median_pair_log_depth_error"], v["name"]))
        groups = {name: sum(row["group"] == name for row in eligible) for name in sorted({row["group"] for row in group})}
        outcome[scope] = dict(total_pairs=len(group), eligible_pairs=len(eligible),
                              eligible_by_group=groups, rankings=by_candidate)

    def passing(side, minimum_pairs):
        scores = side["rankings"]
        if (not scores or side["eligible_pairs"] < minimum_pairs
                or min(side["eligible_by_group"].values(), default=0) < 4):
            return []
        best = scores[0]
        if best["median_pair_pixel_error"] > 2. or best["median_pair_log_depth_error"] > .04:
            return []
        return [v["name"] for v in scores
                if v["median_pair_pixel_error"] <= best["median_pair_pixel_error"] + .25
                and v["median_pair_log_depth_error"] <= best["median_pair_log_depth_error"] + .01]

    selection = passing(outcome["convention_selection"], 15)
    confirmation = passing(outcome["convention_confirmation"], 10)
    matching = [name for name in selection if name in confirmation]
    return dict(**outcome, selection_candidates=selection, confirmation_candidates=confirmation,
                qualified_common_hypotheses=matching,
                status="unique_convention" if len(matching) == 1 else
                       ("equivalent_or_ambiguous_conventions" if matching else "no_confirmed_convention"),
                exact_pose_oracle_passed=False)


def pose_log(frames, path, parser):
    logged = parser(verified(path, POSES_SHA).read_text())
    by_pose = {str(row["timestamp_ns"]): row for row in logged}
    return {row["timestamp_ns"]: by_pose[row["pose_timestamp_ns"]] for row in frames}


def score(frozen, root, observations, observation_sha, stereo, stereo_sha, pose_path, pose_parser, output):
    cv2.setNumThreads(1)
    index, pairs, frames = triplet_index(frozen, root)
    evidence = json.loads(verified(observations, observation_sha).read_bytes())
    if (evidence["query_truth_opened"] is not False or evidence["freeze_sha256"] != FREEZE_SHA
            or evidence["extraction_sha256"] != EXTRACTION_SHA
            or [v["pair_id"] for v in evidence["rows"]] != [p["pair_id"] for p in pairs]):
        raise ValueError("Expected the sealed exact 30-row optical observation set")
    witness = json.loads(verified(stereo, stereo_sha).read_bytes())
    specification = witness["conditional_stereo_depth_convention"]
    if specification is None:
        raise ValueError("Stereo depth geometry remains unresolved")
    by_camera = pose_log(frames, pose_path, pose_parser)
    rows = []
    for pair, observation in zip(pairs, evidence["rows"], strict=True):
        a, b = pair["reference_timestamp_ns"], pair["query_timestamp_ns"]
        storage = [decode_pfm_storage(load_asset(root, index[(t, "depth")])) for t in (a, b)]
        terrain = [decode_semantic_png(load_asset(root, index[(t, "semantic")]))[0] for t in (a, b)]
        xy = [np.asarray(observation[k], float).reshape(-1, 2) for k in ("reference_xy", "query_xy")]
        result = compare_conventions(*xy, *storage, *terrain, by_camera[a], by_camera[b], specification)
        rows.append(dict(pair_id=pair["pair_id"], group=pair["group"], scope=pair["scope"], **result))
    write(output, dict(schema="lusnar_scene1_temporal_convention_audit_v119", scene=1,
                       observation_sha256=observation_sha, stereo_sha256=stereo_sha,
                       calibration_or_final_access=False, rows=rows, assessment=assess(rows)))


def oracle(frozen, root, conventions, convention_sha, stereo, stereo_sha, pose_path, pose_parser, output):
    """Keep the exact-geometry control distinct from observed-track agreement."""
    cv2.setNumThreads(1)
    index, pairs, frames = triplet_index(frozen, root)
    audit = json.loads(verified(conventions, convention_sha).read_bytes())
    if (audit["schema"] != "lusnar_scene1_temporal_convention_audit_v119" or audit["scene"] != 1
            or audit["assessment"]["status"] != "unique_convention" or audit["stereo_sha256"] != stereo_sha
            or [row["pair_id"] for row in audit["rows"]] != [pair["pair_id"] for pair in pairs]):
        raise ValueError("Require a unique confirmed convention before exact-pose controls")
    name, = audit["assessment"]["qualified_common_hypotheses"]
    convention, = [c for c in CONVENTIONS if c.name == name]
    witness = json.loads(verified(stereo, stereo_sha).read_bytes())
    specification = witness["conditional_stereo_depth_convention"]
    if specification is None:
        raise ValueError("Stereo depth remains unresolved")
    k = intrinsics(convention.principal_point)
    poses = pose_log(frames, pose_path, pose_parser)
    rows = []
    for pair in pairs:
        times = tuple(pair[key] for key in ("reference_timestamp_ns", "query_timestamp_ns"))
        depths = [interpret_depth(decode_pfm_storage(load_asset(root, index[(t, "depth")])), specification, k)
                  for t in times]
        masks = [decode_semantic_png(load_asset(root, index[(t, "semantic")]))[0] for t in times]
        cameras = [left_camera(poses[t], convention) for t in times]
        rows.append(dict(pair_id=pair["pair_id"], group=pair["group"],
                         **exact_geometry_pose(*depths, *masks, *cameras, k)))
    eligible = [row for row in rows if row["eligible"]]
    by_group = {g: sum(row["group"] == g for row in eligible) for g in sorted({r["group"] for r in rows})}
    passed = sum(row["passed"] for row in eligible)
    gate = (len(eligible) >= 25 and min(by_group.values(), default=0) >= 4
            and passed * 100 >= 95 * len(eligible))
    write(output, dict(schema="lusnar_scene1_exact_pose_oracle_v119", scene=1, convention=name,
                       convention_audit_sha256=convention_sha, stereo_audit_sha256=stereo_sha,
                       original_pairs=len(rows), eligible_pairs=len(eligible), eligible_by_group=by_group,
                       correct_pose_controls=passed, gate_passed=gate, rows=rows,
                       transfer_claim=False, reserved_scenes_opened=[]))
