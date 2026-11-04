"""Standalone image and independent relative-geometry diagnostics for SPICE-HL3."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
import time
import zlib

import cv2
import numpy as np


SCHEMA = "spice-hl3-optical-transfer-v100"
SENSORS = {
    "flir-blackflys": ((720, 540), "flir-to-marker"),
    "spad": ((512, 512), "spad-to-marker"),
    "zed2:left": ((640, 360), "zed2-left-to-marker"),
}
ZED_MEMBER = re.compile(r"(?:trajectory_C/)?(fast|slow)/stereo/left/stereo_left_([0-9]+\.[0-9]+)_([0-9]+)\.png")
CLOCK = "marker_seconds = camera_seconds + published_t_d"
CALIBRATION_SOURCE = "https://arxiv.org/abs/2403.00976"
DATA_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12987981/"
POSE_TRACK = "qualified_mocap"
IMAGE_TRACK = "image_only"
MAX_BRACKET_SECONDS = Decimal("0.100")
MIN_BASELINE_M = .05
QUATERNION_NORM_TOLERANCE = 2e-4
EPIPOLAR_THRESHOLDS_PX = (1., 2., 4.)
RAW_ROTATION_LIMIT_DEG = 5.
RAW_TRANSLATION_DIRECTION_LIMIT_DEG = 15.
STRICT_ROTATION_LIMIT_DEG = 2.
STRICT_TRANSLATION_DIRECTION_LIMIT_DEG = 10.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def asset(root, record):
    path = Path(record["path"])
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != record["path"]
            or not (root / path).resolve().is_relative_to(root)):
        raise ValueError("Use a canonical relative path inside the manifest directory")
    path = root / path
    if sha(path) != record["sha256"]:
        raise ValueError("Input bytes differ from the frozen manifest: " + str(path))
    return path


def decimal(value):
    if not isinstance(value, str):
        raise ValueError("Time must use decimal text rather than an epoch float")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("Finite decimal time required")
    return result


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rotation(q):
    q = np.asarray(q, float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or abs(norm - 1.) > QUATERNION_NORM_TOLERANCE:
        raise ValueError("Raw mocap xyzw quaternion is outside unit-norm tolerance")
    # Normalize numerical serialization error only, after the original-norm gate.
    v, w = q[:3] / norm, q[3] / norm
    return (w * w - v @ v) * np.eye(3) + 2. * np.outer(v, v) + 2. * w * skew(v)


def pose(r, t):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, t
    return out


def invert(t):
    r = t[:3, :3].T
    return pose(r, -(r @ t[:3, 3]))


def read_extrinsic(path, camera_key):
    _, key = SENSORS[camera_key]
    record = json.loads(Path(path).read_text(), parse_float=Decimal)[key]
    matrix = np.asarray(record["T_c_M"], dtype=float).reshape(4, 4)
    r = matrix[:3, :3]
    # The ZED2 transform is released to only three decimal places.
    # Project its near-rotation to SO(3) within a rounding-level bound.
    tolerance = 1e-3 if camera_key == "zed2:left" else 2e-5
    if (not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or not np.allclose(r.T @ r, np.eye(3), atol=tolerance, rtol=0) or np.linalg.det(r) <= 0):
        raise ValueError("Published marker-to-camera matrix is not a near-rigid transform")
    u, _, v = np.linalg.svd(r)
    r_projected = u @ v
    if np.max(np.abs(r_projected - r)) > tolerance:
        raise ValueError("Marker rotation exceeds rounding-only projection")
    return pose(r_projected, matrix[:3, 3]), decimal(str(record["t_d"]))


def left_jacobian(w):
    angle = np.linalg.norm(w)
    x = skew(w)
    if angle < 1e-6:
        return np.eye(3) + .5 * x + (x @ x) / 6.
    return np.eye(3) + (1. - np.cos(angle)) / angle**2 * x + (angle - np.sin(angle)) / angle**3 * (x @ x)


def interpolate(a, b, fraction):
    relative = invert(a) @ b
    w = cv2.Rodrigues(relative[:3, :3])[0].ravel()
    velocity = np.linalg.solve(left_jacobian(w), relative[:3, 3])
    u = w * fraction
    return a @ pose(cv2.Rodrigues(u)[0], left_jacobian(u) @ (velocity * fraction))


def marker_table(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not {"Time", *("Position" + v for v in "XYZ"),
                            *("Orientation" + v for v in "XYZW")} <= row.keys() for row in rows):
        raise ValueError("Original mocap CSV with Time, PositionXYZ and OrientationXYZW is required")
    times = [decimal(row["Time"]) for row in rows]
    order = sorted(range(len(times)), key=times.__getitem__)
    if any(times[a] >= times[b] for a, b in zip(order, order[1:])):
        raise ValueError("Mocap clock rows must have unique timestamps")
    ranks = {row_id: rank for rank, row_id in enumerate(order)}
    return rows, times, ranks


def marker_pose(row):
    xyz = np.asarray([float(row["Position" + v]) for v in "XYZ"])
    if not np.isfinite(xyz).all():
        raise ValueError("Finite marker position in metres required")
    return pose(rotation([float(row["Orientation" + v]) for v in "XYZW"]), xyz)


def camera_at(frame, rows, times, ranks, marker_to_camera, delay):
    target = decimal(frame["timestamp_s"]) + delay
    i, j = frame["marker_bracket_rows"]
    if (type(i) is not int or type(j) is not int or i not in ranks or j not in ranks
            or not 0 <= ranks[j] - ranks[i] <= 1):
        raise ValueError("Exact or adjacent sorted bracket rows required")
    lo, hi = times[i], times[j]
    if not lo <= target <= hi or hi - lo > MAX_BRACKET_SECONDS:
        raise ValueError("Camera capture has no qualified clock-corrected mocap bracket")
    a = marker_pose(rows[i])
    marker_to_world = a if i == j else interpolate(a, marker_pose(rows[j]), float((target - lo) / (hi - lo)))
    return marker_to_world @ invert(marker_to_camera)


def calibration(path, sensor, crop):
    data = read(path)
    data = data["zed2"]["left_cam"] if sensor == "zed2:left" else data[sensor]
    s, x = data["settings"], data["intrinsics"]
    size, _ = SENSORS[sensor]
    if (s["imageWidth"], s["imageHeight"]) != size:
        raise ValueError("Exact published native sensor size required")
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError("Crop must be integer [x, y, width, height] without resize")
    left, top, width, height = crop
    if min(left, top) < 0 or min(width, height) < 64 or left + width > size[0] or top + height > size[1]:
        raise ValueError("Crop exceeds the calibrated native image")
    k = np.array([[x["fx"], 0., x["cx"] - left], [0., x["fy"], x["cy"] - top], [0., 0., 1.]])
    d = np.zeros(5) if sensor == "zed2:left" else np.array([x["k1"], x["k2"], x["p1"], x["p2"], 0.])
    if not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Finite valid published intrinsics and 2-radial distortion required")
    return size, k, d


def camera_metadata_table(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not {"Timestamp", "Width", "Height", "Encoding"} <= row.keys() for row in rows):
        raise ValueError("Original left-camera frame metadata CSV required")
    times = [decimal(row["Timestamp"]) for row in rows]
    counts = Counter(times)
    return rows, times, counts


def check_zed_binding(frame, metadata):
    rows, times, counts = metadata
    stamp = decimal(frame["timestamp_s"])
    i = frame["camera_metadata_row"]
    match = ZED_MEMBER.fullmatch(frame["archive_member"])
    if (type(i) is not int or not 0 <= i < len(rows) or match is None or match[1] != frame["stream_id"]
            or decimal(match[2]) != stamp or times[i] != stamp or counts[stamp] != 1):
        raise ValueError("Unique exact original camera metadata and archive-member timestamp required")
    row = rows[i]
    if row["Encoding"] != "mono8" or int(row["Height"]) != 360 or int(row["Width"]) != 640:
        raise ValueError("Original mono8 640x360 ZED2 left metadata required")


def verify_archive_schedule(root, schedule_record, frames, cases):
    schedule = read(asset(root, schedule_record))
    expected = {row["id"]: row for row in schedule["frames"]}
    if set(expected) != set(frames) or [{k: row[k] for k in ("id", "reference", "query")} for row in schedule["cases"]] != cases:
        raise ValueError("Original metadata-only frame and case schedule differs")
    source = schedule["source_directory"]
    directory = read(asset(root, source))
    if directory["central_directory_sha256"] != source["central_directory_sha256"]:
        raise ValueError("Original source ZIP directory identity differs")
    for key, frame in frames.items():
        row = expected[key]
        fields = ("timestamp_s", "archive_member", "camera_metadata_row", "marker_bracket_rows")
        if any(row[field] != frame[field] for field in fields) or row["speed"] != frame["stream_id"]:
            raise ValueError("Original metadata-only frame binding changed")
        info = directory["selected"][frame["archive_member"]]
        payload = asset(root, frame).read_bytes()
        if (len(payload) != info["size"] or len(payload) != row["member_size"]
                or zlib.crc32(payload) != info["crc32"] or f"{info['crc32']:08x}" != row["member_crc32"]):
            raise ValueError("Selected original archive PNG payload CRC or length differs")


def validate_manifest(path, *, allow_prepared=False):
    path = Path(path).resolve()
    p = read(path)
    sensor, track = p["camera_key"], p["track"]
    if (p["schema"] != SCHEMA or p["trajectory"] not in ("A", "C") or sensor not in SENSORS
            or track not in (IMAGE_TRACK, POSE_TRACK) or p["calibration_or_final_access"] is not False
            or type(p["prepared_only"]) is not bool or (p["prepared_only"] and not allow_prepared)):
        raise ValueError("Published analog-only A/C transfer manifest required")
    if (p["trajectory"], sensor) not in (("A", "flir-blackflys"), ("C", "spad"), ("C", "zed2:left")):
        raise ValueError("Camera must match the declared trajectory capture path")
    root = path.parent
    size, k, d = calibration(asset(root, p["calibration"]), sensor, p["crop"])
    frames = {}
    for record in p["frames"]:
        key = record["id"]
        if not isinstance(key, str) or not key or key in frames:
            raise ValueError("Unique image frame IDs required")
        decimal(record["timestamp_s"])
        frames[key] = record
        asset(root, record)
    cases = p["cases"]
    used, case_ids = set(), set()
    for case in cases:
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in case_ids:
            raise ValueError("Unique diagnostic case IDs required")
        case_ids.add(case["id"])
        roles = [case["reference"], case["query"]]
        if case.get("template") is not None:
            if p["trajectory"] != "A":
                raise ValueError("Only trajectory A supports the observed dawn template comparison")
            roles.append(case["template"])
            a, b, template = (frames[key] for key in roles)
            if (a["condition"] != "ref" or b["condition"] != "dawn-off"
                    or template["condition"] != "dawn-off" or a["waypoint"] != template["waypoint"]
                    or a["waypoint"] == b["waypoint"] or len({f["exposure_us"] for f in (a, b, template)}) != 1):
                raise ValueError("A template must share reference waypoint and query light/exposure")
        if len(set(roles)) != len(roles) or not set(roles) <= frames.keys():
            raise ValueError("Distinct frozen frame IDs required per case")
        used.update(roles)
    if not cases or used != frames.keys() or len({frames[key]["sha256"] for key in frames}) != len(frames):
        raise ValueError("Manifest must contain exactly the distinct used image bytes")
    if "metadata_schedule" in p:
        verify_archive_schedule(root, p["metadata_schedule"], frames, cases)
    truth = {}
    if track == POSE_TRACK:
        if p["trajectory"] != "C" or sensor != "zed2:left":
            raise ValueError("Only the original rectified trajectory-C ZED2 timestamps qualify")
        if p["time_convention"] != CLOCK:
            raise ValueError("Official clock convention must be frozen")
        m, delay = read_extrinsic(asset(root, p["extrinsics"]), sensor)
        streams = p["mocap_streams"]
        if not streams or not set(streams) <= {"fast", "slow"}:
            raise ValueError("Original ZED2 fast or slow streams required")
        mocap_tables = {key: marker_table(asset(root, value["mocap"])) for key, value in streams.items()}
        camera_tables = {key: camera_metadata_table(asset(root, value["camera_metadata"])) for key, value in streams.items()}
        for key, record in frames.items():
            stream = record["stream_id"]
            check_zed_binding(record, camera_tables[stream])
            truth[key] = camera_at(record, *mocap_tables[stream], m, delay)
        for case in cases:
            a, b = case["reference"], case["query"]
            if frames[a]["stream_id"] != frames[b]["stream_id"]:
                raise ValueError("Mocap scoring is restricted to same-run image pairs")
            distance = np.linalg.norm(truth[a][:3, 3] - truth[b][:3, 3])
            if distance < MIN_BASELINE_M:
                raise ValueError("Every frozen pose case requires at least 5cm nominal mocap baseline")
    return p, frames, size, k, d, truth


def image(root, record, size, crop):
    path = asset(root, record)
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None or values.dtype != np.uint8 or values.shape[:2] != (size[1], size[0]):
        raise ValueError("Verified native-sized uint8 image is required")
    if values.ndim == 3 and values.shape[2] == 3:
        values = cv2.cvtColor(values, cv2.COLOR_BGR2GRAY)
    if values.ndim != 2:
        raise ValueError("Grayscale or three-channel camera content required")
    x, y, w, h = crop
    return np.ascontiguousarray(values[y:y + h, x:x + w])


def sift(a, b):
    descriptor = cv2.SIFT_create(nfeatures=2048)
    ka, da = descriptor.detectAndCompute(a, None)
    kb, db = descriptor.detectAndCompute(b, None)
    if da is None or db is None or len(db) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    all_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    unique = {}
    good = (pair[0] for pair in all_matches if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance)
    for match in sorted(good, key=lambda m: (m.distance, m.queryIdx, m.trainIdx)):
        unique.setdefault(match.trainIdx, match)
    matches = list(unique.values())
    return (np.array([ka[m.queryIdx].pt for m in matches], float).reshape(-1, 2),
            np.array([kb[m.trainIdx].pt for m in matches], float).reshape(-1, 2))


def rectified(points, k, d):
    if not len(points) or not d.any():
        return points.copy()
    return cv2.undistortPoints(points.reshape(-1, 1, 2), k, d, P=k).reshape(-1, 2)


def fit_relative(p, q, k):
    if len(p) < 8:
        return None, None, None, np.zeros(len(p), bool), 0
    cv2.setRNGSeed(20261104)
    e, mask = cv2.findEssentialMat(p, q, k, method=cv2.RANSAC, prob=.999, threshold=1.5, maxIters=2000)
    if e is None or mask is None:
        return None, None, None, np.zeros(len(p), bool), 0
    inlier_count = int(np.count_nonzero(mask))
    best = None
    for model in e.reshape(-1, 3, 3):
        n, r, t, selected = cv2.recoverPose(model, p, q, k, mask=mask.copy())
        if n >= 5 and (best is None or n > best[0]):
            best = n, r, t.ravel(), model, selected.ravel().astype(bool)
    if best is None:
        return None, None, None, np.zeros(len(p), bool), inlier_count
    _, r, t, model, selected = best
    return r, t, model, selected, inlier_count


def epipolar_pixels(p, q, e, k):
    """Symmetric point-to-line RMS in rectified native-pixel units."""
    ik = np.linalg.inv(k)
    f = ik.T @ e @ ik
    a = np.column_stack((p, np.ones(len(p))))
    b = np.column_stack((q, np.ones(len(q))))
    lq, lp = a @ f.T, b @ f
    dot = np.abs(np.einsum("ni,ni->n", b, lq))
    na, nb = np.linalg.norm(lq[:, :2], axis=1), np.linalg.norm(lp[:, :2], axis=1)
    out = np.full(len(p), np.inf)
    ok = (na > 1e-14) & (nb > 1e-14)
    out[ok] = np.sqrt(.5 * ((dot[ok] / na[ok])**2 + (dot[ok] / nb[ok])**2))
    return out


def residual_summary(residual):
    finite = residual[np.isfinite(residual)]
    return dict(returned=len(residual), finite=len(finite),
                within_pixels={str(int(t)): int((residual <= t).sum()) for t in EPIPOLAR_THRESHOLDS_PX},
                median_px=float(np.median(finite)) if len(finite) else None)


def evaluate_geometry(p, q, r, t, k, reference_to_world, query_to_world):
    relative = invert(query_to_world) @ reference_to_world
    true_r, true_t = relative[:3, :3], relative[:3, 3]
    baseline = float(np.linalg.norm(true_t))
    if baseline < MIN_BASELINE_M:
        raise ValueError("Nominal mocap baseline is too small for signed translation direction")
    true_t = true_t / baseline
    e = skew(true_t) @ true_r
    known = residual_summary(epipolar_pixels(p, q, e, k))
    angle_r = angle_t = None
    if r is not None and t is not None:
        angle_r = float(np.rad2deg(np.arccos(np.clip((np.trace(r @ true_r.T) - 1.) * .5, -1., 1.))))
        angle_t = float(np.rad2deg(np.arccos(np.clip(t @ true_t / np.linalg.norm(t), -1., 1.))))
    return dict(baseline_m=baseline, known_essential=e.tolist(), independent_epipolar=known,
                rotation_error_deg=angle_r, signed_translation_direction_error_deg=angle_t,
                recovered=angle_r is not None and angle_r <= RAW_ROTATION_LIMIT_DEG and angle_t <= RAW_TRANSLATION_DIRECTION_LIMIT_DEG,
                strict_recovered=angle_r is not None and angle_r <= STRICT_ROTATION_LIMIT_DEG and angle_t <= STRICT_TRANSLATION_DIRECTION_LIMIT_DEG)


def evaluate_images(reference, query, k, d):
    """Image-only inference. Ground truth and case data never enter this function."""
    p, q = sift(reference, query)
    a, b = rectified(p, k, d), rectified(q, k, d)
    reason = None
    try:
        r, t, e, selected, consensus = fit_relative(a, b, k)
        if r is None:
            reason = "insufficient_matches_or_cheirality"
    except cv2.error:
        r = t = e = None
        selected, consensus = np.zeros(len(a), bool), 0
        reason = "opencv_pose_failure"
    return a, b, r, t, dict(
        pose_failure_reason=reason,
        matched=len(p), reference_xy_crop=p.tolist(), query_xy_crop=q.tolist(),
        reference_rectified_xy_crop=a.tolist(), query_rectified_xy_crop=b.tolist(),
        raw_rotation=None if r is None else r.tolist(),
        raw_translation_direction=None if t is None else t.tolist(), ransac_inliers=consensus,
        cheirality_inliers=np.flatnonzero(selected).tolist(),
        fitted_image_epipolar=None if e is None else residual_summary(epipolar_pixels(a, b, e, k)),
    )


def run(path, output, max_seconds):
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= 600:
        raise ValueError("Explicit finite local diagnostic allowance at most 600 seconds required")
    start = time.monotonic()
    p, frames, size, k, d, truth = validate_manifest(path)
    root = Path(path).resolve().parent
    output = Path(output)
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    row_pins = {}
    for case in p["cases"]:
        for role in ("reference", "template"):
            ref_id = case.get(role)
            if ref_id is None:
                continue
            if time.monotonic() - start >= max_seconds:
                write(output / "TIMEOUT.json", dict(completed_arms=len(results), max_seconds=max_seconds))
                return
            tick = time.monotonic()
            query_id = case["query"]
            reference = image(root, frames[ref_id], size, p["crop"])
            query = image(root, frames[query_id], size, p["crop"])
            a, b, r, t, raw = evaluate_images(reference, query, k, d)
            geometry = evaluate_geometry(a, b, r, t, k, truth[ref_id], truth[query_id]) if truth else None
            row = dict(case_id=case["id"], arm=role, reference=ref_id, query=query_id, raw=raw,
                       stream_id=frames[query_id].get("stream_id"), independent_geometry=geometry,
                       calendar_seconds=time.monotonic() - tick)
            row_name = str(len(results)).zfill(5) + ".json"
            write(output / row_name, row)
            row_pins[row_name] = sha(output / row_name)
            results.append(row)
    if time.monotonic() - start >= max_seconds:
        write(output / "TIMEOUT.json", dict(completed_arms=len(results), max_seconds=max_seconds))
        return
    arms = {}
    for role in ("reference", "template"):
        entries = [r for r in results if r["arm"] == role]
        if entries:
            arms[role] = dict(pairs=len(entries), matched=sum(r["raw"]["matched"] for r in entries),
                              recovered=sum(r["independent_geometry"]["recovered"] for r in entries) if truth else None,
                              strict_recovered=sum(r["independent_geometry"]["strict_recovered"] for r in entries) if truth else None)
    write(output / "summary.json", dict(
        manifest_sha256=sha(path), code_sha256=sha(__file__), trajectory=p["trajectory"], camera_key=p["camera_key"],
        track=p["track"], crop=p["crop"], native_size=size, crop_intrinsics=k.tolist(), distortion=d.tolist(),
        arms=arms, rows=len(results), row_sha256=row_pins, elapsed_seconds=time.monotonic() - start,
        thresholds=dict(rotation_deg=RAW_ROTATION_LIMIT_DEG, signed_direction_deg=RAW_TRANSLATION_DIRECTION_LIMIT_DEG,
                        strict_rotation_deg=STRICT_ROTATION_LIMIT_DEG, strict_signed_direction_deg=STRICT_TRANSLATION_DIRECTION_LIMIT_DEG),
        scope=("Laboratory optical consistency and, if qualified, nominal mocap relative rotation and signed translation "
               "direction. Metric translation/absolute localization and deployment acceptance are outside this track. "
               "Fitted image epipolar scores are in-sample. Independent epipolar consistency does not verify along-line matches."),
        calibration_source=CALIBRATION_SOURCE, dataset_source=DATA_SOURCE, calibration_or_final_access=False,
    ))


def replay(path, result_directory, output):
    """Recompute independent scores from sealed raw matches, without opening pixel arrays."""
    p, frames, _, k, d, truth = validate_manifest(path)
    root = Path(result_directory)
    summary = read(root / "summary.json")
    expected = [(c["id"], role, c[role], c["query"]) for c in p["cases"] for role in ("reference", "template")
                if c.get(role) is not None]
    names = [str(i).zfill(5) + ".json" for i in range(len(expected))]
    if (summary["manifest_sha256"] != sha(path) or summary["code_sha256"] != sha(__file__)
            or summary["rows"] != len(expected) or set(summary["row_sha256"]) != set(names)):
        raise ValueError("Complete source-bound optical result inventory required")
    checked = []
    for name, identity in zip(names, expected, strict=True):
        if sha(root / name) != summary["row_sha256"][name]:
            raise ValueError("Saved raw pair bytes differ")
        row = read(root / name)
        if tuple(row[key] for key in ("case_id", "arm", "reference", "query")) != identity:
            raise ValueError("Saved raw pair identity differs")
        raw = row["raw"]
        points = [np.asarray(raw[key], float).reshape(-1, 2) for key in ("reference_xy_crop", "query_xy_crop")]
        if any(len(a) != raw["matched"] or not np.isfinite(a).all() or (a < 0).any()
               or (a >= np.array(p["crop"][2:])).any() for a in points):
            raise ValueError("Paired finite raw crop pixel centres required")
        a, b = [rectified(x, k, d) for x in points]
        for key, values in (("reference_rectified_xy_crop", a), ("query_rectified_xy_crop", b)):
            if not np.array_equal(np.asarray(raw[key]).reshape(-1, 2), values):
                raise ValueError("Rectified output does not replay from the original raw points")
        r = None if raw["raw_rotation"] is None else np.array(raw["raw_rotation"])
        t = None if raw["raw_translation_direction"] is None else np.array(raw["raw_translation_direction"])
        ref, query = identity[2:]
        geometry = evaluate_geometry(a, b, r, t, k, truth[ref], truth[query]) if truth else None
        if geometry != row["independent_geometry"]:
            raise ValueError("Saved independent score differs")
        checked.append(dict(case_id=identity[0], arm=identity[1], stream_id=frames[query].get("stream_id"),
                            matched=len(a), independently_scored=geometry))
    write(output, dict(passed=True, rows=checked, source_summary_sha256=sha(root / "summary.json"),
                       manifest_sha256=sha(path), track=p["track"], calibration_or_final_access=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--max-seconds", type=float)
    mode.add_argument("--replay-directory", type=Path)
    args = cli.parse_args()
    if args.replay_directory is None:
        run(args.manifest, args.output, args.max_seconds)
    else:
        replay(args.manifest, args.replay_directory, args.output)
