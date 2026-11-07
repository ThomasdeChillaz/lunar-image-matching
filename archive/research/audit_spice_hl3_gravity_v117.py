"""Compare the archived ZED2 camera composition against released IMU metadata."""

import argparse
from bisect import bisect_left, bisect_right
import csv
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

SCHEDULE = "trajectory_C_metadata_cases_v2.json"
SCHEDULE_SHA = "8e5f38aa087a147dc58979c5f2abee2bcf6580bb2bf31a831cab6929e69034f8"
SPEEDS = ("fast", "slow")
IMU_SHA = {
    "fast": "814fcad72f43eb16ed33d1276090bdd758ed75b4f4bd7059a9f3df46000a39d4",
    "slow": "8e1884b52a8f4f4d69f0c06f8660fb7b81dd8ca6ba4c65fb8dc912bbed0b6669",
}
WINDOW = Decimal("0.050")
MIN_SAMPLES = 3
ACCEL_BOUNDS = (9.3, 10.3)
MAX_ANGULAR_RATE = 0.2
WORLD_GRAVITY = np.array([0.0, -1.0, 0.0])
# Fixed diagnostic, selected after observing a near-quarter-turn roll from
# mocap alone. It is not an estimate of the true extrinsic.
OPTICAL_TO_QUARTER_TURN_BASIS = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


def checked(path, expected):
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Archived metadata hash differs for {path.name}")
    return path


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def columns(row, prefix, axes):
    return np.array([float(row[prefix + axis]) for axis in axes])


def slerp(a, b, weight):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cosine = float(a @ b)
    if cosine < 0:
        b, cosine = -b, -cosine
    if cosine > 1 - 1e-8:
        q = (1 - weight) * a + weight * b
        return q / np.linalg.norm(q)
    angle = np.arccos(cosine)
    return (np.sin((1 - weight) * angle) * a + np.sin(weight * angle) * b) / np.sin(angle)


def marker_rotation(rows, frame):
    lo, hi = (rows[i] for i in frame["marker_bracket_rows"])
    t0, t1 = Decimal(lo["Time"]), Decimal(hi["Time"])
    t = Decimal(frame["timestamp_s"])
    if not t0 <= t <= t1 or t1 - t0 > Decimal("0.100"):
        raise ValueError("Camera timestamp is outside its frozen motion-capture bracket")
    weight = float((t - t0) / (t1 - t0)) if t0 != t1 else 0.0
    a, b = (columns(row, "Orientation", "XYZW") for row in (lo, hi))
    if not all(abs(np.linalg.norm(q) - 1) <= 0.001 for q in (a, b)):
        raise ValueError("Invalid motion-capture quaternion")
    x, y, z, w = slerp(a, b, weight)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def calibrated_rotation(record, key):
    transform = np.asarray(record[key]).reshape(4, 4)
    raw = transform[:3, :3]
    if (not np.array_equal(transform[3], [0, 0, 0, 1])
            or np.max(np.abs(raw.T @ raw - np.eye(3))) > 0.002
            or np.linalg.det(raw) <= 0):
        raise ValueError("Expected a serialized rigid rotation")
    u, _, vt = np.linalg.svd(raw)
    return u @ vt


def imu_samples(rows):
    samples = []
    for row in rows:
        a = columns(row, "Linear_Acc", "XYZ")
        rate = np.linalg.norm(columns(row, "Angular_Vel", "XYZ"))
        norm = np.linalg.norm(a)
        if ACCEL_BOUNDS[0] <= norm <= ACCEL_BOUNDS[1] and rate <= MAX_ANGULAR_RATE:
            samples.append((Decimal(row["Time"]), a / norm))
    samples.sort(key=lambda sample: sample[0])
    if len({t for t, _ in samples}) != len(samples):
        raise ValueError("Duplicate IMU timestamps")
    return [sample[0] for sample in samples], [sample[1] for sample in samples]


def optical_gravity(t, times, forces, imu_to_optical):
    lo = bisect_left(times, t - WINDOW)
    hi = bisect_right(times, t + WINDOW)
    if hi - lo < MIN_SAMPLES:
        return None, hi - lo
    # The accelerometer reports proper acceleration, opposite to gravity
    # for a quasi-static camera. No fused orientation is used.
    estimate = -(imu_to_optical @ np.median(forces[lo:hi], axis=0))
    return estimate / np.linalg.norm(estimate), hi - lo


def direction_fields(vector, observed):
    roll = np.degrees(np.arctan2(vector[0], vector[1]))
    pitch = np.degrees(np.arcsin(np.clip(vector[2], -1, 1)))
    error = np.degrees(np.arccos(np.clip(vector @ observed, -1, 1)))
    return dict(roll_deg=float(roll), down_pitch_deg=float(pitch), error_deg=float(error))


def fetch_imu(assets, output):
    from spice_hl3_released_archive_v100 import ARCHIVES, SOURCES, RangeReader, read_member

    schedule = json.loads(checked(assets / SCHEDULE, SCHEDULE_SHA).read_bytes())
    entry = schedule["source_directory"]
    directory = json.loads(checked(assets / entry["path"], entry["sha256"]).read_bytes())
    url = SOURCES["zenodo"] + "trajectory_C.zip"
    size, _ = ARCHIVES["trajectory_C.zip"]
    if (directory["source"] != url or directory["archive_size"] != size
            or directory["central_directory_sha256"] != entry["central_directory_sha256"]):
        raise ValueError("Unexpected original archive identity")
    output.mkdir(parents=True, exist_ok=True)
    reader = RangeReader(url, size, byte_limit=700_000)
    for speed in SPEEDS:
        path = output / f"{speed}_stereo_imu_data.csv"
        if path.exists():
            checked(path, IMU_SHA[speed])
            continue
        name = f"{speed}/stereo/imu_data.csv"
        data = read_member(reader, name, directory["selected"][name])
        if sha256(data).hexdigest() != IMU_SHA[speed]:
            raise ValueError("Released IMU file bytes differ")
        with path.open("xb") as stream:
            stream.write(data)


def measure(assets, imu):
    schedule = json.loads(checked(assets / SCHEDULE, SCHEDULE_SHA).read_bytes())
    extrinsics = schedule["extrinsics"]
    source = json.loads(checked(assets / extrinsics["path"], extrinsics["sha256"]).read_bytes())
    marker_record = source["zed2-left-to-marker"]
    if marker_record["t_d"] != 0.0 or source["zed2Left-to-imu"]["t_d"] != 0.0:
        raise ValueError("This diagnostic assumes the two released zero time offsets")
    marker_to_optical = calibrated_rotation(marker_record, "T_c_M")
    imu_to_optical = calibrated_rotation(source["zed2Left-to-imu"], "T_c_I")
    frames = schedule["frames"]
    cases = schedule["cases"]
    if (len(cases) != 24 or len(frames) != 48
            or {f["id"] for f in frames} != {c[k] for c in cases for k in ("reference", "query")}):
        raise ValueError("Expected the exact 48 frozen camera endpoints")
    by_speed = {}
    for speed in SPEEDS:
        meta = schedule["sequences"][speed]["mocap"]
        mocap = csv_rows(checked(assets / meta["path"], meta["sha256"]))
        imu_rows = csv_rows(checked(imu / f"{speed}_stereo_imu_data.csv", IMU_SHA[speed]))
        times, forces = imu_samples(imu_rows)
        rows = []
        for f in frames:
            if f["speed"] != speed:
                continue
            observed, count = optical_gravity(Decimal(f["timestamp_s"]), times, forces, imu_to_optical)
            row = dict(frame=f["id"], eligible_imu_samples=count)
            if observed is not None:
                marker_gravity = marker_rotation(mocap, f).T @ WORLD_GRAVITY
                nominal = marker_to_optical @ marker_gravity
                row.update(imu_gravity_camera=observed.tolist(), nominal_gravity_camera=nominal.tolist(),
                           imu=direction_fields(observed, observed),
                           nominal=direction_fields(nominal, observed),
                           reversed_marker=direction_fields(marker_to_optical.T @ marker_gravity, observed),
                           quarter_turn=direction_fields(OPTICAL_TO_QUARTER_TURN_BASIS.T @ nominal, observed))
            rows.append(row)
        by_speed[speed] = dict(released_samples=len(imu_rows), quasi_static_samples=len(times), frames=rows)
    return dict(scope="Metadata-only physical reference diagnostic. No image, matcher, pose output or inferred correction.",
                schedule_sha256=SCHEDULE_SHA, imu_sha256=IMU_SHA, extrinsics_sha256=extrinsics["sha256"],
                imu_window_seconds=float(WINDOW), minimum_samples=MIN_SAMPLES, accel_bounds_m_s2=ACCEL_BOUNDS,
                maximum_angular_rate_rad_s=MAX_ANGULAR_RATE, streams=by_speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-imu", action="store_true", help="Fetch only the two pinned IMU CSV members")
    args = parser.parse_args()
    if args.fetch_imu:
        fetch_imu(args.assets, args.imu)
    result = measure(args.assets, args.imu)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
