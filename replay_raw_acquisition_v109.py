"""Replay the sealed raw-render acquisition with OpenCV and independent array reductions."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


STUDY = "reports/solar-raw-acquisition-v109-study.json"
TIMING = "reports/solar-raw-acquisition-v109-timing.json"
COHORT = "reports/solar-clean-rerender-cohort-v106.json"
MASTER = "reports/solar_development_admission_v67.json"
JOURNAL = "runs/solar-compute-reservations-v67.json"
AOVS = ("combined", "direct", "indirect", "albedo")
FIXTURES = ("black_sun_off", "blocked_with_reflector", "unblocked", "unblocked_half")
WEIGHTS = (.2126, .7152, .0722)
JOB_FIELDS = ("key", "scene", "samples", "replicate", "seed")
GEOMETRY_FIELDS = ("axial_depth", "body_origin", "cam2world", "enu_rows", "geometry_source",
                   "intrinsics", "normals", "position", "valid")
STATS = ("aggregate_p01", "aggregate_p995", "aggregate_span", "seed_difference_single_render_rms",
         "across_seed_spatial_covariance", "positive_support_fraction")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(name):
    path = Path(name)
    if (path.is_absolute() or str(path) != name or ".." in path.parts
            or path.resolve() != Path.cwd() / path):
        raise ValueError("Require immutable repository-relative artifact paths")
    return path.read_bytes()


def pinned_bytes(name, expected):
    data = read_bytes(name)
    if digest(data) != expected:
        raise ValueError("Artifact hash differs: " + name)
    return data


def read_json(name):
    data = read_bytes(name)
    return json.loads(data), digest(data)


def write_new(name, text):
    with Path(name).open("x") as stream:
        stream.write(text)


def read_exr_channels(data):
    """Check physical sample type before OpenCV can silently promote half to float."""
    stream = io.BytesIO(data)
    if stream.read(4) != b"\x76\x2f\x31\x01":
        raise ValueError("Expected OpenEXR magic")
    stream.read(4)  # Version and flags.
    metadata = {}

    def cstr(input_):
        value = bytearray()
        while len(value) < 256:
            byte = input_.read(1)
            if byte == b"\0":
                return bytes(value)
            if not byte:
                break
            value.extend(byte)
        raise ValueError("Invalid EXR C string")

    while key := cstr(stream):
        field = cstr(stream)
        length = int.from_bytes(stream.read(4), "little")
        if key in metadata or length > 1024 * 1024:
            raise ValueError("Unexpected EXR header field")
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError("Incomplete EXR header field")
        metadata[key] = (field, payload)
    label, value = metadata[b"dataWindow"]
    if label != b"box2i":
        raise ValueError("Invalid EXR data window type")
    x0, y0, x1, y1 = struct.unpack("<iiii", value)
    channel_type, values = metadata[b"channels"]
    if channel_type != b"chlist":
        raise ValueError("Invalid EXR channel list")
    channel_stream = io.BytesIO(values)
    channels = {}
    while name := cstr(channel_stream):
        if name in channels:
            raise ValueError("Repeated EXR channel")
        pixel_type, x_sampling, y_sampling = struct.unpack("<i4xii", channel_stream.read(16))
        channels[name] = (pixel_type, x_sampling, y_sampling)
    if (x0, y0) != (0, 0) or channels != {c: (2, 1, 1) for c in (b"R", b"G", b"B")}:
        raise ValueError("Expected uncropped, full-rate float32 RGB")
    return (y1 + 1, x1 + 1)


def decode(data, *, exr_size=None):
    if exr_size is not None and read_exr_channels(data) != (exr_size, exr_size):
        raise ValueError("EXR native window mismatch")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    expected = (exr_size or 512, exr_size or 512, 3)
    if bgr is None or bgr.shape != expected or bgr.dtype != (np.float32 if exr_size else np.uint8):
        raise ValueError("Unexpected independent decoded image")
    if not np.isfinite(bgr).all():
        raise ValueError("Nonfinite decoded pixels")
    return bgr[..., ::-1]


def close(actual, expected, label, *, absolute=2e-13, relative=2e-7):
    if not math.isclose(actual, expected, abs_tol=absolute, rel_tol=relative):
        raise ValueError(f"Independent {label} differs: {actual} vs {expected}")


def exr_planes(row, directory, size, fields):
    paths = row["exr"]
    if set(paths) != set(fields):
        raise ValueError("Wrong decoded EXR plane inventory")
    planes = {}
    for name in fields:
        item = paths[name]
        path = directory / (name + "_0001.exr")
        if item["path"] != str(path):
            raise ValueError("EXR path does not match the frozen exposure")
        planes[name] = decode(pinned_bytes(str(path), item["sha256"]), exr_size=size)
    return planes


def reconstructed(planes, row):
    if len(planes) == 1:
        if row["recon_max_abs_error"] is not None:
            raise ValueError("Legacy exposure has an unexpected reconstruction")
        return {"combined": planes["combined"]}
    reflected = planes["albedo"]
    total = reflected * (planes["direct"] + planes["indirect"])
    error = float(np.abs(planes["combined"] - total).max())
    close(error, row["recon_max_abs_error"], "EXR component reconstruction")
    if error > 2e-6 + 4e-5 * float(np.abs(planes["combined"]).max()):
        raise ValueError("Light decomposition exceeds the frozen tolerance")
    return dict(combined=planes["combined"], direct=reflected * planes["direct"],
                indirect=reflected * planes["indirect"])


def packed_mask(mask):
    packed = np.packbits(np.asarray(mask, dtype=np.bool_).ravel(), bitorder="little")
    return {"count": int(np.count_nonzero(mask)), "packed_row_major_sha256": digest(packed.tobytes())}


def geometry_mask(source, master, worker_geometry):
    asset = source["archive_geometry_path"]
    data = pinned_bytes(asset, source["archive_geometry_sha256"])
    role = source["role"]
    camera = master["units"][source["unit_index"]]["camera_pairs"][0][role]
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if sorted(archive.files) != list(GEOMETRY_FIELDS):
            raise ValueError("Unexpected frozen native geometry fields")
        valid = archive["valid"]
        depth = archive["axial_depth"]
        positions = archive["position"]
        normals = archive["normals"]
        reconstructed_valid = (np.isfinite(positions).all(axis=-1) & np.isfinite(normals).all(axis=-1)
                               & np.isfinite(depth) & (depth > 0))
        if (valid.shape != (512, 512) or valid.dtype != np.bool_ or not valid.all()
                or not np.array_equal(reconstructed_valid, valid)
                or not np.array_equal(np.asarray(camera["cam2world"]), archive["cam2world"])
                or not np.array_equal(np.asarray(camera["intrinsics"]), archive["intrinsics"])):
            raise ValueError("Independent archive geometry mask or camera replay failed")
    if (worker_geometry["exact_camera_depth_and_rays"] is not True
            or worker_geometry["native_rays"] != 512 * 512
            or worker_geometry["compared_arrays"] != list(GEOMETRY_FIELDS)):
        raise ValueError("Worker did not reproduce the exact original ray inventory")
    return packed_mask(valid)


def row_job(row):
    return {key: row[key] for key in JOB_FIELDS}


def block_means(values, block):
    """Reduce explicit, nonoverlapping spatial tiles without the renderer reshape."""
    if block == 1:
        return values
    h, w = values.shape
    if h % block or w % block:
        raise ValueError("Partial tile in replay support")
    return np.array([np.mean(values[y:y + block, x:x + block], dtype=np.float64)
                     for y in range(0, h, block) for x in range(0, w, block)]).reshape(h // block, w // block)


def exact_quantile(array, fraction):
    ordered = np.sort(array, axis=None)
    rank = fraction * (len(ordered) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    weight = rank - lower
    return float((1 - weight) * ordered[lower] + weight * ordered[upper])


def seed_statistics(first, second, block):
    def luma(pixels):
        return sum(np.multiply(pixels[..., channel], weight, dtype=np.float64)
                   for channel, weight in enumerate(WEIGHTS))

    left, right = (block_means(luma(image), block).ravel() for image in (first, second))
    mean = .5 * left + .5 * right
    low = exact_quantile(mean, .01)
    high = exact_quantile(mean, .995)
    difference = left - right
    covariance = float(np.dot(left - left.mean(), right - right.mean()) / len(left))
    return dict(aggregate_p01=low, aggregate_p995=high, aggregate_span=high - low,
                seed_difference_single_render_rms=float(np.linalg.norm(difference) / math.sqrt(2 * len(left))),
                across_seed_spatial_covariance=covariance, positive_support_fraction=float(np.count_nonzero(mean > 0)
                                                                                            / len(mean)))


def seed_masks(first, second):
    left, right = (np.any(image > 0, axis=-1) for image in (first, second))
    return dict(seed0_positive=packed_mask(left), seed1_positive=packed_mask(right),
                both_positive=packed_mask(left & right), either_positive=packed_mask(left | right),
                exactly_one_positive=packed_mask(left ^ right))


def paired_seed_replay(pairs, original):
    results = {}
    for key, by_level in pairs.items():
        levels = {}
        for level, seeds in by_level.items():
            if set(seeds) != {0, 1}:
                raise ValueError("Independent replay requires both prescribed seeds")
            first, second = seeds[0], seeds[1]
            replayed = {}
            for name in ("combined", "direct", "indirect"):
                bands = {}
                for block in (1, 32):
                    stats = seed_statistics(first[name], second[name], block)
                    expected = original[key][str(level)][name][str(block)]
                    if set(expected) != set(STATS):
                        raise ValueError("Unexpected original seed statistics")
                    for statistic in STATS:
                        close(stats[statistic], expected[statistic], f"{key}/{level}/{name}/{block}/{statistic}")
                    bands[str(block)] = stats
                replayed[name] = {"stats": bands, "decoded_positive_masks": seed_masks(first[name], second[name])}
            levels[str(level)] = replayed
        results[key] = levels
    if set(results) != set(original) or any(set(values) != set(original[k]) for k, values in results.items()):
        raise ValueError("Unexpected frozen independent seed group")
    return results


def fixture_replay(receipt, folder):
    if receipt["passed"] is not True or set(receipt["controls"]) != set(FIXTURES):
        raise ValueError("Incomplete synthetic fixture receipt")
    values = {}
    crop = np.s_[16:48, 16:48]
    for name in FIXTURES:
        control = receipt["controls"][name]
        planes = exr_planes(control, folder / "fixtures" / name, 64, AOVS)
        reconstructed(planes, control)
        values[name] = dict(
            combined_max=float(np.max(np.abs(planes["combined"][crop]))),
            combined_mean=float(np.mean(planes["combined"][crop], dtype=np.float64)),
            direct_max=float(np.max(np.abs(planes["direct"][crop]))),
            direct_mean=float(np.mean(planes["direct"][crop], dtype=np.float64)),
            indirect_mean=float(np.mean(planes["indirect"][crop], dtype=np.float64)),
        )
        for statistic, expected in values[name].items():
            close(expected, control[statistic], f"{name}/{statistic}")
    black, blocked, bright, half = (values[name] for name in FIXTURES)
    if (black["combined_max"] > 1e-8 or blocked["direct_max"] > 1e-6
            or blocked["indirect_mean"] <= 1e-6 or bright["direct_mean"] <= .001
            or abs(half["combined_mean"] / bright["combined_mean"] - .5) > .01):
        raise ValueError("Decoded physical export controls failed")
    return values


def published_protocol(path, protocol, commit):
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Full frozen publication commit required")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   check=True, timeout=5)
    for name in (path, *protocol["source_sha256"], *protocol["metadata_sha256"]):
        expected = protocol["source_sha256"].get(name, protocol["metadata_sha256"].get(name))
        current = pinned_bytes(name, expected) if expected is not None else read_bytes(name)
        archived = subprocess.check_output(["git", "show", f"{commit}:{name}"], timeout=5)
        if current != archived:
            raise ValueError("Published bytes differ: " + name)


def admitted(phase, path, journal):
    protocol, sha = read_json(path)
    output = Path(f"runs/solar-raw-acquisition-v109-{phase}")
    receipt, receipt_hash = read_json(str(output / "COMPLETE.json"))
    watchdog, watchdog_hash = read_json(str(output.with_name(output.name + "-watchdog.json")))
    caps = {"timing": 240, "study": 900}
    if (protocol["phase"] != phase or protocol["split"] != "model_selection"
            or protocol["calibration_or_final_access"] or protocol["query_truth_in_inference"]
            or protocol["native_shape"] != [512, 512] or protocol["cpu_threads"] != 2
            or protocol["output"] != str(output) or protocol["cap_seconds"] != caps[phase]
            or receipt["schema"] != 1 or receipt["phase"] != phase or not receipt["complete"]
            or receipt["calibration_or_final_access"] or receipt["cpu_threads"] != 2
            or receipt["protocol_sha256"] != sha or receipt["renderer_sha256"] != protocol["renderer_sha256"]
            or receipt["deadline_utc"] != watchdog["absolute_deadline_utc"] or not watchdog["complete"]
            or watchdog["protocol_sha256"] != sha or watchdog["error"] is not None):
        raise ValueError("Frozen run, receipt or watchdog scope mismatch")
    allocation = watchdog["reservation"]
    if (allocation not in journal["reservations"] or allocation["job_id"] != sha + "/raw_acquisition_v109"
            or allocation["category"] != "baseline" or allocation["seconds"] != caps[phase]
            or allocation["refund_on_failure"]):
        raise ValueError("Original reservation missing")
    if [row_job(row) for row in receipt["rows"]] != protocol["render_jobs"]:
        raise ValueError("Sealed rows do not match the exact admitted job list")
    published_protocol(path, protocol, receipt["publication_commit"])
    return protocol, receipt, output, {"protocol_sha256": sha, "receipt_sha256": receipt_hash,
                                       "watchdog_sha256": watchdog_hash}


def role_join(protocol, source_rows):
    join = {}
    for item in protocol["camera_mesh_identity"]:
        key = item["key"]
        source = source_rows[key]
        if key in join or any(source[k] != value for k, value in item.items()):
            raise ValueError("Frozen camera and mesh identity differ from v106")
        if source["unit_index"] not in (13, 25, 27, 61):
            raise ValueError("Unexpected terrain endpoint")
        join[key] = source
    if set(join) != {row["key"] for row in protocol["render_jobs"]}:
        raise ValueError("Incomplete sealed camera identity")
    return join


def replay_native(protocol, receipt, output, source_rows, master):
    sources = role_join(protocol, source_rows)
    masks, artifacts, pairs, legacy = {}, [], defaultdict(lambda: defaultdict(dict)), []
    for row in receipt["rows"]:
        key, scene, spp, replicate, _ = (row[k] for k in JOB_FIELDS)
        folder = output / key / scene / f"{spp}spp-r{replicate}"
        individual, row_sha = read_json(str(folder / "receipt.json"))
        if individual != row:
            raise ValueError("Individual frozen render receipt differs")
        old = scene == "legacy_white"
        if scene not in ("legacy_white", "black_space") or (not old and row["adaptive"]):
            raise ValueError("Unexpected photometric scene or adaptive sampling")
        planes = exr_planes(row, folder, 512, ("combined",) if old else AOVS)
        light = reconstructed(planes, row)
        if key not in masks:
            masks[key] = geometry_mask(sources[key], master, row["geometry"])
        if row["geometry"] != {"exact_camera_depth_and_rays": True, "native_rays": 512 * 512,
                               "compared_arrays": list(GEOMETRY_FIELDS)}:
            raise ValueError("Native ray seal disagrees across exposures")
        if old:
            archive = sources[key]
            original_png = pinned_bytes(archive["archive_path"], archive["archive_sha256"])
            regenerated = read_bytes(str(folder / "legacy_rgb8.png"))
            if not np.array_equal(decode(original_png), decode(regenerated)) or row["exact_legacy_rgb8"] is not True:
                raise ValueError("Independent original RGB8 parity failed")
            legacy.append({"key": key, "archive_png_sha256": archive["archive_sha256"],
                           "regenerated_png_sha256": digest(regenerated), "all_rgb_pixels_equal": True})
        elif protocol["phase"] == "study":
            if row["exact_legacy_rgb8"] is not None:
                raise ValueError("A black-space scene claimed historical RGB8 parity")
            pairs[key][spp][replicate] = light
        artifacts.append({"job": row_job(row), "row_receipt_sha256": row_sha,
                          "exr_sha256": {name: item["sha256"] for name, item in row["exr"].items()}})
    repeated = paired_seed_replay(pairs, receipt["same_camera_repeatability"])
    return {"artifacts": artifacts, "native_valid_masks": masks,
            "original_rgb8_parity": legacy, "same_camera_repeatability": repeated}


def replay():
    cv2.setNumThreads(1)
    journal, journal_hash = read_json(JOURNAL)
    timing, time_receipt, time_root, time_pins = admitted("timing", TIMING, journal)
    study, study_receipt, study_root, study_pins = admitted("study", STUDY, journal)
    if study["timing_basis"]["receipt_sha256"] != time_pins["receipt_sha256"]:
        raise ValueError("Study used a different timing pilot")
    for protocol in (timing, study):
        if protocol["metadata_sha256"]["reports/solar_compute_admission_v67.json"] != journal["ledger_sha256"]:
            raise ValueError("Original ledger identity differs")
    cohort, _ = read_json(COHORT)
    master, _ = read_json(MASTER)
    rows = {row["key"]: row for row in cohort["renders"]}
    timing_results = replay_native(timing, time_receipt, time_root, rows, master)
    study_results = replay_native(study, study_receipt, study_root, rows, master)
    if len(study_results["original_rgb8_parity"]) != 8:
        raise ValueError("The matched study needs eight original archive controls")
    all_jobs = timing["render_jobs"] + study["render_jobs"]
    seeds = [row["seed"] for row in all_jobs if row["scene"] == "black_space"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated black-space Cycles seed")
    return dict(
        schema=1, passed=True, calibration_or_final_access=False,
        decoder=f"OpenCV {cv2.__version__} independent OpenEXR decode, original native row order",
        source_sha256=digest(Path(__file__).read_bytes()), journal_snapshot_sha256=journal_hash,
        synthetic_controls=fixture_replay(time_receipt["fixtures"], time_root),
        timing={**time_pins, **timing_results}, study={**study_pins, **study_results},
    )


def fmt_pair(stats):
    signal = math.sqrt(max(0., stats["across_seed_spatial_covariance"]))
    noise = stats["seed_difference_single_render_rms"]
    return f"{signal:.3g} / {noise:.3g}"


def markdown(replayed):
    repeated = replayed["study"]["same_camera_repeatability"]
    lines = [
        "# Independent raw radiance replay",
        "",
        f"OpenCV independently decoded and verified {len(replayed['timing']['artifacts'])} timing and "
        f"{len(replayed['study']['artifacts'])} study renders, plus all four analytic controls. "
        "The eight legacy RGB8 captures reproduce the original archived pixels exactly. "
        "Every native archived mask contains all 262,144 valid camera pixels. The renderer's original "
        "audit checked the full geometry arrays bit for bit.",
        "",
        "At 32-pixel block scale, each cell is **repeatable spatial RMS / single-render seed-residual RMS** "
        "in relative scene-linear units. The first term is the square root of nonnegative cross-seed "
        "spatial covariance. The second is the RMS difference between independent Cycles seeds divided "
        "by square root of two. Direct and secondary terms are weighted by diffuse color.",
        "",
        "| Case and view | Cycles samples | Direct RMS pair | Secondary RMS pair | Secondary >0 in both seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, levels in repeated.items():
        case, role, _ = key.split("/")
        for level, bands in levels.items():
            direct, indirect = bands["direct"], bands["indirect"]
            overlap = indirect["decoded_positive_masks"]["both_positive"]["count"]
            lines.append(f"| {case} {role} | {level} | {fmt_pair(direct['stats']['32'])} "
                         f"| {fmt_pair(indirect['stats']['32'])} | {overlap / (512 * 512):.2%} |")
    lines.extend((
        "",
        "Positive-support masks describe decoded radiance and have no minimum intensity. The repeatability "
        "test compares renders of the same camera; it does not establish shared signal across viewpoints. "
        "The noise term measures Cycles integration residuals. No camera exposure, photon counts, "
        "read noise, matching or pose estimation was evaluated. Terrain-scattered light is limited "
        "to the declared finite simulation meshes and constant diffuse material.",
        "",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.table).exists():
        raise FileExistsError("Preserve earlier replay artifacts")
    result = replay()
    write_new(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_new(args.table, markdown(result))


if __name__ == "__main__":
    main()
