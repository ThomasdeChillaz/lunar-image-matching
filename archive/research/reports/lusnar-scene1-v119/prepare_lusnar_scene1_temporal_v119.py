"""Prepare disjoint temporal LuSNAR Scene 1 checks and anchor semantic support."""

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import runpy

from prepare_lusnar_scene1_v117 import (
    ARCHIVE, ARCHIVE_BYTES, INVENTORY, METADATA, PARSER, PINS, SOURCE, WORK,
    checked_read, digest, extract_member, format_header, write_json,
)

PARENT_SHA = "305fbd81a394462d81c99e94e507d5601f435dc995df0a422bb3d413306408b6"
WINDOW_NS = 4_500_000_000
DT_NS = (200_000_000, 1_500_000_000)
BANDS = ((.20, .75), (.75, 2.))
PAIR_ROLES = {"left": ("color", ".png"), "depth": ("depth", ".pfm"), "semantic": ("label", ".png")}


def sensor_member(timestamp, role):
    folder, extension = PAIR_ROLES[role]
    return f"Moon_1/image0/{folder}/{timestamp}{extension}"


def relative_angle(a, b):
    cosine = abs(math.fsum(x * y for x, y in zip(a, b))) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(2 * math.acos(min(1., cosine)))


def selection(eligible, poses, anchors):
    taken = set(anchors)
    groups = []
    for index, centre in enumerate(anchors):
        near = [t for t in eligible if abs(t - centre) <= WINDOW_NS and t not in taken]
        all_pairs = []
        for a, b in combinations(near, 2):
            if not DT_NS[0] <= b - a <= DT_NS[1]:
                continue
            first, second = poses[a], poses[b]
            distance = math.dist(first["position"], second["position"])
            angle = relative_angle(first["quaternion_wxyz"], second["quaternion_wxyz"])
            if angle <= 15.:
                all_pairs.append((a, b, distance, angle))
        selected = []
        for band, (lo, hi) in enumerate(BANDS):
            pool = [p for p in all_pairs if lo <= p[2] < hi]
            # Preserve both low-rotation translations and leverage for testing
            # camera origins. The median-rotation control is selected within
            # the remaining pool after the first two disjoint choices.
            for motion in ("turn", "straight", "middle"):
                remaining = [p for p in pool if p[0] not in taken and p[1] not in taken]
                if not remaining:
                    raise ValueError(f"Insufficient metadata-only pairs for window {index}")
                angles = sorted(p[3] for p in remaining)
                middle = angles[len(angles) // 2]
                score = {"turn": lambda p: -p[3], "straight": lambda p: p[3],
                         "middle": lambda p: abs(p[3] - middle)}[motion]
                a, b, distance, angle = min(
                    remaining, key=lambda p: (score(p), abs(p[0] + p[1] - 2 * centre), p[0], p[1]))
                taken.update((a, b))
                selected.append(dict(
                    pair_id=f"scene1-window-{index}-band-{band}-{motion}",
                    reference_timestamp_ns=str(a), query_timestamp_ns=str(b),
                    elapsed_ns=b-a, logged_translation=distance, logged_rotation_deg=angle,
                    nominal_motion=motion, logged_translation_band=[lo, hi],
                ))
        groups.append(dict(group_id=f"scene1-window-{index}", anchor_timestamp_ns=str(centre),
                           scope="convention_selection" if index in (0, 2, 4) else "convention_confirmation",
                           eligible_frame_count=len(near), pairs=selected))
    if len(taken) != 65:
        raise ValueError("Exactly 60 disjoint endpoints and five stereo anchors are required")
    return groups


def build(parent):
    parent_json = json.loads(checked_read(parent, PARENT_SHA))
    if (parent_json["scene"] != 1 or parent_json["reserve_access_allowed"]
            or parent_json["geometry_run_allowed"] or len(parent_json["triplets"]) != 5):
        raise ValueError("Require the frozen development-only stereo anchor selection")
    parser = WORK / PARSER
    checked_read(parser, PINS[PARSER])
    metadata_parser = runpy.run_path(str(parser))
    inventory = json.loads(checked_read(SOURCE / INVENTORY, PINS[INVENTORY]))
    entries = {row["name"]: row for row in inventory}
    if len(entries) != len(inventory):
        raise ValueError("Duplicate ZIP member identity")
    inputs = [checked_read(SOURCE / "development_metadata" / name, parent_json["metadata"][name]["sha256"])
              .decode("utf-8") for name in METADATA]
    rgb = metadata_parser["image_times"](inputs[1], ".png")
    depth = metadata_parser["image_times"](inputs[2], ".pfm")
    if rgb != depth:
        raise ValueError("Exact RGB/depth timestamp identity required")
    poses = metadata_parser["poses"](inputs[0])
    ids, offsets = metadata_parser["associate"](rgb, [p["timestamp_ns"] for p in poses])
    by_time = {t: poses[i] for t, i in zip(rgb, ids)}
    offsets_by_time = dict(zip(rgb, offsets))
    anchors = [int(x["timestamp_ns"]) for x in parent_json["triplets"]]
    eligible = [t for t in rgb if all(entries.get(sensor_member(t, role), {}).get("complete") is True
                                      for role in PAIR_ROLES)]
    groups = selection(eligible, by_time, anchors)
    timestamps = sorted({int(row[key]) for group in groups for row in group["pairs"]
                         for key in ("reference_timestamp_ns", "query_timestamp_ns")})
    frames = [dict(timestamp_ns=str(t), pose_timestamp_ns=str(by_time[t]["timestamp_ns"]),
                   signed_pose_offset_ns=offsets_by_time[t],
                   members={role: entries[sensor_member(t, role)] for role in PAIR_ROLES})
              for t in timestamps]
    support = [dict(timestamp_ns=str(t), member=entries[sensor_member(t, "semantic")]) for t in anchors]
    return dict(
        schema="lusnar_scene1_temporal_local_extraction_v119", scene=1, split="exposed_development",
        reserved_scenes=[7, 8, 9], reserve_access_allowed=False, geometry_run_allowed=False,
        parent_extraction_freeze_sha256=PARENT_SHA, archive_prefix_bytes=ARCHIVE_BYTES,
        archive_full_sha256_verified=False, inventory_sha256=PINS[INVENTORY], parser_sha256=PINS[PARSER],
        source_sha256=digest(Path(__file__).read_bytes()), eligible_frames=len(eligible),
        window_radius_ns=WINDOW_NS, interval_bounds_ns=list(DT_NS), logged_translation_bands=[list(x) for x in BANDS],
        groups=groups, temporal_frames=frames, stereo_anchor_semantic=support,
    )


def extract(parent, frozen_path, output):
    raw = frozen_path.read_bytes()
    frozen = json.loads(raw)
    if frozen != build(parent):
        raise ValueError("Exact metadata-only frozen protocol required")
    path = SOURCE / ARCHIVE
    if path.is_symlink() or path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Require the exact local Scene 1 prefix")
    output.mkdir(exist_ok=False)

    def write_member(archive, root, timestamp, role, member):
        blob = extract_member(archive, member, ARCHIVE_BYTES)
        dest = root / member["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(blob)
        return dict(timestamp_ns=timestamp, role=role, path=member["name"], sha256=digest(blob),
                    bytes=len(blob), local_zip_crc32=member["crc32"], header=format_header(role, blob))

    stereo_root = output / "stereo_semantic"
    stereo_root.mkdir()
    temporal_root = output / "temporal"
    temporal_root.mkdir()
    with path.open("rb") as archive:
        semantic = [write_member(archive, stereo_root, x["timestamp_ns"], "semantic", x["member"])
                    for x in frozen["stereo_anchor_semantic"]]
        temporal = [write_member(archive, temporal_root, frame["timestamp_ns"], role, member)
                    for frame in frozen["temporal_frames"] for role, member in frame["members"].items()]
    evidence = dict(scene=1, complete=True, freeze_sha256=digest(raw), reserved_scenes_opened=[],
                    parent_extraction_freeze_sha256=PARENT_SHA)
    write_json(stereo_root / "COMPLETE.json", dict(
        evidence, schema="lusnar_scene1_semantic_support_v119", entries=semantic, semantic_sample_arrays_decoded=0))
    write_json(temporal_root / "COMPLETE.json", dict(
        evidence, schema="lusnar_scene1_temporal_support_v119", entries=temporal,
        pixel_or_depth_sample_arrays_decoded=0, temporal_pairs=30, distinct_endpoints=60))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.extract is None:
        write_json(args.freeze, build(args.parent))
    else:
        extract(args.parent, args.freeze, args.extract)


if __name__ == "__main__":
    main()
