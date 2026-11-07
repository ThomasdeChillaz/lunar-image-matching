"""Freeze and extract five complete development-only LuSNAR Scene 1 triplets."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import runpy
import struct
import zlib


WORK = Path(__file__).resolve().parent.parent
SOURCE = WORK / "lusnar-metadata"
ARCHIVE = "Moon_1.zip.part"
ARCHIVE_BYTES = 4003463168
INVENTORY = "scene1_partial_member_inventory.json"
PROVENANCE = "scene1_recovered_metadata_provenance.json"
PARSER = "lusnar_metadata_audit_v46.py"
PINS = {
    INVENTORY: "870f26a469288a00622fe1c2e91bf3c1868728d97bce01ae9b5196a53d49fb4b",
    PROVENANCE: "db9768169c1c01cb3269d12b39682fb8b6f5b0d70405ff5d3c11096de21901f8",
    PARSER: "4bf292bb1a7dd93213adf47c1395ecf098236922784c945c3f14bc8d2f4df6de",
}
METADATA = (
    "Moon_1/gt.txt",
    "Moon_1/image0/color_timestamp.txt",
    "Moon_1/image0/depth_timestamp.txt",
)
ROLE = {
    "left": ("Moon_1/image0/color", ".png"),
    "right": ("Moon_1/image1/color", ".png"),
    "depth": ("Moon_1/image0/depth", ".pfm"),
}
LOCAL_HEADER = struct.Struct("<I5H3I2H")
MEMBER_LIMIT = 8 * 1024**2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_read(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Frozen source digest differs: {path.name}")
    return raw


def write_json(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def triplet_names(timestamp):
    return {role: f"{folder}/{timestamp}{suffix}" for role, (folder, suffix) in ROLE.items()}


def build_freeze():
    inventory = json.loads(checked_read(SOURCE / INVENTORY, PINS[INVENTORY]))
    provenance = json.loads(checked_read(SOURCE / PROVENANCE, PINS[PROVENANCE]))
    checked_read(WORK / PARSER, PINS[PARSER])
    parser = runpy.run_path(str(WORK / PARSER))
    if provenance["archive_prefix_bytes"] != ARCHIVE_BYTES:
        raise ValueError("Archive prefix size differs")
    entries = {row["name"]: row for row in inventory}
    if len(entries) != len(inventory):
        raise ValueError("Duplicate inventory member names")
    metadata = {row["member"]: row for row in provenance["members"]}
    contents = {}
    for name in METADATA:
        row = metadata[name]
        raw = checked_read(SOURCE / "development_metadata" / name, row["sha256"])
        if len(raw) != row["bytes"] or row["member_crc_verified"] is not True:
            raise ValueError("Recovered metadata lacks complete byte verification")
        contents[name] = raw.decode("utf-8")
    pose_rows = parser["poses"](contents[METADATA[0]])
    color = parser["image_times"](contents[METADATA[1]], ".png")
    depth = parser["image_times"](contents[METADATA[2]], ".pfm")
    if color != depth:
        raise ValueError("Left image and depth timestamps differ")
    eligible = []
    for timestamp in color:
        if all(entries.get(name, {}).get("complete") is True for name in triplet_names(timestamp).values()):
            eligible.append(timestamp)
    if len(eligible) != 966:
        raise ValueError("Unexpected complete triplet population")
    selected = [eligible[(len(eligible) - 1) * quartile // 4] for quartile in range(5)]
    pose_ids, offsets = parser["associate"](selected, [row["timestamp_ns"] for row in pose_rows])
    triplets = []
    for timestamp, pose_id, offset in zip(selected, pose_ids, offsets):
        triplets.append({
            "timestamp_ns": str(timestamp),
            "pose_timestamp_ns": str(pose_rows[pose_id]["timestamp_ns"]),
            "signed_pose_offset_ns": offset,
            "members": {role: entries[name] for role, name in triplet_names(timestamp).items()},
        })
    return {
        "schema": "lusnar_scene1_exact_local_extraction_v117",
        "source_sha256": {**PINS, Path(__file__).name: digest(Path(__file__).read_bytes())},
        "archive_prefix": {"name": ARCHIVE, "bytes": ARCHIVE_BYTES, "complete_archive_verified": False},
        "scene": 1,
        "split": "exposed_development",
        "reserved_scenes": [7, 8, 9],
        "reserve_access_allowed": False,
        "geometry_run_allowed": False,
        "eligible_triplets": len(eligible),
        "selection": "Sorted exact integer timestamps; floor((N-1)*q/4), q=0,1,2,3,4",
        "triplets": triplets,
        "metadata": {name: {"member": entries[name], "sha256": metadata[name]["sha256"]} for name in METADATA},
    }


def extract_member(archive, row, prefix_bytes):
    """Read one bounded local member without a ZIP central directory."""
    name = row["name"]
    offset, data_offset = row["offset"], row["data_offset"]
    size, compressed = row["bytes"], row["compressed"]
    if (
        row["complete"] is not True or row["method"] not in (0, 8)
        or not 0 < size <= MEMBER_LIMIT or not 0 < compressed <= MEMBER_LIMIT
        or not 0 <= offset < data_offset <= prefix_bytes - compressed
    ):
        raise ValueError("Incomplete, unsupported or oversized local ZIP member")
    archive.seek(offset)
    header = archive.read(LOCAL_HEADER.size)
    if len(header) != LOCAL_HEADER.size:
        raise ValueError("Truncated ZIP header")
    magic, _, flags, method, _, _, crc, zipped, expanded, name_len, extra_len = LOCAL_HEADER.unpack(header)
    if (
        magic != 0x04034B50 or flags not in (0, 0x800) or extra_len != 0
        or offset + LOCAL_HEADER.size + name_len != data_offset
        or (method, crc, zipped, expanded) != (row["method"], row["crc32"], compressed, size)
        or archive.read(name_len).decode("utf-8") != name
    ):
        raise ValueError(f"Local ZIP header differs from inventory: {name}")
    payload = archive.read(compressed)
    if len(payload) != compressed:
        raise ValueError("Truncated ZIP member payload")
    if method == 8:
        inflater = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        raw = inflater.decompress(payload, size + 1)
        if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise ValueError("Deflate stream does not end at frozen member boundary")
    else:
        raw = payload
    if len(raw) != size or zlib.crc32(raw) != crc:
        raise ValueError(f"Expanded size or CRC differs: {name}")
    return raw


def format_header(role, raw):
    """Inspect structural headers and byte counts without decoding image samples."""
    if role != "depth":
        if len(raw) < 33 or raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
            raise ValueError("Expected PNG signature and 13-byte first IHDR")
        if zlib.crc32(raw[12:29]) != int.from_bytes(raw[29:33], "big"):
            raise ValueError("Bad PNG IHDR checksum")
        width, height, bits, color, compression, filters, interlace = struct.unpack(">IIBBBBB", raw[16:29])
        if (width, height) != (1024, 1024) or compression != 0 or filters != 0 or interlace not in (0, 1):
            raise ValueError("Unexpected PNG header")
        return dict(format="PNG", width=width, height=height, bit_depth=bits, color_type=color, interlaced=bool(interlace))
    file = io.BytesIO(raw)
    if file.readline(8).strip() != b"Pf":
        raise ValueError("Expected grayscale PFM")
    width, height = map(int, file.readline(64).split())
    signed_scale = float(file.readline(64))
    pixel_bytes = len(raw) - file.tell()
    if (width, height) != (1024, 1024) or not math.isfinite(signed_scale) or signed_scale == 0 or pixel_bytes != width * height * 4:
        raise ValueError("Unexpected PFM header or float32 sample footprint")
    return dict(format="PFM", width=width, height=height, storage_endian="little" if signed_scale < 0 else "big",
                scale_magnitude=abs(signed_scale), sample_storage_bytes=4, sample_payload_bytes=pixel_bytes)


def extract(freeze_path, output):
    frozen_raw = freeze_path.read_bytes()
    frozen = json.loads(frozen_raw)
    if frozen != build_freeze():
        raise ValueError("Freeze differs from the exact source or inventory policy")
    archive_path = SOURCE / ARCHIVE
    if archive_path.is_symlink() or archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Exact local Scene 1 prefix required")
    output.mkdir(exist_ok=False)
    rows = []
    with archive_path.open("rb") as archive:
        work = [(None, "metadata", record["member"], record["sha256"]) for record in frozen["metadata"].values()]
        work.extend((triplet["timestamp_ns"], role, member, None)
                    for triplet in frozen["triplets"] for role, member in triplet["members"].items())
        for timestamp, role, member, expected in work:
            raw = extract_member(archive, member, ARCHIVE_BYTES)
            checksum = digest(raw)
            header = None
            if role == "metadata":
                if checksum != expected:
                    raise ValueError("Recovered metadata digest differs")
            else:
                header = format_header(role, raw)
            path = output / member["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as file:
                file.write(raw)
            rows.append(dict(timestamp_ns=timestamp, role=role, path=member["name"], sha256=checksum, bytes=len(raw),
                             local_zip_crc32=member["crc32"], header=header))
    write_json(output / "COMPLETE.json", dict(
        schema="lusnar_scene1_extraction_receipt_v117", scene=1, complete=True, freeze_sha256=digest(frozen_raw),
        entries=rows, complete_archive_verified=False, image_sample_arrays_decoded=0, optical_matches_computed=0,
        reserved_scenes_opened=[], depth_units_verified=False, axial_vs_ray_verified=False, camera_pose_verified=False,
        geometry_gate_status="pending_separate_frozen_development_run",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("output", type=Path)
    unpack = commands.add_parser("extract")
    unpack.add_argument("freeze", type=Path)
    unpack.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        write_json(args.output, build_freeze())
    else:
        extract(args.freeze, args.output)


if __name__ == "__main__":
    main()
