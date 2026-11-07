"""Acquire only the SPICE-HL3 members selected by the metadata freeze."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from struct import unpack_from
import sys

from spice_hl3_released_archive_v100 import ARCHIVES, RangeReader, SOURCES, read_member


def digest(raw):
    return sha256(raw).hexdigest()


def write_once(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def acquire(root, schedule_path):
    schedule = json.loads(schedule_path.read_text())
    if schedule["selected_optical_payloads_opened"] != 0:
        raise ValueError("Metadata-only case selection required before extraction")
    directory_path = root / schedule["source_directory"]["path"]
    raw_directory = directory_path.read_bytes()
    inventory = json.loads(raw_directory)
    if (digest(raw_directory) != schedule["source_directory"]["sha256"]
            or inventory["central_directory_sha256"] != schedule["source_directory"]["central_directory_sha256"]):
        raise ValueError("ZIP member inventory changed after frame selection")
    frames = schedule["frames"]
    used = [case[which] for case in schedule["cases"] for which in ("reference","query")]
    if len(used) != 48 or len(set(used)) != 48 or set(used) != {f["id"] for f in frames}:
        raise ValueError("Exact distinct optical endpoints required")
    source = SOURCES["zenodo"] + "trajectory_C.zip"
    size, _ = ARCHIVES["trajectory_C.zip"]
    if inventory["source"] != source or inventory["archive_size"] != size:
        raise ValueError("Different released archive")
    out_dir = root / "frames"
    out_dir.mkdir(exist_ok=False)

    def stream(speed):
        result = []
        with RangeReader(source, size, byte_limit=12*1024**2) as reader:
            for frame in [x for x in frames if x["speed"] == speed]:
                name = frame["archive_member"]
                member = inventory["selected"][name]
                if (not re.fullmatch(rf"C-{speed}-[0-9]{{3}}", frame["id"])
                        or f"{member['crc32']:08x}" != frame["member_crc32"]
                        or member["size"] != frame["member_size"]):
                    raise ValueError("Selected member identity changed")
                raw = read_member(reader, name, member, max_bytes=512*1024)
                if (raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
                        or unpack_from(">IIBB", raw, 16) != (640,360,8,0)):
                    raise ValueError(f"Unexpected original mono8 PNG header for {frame['id']}")
                rel = f"frames/{frame['id']}.png"
                write_once(root / rel, raw)
                result.append(dict(id=frame["id"], path=rel, sha256=digest(raw),
                                   stream_id=speed,timestamp_s=frame["timestamp_s"],
                                   camera_metadata_row=frame["camera_metadata_row"],
                                   marker_bracket_rows=frame["marker_bracket_rows"],
                                   archive_member=name))
        return result,reader.bytes_received

    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = list(pool.map(stream, ("fast","slow")))
    result = dict(schema="spice-hl3-optical-transfer-v100",trajectory="C",camera_key="zed2:left",
                  track="qualified_mocap",prepared_only=True,calibration_or_final_access=False,
                  calibration=schedule["intrinsics"],extrinsics=schedule["extrinsics"],
                  time_convention="marker_seconds = camera_seconds + published_t_d",
                  crop=[0,0,640,360],scope=schedule["scope"],source=schedule["source"],
                  license=schedule["license"],
                  metadata_schedule=dict(path=schedule_path.name,sha256=digest(schedule_path.read_bytes())),
                  acquisition=dict(original_release_native_grayscale_png=True,pixel_arrays_decoded=0,
                                   network_bytes_read=sum(receipt[1] for receipt in streams)),
                  mocap_streams={key:{asset:data[asset] for asset in ("mocap","camera_metadata")}
                                 for key,data in schedule["sequences"].items()},
                  frames=[frame for group,_ in streams for frame in group],
                  cases=[{key:case[key] for key in ("id","reference","query")} for case in schedule["cases"]])
    return result


if __name__ == "__main__":
    root = Path(sys.argv[1])
    result = acquire(root, root/"trajectory_C_metadata_cases_v2.json")
    target = root/"spice_hl3_c_geometry_transfer_prepared.json"
    write_once(target,(json.dumps(result,indent=2)+"\n").encode())
    print(len(result["cases"]),len(result["frames"]),result["acquisition"])
