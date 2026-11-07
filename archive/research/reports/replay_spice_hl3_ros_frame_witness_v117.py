"""Decode a bounded prefix of the original Trajectory C ROS 2 bags for frame evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from urllib.request import Request, urlopen
import zlib


SOURCE = "https://zenodo.org/records/13970078/files/trajectory_C.zip"
SIZE = 12243101038
PREFIX_LIMIT = 3 * 1024**2
EXPANDED_PREFIX_LIMIT = 8 * 1024**2
ANCHORS = {
    # (central-directory local-header offset, compressed size, uncompressed CRC)
    "fast": (1874494588, 763, 1370862673),
    "slow": (11491584248, 775, 3065545014),
}
LOCAL = struct.Struct("<4s5H3L2H")
ROS = "/leo01/zed2/zed_node/"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


class BoundedArchive:
    def __init__(self):
        self.ranges = []

    def read(self, start, count):
        stop = start + count - 1
        require(0 <= start <= stop < SIZE
                and sum(item["bytes"] for item in self.ranges) + count <= 2 * PREFIX_LIMIT + 4096,
                "Read is outside the approved bounded ZIP ranges")
        request = Request(SOURCE, headers={
            "Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity",
            "User-Agent": "SPICE-HL3 ROS calibration witness",
        })
        with urlopen(request, timeout=25) as response:
            content_range = response.headers.get("Content-Range")
            require(response.status == 206 and content_range == f"bytes {start}-{stop}/{SIZE}",
                    "Source did not honor exact range")
            raw = response.read(count + 1)
        require(len(raw) == count, "Source returned a different byte count")
        self.ranges.append(dict(start=start, stop=stop, bytes=count, sha256=digest(raw)))
        return raw


def member_header(block, offset):
    sig, _, flags, method, _, _, crc, packed, raw, name_len, extra_len = LOCAL.unpack_from(block, offset)
    require(sig == b"PK\x03\x04" and flags == 0 and method == 8, "Unexpected local ZIP member envelope")
    begin = offset + LOCAL.size
    end = begin + name_len + extra_len
    require(end <= len(block), "Truncated local ZIP member envelope")
    return block[begin:begin + name_len].decode(), end, packed, raw, crc


def archive_topics(raw):
    topics = {}
    for chunk in raw.decode().split("    - topic_metadata:")[1:]:
        name = re.search(r"^        name: (.+)$", chunk, re.M)[1]
        datatype = re.search(r"^        type: (.+)$", chunk, re.M)[1]
        count = int(re.search(r"^      message_count: (\d+)$", chunk, re.M)[1])
        topics[name] = {"type": datatype, "messages": count}
    return topics


def mcap_string(data, position):
    count, = struct.unpack_from("<I", data, position)
    end = position + 4 + count
    require(end <= len(data), "Truncated MCAP string")
    return data[position + 4:end].decode(), end


def records(data, position, allow_truncated):
    while position < len(data):
        if len(data) - position < 9:
            require(allow_truncated, "Truncated complete MCAP record header")
            return
        opcode, count = struct.unpack_from("<BQ", data, position)
        position += 9
        if position + count > len(data):
            require(allow_truncated, "Truncated complete MCAP record")
            return
        yield opcode, data[position:position + count]
        position += count


def collect_mcap(data, position=8, channels=None, messages=None, chunk_receipts=None):
    top = channels is None
    if top:
        require(data[:8] == b"\x89MCAP0\r\n", "MCAP header differs")
        channels, messages, chunk_receipts = {}, {}, []
    for opcode, raw in records(data, position, allow_truncated=top):
        if opcode == 6:
            first, last, count, crc = struct.unpack_from("<QQQI", raw)
            compression, p = mcap_string(raw, 28)
            length, = struct.unpack_from("<Q", raw, p)
            payload = raw[p + 8:]
            require(not compression and len(payload) == length == count and (not crc or zlib.crc32(payload) == crc),
                    "Unsupported compressed or corrupt MCAP chunk")
            chunk_receipts.append(dict(first_log_ns=first, last_log_ns=last, size=count, crc32=crc,
                                       sha256=digest(payload)))
            collect_mcap(payload, 0, channels, messages, chunk_receipts)
        if opcode == 4:
            channel, _ = struct.unpack_from("<HH", raw)
            name, p = mcap_string(raw, 4)
            encoding, _ = mcap_string(raw, p)
            require(channel not in channels or channels[channel] == (name, encoding), "MCAP channel changed")
            channels[channel] = name, encoding
        if opcode == 5:
            channel, _, log_time, publish_time = struct.unpack_from("<HIQQ", raw)
            topic, encoding = channels[channel]
            if topic.startswith("/tf") or topic.startswith("/vrpn_mocap/") or topic.startswith(ROS):
                require(encoding == "cdr", "ROS message is not CDR")
                messages.setdefault(topic, []).append((log_time, publish_time, raw[22:]))
    return messages, chunk_receipts


def cdr_align(position, width):
    return position + (-(position - 4) % width)


def cdr_string(data, position):
    position = cdr_align(position, 4)
    count, = struct.unpack_from("<I", data, position)
    start = position + 4
    end = start + count
    require(1 <= count <= 300 and end <= len(data) and data[end - 1] == 0, "Invalid CDR string")
    return data[start:end - 1].decode(), end


def header(data, position=4):
    position = cdr_align(position, 4)
    seconds, nanos = struct.unpack_from("<iI", data, position)
    frame, end = cdr_string(data, position + 8)
    return dict(header_stamp_ns=seconds * 10**9 + nanos, frame_id=frame), end


def doubles(data, position, count):
    if not count:
        return (), position
    position = cdr_align(position, 8)
    return struct.unpack_from("<" + "d" * count, data, position), position + 8 * count


def transform(data, position):
    prefix, end = header(data, position)
    child, end = cdr_string(data, end)
    values, end = doubles(data, end, 7)
    return dict(prefix, child_frame_id=child, translation_m=values[:3], quaternion_xyzw=values[3:]), end


def camera_info(data, prefix, end):
    position = cdr_align(end, 4)
    height, width = struct.unpack_from("<II", data, position)
    model, end = cdr_string(data, position + 8)
    position = cdr_align(end, 4)
    count, = struct.unpack_from("<I", data, position)
    require(0 <= count <= 14, "Camera distortion count is unexpected")
    distortion, end = doubles(data, position + 4, count)
    intrinsics, end = doubles(data, end, 9)
    rectification, end = doubles(data, end, 9)
    projection, _ = doubles(data, end, 12)
    return dict(prefix, image_height=height, image_width=width, distortion_model=model,
                D=distortion, K=intrinsics, R=rectification, P=projection)


def decode(topic, message):
    log, publish, data = message
    require(data[:4] == b"\x00\x01\x00\x00", "Unexpected ROS 2 serialization endianness")
    if topic.startswith("/tf"):
        count, = struct.unpack_from("<I", data, 4)
        require(count <= 100, "Transform count exceeds bound")
        fields, position = [], 8
        for _ in range(count):
            field, position = transform(data, position)
            fields.append(field)
        decoded = {"transforms": fields}
    elif topic.endswith("left_cam_imu_transform"):
        decoded, _ = transform(data, 4)
    else:
        decoded, position = header(data)
        if topic.endswith("camera_info"):
            decoded = camera_info(data, decoded, position)
    return dict(decoded, log_time_ns=log, publish_time_ns=publish, payload_sha256=digest(data))


def unique_edges(decoded):
    unique = {}
    for item in decoded:
        for tf in item["transforms"]:
            key = (tf["frame_id"], tf["child_frame_id"], tf["translation_m"], tf["quaternion_xyzw"])
            unique.setdefault(key, dict(tf, first_payload_sha256=item["payload_sha256"]))
    return list(unique.values())


def inspect(archive, speed):
    name = f"{speed}/stereo/rosbag/metadata.yaml"
    offset, expected_packed, expected_crc = ANCHORS[speed]
    block = archive.read(offset, 2048)
    recorded, position, packed, raw_size, crc = member_header(block, 0)
    require(recorded == name and packed == expected_packed and crc == expected_crc and raw_size < 10000,
            "Known central-directory metadata anchor changed")
    raw = zlib.decompress(block[position:position + packed], -15)
    require(len(raw) == raw_size and zlib.crc32(raw) == crc, "ROS metadata size or CRC differs")
    bag, begin, packed, raw_size, crc = member_header(block, position + packed)
    require(bag == f"{speed}/stereo/rosbag/trajectoryC_stereo_{speed}.mcap"
            and packed > PREFIX_LIMIT, "Unexpected adjacent MCAP ZIP member")
    # Validate every fully present MCAP chunk. The final incomplete chunk is skipped.
    prefix = archive.read(offset + begin, PREFIX_LIMIT)
    mcap = zlib.decompressobj(-15).decompress(prefix, EXPANDED_PREFIX_LIMIT)
    messages, chunks = collect_mcap(mcap)
    decoded = {topic: [decode(topic, message) for message in records]
               for topic, records in messages.items()}
    static = decoded["/tf_static"]
    topics = archive_topics(raw)
    require(0 < len(static) <= topics["/tf_static"]["messages"], "Invalid static transform coverage")
    mocap = "/vrpn_mocap/ZED2_camera_marker/pose"
    require(mocap in topics, "Missing camera marker pose in recorded topic inventory")
    required = [ROS + "left/camera_info", ROS + "left_cam_imu_transform", ROS + "imu/data", ROS + "imu/data_raw"]
    if speed == "slow":
        required.extend((mocap, ROS + "left_gray/camera_info", ROS + "right_gray/camera_info"))
    for topic in required:
        require(topic in decoded and decoded[topic], "Missing required recorded topic: " + topic)
    selected = {}
    for topic, values in decoded.items():
        selected[topic] = {"messages_in_verified_prefix": len(values), "first": values[0],
                           "distinct_header_frames": sorted({value["frame_id"] for value in values if "frame_id" in value})}
    return dict(metadata_member=name, metadata_sha256=digest(raw), metadata_crc32=expected_crc,
                archive_topic_inventory=topics, mcap_member=bag, mcap_payload_zip_offset=offset + begin,
                mcap_zip_member_compressed_bytes=packed, mcap_zip_member_uncompressed_bytes=raw_size,
                mcap_full_zip_crc32_unverified=crc, bounded_inflated_bytes=len(mcap), verified_chunks=chunks,
                compressed_prefix_sha256=digest(prefix), decoded_topics=selected,
                static_messages_all_observed=len(static) == topics["/tf_static"]["messages"],
                observed_static_edges=unique_edges(static),
                observed_dynamic_edges=unique_edges(decoded.get("/tf", [])))


def evidence():
    archive = BoundedArchive()
    runs = {speed: inspect(archive, speed) for speed in ANCHORS}
    return dict(schema="spice_hl3_trajectory_C_ros_frame_witness_v117", archive_url=SOURCE,
                archive_bytes=SIZE, archive_published_md5="c20a8167c3fb570880e17874aabbfcd5",
                archive_full_md5_verified=False, total_range_bytes=sum(item["bytes"] for item in archive.ranges),
                ranges=archive.ranges, recordings=runs,
                scope="Original range-decoded ROS message headers and static/dynamic transform observations. No image matching, calibration fit, or replacement of the v101 nominal ground truth.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = evidence()
    text = json.dumps(value, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        with args.output.open("x") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    main()
