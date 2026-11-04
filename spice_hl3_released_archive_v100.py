"""Fetch selected SPICE-HL3 ZIP records without downloading either full archive."""

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import struct
from urllib.request import Request, urlopen
from zlib import crc32, decompressobj


RECORD = "13970078"
SOURCES = {
    "zenodo": f"https://zenodo.org/records/{RECORD}/files/",
    "author_huggingface_mirror": (
        "https://huggingface.co/datasets/UMASpaceRobotics/spice-hl3/"
        "resolve/ad39cd208acf5d0bea44a5363a8381ef75567132/"
    ),
}
ARCHIVES = {
    "trajectory_A.zip": (15686762753, "8d8a475b0ddc1c96c5253703b657d07a"),
    "trajectory_C.zip": (12243101038, "c20a8167c3fb570880e17874aabbfcd5"),
    "calibration.zip": (302301787, "a81b2c7e693b94069e5992c844f0698b"),
}
BYTE_LIMIT = 16 * 1024**2


class RangeReader(io.RawIOBase):
    """Expose an archive to ZipFile without silently downloading a full response."""

    def __init__(self, url, size, *, byte_limit=BYTE_LIMIT):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0
        self.bytes_received = 0
        self.requests = 0
        self.byte_limit = byte_limit

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = origin[whence] + offset
        if self.position < 0:
            raise ValueError("Negative archive seek")
        return self.position

    def read(self, count=-1):
        remaining = max(0, self.size - self.position)
        count = min(count, remaining) if count >= 0 else remaining
        if not count:
            return b""
        if count + self.bytes_received > self.byte_limit or self.requests >= 64:
            raise ValueError("Archive metadata transfer exceeds its budget")
        start, stop = self.position, self.position + count - 1
        expected = f"bytes {start}-{stop}/{self.size}"
        request = Request(self.url, headers={
            "Range": f"bytes={start}-{stop}",
            "Accept-Encoding": "identity",
            "User-Agent": "Lunar research archive metadata inventory",
        })
        self.requests += 1
        with urlopen(request, timeout=25) as response:
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Archive server did not honor the exact requested byte range")
            result = response.read(count + 1)
        if len(result) != count:
            raise ValueError("Archive server returned a different byte count")
        self.position += count
        self.bytes_received += count
        return result

EOCD = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR = struct.Struct("<4sLQL")
ZIP64_EOCD = struct.Struct("<4sQ2H2L4Q")
CENTRAL = struct.Struct("<4s6H3L5H2L")
LOCAL = struct.Struct("<4s5H3L2H")
CHUNK_BYTES = 8 * 1024**2


def directory_info(reader):
    suffix_size = min(reader.size, 65557)
    reader.seek(reader.size - suffix_size)
    tail = reader.read(suffix_size)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0 or len(tail) - pos < EOCD.size:
        raise ValueError("ZIP end record unavailable")
    _, disk, central_disk, n_disk, n_total, size, start, comment = EOCD.unpack_from(tail, pos)
    if pos + EOCD.size + comment != len(tail) or disk != central_disk or disk != 0 or n_disk != n_total:
        raise ValueError("Unsupported split ZIP or invalid end record")
    locator_start = pos - ZIP64_LOCATOR.size
    directory_end = reader.size - len(tail) + pos
    if locator_start >= 0 and tail[locator_start:locator_start + 4] == b"PK\x06\x07":
        _, zip64_disk, eocd_start, disks = ZIP64_LOCATOR.unpack_from(tail, locator_start)
        if zip64_disk != 0 or disks != 1:
            raise ValueError("Unsupported split ZIP64")
        reader.seek(eocd_start)
        marker, body_size, _, _, disk, central_disk, n_disk, n_total, size, start = ZIP64_EOCD.unpack(
            reader.read(ZIP64_EOCD.size)
        )
        if (marker != b"PK\x06\x06" or body_size < 44 or disk != central_disk or disk != 0
                or n_disk != n_total or eocd_start + 12 + body_size != directory_end - ZIP64_LOCATOR.size):
            raise ValueError("Invalid ZIP64 end record")
        directory_end = eocd_start
    if not 0 < n_total <= 2_000_000 or not 0 < start < directory_end or start + size != directory_end:
        raise ValueError("Central directory bounds differ")
    return start, size, n_total


def zip64_values(header, extra):
    disk = header[13]
    values = [header[9], header[8], header[16]]
    extended = [i for i, value in enumerate(values) if value == 0xffffffff]
    if disk != 0:
        raise ValueError("Member belongs to another disk")
    if extended:
        cursor = 0
        while cursor + 4 <= len(extra):
            tag, length = struct.unpack_from("<HH", extra, cursor)
            cursor += 4
            if cursor + length > len(extra):
                raise ValueError("Truncated ZIP extra record")
            if tag == 1:
                if length < 8 * len(extended):
                    raise ValueError("ZIP64 member record is incomplete")
                for index, value in zip(extended, struct.unpack_from("<" + "Q" * len(extended), extra, cursor)):
                    values[index] = value
                return values
            cursor += length
        raise ValueError("Required ZIP64 member record is absent")
    return values


def retain(path, archive):
    suffix = path.rsplit(b".", 1)[-1].lower()
    if suffix in (b"json", b"csv", b"yaml", b"yml", b"txt"):
        return True
    if archive == "trajectory_A.zip":
        return b"/mono/" in path and suffix in (b"jpg", b"png")
    if archive == "trajectory_C.zip":
        return b"/mono/" in path and suffix == b"png"
    return False


def inventory(reader, archive, *, all_stereo_png=False):
    start, size, n_total = directory_info(reader)
    if size > 152 * 1024**2:
        raise ValueError("Archive directory exceeds the declared transfer envelope")
    reader.seek(start)
    pending = b""
    remaining = size
    seen = 0
    selected = {}
    stereo_prefixes = Counter()
    digest = sha256()
    while remaining:
        block = reader.read(min(remaining, CHUNK_BYTES))
        digest.update(block)
        remaining -= len(block)
        data = pending + block
        position = 0
        while len(data) - position >= CENTRAL.size:
            fields = CENTRAL.unpack_from(data, position)
            if fields[0] != b"PK\x01\x02":
                raise ValueError("Invalid ZIP central directory member")
            name_size, extra_size, comment_size = fields[10:13]
            begin = position + CENTRAL.size
            end = begin + name_size + extra_size + comment_size
            if end > len(data):
                break
            seen += 1
            name = data[begin:begin + name_size]
            stereo_png = all_stereo_png and b"/stereo/" in name and name.endswith(b".png")
            if stereo_png:
                stereo_prefixes[name.rsplit(b"/",1)[0].decode("utf-8")] += 1
            if retain(name, archive) or stereo_png:
                path = name.decode("utf-8")
                parsed = PurePosixPath(path)
                if parsed.is_absolute() or ".." in parsed.parts or path in selected:
                    raise ValueError("Invalid retained ZIP member name")
                raw_size, packed_size, offset = zip64_values(
                    fields, data[begin + name_size:begin + name_size + extra_size]
                )
                if offset + LOCAL.size + packed_size > start or fields[3] & 1:
                    raise ValueError("Invalid or encrypted retained member")
                selected[path] = dict(size=raw_size, packed_size=packed_size,
                                      header_offset=offset, method=fields[4], crc32=fields[7])
            position = end
        pending = data[position:]
    if pending or seen != n_total:
        raise ValueError("Incomplete ZIP member inventory")
    result = dict(total_members=seen, central_directory_offset=start,
                central_directory_bytes=size, central_directory_sha256=digest.hexdigest(),
                selected=selected)
    if all_stereo_png:
        result.update(archive=archive, archive_size=reader.size, source=reader.url,
                      prefixes=stereo_prefixes.most_common(30), member_payloads_opened=0)
    return result


def read_member(reader, name, member, max_bytes=2 * 1024**2):
    if not 0 <= member["size"] <= max_bytes or not 0 <= member["packed_size"] <= max_bytes:
        raise ValueError("Selected member exceeds its extraction envelope")
    reader.seek(member["header_offset"])
    prefix = reader.read(LOCAL.size + len(name.encode("utf-8")))
    fields = LOCAL.unpack_from(prefix)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != member["method"]:
        raise ValueError("ZIP local member identity differs")
    name_size, extra_size = fields[-2:]
    if name_size != len(prefix) - LOCAL.size or prefix[LOCAL.size:].decode("utf-8") != name:
        raise ValueError("ZIP local member path differs")
    reader.seek(extra_size, 1)
    packed = reader.read(member["packed_size"])
    if member["method"] == 0:
        raw = packed
    elif member["method"] == 8:
        decoder = decompressobj(-15)
        raw = decoder.decompress(packed, max_bytes + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Selected ZIP member did not fit the decompression envelope")
    else:
        raise ValueError("Unsupported ZIP compression method")
    if len(raw) != member["size"] or crc32(raw) != member["crc32"]:
        raise ValueError("Selected member length or CRC differs")
    return raw



def acquire_metadata(root):
    """Create the reproducible C member catalogue and six calibration/pose inputs."""
    root.mkdir(parents=True, exist_ok=False)
    calib = ("calibration/camera_intrinsics.json", "calibration/sensor_extrinsics.json")
    for archive_name in ("trajectory_C.zip", "calibration.zip"):
        size, _ = ARCHIVES[archive_name]
        is_camera = archive_name == "trajectory_C.zip"
        with RangeReader(SOURCES["zenodo"] + archive_name, size, byte_limit=17 * 1024**2) as reader:
            directory = inventory(reader, archive_name, all_stereo_png=is_camera)
            if is_camera:
                with (root / "trajectory_C_stereo_directory.json").open("x") as f:
                    json.dump(directory, f, indent=2)
                    f.write("\n")
                members = [(f"{speed}/stereo/{name}", f"{speed}_stereo_{name}")
                           for speed in ("fast", "slow")
                           for name in ("mocap_data.csv", "left_metadata.csv")]
            else:
                members = [(name, PurePosixPath(name).name) for name in calib]
            for name, target in members:
                raw = read_member(reader, name, directory["selected"][name])
                with (root / target).open("xb") as f:
                    f.write(raw)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--metadata-output", type=Path, required=True)
    acquire_metadata(cli.parse_args().metadata_output)
