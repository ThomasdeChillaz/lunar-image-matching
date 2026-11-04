"""Verify LunaPolaris release sidecars and a bounded timestamp-index window."""

import hashlib
import json
import re
import urllib.request


RECORDS = (
    (15737938, "01", 33827237888, "be6629e277e209de6fb9e479ffd1ec98", 8917, "c98a57bad7d8e53e4f28132c97858401"),
    (15745932, "07", 29695537152, "7dff727dc279bd16e4cead8aaa0b04e3", 9520, "f3098f8e18656b4099603a1ea5de8e26"),
    (15747096, "10", 34111774720, "a26b14273ed087cc7583dad1c29f0ffb", 9520, "03f44576af8ec238be1658b2eb50434d"),
)
PAGE_SIZE = 4096
MAX_PAGES_PER_RECORD = 12
HOST = "https://zenodo.org/api/records/"


def manifest(record):
    with urllib.request.urlopen(HOST + str(record), timeout=12) as response:
        data = json.load(response)
    if data["id"] != record or data["metadata"]["version"] != "1.0.0":
        raise ValueError("Expected exact published record version")
    return data


def file_spec(data, name, size, digest):
    item = next(row for row in data["files"] if row["key"] == name)
    exact = f"{HOST}{data['id']}/files/{name}/content"
    if (item["size"], item["checksum"], item["links"]["self"]) != (size, "md5:" + digest, exact):
        raise ValueError("Published file metadata changed")
    return dict(key=name, size=size, publisher_md5=digest, url=exact)


def sidecar(item):
    with urllib.request.urlopen(item["url"], timeout=12) as response:
        raw = response.read(item["size"] + 1)
    if len(raw) != item["size"] or hashlib.md5(raw).hexdigest() != item["publisher_md5"]:
        raise ValueError("Published sidecar bytes changed")
    return raw, dict(item, sha256=hashlib.sha256(raw).hexdigest())


def uint(raw):
    return int.from_bytes(raw, "big")


def vint(raw, pos):
    value = 0
    for i in range(9):
        byte = raw[pos + i]
        if i == 8:
            return (value << 8) | byte, pos + 9
        value = (value << 7) | (byte & 127)
        if byte < 128:
            return value, pos + i + 1


def record_fields(raw):
    header, pos = vint(raw, 0)
    serial_types = []
    while pos < header:
        code, pos = vint(raw, pos)
        serial_types.append(code)
    fields = []
    for code in serial_types:
        if code in (0, 8, 9):
            fields.append(None if code == 0 else code - 8)
            continue
        lengths = (0, 1, 2, 3, 4, 6, 8)
        if 1 <= code <= 6:
            count = lengths[code]
            fields.append(int.from_bytes(raw[header:header + count], "big", signed=True))
        elif code >= 13 and code % 2:
            count = (code - 13) // 2
            fields.append(raw[header:header + count].decode("utf-8"))
        else:
            raise ValueError("Unexpected record value in metadata index")
        header += count
    if header != len(raw):
        raise ValueError("Truncated SQLite record")
    return fields


def pointers(raw, offset, header_size):
    count = uint(raw[offset+3:offset+5])
    start = offset + header_size
    return [uint(raw[i:i+2]) for i in range(start, start + count*2, 2)]


def schema(raw):
    if raw[:16] != b"SQLite format 3\0" or uint(raw[16:18]) != PAGE_SIZE or raw[18:20] != b"\1\1":
        raise ValueError("Expected checkpointed 4096-byte rollback-format SQLite")
    if raw[100] != 13:
        raise ValueError("Expected local table-leaf sqlite_schema")
    output = {}
    for ptr in pointers(raw, 100, 8):
        count, pos = vint(raw, ptr)
        _, pos = vint(raw, pos)
        if count > 2000:
            raise ValueError("Unexpected overflowing schema")
        kind, name, table, root, sql = record_fields(raw[pos:pos+count])
        output[name] = dict(kind=kind, table=table, root_page=root, sql=sql)
    return output


def index_range(page, root, lower, upper):
    raw = page(root)
    flag = raw[0]
    if flag not in (2, 10):
        raise ValueError("Expected SQLite timestamp-index B-tree")
    interior = flag == 2
    result = []
    for ptr in pointers(raw, 0, 12 if interior else 8):
        child = uint(raw[ptr:ptr+4]) if interior else None
        count, pos = vint(raw, ptr+4 if interior else ptr)
        if not 0 < count <= 40:
            raise ValueError("Expected local timestamp/rowid index records")
        timestamp, rowid = record_fields(raw[pos:pos+count])
        if interior and timestamp >= lower:
            result.extend(index_range(page, child, lower, upper))
        if timestamp >= upper:
            return result
        if timestamp >= lower:
            result.append((timestamp, rowid))
    if interior:
        result.extend(index_range(page, uint(raw[8:12]), lower, upper))
    return result


def one(spec):
    record, sequence, db_size, db_md5, yaml_size, yaml_md5 = spec
    meta = manifest(record)
    archive = file_spec(meta, f"sequence_{sequence}.db3", db_size, db_md5)
    raw, yml = sidecar(file_spec(meta, "metadata.yaml", yaml_size, yaml_md5))
    text = raw.decode("utf-8")
    sections = re.split(r"^    - topic_metadata:\s*$", text, flags=re.M)[1:]
    topics = []
    for chunk in sections:
        name = re.search(r"^        name: (.+)$", chunk, re.M)[1]
        kind = re.search(r"^        type: (.+)$", chunk, re.M)[1]
        count = int(re.search(r"^      message_count: (\d+)$", chunk, re.M)[1])
        topics.append(dict(name=name, type=kind, messages=count))
    total = int(re.search(r"^  message_count: (\d+)$", text, re.M)[1])
    duration = int(re.search(r"^    nanoseconds: (\d+)$", text, re.M)[1])
    start = int(re.search(r"^    nanoseconds_since_epoch: (\d+)$", text, re.M)[1])
    if len(topics) != 17 or total != sum(x["messages"] for x in topics) or 'compression_mode: ""' not in text:
        raise ValueError("Expected exactly one uncompressed, consistently counted ROS bag")
    pages = {}
    identities = {}

    def page(n):
        if n in pages:
            return pages[n]
        if not 1 <= n <= db_size // PAGE_SIZE or len(pages) >= MAX_PAGES_PER_RECORD:
            raise ValueError("Metadata-only byte-range budget exceeded")
        lo, hi = (n-1)*PAGE_SIZE, n*PAGE_SIZE-1
        req = urllib.request.Request(
            archive["url"] + f"?probe=v104-lunapolaris-metadata-page-{n}",
            headers={"Range": f"bytes={lo}-{hi}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(req, timeout=12) as res:
            content_range = res.headers.get("Content-Range")
            if res.status != 206 or content_range != f"bytes {lo}-{hi}/{db_size}":
                raise ValueError("Exact publisher Content-Range required")
            content = res.read(PAGE_SIZE + 1)
            if len(content) != PAGE_SIZE:
                raise ValueError("Exact bounded SQLite page required")
            etag = res.headers.get("ETag")
        pages[n] = content
        identities[n] = dict(content_range=content_range, sha256=hashlib.sha256(content).hexdigest(), etag=etag)
        return content

    first = page(1)
    if uint(first[28:32]) * PAGE_SIZE != db_size:
        raise ValueError("Database header page count differs from official size")
    objects = schema(first)
    index = objects["timestamp_idx"]
    if index["sql"] != "CREATE INDEX timestamp_idx ON messages (timestamp ASC)":
        raise ValueError("Expected exact covering timestamp index")
    lower = start + duration//2
    rows = index_range(page, index["root_page"], lower, lower + 1_000_000_000)
    if not rows or rows != sorted(set(rows)) or any(not lower <= t < lower + 1_000_000_000 for t, _ in rows):
        raise ValueError("Bounded timestamp-only rows failed index ordering/range")
    return dict(
        sequence=sequence, record=record, doi=meta["doi"], version=meta["metadata"]["version"],
        archive=archive, metadata=yml, bag_duration_ns=duration, bag_start_ns=start, messages=total, topics=topics,
        schema=objects, schema_page=identities[1], pages=identities,
        index_window=dict(start_ns=lower, duration_ns=1_000_000_000, count=len(rows),
                          first_timestamp_rowid=rows[0], last_timestamp_rowid=rows[-1]),
    )


def main():
    metadata = manifest(16728838)
    raw, item = sidecar(file_spec(metadata, "calibration_results.json", 4046, "b80161a348720252f5d5335cc0af56d9"))
    calibration = json.loads(raw)
    print(json.dumps(dict(
        scope="Sidecars, SQLite schema and timestamp index only. No image, point cloud, transform or pose payloads fetched.",
        calibration=dict(record=metadata["id"], doi=metadata["doi"], version=metadata["metadata"]["version"],
                         file=item, dimensions=[calibration["width"], calibration["height"]],
                         fields=sorted(calibration)),
        sequences=[one(spec) for spec in RECORDS],
    ), indent=2))


if __name__ == "__main__":
    main()
