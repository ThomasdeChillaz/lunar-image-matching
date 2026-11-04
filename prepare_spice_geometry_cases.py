"""Freeze geometry-only SPICE-HL3 ZED2 C frame choices before image reads."""

from bisect import bisect_right
import csv
from decimal import Decimal
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def norm(vec):
    return math.sqrt(sum(x*x for x in vec))


def separation(a, b):
    return norm([x-y for x,y in zip(a,b)])


def inventory_speed(root, inventory, speed):
    markers_path = root / f"{speed}_stereo_mocap_data.csv"
    camera_path = root / f"{speed}_stereo_left_metadata.csv"
    markers = rows(markers_path)
    indexed = sorted((Decimal(row["Time"]), index, row) for index, row in enumerate(markers))
    times = [entry[0] for entry in indexed]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate motion-capture clock readings")
    for row in markers:
        orientation = [float(row["Orientation"+c]) for c in "XYZW"]
        if not abs(norm(orientation)-1.) <= .001:
            raise ValueError("Invalid motion-capture quaternion norm")
    cameras = rows(camera_path)
    usable = {Decimal(row["Timestamp"]): index for index,row in enumerate(cameras)
              if (row["Width"], row["Height"], row["Encoding"]) == ("640","360","mono8")}
    if len(usable) != len(cameras):
        raise ValueError("Duplicate or inconsistent camera metadata")
    regex = re.compile(rf"{speed}/stereo/left/stereo_left_([0-9.]+)_([0-9]+)\.png")
    candidates = []
    for member, receipt in inventory["selected"].items():
        match = regex.fullmatch(member)
        if match is None:
            continue
        text_time, frame = match.groups()
        image_time = Decimal(text_time)
        if image_time not in usable:
            continue
        high = bisect_right(times, image_time)
        if high < 1 or high == len(times):
            continue
        lo_time, lo_index, lo_row = indexed[high-1]
        hi_time, hi_index, hi_row = indexed[high]
        if hi_time-lo_time > Decimal("0.100"):
            continue
        weight = float((image_time-lo_time)/(hi_time-lo_time))
        position = [(1-weight)*float(lo_row["Position"+c])+weight*float(hi_row["Position"+c]) for c in "XYZ"]
        quat = [float(lo_row["Orientation"+c]) for c in "XYZW"]
        candidates.append(dict(
            id=f"C-{speed}-{int(frame):03d}", timestamp_s=text_time, archive_member=member,
            member_crc32=f"{receipt['crc32']:08x}", member_size=receipt["size"],
            camera_metadata_row=usable[image_time],
            marker_bracket_rows=[lo_index,hi_index], marker_interval_seconds=str(hi_time-lo_time),
            marker_position_m=position, approximate_marker_xyzw=quat,
        ))
    candidates.sort(key=lambda row: Decimal(row["timestamp_s"]))
    if len({x["timestamp_s"] for x in candidates}) != len(candidates):
        raise ValueError("Duplicate selected camera timestamps")
    return candidates, dict(mocap={"path":markers_path.name,"sha256":digest(markers_path)},
                            camera_metadata={"path":camera_path.name,"sha256":digest(camera_path)},
                            metadata_records=len(cameras), marker_records=len(markers))


def angle(a,b):
    dot = abs(sum(x*y for x,y in zip(a,b))) / (norm(a)*norm(b))
    return math.degrees(2*math.acos(min(1.,dot)))


def choose_cases(candidates, speed):
    times = [float(f["timestamp_s"]) for f in candidates]
    chosen = []
    used = set()
    for band, limits in enumerate(((.10,.22),(.22,.44))):
        lo, hi = limits
        pairs = []
        for (i,a),(j,b) in combinations(enumerate(candidates),2):
            delta = times[j]-times[i]
            if not .25<=delta<=8.:
                continue
            gap = separation(a["marker_position_m"], b["marker_position_m"])
            degrees = angle(a["approximate_marker_xyzw"], b["approximate_marker_xyzw"])
            if lo<=gap<hi and degrees<=20.:
                pairs.append((i,j,gap,degrees))
        for third in range(6):
            anchor = times[0] + (times[-1]-times[0])*(third+.5)/6
            available = [p for p in pairs if p[0] not in used and p[1] not in used]
            if not available:
                raise ValueError("Insufficient disjoint endpoints in a frozen baseline band")
            i,j,gap,degrees = min(
                available, key=lambda p: (abs((times[p[0]]+times[p[1]])/2-anchor),
                                           abs(p[2]-(lo+hi)/2),p[0],p[1])
            )
            used.update((i,j))
            chosen.append(dict(id=f"C-{speed}-b{band+1}-{third+1:02d}",
                reference=candidates[i]["id"], query=candidates[j]["id"], speed=speed,
                marker_baseline_m=gap, approximate_marker_rotation_deg=degrees,
                baseline_band_m=list(limits)))
    return chosen, [row for row in candidates if row["id"] in {candidates[index]["id"] for index in used}]


def prepare(root):
    directory = root / "trajectory_C_stereo_directory.json"
    inventory = json.loads(directory.read_text())
    intrinsics = root / "camera_intrinsics.json"
    extrinsics = root / "sensor_extrinsics.json"
    calibration = json.loads(extrinsics.read_text())
    if calibration["zed2-left-to-marker"]["t_d"] != 0.:
        raise ValueError("Metadata selector requires the released zero ZED2 time offset")
    cases, frames, datasets = [],[],{}
    for speed in ("fast","slow"):
        candidates, evidence = inventory_speed(root,inventory,speed)
        selected, assets = choose_cases(candidates,speed)
        for row in assets:
            row["speed"] = speed
        cases.extend(selected)
        frames.extend(assets)
        datasets[speed] = dict(**evidence,eligible_camera_records=len(candidates))
    return dict(
        scope="Supplementary physical lunar-analog geometry diagnostic on one lab site, two traversals. "
              "Case selection uses metadata only and does not claim illumination transfer.",
        source="https://doi.org/10.5281/zenodo.13970078",
        license="CC BY 4.0",
        source_directory=dict(path=directory.name,sha256=digest(directory),
            central_directory_sha256=inventory["central_directory_sha256"]),
        intrinsics=dict(path=intrinsics.name,sha256=digest(intrinsics)),
        extrinsics=dict(path=extrinsics.name,sha256=digest(extrinsics)),
        calibration_convention="Rectified ZED2 left native 640x360. Compose T_world_marker @ inverse(T_camera_marker).",
        camera_clock_rule="Exact timestamp intersection with archived mono8 metadata. Adjacent sorted mocap gap <= 0.100 s.",
        case_selection="Six geometry-only choices per baseline band and speed. All 48 endpoints distinct.",
        selected_optical_payloads_opened=0,
        frames=frames,cases=cases,sequences=datasets,
    )


if __name__ == "__main__":
    base = Path(sys.argv[1])
    data = prepare(base)
    with Path(sys.argv[2]).open("x") as stream:
        json.dump(data,stream,indent=2)
        stream.write("\n")
    print(len(data["cases"]),len(data["frames"]),{k:v["eligible_camera_records"] for k,v in data["sequences"].items()})
