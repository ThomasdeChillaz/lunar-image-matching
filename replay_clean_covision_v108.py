"""Independently replay v108 with explicit patch indexing and rank interpolation."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def verified(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Asset checksum differs: {path}")
    return raw


def linear_image(item):
    pixels = cv2.imdecode(np.frombuffer(verified(item["path"], item["sha256"]), dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED)
    if pixels.shape != (512, 512, 3) or pixels.dtype != np.uint16:
        raise ValueError("Expected unchanged native RGB16")
    image = pixels @ np.array([.114, .587, .299])
    image /= 65535
    linear = image / 12.92
    high = image > .04045
    linear[high] = ((image[high] + .055) / 1.055) ** 2.4
    return linear


def contrasts(image, xy):
    """Use explicit order statistics instead of the measurement's quantile API."""
    result = []
    for x, y in xy:
        samples = image[y-16:y+17, x-16:x+17].flatten()
        ranks = (54, 55, 1033, 1034, 1077, 1078)
        samples = np.partition(samples, ranks)
        low = samples[54]*.6 + samples[55]*.4
        hi95 = samples[1033]*.4 + samples[1034]*.6
        hi99 = samples[1077]*.88 + samples[1078]*.12
        result.append([hi95 - low, hi99 - low])
    return np.array(result)


def compare(report, left, right, left_xy, right_xy, sigma, include_tiles):
    for index, name in enumerate(("p95_minus_p5", "p99_minus_p5")):
        for multiple in (1, 3, 6):
            positive_l, positive_r = (side[:, index] >= sigma*multiple for side in (left, right))
            both = positive_l & positive_r
            expected = report[name][str(multiple)]
            got = {
                "reference": int(positive_l.sum()), "query": int(positive_r.sum()), "both": int(both.sum()),
                "either": int((positive_l | positive_r).sum()), "neither": int((~(positive_l | positive_r)).sum()),
            }
            if include_tiles and index == 0 and multiple == 3:
                for role, points in (("reference", left_xy), ("query", right_xy)):
                    tiles = [[0]*4 for _ in range(4)]
                    for x, y in points[both]:
                        tiles[y//128][x//128] += 1
                    got[role + "_tiles_yx"] = tiles
            if got != expected:
                raise ValueError(f"Independent count differs: {name} at {multiple} sigma")


def inside(xy):
    x, y = xy
    return 16 <= x <= 495 and 16 <= y <= 495


def main(protocol_path, result_path, out):
    raw = Path(result_path).read_bytes()
    result = json.loads(raw)
    protocol = json.loads(verified(protocol_path, result["protocol_sha256"]))
    cv2.setNumThreads(2)
    for item, row in zip(protocol["items"], result["rows"], strict=True):
        if item["case_id"] != row["case_id"]:
            raise ValueError("Case order differs")
        seal = item["scoring_labels"]
        with np.load(BytesIO(verified(seal["path"], seal["sha256"])), allow_pickle=False) as labels:
            p, q, vis = [labels[k] for k in ("reference_xy", "query_xy", "visible")]
        projected = [(tuple(int(x+.5) for x in left), tuple(int(x+.5) for x in right))
                     for left, right, valid in zip(p, q, vis, strict=True) if valid]
        aligned = [(a, b) for a, b in projected if inside(a) and inside(b)]
        shifts = [tuple(v) for v in protocol["query_offsets_pixels"]]
        common = [(a, (x, y)) for a, (x, y) in aligned if all(inside((x+dx, y+dy)) for dx, dy in shifts)]
        if [len(p), len(projected), len(aligned), len(common)] != [
                row[k] for k in ("grid_anchors", "exact_visible_centres", "aligned_supported", "common_supported")]:
            raise ValueError("Geometry support differs")
        ref, query = [linear_image(item["exposures"][role]) for role in ("reference", "query")]
        a, b = (np.array(v) for v in zip(*aligned, strict=True))
        sigma = protocol["linear_noise_sigma"]
        compare(row["aligned"], contrasts(ref, a), contrasts(query, b), a, b, sigma, True)
        a, b = (np.array(v) for v in zip(*common, strict=True))
        r = contrasts(ref, a)
        for (dx, dy), control in zip(shifts, row["common"], strict=True):
            if list((dx, dy)) != control["query_offset_xy"]:
                raise ValueError("Shift order differs")
            b_shifted = b + [dx, dy]
            compare(control["overlap"], r, contrasts(query, b_shifted), a, b_shifted, sigma, False)
    audit = dict(schema="solar_clean_covision_replay_v108", passed=True, cases_replayed=len(result["rows"]),
                 comparison_tables_replayed=6*len(result["rows"]),
                 method="Explicit patch extraction and rank interpolation, independent support and contrast counts.",
                 result_sha256=sha256(raw).hexdigest(), protocol_sha256=result["protocol_sha256"],
                 calibration_or_final_access=False)
    with Path(out).open("x") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
