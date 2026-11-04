"""Read-only Python/libm replay of frozen lunar archive catalogue metadata."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

MAX_VECTOR_ROUNDOFF = 2e-15


def read_json(path):
    with Path(path).open("r") as source:
        return json.load(source)


def expected_catalogue(latitude):
    result = []
    for di in range(-6, 7):
        for hi in range(-72, 73):
            hour, decl = 1.25 * hi, 0.25 * di
            p, h, d = map(math.radians, (latitude, hour, decl))
            sun = [
                -math.cos(d) * math.sin(h),
                math.cos(p) * math.sin(d) - math.sin(p) * math.cos(d) * math.cos(h),
                math.sin(p) * math.sin(d) + math.cos(p) * math.cos(d) * math.cos(h),
            ]
            if sun[2] > 0:
                result.append([hour, decl, hi % 4 == 0 and di % 2 == 0, sun])
    return result


def vector_distance(first, second):
    if not isinstance(first, list) or len(first) != 3 or not all(
        type(component) is float and math.isfinite(component) for component in first
    ):
        return math.inf
    if first[2] <= 0 or abs(math.hypot(*first) - 1.0) > MAX_VECTOR_ROUNDOFF:
        return math.inf
    return max(abs(a - b) for a, b in zip(first, second))


def compare(rows, expected, availability):
    geometry = lambda table: [(row[0], row[1], row[2]) for row in table]
    exact_membership = (
        len(rows) == len(expected)
        and all(len(row) == 4 and type(row[2]) is bool for row in rows)
        and geometry(rows) == geometry(expected)
    )
    exact_availability = len(availability) == len(rows) and all(
        materialized is True for materialized in availability
    )
    deltas = [vector_distance(row[3], truth[3]) for row, truth in zip(rows, expected)]
    mismatch_rows = sum(delta != 0 for delta in deltas)
    max_abs = max(deltas, default=0)
    return {
        "catalogue_membership_order_and_coarse_exact": exact_membership,
        "all_availability_exact": exact_availability,
        "historical_vector_equality_pass": exact_membership and exact_availability and mismatch_rows == 0,
        "bounded_vector_check_pass": exact_membership and exact_availability and max_abs <= MAX_VECTOR_ROUNDOFF,
        "vector_mismatch_rows": mismatch_rows,
        "max_abs_vector_delta": max_abs if math.isfinite(max_abs) else None,
    }


def export(root, output):
    import numpy as np

    evidence_path = root / "reports/solar-modern-arm-v98/portable-reference/evidence.json"
    protocol_path = root / "reports/solar-modern-arm-v98/pilot-protocol.json"
    evidence, protocol = read_json(evidence_path), read_json(protocol_path)
    contract = read_json(root / evidence["contract"])
    manifest = read_json(root / protocol["manifest"])
    pilot_ids = set(protocol["pilot_case_ids"])
    pilot_references = {
        case["reference_geometry"] for case in manifest["cases"] if case["case_id"] in pilot_ids
    }
    archives = {}
    for index in contract["selection_units"]:
        unit = contract["reference_units"][str(index)]
        path, = [
            path for path in evidence["result"]["reference_archives"]
            if f"/u{index:03d}" in path
        ]
        receipt_path = path + ".json"
        receipt = read_json(root / receipt_path)
        catalogue = receipt["sun_catalogue"]
        expected_parameters = [[row["hour_angle_deg"], row["subsolar_latitude_deg"]] for row in catalogue]
        expected_suns = [row["sun_enu"] for row in catalogue]
        expected_coarse = [row["coarse"] for row in catalogue]
        with np.load(root / path, allow_pickle=False) as archive:
            assert archive["sun_vectors"].dtype == np.float64
            assert archive["coarse_sun_mask"].dtype == archive["solar_materialized_mask"].dtype == np.bool_
            assert np.array_equal(archive["sun_parameters"], expected_parameters)
            assert np.array_equal(archive["sun_vectors"], expected_suns)
            assert np.array_equal(archive["coarse_sun_mask"], expected_coarse)
            assert np.array_equal(archive["solar_materialized_mask"], np.ones(len(catalogue), bool))
        rows = [
            [parameters[0], parameters[1], coarse, sun]
            for parameters, coarse, sun in zip(expected_parameters, expected_coarse, expected_suns)
        ]
        catalogue_bytes = json.dumps(rows, separators=(",", ":")).encode()
        catalogue_hash = hashlib.sha256(catalogue_bytes).hexdigest()
        record = archives.setdefault((unit["latitude_deg"], catalogue_hash), {
            "latitude": unit["latitude_deg"], "rows": rows, "catalogue_sha256": catalogue_hash,
            "archives": [],
        })
        record["archives"].append({
            "unit": index, "pilot_reference": path in pilot_references,
            "archive_sha256": evidence["pinned_inputs"][path],
            "receipt_sha256": evidence["pinned_inputs"][receipt_path],
        })
    fixtures = {"records": list(archives.values()), "source": "frozen run 41926059385"}
    data = json.dumps(fixtures, sort_keys=True, separators=(",", ":")).encode()
    with Path(output).open("xb") as dest:
        dest.write(gzip.compress(data, mtime=0))
    print(json.dumps({
        "records": len(archives), "archives": sum(len(record["archives"]) for record in archives.values()),
        "pilot_reference_archives": len(pilot_references), "gzip_bytes": Path(output).stat().st_size,
        "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
    }))


def run_controls(expected):
    import copy

    def passes(rows, mask):
        return compare(rows, expected, mask)["bounded_vector_check_pass"]

    baseline = [True] * len(expected)
    assert passes(expected, baseline)
    altered = copy.deepcopy(expected)
    altered[0][3][0] = math.nextafter(altered[0][3][0], math.inf)
    assert passes(altered, baseline) and not compare(altered, expected, baseline)["historical_vector_equality_pass"]
    altered[0][3][0] += 1e-10
    assert not passes(altered, baseline)
    altered = copy.deepcopy(expected)
    altered[0][3][0] = math.nan
    assert not passes(altered, baseline)
    assert not passes(expected[:-1], baseline[:-1])
    assert not passes(expected + [expected[-1]], baseline + [True])
    assert not passes(expected[::-1], baseline)
    altered = copy.deepcopy(expected)
    altered[0][0] += 1.25
    assert not passes(altered, baseline)
    altered = copy.deepcopy(expected)
    altered[0][2] = not altered[0][2]
    assert not passes(altered, baseline)
    assert not passes(expected, [False] + baseline[1:])
    assert not passes(expected, [1] + baseline[1:])


def run(fixture, require_linux_arm):
    machine = platform.machine()
    os_name = platform.system()
    if require_linux_arm and (os_name != "Linux" or machine not in ("aarch64", "arm64")):
        raise SystemExit("Use a real Linux ARM worker for the cross-platform probe")
    with gzip.open(fixture, "rt") as source:
        records = json.load(source)["records"]
    outcomes = []
    for item in records:
        expected = expected_catalogue(item["latitude"])
        run_controls(expected)
        rows = item["rows"]
        data = json.dumps(rows, separators=(",", ":")).encode()
        assert hashlib.sha256(data).hexdigest() == item["catalogue_sha256"]
        outcome = compare(rows, expected, [True] * len(rows))
        outcome.update({
            "latitude": item["latitude"],
            "units": [archive["unit"] for archive in item["archives"]],
            "pilot_reference_units": [archive["unit"] for archive in item["archives"] if archive["pilot_reference"]],
            "stored_row_count": len(rows), "recomputed_row_count": len(expected),
            "smallest_archived_up_component": min(row[3][2] for row in rows),
        })
        outcomes.append(outcome)
    print(json.dumps({
        "system": os_name, "machine": machine, "python": platform.python_version(),
        "libc": platform.libc_ver(), "fixture_sha256": hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
        "atol": MAX_VECTOR_ROUNDOFF,
        "controls_passed": True, "catalogues": outcomes,
    }, indent=2, allow_nan=False))
    if not all(outcome["bounded_vector_check_pass"] for outcome in outcomes):
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    exporter = subcommands.add_parser("export")
    exporter.add_argument("root", type=Path)
    exporter.add_argument("output", type=Path)
    probe = subcommands.add_parser("run")
    probe.add_argument("fixture", type=Path)
    probe.add_argument("--require-linux-arm", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        export(args.root, args.output)
        return 0
    return run(args.fixture, args.require_linux_arm)


if __name__ == "__main__":
    sys.exit(main())
