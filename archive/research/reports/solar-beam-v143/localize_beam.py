"""Experimental persistent-beam command; existing matching commands are unchanged.

Only inference-case metadata and reference geometry are accepted. Public pose is
null until calibration; --research-output is required to expose raw hypotheses.
Any agent-run lunar experiment additionally requires its admitted study protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/"solar-blind-v139"))
from beam_matcher_v143 import POLICIES, run
from catalogue_v138 import Catalogue
from engine_v139 import decode_clean16

CASE_KEYS = {"id", "reference_geometry", "catalogue", "query_image", "query_intrinsics"}
REFERENCE_KEYS = {"position", "normals", "valid", "axial_depth", "cam2world", "intrinsics",
                  "geometry_source", "body_origin", "enu_rows"}
REQUIRED_REFERENCE_KEYS = {"valid", "axial_depth", "cam2world", "intrinsics"}


def validate_case(case):
    if not isinstance(case, dict) or set(case) != CASE_KEYS or not isinstance(case["catalogue"], list):
        raise ValueError("Inference-only case required; query pose/depth/Sun metadata is forbidden")
    if not 1 <= len(case["catalogue"]) <= 24 or any(set(c) != {"light_id", "sun_world", "image"} for c in case["catalogue"]):
        raise ValueError("Explicit reference catalogue images and shared reference-world lighting required")
    return case


def checked_file(record):
    if (not isinstance(record, dict) or not {"path", "sha256"} <= set(record)
        or not set(record) <= {"path", "sha256", "bit_depth", "bytes", "native_shape"}):
        raise ValueError("Input path and exact SHA256 required")
    path = Path(record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Inference input hash differs")
    return path


def load_case(case):
    validate_case(case)
    catalogue = Catalogue(np.array([decode_clean16(checked_file(c["image"])) for c in case["catalogue"]]),
        np.array([c["sun_world"] for c in case["catalogue"]]), tuple(c["light_id"] for c in case["catalogue"]))
    query = decode_clean16(checked_file(case["query_image"]))
    with np.load(checked_file(case["reference_geometry"]), allow_pickle=False) as geometry:
        if not REQUIRED_REFERENCE_KEYS <= set(geometry.files) <= REFERENCE_KEYS:
            raise ValueError("Reference-only geometry archive required; query truth fields are forbidden")
        depth, k, pose, valid = (geometry[n].copy() for n in ("axial_depth", "intrinsics", "cam2world", "valid"))
    return catalogue, query, depth, k, pose, np.asarray(case["query_intrinsics"]), valid


def field_count(text):
    try:
        count = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Integer native-field budget required") from error
    if not 0 <= count <= 128:
        raise argparse.ArgumentTypeError("Native-field budget must be between0and128")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field-budget", type=field_count, default=128, metavar="0..128")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--research-output", action="store_true")
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    if not isinstance(entries, list) or len({r["id"] for r in entries}) != len(entries):
        raise ValueError("Distinct inference-case IDs required")
    selected = [r for r in entries if r["id"] == args.case_id]
    if len(selected) != 1:
        raise ValueError("One matching inference case required")
    cat, query, depth, k, pose, qk, valid = load_case(selected[0])
    result = run(cat, query, depth, k, pose, qk, reference_valid=valid,
        height_direction_reference=pose[:3, :3].T @ np.array([0., 0., 1.]),
        policies=tuple(args.policies), field_budget=args.field_budget, research_output=args.research_output)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False); output.write("\n")
