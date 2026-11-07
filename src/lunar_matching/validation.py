"""Reference-only native input contracts, independent of the historical imports."""
from copy import deepcopy
from pathlib import Path
import re
import struct

import cv2
import numpy as np

from .artifacts import read_json, sha256

CASE_KEYS = {"id", "reference_geometry", "catalogue", "query_image", "query_intrinsics"}
FILE_KEYS = {"path", "sha256", "bytes", "bit_depth", "native_shape"}
REFERENCE_KEYS = {"position", "normals", "valid", "axial_depth", "cam2world", "intrinsics",
                  "geometry_source", "body_origin", "enu_rows"}
REQUIRED_REFERENCE = {"valid", "axial_depth", "cam2world", "intrinsics"}


def intrinsics(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-8)
            or not np.isclose(matrix[1, 0], 0, atol=1e-8)):
        raise ValueError("Finite calibrated 3x3 pixel intrinsics are required.")
    return matrix


def camera_to_world(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("A finite rigid camera-to-world pose is required.")
    return matrix


def checked_file(record: dict, base: Path) -> Path:
    if (not isinstance(record, dict) or not {"path", "sha256"} <= set(record) <= FILE_KEYS
            or not isinstance(record["path"], str)
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise ValueError("An input path and lowercase SHA-256 are required; extra fields are forbidden.")
    path = Path(record["path"])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or changed input: {path.name}")
    if "bytes" in record and (type(record["bytes"]) is not int or record["bytes"] != path.stat().st_size):
        raise ValueError("The declared input length differs.")
    return path


def native_rgb16(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
            or struct.unpack(">IIBB", header[16:26]) != (512, 512, 16, 2)):
        raise ValueError("The image must be a native 512x512 RGB16 PNG; RGB8 conversion is not equivalent.")
    pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint16 or pixels.shape != (512, 512, 3):
        raise ValueError("The native RGB16 image cannot be decoded completely.")


def reference_geometry(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        if not REQUIRED_REFERENCE <= set(data.files) <= REFERENCE_KEYS:
            raise ValueError("Reference geometry only: query pose, depth and scoring labels are forbidden.")
        valid, depth = data["valid"], data["axial_depth"]
        if (valid.shape != (512, 512) or valid.dtype != bool or depth.shape != valid.shape
                or not np.isfinite(depth[valid]).all() or (depth[valid] <= 0).any()):
            raise ValueError("Native axial depth and an explicit Boolean reference-validity mask are required.")
        intrinsics(data["intrinsics"])
        camera_to_world(data["cam2world"])


def prepare_case(manifest: Path, case_id: str, base: Path) -> dict:
    entries = read_json(manifest)
    if (not isinstance(entries, list) or not entries
            or any(not isinstance(row, dict) or set(row) != CASE_KEYS
                   or not isinstance(row["id"], str) for row in entries)
            or len({row["id"] for row in entries}) != len(entries)):
        raise ValueError("Distinct inference-only cases are required; query truth is forbidden.")
    selected = [row for row in entries if row["id"] == case_id]
    if len(selected) != 1:
        raise ValueError("The requested case is absent from the manifest.")
    case = deepcopy(selected[0])
    intrinsics(case["query_intrinsics"])
    catalogue = case["catalogue"]
    if not isinstance(catalogue, list) or not 1 <= len(catalogue) <= 24:
        raise ValueError("Between one and 24 reference lighting images are required.")
    ids = set()
    for entry in catalogue:
        if not isinstance(entry, dict) or set(entry) != {"light_id", "sun_world", "image"}:
            raise ValueError("Only reference lighting and its image are permitted in the catalogue.")
        if type(entry["light_id"]) is not int or entry["light_id"] in ids:
            raise ValueError("Reference light IDs must be distinct integers.")
        ids.add(entry["light_id"])
        sun = np.asarray(entry["sun_world"], dtype=float)
        if sun.shape != (3,) or not np.isfinite(sun).all() or not np.isclose(np.linalg.norm(sun), 1, atol=1e-5):
            raise ValueError("Reference Sun directions must be finite world-frame unit vectors.")
    for record in [case["query_image"], *[row["image"] for row in catalogue]]:
        path = checked_file(record, base)
        if ("bit_depth" in record and record["bit_depth"] != 16
                or "native_shape" in record and record["native_shape"] not in ([512, 512], [512, 512, 3])):
            raise ValueError("Image metadata disagrees with native RGB16 input.")
        native_rgb16(path)
        record["path"] = str(path)
    record = case["reference_geometry"]
    path = checked_file(record, base)
    reference_geometry(path)
    record["path"] = str(path)
    return case
