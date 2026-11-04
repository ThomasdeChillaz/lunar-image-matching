"""Bind frozen reference identities for Direct on a materialized ARM worker."""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType, SimpleNamespace

from solar_evaluation_v70 import InferenceInput
from solar_portable_evidence_v75 import file, safe

REFERENCE_PUBLICATION = "b90e2b14308f45c55c8eda6e03dad86febc2d355"
MAX_LOOKUP_METADATA_BYTES = 16 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def git_blob(root, name):
    """The immutable, already-public v80 lookup bytes in the checked-out history."""
    return subprocess.check_output(
        ["git", "-C", str(root), "show", REFERENCE_PUBLICATION + ":" + name],
        timeout=5, stderr=subprocess.DEVNULL,
    )


def _check_pin(path, digest, case_pins, execution_pins):
    safe(path)
    if (not isinstance(digest, str) or not DIGEST.fullmatch(digest)
            or case_pins.get(path) != digest or execution_pins.get(path) != digest):
        raise ValueError("Case and execution must pin the same reference metadata input: " + path)


def reference_identity(case, execution_pins, root=".", public_bytes=git_blob):
    """Authenticate reference-only metadata; return three immutable digests."""
    if case.get("split") != "model_selection":
        raise ValueError("Direct ARM bridge is limited to the frozen development cohort")
    meta = case["reference_asset_metadata"]
    name = safe(case["solar_lookup_protocol"])
    if (meta["solar_lookup_protocol"] != name or meta["unit_index"] != case["unit_index"]
            or meta["component"] != case["component"]
            or meta["reference_geometry"] != case["reference_geometry"]):
        raise ValueError("Original reference unit identity differs")
    case_pins = case["input_sha256"]
    _check_pin(name, meta["solar_lookup_protocol_sha256"], case_pins, execution_pins)
    _check_pin(case["reference_geometry"], meta["reference_geometry_sha256"], case_pins, execution_pins)
    source_geometry = meta["source_reference_geometry_path"]
    mesh = meta["scene_mesh_path"]
    _check_pin(source_geometry, meta["source_reference_geometry_sha256"], case_pins, execution_pins)
    _check_pin(mesh, meta["scene_mesh_sha256"], case_pins, execution_pins)

    path = file(Path(root), name)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_LOOKUP_METADATA_BYTES:
        raise ValueError("Expected bounded public reference metadata JSON")
    raw = path.read_bytes()
    if (len(raw) > MAX_LOOKUP_METADATA_BYTES
            or hashlib.sha256(raw).hexdigest() != meta["solar_lookup_protocol_sha256"]
            or raw != public_bytes(Path(root), name)):
        raise ValueError("Lookup bytes differ from the case hash or original public v80 commit")
    lookup = json.loads(raw)

    expected = dict(
        schema_version=70, cpu_threads=2, query_truth_used=False,
        reference_geometry=source_geometry,
        reference_geometry_sha256=meta["source_reference_geometry_sha256"],
        mesh=mesh, mesh_sha256=meta["scene_mesh_sha256"],
        scene_mesh=mesh, scene_mesh_sha256=meta["scene_mesh_sha256"],
        approved_reference_key=case["component"] + "/" + Path(source_geometry).parent.parent.name,
        geometry_audit_sha256=meta["source_audit_sha256"],
    )
    if lookup.get("query_truth_used") is not False or any(lookup.get(key) != value for key, value in expected.items()):
        raise ValueError("Published lookup does not match the original full reference map and camera")
    _check_pin(lookup["geometry_audit"], meta["source_audit_sha256"], case_pins, execution_pins)
    return MappingProxyType({
        name: expected[name] for name in ("reference_geometry_sha256", "mesh_sha256", "scene_mesh_sha256")
    })


@dataclass(frozen=True)
class BoundDirectArmCase:
    direct: object
    identity: MappingProxyType

    def infer(self, inp):
        if type(inp) is not InferenceInput or inp.known_query_sun is not None or inp.solar_lookup is not None:
            raise ValueError("ARM direct inference requires the primary truth-free materialized input")
        # DirectRenderAdapter reads only .protocol for its existing identity
        # checks. The evaluator sees the original input, with no online provider.
        return self.direct.infer(replace(inp, solar_lookup=SimpleNamespace(protocol=self.identity)))


def bind_case(direct, case, execution_pins, root=".", public_bytes=git_blob):
    return BoundDirectArmCase(direct, reference_identity(case, execution_pins, root, public_bytes))
