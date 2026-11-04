"""Deterministic, allowlisted Direct data supplement over the v110 ARM closure."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

from solar_capsule_v75 import MAX_FILES, MAX_MEMBER_BYTES, allowed
from solar_portable_evidence_v75 import file, pin
from solar_public_transport_v98 import PATH as HISTORICAL_CONTRACT, TEXT_SUFFIXES, scan_public_text
from solar_remote_relay_v98 import RELEASE_URL
from solar_capsule_overlay_v110 import BASE
from plan_solar_matched_arm_v113 import ARM_PROTOCOL, COHORT, DIRECT_TRANSPORT_PREFLIGHT, pinned

SCHEMA = "solar_direct_arm_transport_v113"
RELEASE_TAG = "solar-arm-direct-v113-transport"
ASSET = "solar-arm-direct-v113-supplement.zip"
ASSET_URL = "https://github.com/ThomasdeChillaz/lunar-image-matching/releases/download/" + RELEASE_TAG + "/" + ASSET
MAX_SUPPLEMENT_BYTES = 160 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
BLOCK = 1024 * 1024


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def hash_path(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    protocol = pinned(root, ARM_PROTOCOL)
    preflight = pinned(root, DIRECT_TRANSPORT_PREFLIGHT)
    cohort = pinned(root, COHORT)
    if (
        protocol["pilot_case_ids"] != cohort["pilot_case_ids"] or len(protocol["case_ids"]) != 384
        or preflight["original_reference_cameras"] != 32 or preflight["exact_native_reference_pngs"] != 768
    ):
        raise ValueError("Original eight-case qualification and complete reference library required")
    base = protocol["input_sha256"] | protocol["source_sha256"]
    direct = preflight["direct_inputs"] | preflight["direct_sources"]
    if any(base[name] != direct[name] for name in base.keys() & direct.keys()):
        raise ValueError("Direct supplement would overwrite an inherited scientific pin")
    additions = {name: digest for name, digest in preflight["direct_inputs"].items() if name not in base}
    sources = {name: digest for name, digest in preflight["direct_sources"].items() if name not in base}
    if len(additions) != 677 or len(sources) != 7 or len(additions) > MAX_FILES:
        raise ValueError("Original Direct-only additions have changed")
    return dict(sorted(additions.items())), dict(sorted(sources.items()))


def _zip_info(name):
    row = zipfile.ZipInfo(name)
    row.compress_type = zipfile.ZIP_STORED
    row.create_system = 3
    row.external_attr = (stat.S_IFREG | 0o644) << 16
    return row


def _stream(source, target, expected, limit, check):
    digest = hashlib.sha256()
    total = 0
    while block := source.read(BLOCK):
        check()
        total += len(block)
        if total > limit:
            raise ValueError("Direct member exceeds its exact frozen byte count")
        digest.update(block)
        target.write(block)
    if total != limit or digest.hexdigest() != expected:
        raise ValueError("Direct member bytes differ from their frozen size and hash")


def build(root, output, receipt):
    root, output, receipt = map(Path, (root, output, receipt))
    root = root.resolve()
    if output.resolve().is_relative_to(root) or receipt.resolve().is_relative_to(root):
        raise ValueError("Build the supplement and receipt outside the scientific checkout")
    if output.exists() or receipt.exists() or output.name != ASSET:
        raise FileExistsError("Use one fresh exact supplemental archive and receipt")
    additions, sources = inventory(root)
    historical = json.loads(file(root, HISTORICAL_CONTRACT).read_bytes())
    members = {}
    for name, digest in additions.items():
        allowed(name, "input")
        source = pin(root, name, digest)
        size = source.stat().st_size
        if not 0 < size <= MAX_MEMBER_BYTES:
            raise ValueError("Direct member size outside original transport cap")
        if name.endswith(TEXT_SUFFIXES):
            scan_public_text(name, source.read_bytes(), historical)
        members[name] = {"bytes": size, "sha256": digest}
    for name, digest in sources.items():
        source = pin(root, name, digest)
        allowed(name, "source")
        scan_public_text(name, source.read_bytes(), historical)
    logical = sum(row["bytes"] for row in members.values())
    if logical >= MAX_SUPPLEMENT_BYTES:
        raise ValueError("Supplement exceeds its separate limit")
    manifest = {
        "schema": SCHEMA, "base_archive_sha256": BASE["archive_sha256"],
        "v110_protocol_sha256": ARM_PROTOCOL[1], "cohort_sha256": COHORT[1],
        "preflight_sha256": DIRECT_TRANSPORT_PREFLIGHT[1], "files": members,
    }
    manifest_bytes = encode(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("Direct manifest exceeds its metadata cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", allowZip64=False) as archive:
        archive.writestr(_zip_info("manifest.json"), manifest_bytes)
        for name, row in members.items():
            with file(root, name).open("rb") as source, archive.open(_zip_info("payload/" + name), "w") as target:
                _stream(source, target, row["sha256"], row["bytes"], lambda: None)
    result = {
        "schema": SCHEMA, "publication": False, "reservation": False, "scientific_execution": False,
        "asset": {"url": ASSET_URL, "bytes": output.stat().st_size, "sha256": hash_path(output)},
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "logical_bytes": logical, "input_members": len(members), "additional_git_source_pins": sources,
    }
    validate_binding(result["asset"], result["manifest_sha256"])
    with receipt.open("xb") as stream:
        stream.write(encode(result))
    return result


def validate_binding(part, manifest_sha256):
    if (
        set(part) != {"url", "bytes", "sha256"} or part["url"] != ASSET_URL
        or RELEASE_URL.fullmatch(part["url"]) is None or type(part["bytes"]) is not int
        or not 0 < part["bytes"] <= MAX_SUPPLEMENT_BYTES
        or any(not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
               for digest in (part["sha256"], manifest_sha256))
    ):
        raise ValueError("Exact bounded Direct supplemental release identity required")


def extract(archive_path, destination, part, manifest_sha256, additions, check=lambda: None):
    validate_binding(part, manifest_sha256)
    archive_path, destination = map(Path, (archive_path, destination))
    if archive_path.stat().st_size != part["bytes"] or hash_path(archive_path) != part["sha256"]:
        raise ValueError("Direct archive differs from its exact published bytes")
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != {"manifest.json", *("payload/" + name for name in additions)}:
            raise ValueError("Direct supplement has duplicate, missing or extra entries")
        for entry in entries:
            if (
                entry.is_dir() or entry.compress_type != zipfile.ZIP_STORED
                or not stat.S_ISREG(entry.external_attr >> 16)
                or PurePosixPath(entry.filename).as_posix() != entry.filename
            ):
                raise ValueError("Only stored regular allowlisted supplemental files are permitted")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Direct supplement manifest exceeds its bound")
        raw = archive.read("manifest.json")
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Direct supplemental manifest hash changed")
        manifest = json.loads(raw)
        if (
            set(manifest) != {"schema", "base_archive_sha256", "v110_protocol_sha256", "cohort_sha256", "preflight_sha256", "files"}
            or manifest["schema"] != SCHEMA or manifest["base_archive_sha256"] != BASE["archive_sha256"]
            or manifest["v110_protocol_sha256"] != ARM_PROTOCOL[1] or manifest["cohort_sha256"] != COHORT[1]
            or manifest["preflight_sha256"] != DIRECT_TRANSPORT_PREFLIGHT[1]
            or set(manifest["files"]) != set(additions)
        ):
            raise ValueError("Supplement manifest provenance or closure changed")
        for name, row in manifest["files"].items():
            allowed(name, "input")
            if (
                set(row) != {"bytes", "sha256"} or row["sha256"] != additions[name]
                or type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_MEMBER_BYTES
                or archive.getinfo("payload/" + name).file_size != row["bytes"]
            ):
                raise ValueError("Supplement member does not bind the original input pin")
        if sum(row["bytes"] for row in manifest["files"].values()) >= MAX_SUPPLEMENT_BYTES:
            raise ValueError("Supplement expands above its exact safety cap")
        destination.mkdir(parents=True, exist_ok=False)
        for name, row in manifest["files"].items():
            target = file(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("payload/" + name) as source, target.open("xb") as output:
                _stream(source, output, row["sha256"], row["bytes"], check)
    return {"members": len(additions), "manifest_sha256": manifest_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repository, args.archive, args.receipt)
    print(json.dumps({key: result[key] for key in ("asset", "manifest_sha256", "logical_bytes", "input_members")}, indent=2))


if __name__ == "__main__":
    main()
