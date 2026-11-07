"""Public transport projection of one frozen v75 execution capsule."""

import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import zipfile

from solar_capsule_v75 import (
    MAX_FILES, MAX_JSON_BYTES, MAX_MANIFEST_BYTES, MAX_MEMBER_BYTES, MAX_TOTAL_BYTES,
    RELEASE_ASSET_MAX_BYTES, file, read, safe, sha, validate_capsule,
)
from solar_portable_evidence_v75 import git_oid
from solar_weight_transport_v96 import (
    BLOCK, DISK_HEADROOM, LOCK, _extract_member, declaration, download_weight,
    required_members, validate_declaration,
)

SOURCE = "solar_public_transport_v98.py"
PATH = "reports/solar_reference_contract_v80.json"
PUBLIC_BLOB = dict(
    commit="b90e2b14308f45c55c8eda6e03dad86febc2d355",
    git_blob="76b0886d09a70379e8b7cf1e7439b7ee194c0cf7",
    bytes=163568,
    sha256="918e4a77630c995c4448d9440b3d09fccaf74fc6cea0d52e245696705321be98",
)
REPO_RAW = "https://raw.githubusercontent.com/ThomasdeChillaz/lunar-image-matching/"
NOTICE_ROOT = "reports/solar-modern-arm-v98/redistribution-notices/"
NOTICE_LOCK = NOTICE_ROOT + "NOTICE_SOURCES.json"
NOTICE_LOCK_SHA256 = "d029e08a0ffa0ed4f15955e8a6b2b01d6fb69f793b7ce4f34273ddfeb68bed95"
EXECUTABLE_FIELDS = ("python_executable", "blender_executable")
JOURNAL_FIELD = "canonical_journal_absolute"
TEXT_SUFFIXES = (".json", ".py", ".md", ".txt", ".yml", ".yaml")
SECRET_PATTERNS = re.compile(
    rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})"
    rb"|-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"
    rb"|(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
)
MACHINE_PATH = re.compile(rb"(?:/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+\\)")



def private_bindings(contract, evidence_path):
    return {
        **{(PATH, name): contract[name] for name in (*EXECUTABLE_FIELDS, JOURNAL_FIELD)},
        **{(evidence_path, "historical_executable_identities/" + name + "/path"): contract[name]
           for name in EXECUTABLE_FIELDS},
        (evidence_path, "original_canonical_absolute_identity"): contract[JOURNAL_FIELD],
    }


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def verify_contract_blob(data):
    if (len(data) != PUBLIC_BLOB["bytes"] or sha256(data).hexdigest() != PUBLIC_BLOB["sha256"]
            or git_oid("blob", data) != PUBLIC_BLOB["git_blob"]):
        raise ValueError("Public original v80 contract bytes differ from the exact Git blob pin")
    result = json.loads(data)
    if any(not isinstance(result.get(name), str) or not result[name].startswith("/")
           for name in (*EXECUTABLE_FIELDS, JOURNAL_FIELD)):
        raise ValueError("The pinned public object lacks its expected historical fields")
    return result


def _touch(holder, key, original, encode):
    expected, replacement = (original, None) if encode else (None, original)
    if key not in holder or holder[key] != expected:
        raise ValueError("Unexpected source or public projection of an opaque identity")
    holder[key] = replacement


def project_evidence(evidence, contract, encode):
    if (evidence["contract"] != PATH or evidence["pinned_inputs"].get(PATH) != PUBLIC_BLOB["sha256"]
            or set(evidence["historical_executable_identities"]) != set(EXECUTABLE_FIELDS)):
        raise ValueError("Only the frozen v80 historical evidence can be projected")
    for name in EXECUTABLE_FIELDS:
        row = evidence["historical_executable_identities"][name]
        if row["sha256"] != contract[name + "_sha256"]:
            raise ValueError("Historical executable identity digest changed")
        _touch(row, "path", contract[name], encode)
    _touch(evidence, "original_canonical_absolute_identity", contract[JOURNAL_FIELD], encode)


def project_manifest(manifest, contract, encode):
    evidence_path = manifest["portable_reference_evidence"]["path"]
    expected = private_bindings(contract, evidence_path)
    rows = manifest["opaque_completed_provenance"]
    observed = {(row["file"], row["pointer"]) for row in rows}
    if len(rows) != len(expected) or observed != set(expected):
        raise ValueError("An unreviewed path appeared in the proposed public manifest")
    for row in rows:
        _touch(row, "value", expected[row["file"], row["pointer"]], encode)


def notice_pins(root):
    if sha(file(root, NOTICE_LOCK)) != NOTICE_LOCK_SHA256:
        raise ValueError("The reviewed redistribution notice inventory changed")
    notice = read(file(root, NOTICE_LOCK))
    pins = {NOTICE_LOCK: NOTICE_LOCK_SHA256}
    for name, entry in notice["files"].items():
        if "/" in name or safe(name) != name:
            raise ValueError("Unexpected legal notice filename")
        path = NOTICE_ROOT + name
        target = file(root, path)
        if target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]:
            raise ValueError("Missing or changed exact redistribution notice: " + name)
        pins[path] = entry["sha256"]
    for path, digest in notice["separately_copied_license_sha256"].items():
        if sha(file(root, path)) != digest:
            raise ValueError("Separately copied upstream agreement differs")
        pins[path] = digest
    return pins


def validate_release_inputs(protocol, root):
    if (protocol.get("external_public_git_blob_v98") != {PATH: PUBLIC_BLOB}
            or protocol["input_sha256"].get(PATH) != PUBLIC_BLOB["sha256"]
            or protocol["source_sha256"].get(SOURCE) != sha(__file__)):
        raise ValueError("Exact known public original Git blob pointer and source required")
    for path, digest in notice_pins(root).items():
        if protocol["input_sha256"].get(path) != digest:
            raise ValueError("Redistribution notice missing from frozen execution inventory: " + path)


def _private_value_present(data, contract):
    return any(contract[name].encode() in data for name in (*EXECUTABLE_FIELDS, JOURNAL_FIELD))


def scan_public_text(path, content, contract):
    """A release preflight, separate from scientific path/byte validation."""
    if _private_value_present(content, contract) or SECRET_PATTERNS.search(content) or MACHINE_PATH.search(content):
        raise ValueError("Unexpected secret or machine path in public text: " + path)


def make_thin_zip(capsule_directory, output_path):
    """A public archive transports no literal local paths or model checkpoints."""
    root = Path(capsule_directory)
    payload = root / "payload"
    manifest = validate_capsule(root)
    protocol = read(file(payload, manifest["protocol_path"]))
    external = validate_declaration(protocol, read(file(payload, LOCK)))
    validate_release_inputs(protocol, payload)
    members = required_members(manifest, protocol)
    evidence_path = manifest["portable_reference_evidence"]["path"]
    evidence = read(file(payload, evidence_path))
    contract = verify_contract_blob(file(payload, PATH).read_bytes())
    if json_bytes(manifest) != (root / "manifest.json").read_bytes() or json_bytes(evidence) != file(payload, evidence_path).read_bytes():
        raise ValueError("Original v75 serialization must round trip exactly")
    for name, entry in external.items():
        if file(payload, name).stat().st_size != entry["bytes"]:
            raise ValueError("Original staged checkpoint size differs")
    logical_manifest_sha = sha(root / "manifest.json")
    project_evidence(evidence, contract, True)
    project_manifest(manifest, contract, True)
    clean_evidence, clean_manifest = json_bytes(evidence), json_bytes(manifest)
    scan_public_text(evidence_path, clean_evidence, contract)
    scan_public_text("manifest.json", clean_manifest, contract)
    for object_id, obj in evidence["publication_proof"]["objects"].items():
        scan_public_text("public Git proof object " + object_id,
                         base64.b64decode(obj["data_base64"], validate=True), contract)
    authorization_path = manifest["pre_reservation_authorization"]["path"]
    authorization = read(file(payload, authorization_path))
    scan_public_text("decoded canonical journal",
                     base64.b64decode(authorization["canonical_journal_bytes_base64"], validate=True), contract)
    omitted = set(external) | {PATH}
    packed = sorted(set(members) - omitted - {evidence_path})
    for name in packed:
        if name.endswith(TEXT_SUFFIXES):
            scan_public_text(name, file(payload, name).read_bytes(), contract)
    output = Path(output_path)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("manifest.json", clean_manifest)
        archive.writestr("payload/" + evidence_path, clean_evidence)
        for name in packed:
            archive.write(file(payload, name), "payload/" + name)
    size = output.stat().st_size
    if size > RELEASE_ASSET_MAX_BYTES:
        raise ValueError("Thin archive exceeds the existing single release asset limit")
    return dict(path=str(output), bytes=size, sha256=sha(output), manifest_sha256=logical_manifest_sha,
                public_projected_manifest_sha256=sha256(clean_manifest).hexdigest(),
                external_weight_count=len(external), public_git_blob_count=1)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Exact pinned raw public Git blob URL redirected")


def download_public_contract(check, remaining_seconds):
    url = REPO_RAW + PUBLIC_BLOB["commit"] + "/" + PATH
    opener = build_opener(ProxyHandler({}), NoRedirect())
    request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "solar-public-blob-v98"})
    check()
    try:
        with opener.open(request, timeout=max(.001, min(10., remaining_seconds()))) as response:
            if (response.status != 200 or response.geturl() != url
                    or response.headers.get("Content-Encoding") not in (None, "identity")):
                raise ValueError("Original public Git blob was not served directly")
            data = bytearray()
            while chunk := response.read1(min(BLOCK, PUBLIC_BLOB["bytes"] + 1 - len(data))):
                check()
                data.extend(chunk)
                if len(data) > PUBLIC_BLOB["bytes"]:
                    raise ValueError("Original public Git blob exceeds its exact length")
    except URLError:
        raise ConnectionError("Exact original public Git blob unavailable") from None
    check()
    return bytes(data)


def _write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(content)


def hydrate_thin_zip(archive_path, destination, manifest_sha256, protocol_sha256, check, remaining_seconds):
    root = Path(destination)
    payload = root / "payload"
    if root.exists() or root.is_symlink():
        raise FileExistsError("Preserve existing capsule directory")
    if Path(archive_path).stat().st_size > RELEASE_ASSET_MAX_BYTES:
        raise ValueError("Thin ZIP exceeded one release asset")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = {item.filename for item in infos}
        if (len(infos) != len(names) or len(infos) > MAX_FILES + 1 or "manifest.json" not in names
                or any(safe(item.filename) != item.filename or item.is_dir() or
                       (item.external_attr >> 16) & 0o170000 == 0o120000 for item in infos)):
            raise ValueError("Thin ZIP contains unexpected members")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
            raise ValueError("Public ZIP manifest exceeds its original byte limit")
        raw_manifest = archive.read("manifest.json")
        manifest = json.loads(raw_manifest)
        if (manifest.get("schema_version") != 75 or manifest.get("complete") is not True
                or manifest.get("capsule_kind") != "model_selection_primary_full_sun"
                or manifest["protocol_sha256"] != protocol_sha256):
            raise ValueError("Original v75 capsule and ticket protocol identity required")
        external = declaration()
        omitted = set(external) | {PATH}
        members = manifest["files"]
        evidence_path = safe(manifest["portable_reference_evidence"]["path"])
        for name in members:
            safe(name)
        expected = {"manifest.json"} | {"payload/" + name for name in members.keys() - omitted}
        if names != expected or not omitted <= members.keys() or len(members) > MAX_FILES or evidence_path in omitted:
            raise ValueError("Only the exact four checkpoints and one public Git blob may be absent")
        packed = [item for item in infos if item.filename != "manifest.json"]
        if any(item.file_size > MAX_MEMBER_BYTES or item.filename.endswith(".json") and
               item.file_size > MAX_JSON_BYTES for item in packed):
            raise ValueError("Public ZIP member exceeded original v75 bound")
        projected_size = sum(item.file_size for item in packed)
        missing_size = PUBLIC_BLOB["bytes"] + sum(entry["bytes"] for entry in external.values())
        total = manifest["total_bytes"]
        if type(total) is not int or not projected_size + missing_size <= total <= MAX_TOTAL_BYTES:
            raise ValueError("Complete hydrated payload exceeds its original size bound")
        if shutil.disk_usage(root.parent).free < 2 * total + DISK_HEADROOM:
            raise OSError("Insufficient free disk for isolated hydration and verified installation")

        root.mkdir()
        for name in sorted(members.keys() - omitted - {evidence_path}):
            _extract_member(archive, "payload/" + name, file(payload, name), members[name], check)
        raw_evidence = archive.read("payload/" + evidence_path)

    if members.get(manifest["protocol_path"]) != protocol_sha256:
        raise ValueError("Downloaded protocol differs from the one-use ticket")
    protocol = read(file(payload, manifest["protocol_path"]))
    validate_declaration(protocol, read(file(payload, LOCK)))
    validate_release_inputs(protocol, payload)
    required_members(manifest, protocol)
    original_blob = download_public_contract(check, remaining_seconds)
    contract = verify_contract_blob(original_blob)
    if any(_private_value_present(data, contract) for data in (raw_evidence, raw_manifest)):
        raise ValueError("An opaque identity appeared in the public JSON")
    evidence = json.loads(raw_evidence)
    project_evidence(evidence, contract, False)
    project_manifest(manifest, contract, False)
    legacy_evidence, legacy_manifest = json_bytes(evidence), json_bytes(manifest)
    if (sha256(legacy_evidence).hexdigest() != members[evidence_path]
            or sha256(legacy_manifest).hexdigest() != manifest_sha256
            or projected_size + missing_size - len(raw_evidence) + len(legacy_evidence) != total):
        raise ValueError("Original full manifest/evidence hashes and exact size must reconstruct")
    _write_new(file(payload, PATH), original_blob)
    _write_new(file(payload, evidence_path), legacy_evidence)
    _write_new(root / "manifest.json", legacy_manifest)
    receipts = [download_weight(file(payload, name), entry, check, remaining_seconds)
                for name, entry in external.items()]
    check()
    return dict(official_checkpoint_receipts=receipts,
                public_contract_git_blob=PUBLIC_BLOB["git_blob"],
                public_contract_sha256=PUBLIC_BLOB["sha256"])
