"""Separate a v75 capsule's full execution inventory from its release ZIP."""

import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import zipfile

from solar_capsule_v75 import (
    MAX_FILES, MAX_JSON_BYTES, MAX_MANIFEST_BYTES, MAX_MEMBER_BYTES, MAX_TOTAL_BYTES,
    RELEASE_ASSET_MAX_BYTES, file, read, safe, sha, validate_capsule,
)

CACHE = "data/models/modern-v2-torch-cache/hub/checkpoints/"
SOURCE = "solar_weight_transport_v96.py"
LOCK = "reports/compact_baseline_lock_v2.json"
OFFICIAL = {
    "dad.pth": ("https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth",
                26_075_286, "7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f"),
    "dinov2_vitl14_pretrain.pth": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        1_217_586_395, "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"),
    "loma_R.pth": ("https://github.com/davnords/storage/releases/download/loma/loma_R.pth",
                   757_905_869, "ade1379fb52143da65c36730f24b69eaa49618bb277890c6d2a0fb1630d33d67"),
    "romav2.0.1.pt": ("https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
                      1_095_883_548, "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"),
}
BLOCK = 1 << 20
DISK_HEADROOM = 512 << 20


def declaration():
    return {CACHE + name: dict(url=url, bytes=size, sha256=digest)
            for name, (url, size, digest) in OFFICIAL.items()}


def validate_declaration(protocol, lock):
    selected = {}
    for method in ("loma_r", "romav2"):
        selected.update({CACHE + name: digest for name, digest in lock["modern"][method]["weights"].items()})
    official = declaration()
    if (protocol.get("external_weight_transport_v96") != official
            or protocol.get("selected_weight_inputs") != selected
            or selected != {path: entry["sha256"] for path, entry in official.items()}
            or any(protocol["input_sha256"].get(path) != digest for path, digest in selected.items())
            or protocol["source_sha256"].get(SOURCE) != sha(__file__)):
        raise ValueError("Exact four official transport pins and unchanged full execution inputs required")
    return official


def required_members(manifest, protocol):
    expected = {**protocol["source_sha256"], **protocol["input_sha256"]}
    if any(expected[name] != digest for name, digest in protocol["source_sha256"].items()):
        raise ValueError("Conflicting execution input/source hashes")
    expected[manifest["protocol_path"]] = manifest["protocol_sha256"]
    auth = manifest["pre_reservation_authorization"]
    expected[auth["path"]] = auth["sha256"]
    if expected != manifest["files"]:
        raise ValueError("Full capsule closure changed")
    return expected


def make_thin_zip(capsule_directory, output_path):
    """Package the already validated full capsule without four public weights."""
    root = Path(capsule_directory)
    manifest = validate_capsule(root)
    protocol = read(file(root / "payload", manifest["protocol_path"]))
    external = validate_declaration(protocol, read(file(root / "payload", LOCK)))
    members = required_members(manifest, protocol)
    for name, entry in external.items():
        if file(root / "payload", name).stat().st_size != entry["bytes"]:
            raise ValueError("Original staged checkpoint size differs: " + name)
    output = Path(output_path)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(root / "manifest.json", "manifest.json")
        for name in sorted(members.keys() - external.keys()):
            archive.write(file(root / "payload", name), "payload/" + name)
    size = output.stat().st_size
    if size > RELEASE_ASSET_MAX_BYTES:
        raise ValueError("Thin archive exceeds the existing single release asset limit")
    return dict(path=str(output), sha256=sha(output), bytes=size, external_weight_count=len(external),
                external_weight_bytes=sum(item["bytes"] for item in external.values()),
                manifest_sha256=sha(root / "manifest.json"), full_payload_bytes=manifest["total_bytes"])


def _host(url, allowed):
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc not in allowed
            or parts.username is not None or parts.password is not None or parts.fragment):
        raise ValueError("Official checkpoint redirect left the allowed HTTPS origin")
    return parts.netloc


class OfficialRedirect(HTTPRedirectHandler):
    max_redirections = 3

    def __init__(self, allowed):
        self.allowed = allowed

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        _host(newurl, self.allowed)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download_weight(destination, entry, check, remaining_seconds):
    url = entry["url"]
    target = Path(destination)
    if declaration().get(CACHE + target.name) != entry:
        raise ValueError("Only an exact official checkpoint manifest entry can be fetched")
    origin = _host(url, {"github.com", "dl.fbaipublicfiles.com"})
    allowed = {"release-assets.githubusercontent.com"} if origin == "github.com" else {origin}
    opener = build_opener(ProxyHandler({}), OfficialRedirect(allowed))
    if target.exists() or target.is_symlink():
        raise FileExistsError("Isolated checkpoint destination must be unused")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".acquiring")
    digest = hashlib.sha256()
    count = 0
    started = monotonic()
    check()
    request = Request(url, headers={"User-Agent": "solar-official-weight-v96", "Accept-Encoding": "identity"})
    try:
        with opener.open(request, timeout=max(0.001, min(10., remaining_seconds()))) as response:
            final_host = _host(response.geturl(), allowed)
            length = response.headers.get("Content-Length")
            if (response.status != 200 or response.headers.get("Content-Encoding") not in (None, "identity")
                    or length is not None and length != str(entry["bytes"])):
                raise ValueError("Official checkpoint server did not provide the exact unencoded response")
            with partial.open("xb") as out:
                while chunk := response.read1(min(BLOCK, entry["bytes"] - count + 1)):
                    check()
                    count += len(chunk)
                    if count > entry["bytes"]:
                        raise ValueError("Official checkpoint exceeded its exact byte count")
                    digest.update(chunk)
                    out.write(chunk)
        check()
        if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Official checkpoint bytes differ from the published pin")
        partial.rename(target)
    except (HTTPError, URLError) as exc:
        # A signed GitHub CDN URL can appear in nested exceptions.
        status = getattr(exc, "code", None)
        raise ConnectionError(f"Official checkpoint request failed ({status or type(exc).__name__})") from None
    return dict(filename=target.name, source=url, final_host=final_host, bytes=count,
                sha256=digest.hexdigest(), calendar_seconds=monotonic() - started)


def _extract_member(archive, name, target, expected_hash, check):
    check()
    digest = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, target.open("xb") as output:
        while chunk := source.read(BLOCK):
            check()
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Thin ZIP member hash differs: " + name)


def hydrate_thin_zip(archive_path, destination, manifest_sha256, protocol_sha256, check, remaining_seconds):
    """Verify transport and restore the exact full payload for v75 installation."""
    root = Path(destination)
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
            raise ValueError("Thin ZIP manifest exceeds its byte limit")
        raw = archive.read("manifest.json")
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Thin ZIP manifest differs from the one-use ticket")
        manifest = json.loads(raw)
        if (manifest.get("schema_version") != 75 or manifest.get("complete") is not True
                or manifest.get("capsule_kind") != "model_selection_primary_full_sun"
                or manifest["protocol_sha256"] != protocol_sha256):
            raise ValueError("Original v75 capsule and ticket protocol identity required")
        members = manifest["files"]
        official = declaration()
        for name in members:
            safe(name)
        expected = {"manifest.json"} | {"payload/" + name for name in members.keys() - official.keys()}
        if names != expected or not official.keys() <= members.keys() or len(members) > MAX_FILES:
            raise ValueError("Thin ZIP must omit exactly the four declared checkpoint paths")
        packed = [item for item in infos if item.filename != "manifest.json"]
        if any(item.file_size > MAX_MEMBER_BYTES or item.filename.endswith(".json") and
               item.file_size > MAX_JSON_BYTES for item in packed):
            raise ValueError("Thin ZIP member exceeded original v75 bound")
        total = sum(item.file_size for item in packed) + sum(entry["bytes"] for entry in official.values())
        if total != manifest["total_bytes"] or total > MAX_TOTAL_BYTES:
            raise ValueError("Complete hydrated payload must fit its original size bound")
        if shutil.disk_usage(root.parent).free < 2 * total + DISK_HEADROOM:
            raise OSError("Insufficient free disk for isolated hydration and verified v75 installation")
        root.mkdir()
        (root / "manifest.json").write_bytes(raw)
        for name in sorted(members.keys() - official.keys()):
            _extract_member(archive, "payload/" + name, file(root / "payload", name), members[name], check)

    protocol = read(file(root / "payload", manifest["protocol_path"]))
    if members.get(manifest["protocol_path"]) != protocol_sha256:
        raise ValueError("Downloaded protocol differs from the one-use ticket")
    validate_declaration(protocol, read(file(root / "payload", LOCK)))
    required_members(manifest, protocol)
    receipts = []
    for name, entry in official.items():
        receipts.append(download_weight(file(root / "payload", name), entry, check, remaining_seconds))
    check()
    return receipts
