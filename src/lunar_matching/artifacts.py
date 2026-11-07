"""Byte-level provenance and exclusive, complete JSON writes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value) -> None:
    """Publish complete JSON without overwriting an existing result, even on a race."""
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".lunar-result-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def audit_archive(root: Path) -> dict:
    manifest = read_json(root / "docs/frozen-files.json")
    archive = (root / "archive/research").resolve()
    for name, record in manifest["files"].items():
        path = (archive / name).resolve()
        if (not path.is_relative_to(archive) or not path.is_file()
                or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]):
            raise ValueError(f"Frozen research file differs: {name}")
    return {"passed": True, "preserved_files": len(manifest["files"]),
            "original_public_commit": manifest["base_commit"], "all_bytes_identical": True}
