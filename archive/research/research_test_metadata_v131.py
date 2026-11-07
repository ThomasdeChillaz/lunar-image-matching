"""Restore immutable JSON-only research fixtures for source-only software tests."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

SNAPSHOT_SHA256="4f2328701b680d65a11add62da00d9f2f1bfa66b18f7963839fd1e7a22f8ec5f"
NAMES={"protocol.json","manifest.json","summary.json","COMPLETE.json","FAILED.json","receipt.json","PREDICTIONS_SEALED.json"}


def restore(root, archive, expected_sha256):
    if sha256(archive).hexdigest()!=expected_sha256:
        raise ValueError("Test metadata archive digest differs")
    root=Path(root).resolve()
    with ZipFile(BytesIO(archive)) as z:
        inventory=json.loads(z.read("inventory.json"))
        if set(z.namelist())!={"inventory.json",*inventory} or len(z.namelist())!=len(inventory)+1:
            raise ValueError("Unexpected test metadata archive entries")
        pending=[]
        for name,digest in inventory.items():
            relative=PurePosixPath(name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or not relative.parts or relative.parts[0]!="runs" or relative.name not in NAMES
                    or "reservations" in name or "journal" in name or "ticket" in name
                    or not re.fullmatch(r"[0-9a-f]{64}",digest)):
                raise ValueError("Only allowlisted immutable JSON metadata may be restored")
            if z.getinfo(name).file_size>4*1024**2:
                raise ValueError("Test metadata entry is too large")
            data=z.read(name)
            if sha256(data).hexdigest()!=digest:
                raise ValueError("Test metadata content digest differs")
            json.loads(data)
            destination=root.joinpath(*relative.parts)
            if destination.is_symlink() or not destination.resolve().is_relative_to(root):
                raise ValueError("Test metadata path leaves the checkout")
            if destination.exists():
                if sha256(destination.read_bytes()).hexdigest()!=digest:
                    raise ValueError("Existing research metadata differs; never overwrite it")
            else:
                pending.append((destination,data))
    for destination,data in pending:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("xb") as f:f.write(data)
    return len(pending)
