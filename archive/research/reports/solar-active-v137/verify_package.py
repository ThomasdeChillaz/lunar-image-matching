"""Verify public replay bytes without inference or private compute authority."""
from pathlib import Path, PurePosixPath
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    root=Path.cwd().resolve()
    manifest=json.loads((root/"ACTIVE_REPLAY_INVENTORY_V137.json").read_text())
    if manifest["schema"] != "active_replay_inventory_v137":
        raise ValueError("Wrong public inventory schema")
    for name,record in manifest["files"].items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Invalid inventory path")
        p=root/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size != record["bytes"] or sha(p) != record["sha256"]:
            raise ValueError("Changed or missing inventory member: "+name)
    home=root/"reports/solar-active-v137"
    protocol=json.loads((home/"protocol.json").read_text())
    for name,digest in {**protocol["sources"],**protocol["input_sha256"]}.items():
        if sha(root/name) != digest:
            raise ValueError("Changed frozen dependency: "+name)
    run=root/"runs/solar-active-v137/comparison"
    seal=json.loads((run/"PREDICTIONS_SEALED.json").read_text())
    complete=json.loads((run/"COMPLETE.json").read_text())
    if seal["sha256"] != sha(run/"predictions.json") or seal["rows"] != 32 or not complete["complete"]:
        raise ValueError("Incomplete or changed scientific seal")
    return dict(passed=True,payload_files=len(manifest["files"]),
                scientific_source=complete["commit"],method_case_cells=832,
                inference=False,private_compute_authority_required=False)


if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
