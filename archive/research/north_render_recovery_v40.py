"""Immutable component reuse for a separately budgeted render recovery."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_component(folder, record):
    folder=Path(folder)
    actual={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    if actual != record['files_sha256']:
        raise ValueError('Prior complete component bytes changed')
    summary=json.loads((folder/'summary.json').read_text())
    if (summary['component'] != record['component'] or len(summary['images']) != 24
            or len(summary['exact_supervision']) != 15 or len(summary['geometry_sha256']) != 4):
        raise ValueError('Prior complete component is incomplete')
    for image in summary['images']:
        p=folder/f'view-{image["view"]}'/f'light-{image["light"]}'/'image.png'
        if sha(p) != image['image_sha256']:raise ValueError('Prior image differs from measured summary')
    for name,digest in summary['geometry_sha256'].items():
        if sha(folder/name) != digest:raise ValueError('Prior geometry differs from measured summary')
    for item in summary['exact_supervision']:
        if sha(folder/item['file']) != item['sha256']:raise ValueError('Prior BVH label differs from measured summary')
    for item in summary['labels']:
        if sha(folder/f'correspondences-0-{item["query_view"]}.npz') != item['sha256']:
            raise ValueError('Prior dense label differs from measured summary')
    return summary


def copy_component(source, destination, record):
    """Copy, do not link: new output cannot mutate original failed-run evidence."""
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError('Preserve every recovery attempt')
    verify_component(source,record)
    shutil.copytree(source,destination)
    return verify_component(destination,record)
