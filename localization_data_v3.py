"""Full released camera inventory and conservative pretraining freshness gate.

No RGB/depth is decoded and no model is trained. Released acquisition-mode
folders are conservative fallback groups, NOT verified trajectory/batch IDs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from dataset import sha256
from lunar_geometry_v2 import camera_check
from lunargeo_dataset import MODES, read_array, write_new


def freshness_gate(groups, exposed_groups, minimum=12):
    groups, exposed_groups = set(groups), set(exposed_groups)
    fresh = sorted(groups-exposed_groups)
    return dict(passed=len(fresh) >= minimum, required_fresh_groups=minimum,
                available_groups=len(groups), exposed_groups=len(groups & exposed_groups),
                fresh_groups=fresh, fresh_group_count=len(fresh))


def require_training_gate(report):
    # Folder-count sufficiency alone cannot establish fresh spatial footprints.
    if not (report.get('gate', {}).get('passed') and
            report.get('verified_fresh_footprint_count', 0) >= 12 and
            report.get('split_audit_passed') is True and
            report.get('trajectory_or_batch_grouping_verified') is True):
        raise ValueError('Fresh-data gate failed: training and final-test inference are prohibited')


def inventory(root, exposed_manifest, output, records_path):
    root, output, records_path = Path(root), Path(output), Path(records_path)
    if output.exists() or records_path.exists():
        raise FileExistsError('Inventory outputs are immutable; choose new paths')
    old = json.loads(Path(exposed_manifest).read_text())
    exposed = {t['mode']+'/'+f['camera'].rsplit('/', 1)[0]
               for t in old['tiles'] for f in t['frames']}
    records_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = {}; groups = set(); field_counts = Counter(); total = 0
    with records_path.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            for mode in MODES:
                path = root/(mode+'.zip')
                provenance = json.loads(path.with_suffix('.provenance.json').read_text())
                digest = sha256(path)
                if digest != provenance['sha256']:
                    raise ValueError('Source archive changed: '+str(path))
                cameras = {}; folders = Counter(); geometry = Counter(); shapes = Counter()
                with zipfile.ZipFile(path) as archive:
                    for name in sorted(n for n in archive.namelist() if n.endswith('.npz')):
                        with read_array(archive, name) as values:
                            fields = sorted(values.files); field_counts.update(fields)
                            k, pose = camera_check(values['intrinsics'], values['cam2world'])
                            # Preserve every numeric field, including any released ID or
                            # Sun metadata, instead of assuming the previous sample's schema.
                            metadata = {}
                            for key in fields:
                                value = values[key]
                                if value.dtype.hasobject:
                                    raise ValueError('Unsafe object metadata is not independently inspectable')
                                if value.size <= 32:
                                    metadata[key] = value.tolist()
                                shapes[str((key, value.shape, str(value.dtype)))] += 1
                        folder, stem = name.rsplit('/', 1)
                        index = int(stem.split('_')[-1].split('.')[0])
                        signature = hashlib.sha256(k.tobytes()+pose.tobytes()).hexdigest()
                        cameras[(folder, index)] = signature
                        group = mode+'/'+folder; groups.add(group); folders[folder] += 1
                        row = dict(mode=mode, camera=name, archive_sha256=digest,
                                   camera_sha256=hashlib.sha256(archive.read(name)).hexdigest(),
                                   fallback_group=group, metadata=metadata)
                        compressed.write((json.dumps(row, sort_keys=True, allow_nan=False)+'\n').encode())
                        total += 1
                    for (folder, index), signature in cameras.items():
                        if index % 2 == 0:
                            partner = cameras.get((folder, index+1))
                            if partner is None:
                                raise ValueError('Incomplete released stereo camera pair')
                            geometry[(folder, signature, partner)] += 1
                summaries[mode] = dict(camera_files=len(cameras), folders=dict(folders),
                    distinct_stereo_geometries=len(geometry),
                    lighting_observation_counts=dict(Counter(geometry.values())),
                    complete_three_light_geometries=sum(n == 3 for n in geometry.values()),
                    archive_sha256=digest, field_shapes=dict(shapes))
                print(mode, 'all camera files', len(cameras), 'geometries', len(geometry), flush=True)
    gate = freshness_gate(groups, exposed)
    result = dict(schema_version=3, recorded_utc=datetime.now(timezone.utc).isoformat(),
        source_doi='10.5281/zenodo.18415119', scope='All NPZ files in all three released SouthPole inner archives',
        camera_files=total, archives=summaries, metadata_field_counts=dict(field_counts),
        released_folder_groups=sorted(groups), previously_exposed_groups=sorted(exposed), gate=gate,
        grouping='Entire released mode/folder retained together because verified trajectory and generation-batch identifiers are unavailable.',
        caveat='Acquisition modes/folders are fallback grouping units, not verified trajectory or batch identities. No fresh footprint count is inferred from them.',
        consequence='Do not train compact-v3 or run final inference unless a new independently audited data gate passes.',
        source_sha256=sha256(__file__), exposed_manifest=str(exposed_manifest),
        exposed_manifest_sha256=sha256(exposed_manifest), records_path=str(records_path),
        records_sha256=sha256(records_path), training_started=False, final_test_started=False)
    write_new(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/lunargeo')
    p.add_argument('--exposed-manifest', default='reports/compact_dataset_v2.json')
    p.add_argument('--output', default='reports/localization_data_gate_v3.json')
    p.add_argument('--records', default='runs/localization-v3/full_camera_inventory.jsonl.gz')
    a = p.parse_args()
    result = inventory(a.root, a.exposed_manifest, a.output, a.records)
    print(json.dumps(result['gate'], indent=2))
