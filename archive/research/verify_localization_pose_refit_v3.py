"""Independent saved-pose and aggregate replay for the development LM ablation."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def audit(path, output):
    doc = json.loads(Path(path).read_text())
    if sha256('localization_pose_refit_v3.py') != doc['source_sha256']:
        raise ValueError('Pose ablation source differs')
    source = Path(doc['source_run'])
    if sha256(source/'results.json.gz') != doc['source_results_sha256']:
        raise ValueError('Input correspondences differ')
    with gzip.open(source/'results.json.gz', 'rt') as f:
        inputs = json.load(f)
    protocol = json.loads((source/'protocol.json').read_text())
    manifest = json.loads(Path(protocol['manifest_path']).read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles']}
    if len(inputs) != len(doc['rows']):
        raise ValueError('Incomplete pose ablation')
    counts = {}
    for i, (original, row) in enumerate(zip(inputs, doc['rows'])):
        if row['source_row'] != i or row['tile_id'] != original['tile_id'] or row['experiment'] != original['experiment']:
            raise ValueError('Case identity differs')
        truth = np.array(tiles[row['tile_id']]['cameras'][1]['cam2world'])
        pose = row['estimated_cam2world']; correct = strict = False
        if pose is not None:
            pose = np.array(pose)
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T@truth[:3, :3])-1)/2, -1, 1))))
            distance = float(np.linalg.norm(pose[:3, 3]-truth[:3, 3]))
            if not np.allclose([rotation, distance], [row['pose_error']['rotation_deg'], row['pose_error']['translation_m']], atol=1e-6):
                raise ValueError('Pose error differs')
            correct = rotation <= 2 and distance <= .01*original['reference_range_m']
            strict = rotation <= .5 and distance <= .001*original['reference_range_m']
        if correct != row['pose_correct'] or strict != row['strict_pose_correct']:
            raise ValueError('Pose outcome differs')
        c = counts.setdefault(row['experiment'], dict(cases=0, original_correct=0, refined_correct=0,
                                                    refined_strict_correct=0, refined_estimated=0))
        c['cases'] += 1; c['original_correct'] += original['pose_correct']; c['refined_correct'] += int(correct)
        c['refined_strict_correct'] += int(strict); c['refined_estimated'] += pose is not None
    if counts != doc['results']:
        raise ValueError('Aggregate table differs')
    result = dict(passed=True, rows=len(inputs), tables=len(counts), saved_pose_errors_replayed=True,
                  source_sha256=sha256(__file__), results_sha256=sha256(path))
    write_new(output, result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='reports/localization_pose_refit_v3.json')
    p.add_argument('--output', default='reports/localization_pose_refit_audit_v3.json')
    a = p.parse_args(); print(json.dumps(audit(a.input, a.output), indent=2))
