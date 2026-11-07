"""Independent native geometry/pose replay and complete-profile denominators."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from verify_surface_learning_v13 import replay_pose


def audit(run, output):
    root = Path(run); protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text()); cv2.setNumThreads(1)
    if not json.loads((root/'controls.json').read_text())['passed']:raise ValueError('Actual checkpoint control failed')
    if protocol['calibration_or_test_pixels_loaded'] or summary['calibration_or_test_pixels_loaded']:
        raise ValueError('Diagnostic scope changed')
    for name, want in protocol['source_sha256'].items():
        if sha256(name) != want or sha256(root/'source'/name) != want:raise ValueError('Source changed: '+name)
        if subprocess.check_output(['git', 'show', protocol['source_commit']+':'+name]) != Path(name).read_bytes():
            raise ValueError('Public source differs')
    if sha256(root/'protocol.json') != summary['protocol_sha256']:raise ValueError('Protocol changed')
    files = {p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))}
    if files != summary['case_sha256']:raise ValueError('Case files changed')
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != protocol['manifest_sha256']:raise ValueError('Manifest changed')
    cases = {c['pair_id']: c for c in manifest['cases'] if c['pair_id'] in protocol['case_ids']}
    if set(cases) != set(protocol['case_ids']) or len(cases) != len(protocol['case_ids']):raise ValueError('Case denominator differs')
    if any(c['split'] != 'model_selection' for c in cases.values()):raise ValueError('Wrong split')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for name, want in protocol['input_sha256'].items():
        if sha256(name) != want:raise ValueError('Native input changed')
    for key in {c['tile_id'] for c in cases.values()}:
        with np.load(tiles[key]['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    rows = [json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    expected_methods = {'fine_scales_'+'_'.join(map(str, s)) for s in protocol['profiles']}
    if {r['method'] for r in rows} != expected_methods:raise ValueError('Unexpected/missing method')
    for row in rows:
        if row['pair_id'] not in cases or any(row[k] != v for k, v in cases[row['pair_id']].items()):raise ValueError('Case metadata changed')
        replay_pose(row, tiles[row['tile_id']], data[row['tile_id']])
    for method in expected_methods:
        part = [r for r in rows if r['method'] == method]
        if len(part) != len(cases) or {r['pair_id'] for r in part} != set(cases):raise ValueError('Incomplete/duplicate profile')
        actual = dict(cases=len(part), correct=sum(r['pose_correct'] for r in part),
            strict=sum(r['strict_pose_correct'] for r in part), estimated=sum(r['estimated_cam2world'] is not None for r in part),
            correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in part), returned=sum(r['returned'] for r in part))
        if actual != summary['results'][method]:raise ValueError('Aggregate changed')
    proof = dict(passed=True, native_metric_and_common_pose_replay=True, rows=len(rows), eligible_pairs=len(cases),
        profiles=len(expected_methods), summary_sha256=sha256(root/'summary.json'), source_sha256=sha256(__file__),
        scope='Independent saved-metric replay only; same exposed development terrain, not a fresh test or independent image-inference implementation.')
    write_new(output, proof); print(json.dumps(proof, indent=2)); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
