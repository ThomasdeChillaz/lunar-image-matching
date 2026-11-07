"""Replay saved MOONSt3R native metrics and common reference-geometry poses."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose
from verify_coarse_training_v5 import replay_row


def audit(run, output):
    root = Path(run); cv2.setNumThreads(1)
    protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    if sha256(root/'protocol.json') != summary['protocol_sha256']:raise ValueError('Protocol changed')
    for name, want in protocol['source_sha256'].items():
        if sha256(name) != want or sha256(root/'source'/name) != want:raise ValueError('Source changed: '+name)
        if subprocess.check_output(['git', 'show', protocol['source_commit']+':'+name]) != Path(name).read_bytes():
            raise ValueError('Public source differs')
    actual_files = {p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))}
    if actual_files != summary['case_sha256']:raise ValueError('Case files changed')
    provenance = json.loads((root/'provenance.json').read_text())
    if (provenance['source_revision'], provenance['safe_weight_sha256'], provenance['native_input_size'], provenance['profile']) != (
        'ed6ea257475abb8a5235dff3a4d6167214b6f697', 'f1f356da8f400d503f2c675623775e08f736bb24066e75f48301a1aff8cb3d5f',
        512, 'common_native512_full_field_of_view'):raise ValueError('Pinned actual baseline differs')
    if provenance['source_lock_sha256'] != sha256('moonst3r_model_lock_v19.json') or provenance['query_geometry_in_interface']:
        raise ValueError('Pretrained lock or inference contract differs')
    controls = Path('runs/localization-v19-moonst3r-native-control')
    if sha256(controls/'summary.json') != protocol['actual_gate_summary_sha256']:raise ValueError('Actual gate changed')
    gate = json.loads((controls/'summary.json').read_text())
    if not gate['passed'] or not gate['actual_native_pretrained_forward_attempted']:raise ValueError('Actual gate failed')
    cp = json.loads((controls/'controls.json').read_text())
    p, q = [np.asarray(cp[k], float).reshape(-1, 2) for k in ('reference_points_native', 'query_points_native')]
    error = np.linalg.norm(q-p-[8, -4], axis=1)
    if len(p) < 6 or len(p) != cp['matches'] or not np.isclose(np.median(error), cp['median_native_error_px']):
        raise ValueError('Actual control metrics differ')
    if np.median(error) > 1 or not cp['blank_empty'] or not cp['invalid_mask_empty']:raise ValueError('Input/transform controls failed')
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != protocol['manifest_sha256']:raise ValueError('Manifest changed')
    cases = {c['pair_id']: c for c in manifest['cases'] if c['split'] == 'model_selection' and
        (not protocol['bank_only'] or c['pair_id'] in manifest['selection_bank'])}
    if set(cases) != set(protocol['case_ids']) or len(cases) != len(protocol['case_ids']):raise ValueError('Eligibility differs')
    for path, want in protocol['input_sha256'].items():
        if sha256(path) != want:raise ValueError('Native input changed')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for key in {c['tile_id'] for c in cases.values()}:
        with np.load(tiles[key]['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    rows = [json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    if len(rows) != len(cases) or {r['pair_id'] for r in rows} != set(cases):raise ValueError('Incomplete/duplicate denominator')
    for row in rows:
        if row['method'] != 'moonst3r_native' or any(row[k] != v for k, v in cases[row['pair_id']].items()):raise ValueError('Case identity differs')
        tile = tiles[row['tile_id']]; z = data[row['tile_id']]; replay_row(row, tile, z)
        a, b = tile['cameras']; p, q = [np.asarray(row[k], float).reshape(-1, 2) for k in ('reference_points_native', 'query_points_native')]
        pose, info = propose_pose(p, q, z['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics'])
        saved = row['estimated_cam2world']
        if (pose is None) != (saved is None) or (pose is not None and not np.allclose(pose, saved, atol=1e-8, rtol=0)):
            raise ValueError('Common pose replay differs')
        if row['inference_error'] is None and info != row['pose_info']:raise ValueError('Common pose diagnostics differ')
        if not np.isfinite(row['elapsed_seconds']) or row['elapsed_seconds'] <= 0:raise ValueError('Invalid elapsed time')
    counts = dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows), strict=sum(r['strict_pose_correct'] for r in rows),
        wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows),
        inference_errors=sum(r['inference_error'] is not None for r in rows),
        median_inference_seconds=float(np.median([r['elapsed_seconds'] for r in rows])))
    if any(summary[k] != v for k, v in counts.items()):raise ValueError('Summary differs')
    proof = dict(passed=True, records=len(rows), eligible_pairs=len(cases), native_metric_and_common_pose_replay=True,
        actual_pretrained_control_replayed=True, exact_training_overlap_known=False, source_sha256=sha256(__file__),
        summary_sha256=sha256(root/'summary.json'),
        scope='Exposed development, unknown pretrained training overlap. Saved-metric/source replay, not independent neural inference, fresh transfer or calibrated acceptance.')
    write_new(output, proof); print(json.dumps(proof, indent=2)); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
