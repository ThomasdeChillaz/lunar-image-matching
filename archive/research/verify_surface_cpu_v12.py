"""Replay all surface CPU scores and fixed-bank paired timings/controls."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from combine_native_baselines_v8 import check_control
from dataset import sha256
from lunargeo_dataset import write_new
from modern_baselines_v2 import SOURCES
from verify_coarse_training_v5 import replay_row


def interval(rows, field, seed=20261101):
    groups = sorted({r['component'] for r in rows})
    parts = [[r[field] for r in rows if r['component'] == g] for g in groups]
    rng = np.random.default_rng(seed); samples = []
    for _ in range(10000):
        draw = [v for i in rng.integers(len(parts), size=len(parts)) for v in parts[i]]
        samples.append(float(np.median(draw)))
    return np.percentile(samples, [2.5, 97.5]).tolist()


def audit(root, output):
    root = Path(root); protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    if protocol['method'] != 'romav2' or protocol['profile'] != 'common' or protocol['cpu_threads'] != 2 or str(protocol['public_execution_id']) != '41262874858':
        raise ValueError('Unplanned same-worker comparison')
    for p, h in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(p) != h:
            raise ValueError('Executed CPU source/input differs: '+p)
    provenance = json.loads((root/'provenance.json').read_text())
    if provenance['revision'] != SOURCES['romav2']['revision'] or provenance['method'] != 'romav2' or provenance['profile'] != 'common':
        raise ValueError('Pinned author configuration differs')
    lock = json.loads(Path('reports/compact_baseline_lock_v2.json').read_text())['modern']['romav2']['weights']
    if any(provenance['weights'].get(p) != h for p, h in lock.items()):
        raise ValueError('Official weights differ')
    check_control(json.loads((root/'modern_controls.json').read_text()))
    control = json.loads((root/'surface_controls.json').read_text()); cp = np.array(control['estimated_cam2world'])
    expected = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    rotation = float(np.linalg.norm(cp[:3, :3]-expected)); translation = float(np.linalg.norm(cp[:3, 3]))
    if not control['passed'] or not control['blank_pose_null'] or rotation >= .01 or translation >= .05 or not np.allclose([rotation, translation], [control['rotation_matrix_error'], control['translation_error_m']]):
        raise ValueError('Actual surface quarter-turn/blank control differs')
    path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(path.read_text())
    if sha256(path) != protocol['manifest_sha256'] or sha256(Path(manifest['coarse_model'])/'selected.pt') != protocol['frozen_checkpoint_sha256']:
        raise ValueError('Common manifest/checkpoint differs')
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection']; bank = set(manifest['selection_bank'])
    if protocol['case_ids'] != [c['pair_id'] for c in cases] or protocol['paired_timing_case_ids'] != [c['pair_id'] for c in cases if c['pair_id'] in bank]:
        raise ValueError('Fixed full/paired case definitions differ')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for key, t in tiles.items():
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    rows = [json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    expected_pairs = {(c['pair_id'], 'compact_surface') for c in cases} | {(p, 'romav2') for p in bank}
    if len(rows) != 88 or {(r['pair_id'], r['method']) for r in rows} != expected_pairs:
        raise ValueError('Incomplete/duplicated CPU comparison')
    case_map = {c['pair_id']: c for c in cases}
    for row in rows:
        if any(row[k] != v for k, v in case_map[row['pair_id']].items()) or row['paired_timing'] != (row['pair_id'] in bank):
            raise ValueError('Case or timing identity differs')
        replay_row(row, tiles[row['tile_id']], data[row['tile_id']])
        if not np.isfinite(row['elapsed_ms']) or row['elapsed_ms'] <= 0:
            raise ValueError('Invalid CPU timing')
    for method in ('romav2', 'compact_surface'):
        part = [r for r in rows if r['method'] == method]; timed = [r for r in part if r['paired_timing']]
        saved = summary['results'][method]
        calculated = dict(cases=len(part), correct=sum(r['pose_correct'] for r in part), strict=sum(r['strict_pose_correct'] for r in part),
            estimated=sum(r['estimated_cam2world'] is not None for r in part), paired_timing_cases=len(timed),
            median_paired_inference_ms=float(np.median([r['elapsed_ms'] for r in timed])))
        if calculated != saved:
            raise ValueError('CPU aggregate differs')
    lookup = {(r['pair_id'], r['method']): r for r in rows}
    ratios = [dict(component=case_map[p]['component'], ratio=lookup[p, 'romav2']['elapsed_ms']/lookup[p, 'compact_surface']['elapsed_ms']) for p in sorted(bank)]
    speedup = float(np.median([r['ratio'] for r in ratios]))
    if not np.isclose(speedup, summary['median_paired_surface_speedup']):
        raise ValueError('Same-case paired speedup differs')
    local = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    with gzip.open(local, 'rt') as f:
        local_rows = {r['pair_id']: r for r in json.load(f)}
    agreements = sum((r['pose_correct'], r['strict_pose_correct']) == (local_rows[r['pair_id']]['pose_correct'], local_rows[r['pair_id']]['strict_pose_correct']) for r in rows if r['method'] == 'compact_surface')
    result = dict(passed=True, scope='Complete72-case surface CPU scoring and16-case paired timing replay on exposed development terrain; no calibrated acceptance or final-test claim.',
        results=summary['results'], median_paired_surface_speedup=speedup,
        descriptive_component_speedup_interval=interval(ratios, 'ratio'), paired_timing_cases=16,
        local_surface_primary_and_strict_status_agreements=agreements,
        memory_note=summary['memory_note'], peak_joint_process_rss_bytes=summary['peak_joint_process_rss_bytes'],
        interval_note='Descriptive resampling of four exposed terrain components, not independent deployment or hardware confidence.',
        source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'), protocol_sha256=sha256(root/'protocol.json'),
        case_sha256={p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))}, calibration_or_test_pixels_loaded=0)
    write_new(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'case_sha256'}, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
