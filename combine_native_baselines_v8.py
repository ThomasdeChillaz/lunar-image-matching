"""Replay the predeclared 36+36 native comparison without cherry-picking runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from verify_coarse_training_v5 import replay_row


TAIL_RUNS = {'loma_r': 41259046837, 'romav2': 41259085058}
TAIL_COMMIT = 'e408c3ffa12b7328b2949523d714c1a778d3840f'


def select_assigned(original, tail, case_ids, method):
    """The assignment is by frozen position, never correctness or run speed."""
    if len(case_ids) != 72 or len(set(case_ids)) != 72:
        raise ValueError('Expected all72 unique development cases')
    selected = []
    for rows, assigned, allowed in ((original, case_ids[:36], case_ids),
                                     (tail, case_ids[36:], case_ids[36:])):
        lookup = {}
        for row in rows:
            key = row['pair_id'], row['method']
            if row['pair_id'] not in allowed or row['method'] not in (method, 'compact_orientation') or key in lookup:
                raise ValueError('Unexpected or duplicated native case')
            lookup[key] = row
        for pair in assigned:
            for which in (method, 'compact_orientation'):
                if (pair, which) not in lookup:
                    raise ValueError('Missing assigned native case; comparison remains incomplete')
                selected.append(lookup[pair, which])
    return selected


def check_control(control):
    p, q = [np.asarray(control[k], dtype=float).reshape(-1, 2)
            for k in ('reference_points_native', 'query_points_native')]
    if p.shape != q.shape or len(p) < 6 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid actual pretrained control points')
    error = np.linalg.norm(q-p-np.array([8., -4.]), axis=1)
    pose = np.asarray(control['estimated_cam2world'], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Invalid native control pose')
    translation = float(np.linalg.norm(pose[:3, 3]-[-.2, .1, 0.]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3])-1)/2, -1, 1))))
    if (not control['passed'] or control['known_translation_native'] != [8., -4.]
            or control['image_seed'] != 20261101 or control['matches'] != len(p)
            or not np.isclose(np.median(error), control['median_native_error_px'])
            or np.median(error) > 1 or translation > .05 or rotation > .5
            or not np.allclose([translation, rotation], [control['translation_error_m'], control['rotation_error_deg']])
            or not control['blank_empty'] or not control['invalid_mask_empty']):
        raise ValueError('Actual pretrained native control replay failed')


def aggregates(rows, method):
    result = {}
    for which in (method, 'compact_orientation'):
        part = [r for r in rows if r['method'] == which]
        result[which] = dict(cases=len(part), correct=sum(r['pose_correct'] for r in part),
            strict=sum(r['strict_pose_correct'] for r in part),
            estimated=sum(r['estimated_cam2world'] is not None for r in part),
            median_full_inference_ms=float(np.median([r['elapsed_ms'] for r in part])),
            inference_errors=sum(r['inference_error'] is not None for r in part))
    return result


def audit(original, tail, output):
    from modern_baselines_v2 import SOURCES
    plan_path = Path('reports/native_case_range_plan_v8.json')
    plan = json.loads(plan_path.read_text())
    manifest_path = Path('reports/localization_training_data_v5.json')
    manifest = json.loads(manifest_path.read_text())
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection']
    case_ids = [c['pair_id'] for c in cases]
    case_map = {c['pair_id']: c for c in cases}
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    bundles = []
    for path in (Path(original), Path(tail)):
        protocol = json.loads((path/'protocol.json').read_text())
        provenance = json.loads((path/'provenance.json').read_text())
        controls = json.loads((path/'controls.json').read_text())
        rows = [json.loads(p.read_text()) for p in sorted((path/'cases').glob('*.json'))]
        method = protocol['method']
        if method not in TAIL_RUNS or protocol['profile'] != 'common' or protocol['cpu_threads'] != 2:
            raise ValueError('Unplanned native comparison settings')
        family = 'romav2' if method == 'romav2' else 'loma'
        if (provenance['revision'] != SOURCES[family]['revision'] or provenance['method'] != method
                or provenance['profile'] != 'common' or protocol['manifest_sha256'] != sha256(manifest_path)):
            raise ValueError('Author settings or common manifest differs')
        expected_weights = json.loads(Path('reports/compact_baseline_lock_v2.json').read_text())['modern'][method]['weights']
        if any(provenance['weights'].get(p) != h for p, h in expected_weights.items()):
            raise ValueError('Official pretrained weight fingerprint differs')
        for p, h in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
            if sha256(p) != h:
                raise ValueError('Executed native source/input differs: '+p)
        if protocol['input_sha256'] != {t['native_cache']: t['native_cache_sha256'] for t in tiles.values()}:
            raise ValueError('Native512px common input list differs')
        check_control(controls)
        for row in rows:
            c = case_map.get(row['pair_id'])
            if c is None or any(row[k] != v for k, v in c.items()):
                raise ValueError('Frozen case definition differs')
            if not np.isfinite(row['elapsed_ms']) or row['elapsed_ms'] < 0 or row['acceptance_policy'] is not None:
                raise ValueError('Invalid timing or raw pose relabelled as accepted')
        if (path/'summary.json').exists():
            if aggregates(rows, method) != json.loads((path/'summary.json').read_text())['results']:
                raise ValueError('Per-job saved aggregate differs')
        bundles.append((path, protocol, provenance, rows))
    a, b = bundles
    method = a[1]['method']
    if method != b[1]['method'] or a[2] != b[2]:
        raise ValueError('The two ranges use different pretrained configurations')
    if (a[1]['case_ids'] != case_ids or int(a[1]['public_execution_id']) != plan['original_runs'][method]
            or a[1]['source_commit'] != plan['original_source_commit']
            or b[1]['case_ids'] != case_ids[36:] or b[1]['all_case_ids'] != case_ids
            or (b[1]['case_start'], b[1]['case_count']) != (36, 36)
            or int(b[1]['public_execution_id']) != TAIL_RUNS[method] or b[1]['source_commit'] != TAIL_COMMIT):
        raise ValueError('Sources do not follow the predeclared first36/last36 assignment')
    for p, h in a[1]['source_sha256'].items():
        if b[1]['source_sha256'].get(p) != h:
            raise ValueError('Shared implementation changed between execution ranges')
    selected = select_assigned(a[3], b[3], case_ids, method)
    depths = {}
    for key, t in tiles.items():
        with np.load(t['native_cache'], allow_pickle=False) as z:
            depths[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    for row in selected:
        replay_row(row, tiles[row['tile_id']], depths[row['tile_id']])
    timing = {(r['pair_id'], r['method']): r['elapsed_ms'] for r in selected}
    result = dict(passed=True, scope='Complete exposed development native comparison; no calibrated acceptance, fresh final or global SOTA claim.',
        method=method, profile='common', eligible_pairs=72, replayed_records=144,
        results=aggregates(selected, method),
        median_paired_compact_speedup=float(np.median([timing[c, method]/max(timing[c, 'compact_orientation'], 1e-9) for c in case_ids])),
        assignment_plan_sha256=sha256(plan_path), source_sha256=sha256(__file__),
        source_runs=[dict(path=str(path), execution_id=p['public_execution_id'],
            preserved_rows=len(rows), protocol_sha256=sha256(path/'protocol.json'),
            controls_sha256=sha256(path/'controls.json'),
            case_sha256={f.name: sha256(f) for f in sorted((path/'cases').glob('*.json'))})
            for path, p, _, rows in bundles],
        memory_note='Joint-process memory includes both models; no isolated method memory ranking.',
        calibration_or_test_pixels_loaded=0)
    write_new(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_runs'}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', required=True); p.add_argument('--tail', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.original, a.tail, a.output)
