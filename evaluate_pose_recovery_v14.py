"""Frozen-weight development control for broad initialization plus strict refinement."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score
from pose_recovery_v14 import recover
from synthetic import texture

SOURCES = ['pose_recovery_v14.py', 'evaluate_pose_recovery_v14.py', 'surface_learning_v13.py',
    'geometry_warp_native_v11.py', 'coarse_pipeline_v5.py', 'native_refinement_v7.py',
    'localization_refiner_v3.py', 'native_baseline_pose_v7.py', 'compact_lunar.py',
    'contrastive_terrain.py', 'localization_diagnostics_v3.py', 'localization_robust_pose_v4.py',
    'localization_pose_refit_v3.py', 'lunar_geometry_v5.py', 'lunar_geometry_v2.py',
    'rotation_bank_pose_v4.py', 'train_balanced_localization_v3.py', 'prepare_localization_training_v3.py']
BASELINE = Path('runs/localization-v11-surface-warp-development')


def controls(coarse):
    image = texture(733, 512); query = np.rot90(image, 1).copy(); blank = np.zeros_like(image)
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    result, _ = recover(coarse, image, query, np.full(image.shape, 10.), k, np.eye(4), k)
    invalid, _ = recover(coarse, blank, blank, np.full(image.shape, 10.), k, np.eye(4), k)
    expected = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    pose = result['pose']; rotation = None if pose is None else float(np.linalg.norm(pose[:3, :3]-expected))
    translation = None if pose is None else float(np.linalg.norm(pose[:3, 3]))
    passed = pose is not None and rotation < .01 and translation < .05 and invalid['pose'] is None
    return dict(passed=bool(passed), rotation_matrix_error=rotation, translation_error_m=translation,
        blank_pose_null=invalid['pose'] is None, estimated_cam2world=None if pose is None else pose.tolist(),
        scope='Actual frozen-weight native512px quarter-turn/blank controls. Broad-final-pose suppression is separately unit-tested.')


def run(output, full=False):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve every completed/failed development attempt')
    root.mkdir(parents=True); torch.set_num_threads(2); cv2.setNumThreads(1)
    manifest_path = Path('reports/localization_training_data_v5.json')
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(Path('reports/native_surface_audit_v11.json').read_text())
    if not audit['passed'] or audit['results_sha256'] != sha256(BASELINE/'results.json.gz'):
        raise ValueError('Audited historical surface control required')
    with gzip.open(BASELINE/'results.json.gz', 'rt') as stream:
        baseline = {r['pair_id']: r for r in json.load(stream)}
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (full or c['pair_id'] in manifest['selection_bank'])]
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    protocol = dict(scope='Development-only missing-pose recovery with frozen weights. Engineering control, no learned gain or accepted-pose claim.',
        source_sha256={p: sha256(p) for p in SOURCES}, manifest_sha256=sha256(manifest_path),
        checkpoint_sha256=sha256(checkpoint), input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        baseline_results_sha256=sha256(BASELINE/'results.json.gz'), baseline_audit_sha256=sha256('reports/native_surface_audit_v11.json'),
        case_ids=[c['pair_id'] for c in cases], full=full, initial_threshold_px=12., final_threshold_px=3.,
        route='Preserve every historical estimated surface pose. Only previously missing poses trigger four broad native branches, ranked by12px support/residual then rotation. Require at least one successful unchanged3px surface update; otherwise pose:null.',
        query_geometry='Query depth and true query pose are scoring only. Inference uses reference depth/pose and intrinsics.',
        cpu_threads=2, timing='Concurrent with a separate two-thread training job; no isolated timing claim.',
        calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json', protocol)
    coarse = CompactMatcher(manifest['coarse_model']); control = controls(coarse)
    write_new(root/'controls.json', control)
    if not control['passed']:
        raise ValueError('Actual frozen-model controls failed before lunar inference')
    rows = []; data = {}; features = {}; previous = None
    for c in cases:
        old = baseline[c['pair_id']]
        if old['estimated_cam2world'] is not None:
            rows.append(dict(old, method='broad_init_surface', reused_unchanged_baseline=True,
                recovery=dict(attempted=False, recovered=False, final_threshold_px=3.)))
        else:
            tile = tiles[c['tile_id']]
            if c['tile_id'] not in data:
                if sha256(tile['native_cache']) != tile['native_cache_sha256']:
                    raise ValueError('Development input changed')
                with np.load(tile['native_cache'], allow_pickle=False) as z:
                    data[c['tile_id']] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
            if previous != c['tile_id']:
                features.clear(); previous = c['tile_id']
            z = data[c['tile_id']]; a, b = tile['cameras']; ia, ib = 2*c['light_a'], 2*c['light_b']+1
            result, _ = recover(coarse, z['images'][ia], z['images'][ib], z['depth_a'],
                a['intrinsics'], a['cam2world'], b['intrinsics'], features,
                ((c['tile_id'], ia), (c['tile_id'], ib)))
            if not result['recovery']['attempted']:
                raise ValueError('Historical missing-pose route changed under identical inputs')
            rows.append(dict(c, method='broad_init_surface', reused_unchanged_baseline=False,
                **score(result['p'], result['q'], result['pose'], z, tile['cameras']),
                quarter_turn=result['quarter_turn'], pose_info=result['info'],
                geometry_history=result.get('geometry_history', []), recovery=result['recovery']))
        print('broad recovery', len(rows), '/', len(cases), 'correct', sum(r['pose_correct'] for r in rows), flush=True)
    with gzip.open(root/'results.json.gz', 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)
    summary = dict(scope=protocol['scope'], cases=len(rows), correct=sum(r['pose_correct'] for r in rows),
        strict=sum(r['strict_pose_correct'] for r in rows), estimated=sum(r['estimated_cam2world'] is not None for r in rows),
        wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows),
        attempted=sum(r['recovery']['attempted'] for r in rows), recovered=sum(r['recovery']['recovered'] for r in rows),
        newly_correct=sum(r['pose_correct'] and r['recovery']['attempted'] for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows), results_sha256=sha256(root/'results.json.gz'))
    write_new(root/'summary.json', summary); print(json.dumps(summary, indent=2)); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--full', action='store_true'); args = parser.parse_args(); run(args.output, args.full)
