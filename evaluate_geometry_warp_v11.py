"""Fixed two-iteration reference-surface warp development control."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from geometry_warp_native_v11 import infer
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score


SOURCES = ['geometry_warp_native_v11.py', 'evaluate_geometry_warp_v11.py', 'native_baseline_pose_v7.py',
    'coarse_pipeline_v5.py', 'localization_diagnostics_v3.py', 'compact_lunar.py', 'contrastive_terrain.py',
    'localization_robust_pose_v4.py', 'localization_pose_refit_v3.py', 'lunar_geometry_v5.py', 'lunar_geometry_v2.py', 'rotation_bank_pose_v4.py']


def control(coarse):
    from synthetic import texture
    image = texture(733, 512); query = np.rot90(image, 1).copy()
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    selected, _ = infer(coarse, image, query, np.full(image.shape, 10.), k, np.eye(4), k)
    expected = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    rotation = None if selected is None else float(np.linalg.norm(selected['pose'][:3, :3]-expected))
    translation = None if selected is None else float(np.linalg.norm(selected['pose'][:3, 3]))
    blank, _ = infer(coarse, np.zeros_like(image), np.zeros_like(image), np.full(image.shape, 10.), k, np.eye(4), k)
    passed = selected is not None and rotation < .01 and translation < .05 and blank is None
    return dict(passed=bool(passed), rotation_matrix_error=rotation, translation_error_m=translation,
        blank_pose_null=blank is None, estimated_cam2world=None if selected is None else selected['pose'].tolist(),
        geometry_history=None if selected is None else selected['geometry_history'], image_seed=733)


def run(output, bank_only=True):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve every completed/failed run')
    root.mkdir(parents=True); torch.set_num_threads(2); cv2.setNumThreads(1)
    path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(path.read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (not bank_only or c['pair_id'] in manifest['selection_bank'])]
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse checkpoint changed')
    protocol = dict(scope='Development-only deterministic pose-guided3D surface normalization; no new learning or accepted-pose claim.',
        bank_only=bank_only, case_ids=[c['pair_id'] for c in cases], manifest_sha256=sha256(path),
        source_sha256={p: sha256(p) for p in SOURCES}, frozen_checkpoint_sha256=sha256(checkpoint),
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        iterations=2, selection='Initial four-orientation native pose; fixed two successful refinement updates. Keep last successful pose if a later refinement fails. No query truth for any choice.',
        geometry='Reference axial-depth surface only; project using the image-estimated query pose and known intrinsics. Query depth and true query pose enter scoring only.',
        matching='Frozen native descriptors,33px rectified search and57px valid warp support; same3nativepx AP3P estimator. Surface normalization and search extent change together.',
        occlusion='No query-depth visibility available in inference. Warp validity uses finite/front-facing reference projections and query image bounds, never brightness.',
        cpu_threads=2, timing='Concurrent local diagnostic; no isolated speed claim.', calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json', protocol)
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval()
    controls = control(coarse); write_new(root/'controls.json', controls)
    if not controls['passed']:
        raise ValueError('Actual frozen-model surface-warp control failed; no lunar inference')
    rows = []; data = {}; features = {}
    for key, t in tiles.items():
        if sha256(t['native_cache']) != t['native_cache_sha256']:
            raise ValueError('Native development inputs differ')
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
    for i, c in enumerate(cases):
        t = tiles[c['tile_id']]; z = data[c['tile_id']]; a, b = t['cameras']
        ia, ib = 2*c['light_a'], 2*c['light_b']+1
        selected, branches = infer(coarse, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], features, ((c['tile_id'], ia), (c['tile_id'], ib)))
        branch = selected if selected is not None else branches[0]
        rows.append(dict(c, method='frozen_surface_warp',
            **score(branch['p'], branch['q'], None if selected is None else branch['pose'], z, t['cameras']),
            quarter_turn=branch['quarter_turn'], pose_info=branch['info'], geometry_history=branch.get('geometry_history', [])))
        print('surface warp', i+1, '/', len(cases), 'correct', sum(r['pose_correct'] for r in rows), flush=True)
    with gzip.open(root/'results.json.gz', 'wt') as f:
        json.dump(rows, f, allow_nan=False)
    result = dict(scope=protocol['scope'], cases=len(rows), correct=sum(r['pose_correct'] for r in rows),
        strict=sum(r['strict_pose_correct'] for r in rows), estimated=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows), results_sha256=sha256(root/'results.json.gz'))
    write_new(root/'summary.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--full', action='store_true')
    a = p.parse_args(); run(a.output, not a.full)
