"""Development-only deterministic affine preconditioning experiment."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from affine_native_v10 import infer
from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score


SOURCES = ['affine_native_v10.py', 'evaluate_affine_native_v10.py', 'native_baseline_pose_v7.py',
    'localization_diagnostics_v3.py', 'coarse_pipeline_v5.py', 'compact_lunar.py', 'contrastive_terrain.py',
    'localization_robust_pose_v4.py', 'localization_pose_refit_v3.py', 'lunar_geometry_v5.py',
    'lunar_geometry_v2.py', 'rotation_bank_pose_v4.py']


def control(coarse):
    from synthetic import texture
    image = texture(20261101, 512)
    query = cv2.warpAffine(image, np.array([[1., 0, 8.], [0, 1., -4.]]), (512, 512))
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    depth = np.full(image.shape, 10.)
    selected, _ = infer(coarse, image, query, depth, k, np.eye(4), k)
    if selected is None:
        raise ValueError('Actual frozen matcher failed affine known-transform control')
    pose = selected['pose']
    pixel = np.linalg.norm(selected['q']-selected['p']-[8., -4.], axis=1)
    translation = float(np.linalg.norm(pose[:3, 3]-[-.2, .1, 0.]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3])-1)/2, -1, 1))))
    blank, _ = infer(coarse, np.zeros_like(image), np.zeros_like(image), depth, k, np.eye(4), k)
    passed = bool(len(pixel) >= 6 and np.median(pixel) <= 1 and translation <= .05 and rotation <= .5 and blank is None)
    return dict(passed=passed, median_native_error_px=float(np.median(pixel)), translation_error_m=translation,
        rotation_error_deg=rotation, blank_pose_null=blank is None,
        reference_points_native=selected['p'].tolist(), query_points_native=selected['q'].tolist(),
        estimated_cam2world=pose.tolist(), image_seed=20261101, known_translation_native=[8., -4.])


def run(output, bank_only=True):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve previous experiments')
    root.mkdir(parents=True); torch.set_num_threads(2); cv2.setNumThreads(1)
    path = Path('reports/localization_training_data_v5.json')
    manifest = json.loads(path.read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (not bank_only or c['pair_id'] in manifest['selection_bank'])]
    protocol = dict(scope='Exposed development deterministic engineering hypothesis; no learning, calibrated acceptance or fresh test.',
        bank_only=bank_only, case_ids=[c['pair_id'] for c in cases], cpu_threads=2,
        manifest_sha256=sha256(path), source_sha256={p: sha256(p) for p in SOURCES},
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        hypothesis='Viewpoint scale/shear reduces frozen local descriptor precision; affine preconditioning uses coarse image matches only.',
        settings='4quarter-turns; affine RANSAC12nativepx,2000iterations,.99,6inliers,min/maxsingular0.5/2; native33px rectified search,12px valid feature support; same3nativepx AP3P estimator.',
        query_truth='No query depth/true pose accepted by inference; scoring only.',
        interpretation='Affine-normalized33px search differs from a33native-query-pixel window. No formal oracle bound or speed claim; concurrent local work.',
        calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json', protocol)
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval()
    controls = control(coarse); write_new(root/'controls.json', controls)
    if not controls['passed']:
        raise ValueError('Actual frozen-model controls failed; no lunar inference')
    rows = []; data = {}; features = {}
    for key, t in tiles.items():
        if sha256(t['native_cache']) != t['native_cache_sha256']:
            raise ValueError('Native input cache differs')
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
    for i, c in enumerate(cases):
        t = tiles[c['tile_id']]; z = data[c['tile_id']]; a, b = t['cameras']
        ia, ib = 2*c['light_a'], 2*c['light_b']+1; tick = perf_counter()
        selected, branches = infer(coarse, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], features, ((c['tile_id'], ia), (c['tile_id'], ib)))
        branch = selected if selected is not None else branches[0]
        elapsed = (perf_counter()-tick)*1000
        row = dict(c, method='frozen_affine_native', elapsed_ms=elapsed,
            **score(branch['p'], branch['q'], None if selected is None else branch['pose'], z, t['cameras']),
            quarter_turn=branch['quarter_turn'], pose_info=branch['info'], affine=branch['affine'],
            all_branches=[dict(quarter_turn=v['quarter_turn'], affine=v['affine'], info=v['info'],
                estimated=v['pose'] is not None, median_inlier_residual=v['median_inlier_residual']) for v in branches])
        rows.append(row)
        print('affine native', i+1, '/', len(cases), 'correct', sum(r['pose_correct'] for r in rows), flush=True)
    with gzip.open(root/'results.json.gz', 'wt') as f:
        json.dump(rows, f, allow_nan=False)
    result = dict(scope=protocol['scope'], cases=len(rows), correct=sum(r['pose_correct'] for r in rows),
        strict=sum(r['strict_pose_correct'] for r in rows), estimated=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        results_sha256=sha256(root/'results.json.gz'), protocol_sha256=sha256(root/'protocol.json'))
    write_new(root/'summary.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--full', action='store_true')
    a = p.parse_args(); run(a.output, not a.full)
