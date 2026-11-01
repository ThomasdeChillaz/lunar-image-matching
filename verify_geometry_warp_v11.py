"""Independent native-error, axial-depth pose and saved control replay."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from localization_robust_pose_v4 import CONFIGS, estimate
from lunargeo_dataset import write_new
from verify_coarse_training_v5 import replay_row


def audit(root, output):
    root = Path(root); protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text()); control = json.loads((root/'controls.json').read_text())
    for p, h in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(p) != h:
            raise ValueError('Executed surface-warp source/input differs')
    path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(path.read_text())
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if (sha256(path) != protocol['manifest_sha256'] or sha256(checkpoint) != protocol['frozen_checkpoint_sha256']
            or protocol['frozen_checkpoint_sha256'] != manifest['coarse_checkpoint_sha256']
            or protocol['calibration_or_test_pixels_loaded'] != 0 or protocol['iterations'] != 2):
        raise ValueError('Surface-warp manifest/checkpoint/scope differs')
    expected_rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    cp = np.asarray(control['estimated_cam2world'])
    rotation = float(np.linalg.norm(cp[:3, :3]-expected_rotation)); translation = float(np.linalg.norm(cp[:3, 3]))
    if (not control['passed'] or not control['blank_pose_null'] or rotation >= .01 or translation >= .05
            or not np.allclose([rotation, translation], [control['rotation_matrix_error'], control['translation_error_m']])):
        raise ValueError('Actual frozen-model surface-warp control differs')
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (not protocol['bank_only'] or c['pair_id'] in manifest['selection_bank'])]
    if protocol['case_ids'] != [c['pair_id'] for c in cases] or sha256(root/'results.json.gz') != summary['results_sha256']:
        raise ValueError('Eligible surface-warp cases/results changed')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for key, t in tiles.items():
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    with gzip.open(root/'results.json.gz', 'rt') as f:
        rows = json.load(f)
    if len(rows) != len(cases) or [r['pair_id'] for r in rows] != protocol['case_ids']:
        raise ValueError('Missing/repeated surface-warp cases')
    cv2.setNumThreads(1); config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    for row, case in zip(rows, cases):
        if any(row[k] != v for k, v in case.items()):
            raise ValueError('Surface-warp case definition differs')
        t = tiles[row['tile_id']]; z = data[row['tile_id']]; replay_row(row, t, z)
        history = row['geometry_history']
        if len(history) > 2 or [h['iteration'] for h in history] != list(range(1, len(history)+1)):
            raise ValueError('Fixed geometry iteration budget differs')
        if any(not h['estimated'] for h in history[:-1]):
            raise ValueError('Refinement continued after failed pose')
        p = np.array(row['reference_points_native']).reshape(-1, 2); q = np.array(row['query_points_native']).reshape(-1, 2)
        indices = np.rint(p).astype(int)
        if not ((indices >= 0) & (indices < 512)).all():
            raise ValueError('Invalid native reference centers')
        a, b = t['cameras']; ca = np.array(a['cam2world'])
        rays = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(np.array(a['intrinsics'])).T
        world = (rays*z['depth_a'][indices[:, 1], indices[:, 0]][:, None])@ca[:3, :3].T+ca[:3, 3]
        pose, _ = estimate(world, q, b['intrinsics'], config); saved = row['estimated_cam2world']
        if (pose is None) != (saved is None) or (pose is not None and not np.allclose(pose, saved, atol=.001, rtol=0)):
            raise ValueError('Surface-warp common pose estimator replay differs')
        if history and not history[-1]['estimated'] and not np.allclose(saved, history[-1]['input_estimated_pose']):
            raise ValueError('Failed update did not retain last successful raw pose')
    counts = dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows), strict=sum(r['strict_pose_correct'] for r in rows),
        estimated=sum(r['estimated_cam2world'] is not None for r in rows), correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows))
    if any(summary[k] != v for k, v in counts.items()):
        raise ValueError('Surface-warp aggregate differs')
    result = dict(passed=True, scope='Independent scoring/control/pose-estimator replay of a deterministic development hypothesis; not a learned, accepted-pose or fresh-test result.',
        results=counts, source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'), results_sha256=summary['results_sha256'],
        verification_limits='Does not independently reimplement learned image inference or assert query-depth visibility. Surface warp uses only estimated query poses.',
        calibration_or_test_pixels_loaded=0)
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
