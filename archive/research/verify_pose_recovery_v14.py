"""Audit broad-initialization routing and independently replay strict final poses."""
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


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def audit(root, output):
    root = Path(root); cv2.setNumThreads(1)
    protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    controls = json.loads((root/'controls.json').read_text())
    if protocol['calibration_or_test_pixels_loaded'] or protocol['initial_threshold_px'] != 12 or protocol['final_threshold_px'] != 3:
        raise ValueError('Scope or final estimator threshold differs')
    for name, digest in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(name) != digest:
            raise ValueError('Executed source/input changed: '+name)
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != protocol['manifest_sha256'] or sha256(Path(manifest['coarse_model'])/'selected.pt') != protocol['checkpoint_sha256']:
        raise ValueError('Frozen geometry/checkpoint changed')
    old_path = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    if sha256(old_path) != protocol['baseline_results_sha256'] or sha256('reports/native_surface_audit_v11.json') != protocol['baseline_audit_sha256']:
        raise ValueError('Historical control changed')
    expected_rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    pose = np.asarray(controls['estimated_cam2world'])
    if (not controls['passed'] or not controls['blank_pose_null'] or np.linalg.norm(pose[:3, :3]-expected_rotation) >= .01
            or np.linalg.norm(pose[:3, 3]) >= .05):
        raise ValueError('Actual software controls differ')
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (protocol['full'] or c['pair_id'] in manifest['selection_bank'])]
    rows = read(root/'results.json.gz'); baseline = {r['pair_id']: r for r in read(old_path)}
    if ([c['pair_id'] for c in cases] != protocol['case_ids'] or [r['pair_id'] for r in rows] != protocol['case_ids']
            or sha256(root/'results.json.gz') != summary['results_sha256']):
        raise ValueError('Missing/repeated/changed cases')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}; initial_errors = []
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    for row, case in zip(rows, cases):
        if any(row[k] != v for k, v in case.items()):
            raise ValueError('Case definition differs')
        old = baseline[row['pair_id']]; route = row['recovery']; attempted = old['estimated_cam2world'] is None
        if route['attempted'] != attempted or row['reused_unchanged_baseline'] == attempted or route['final_threshold_px'] != 3:
            raise ValueError('Availability-only recovery routing differs')
        if not attempted:
            if any(row[k] != v for k, v in old.items() if k != 'method'):
                raise ValueError('Historical estimated baseline was modified')
        else:
            branches = route['broad_branches']
            if [b['quarter_turn'] for b in branches] != [0, 1, 2, 3]:
                raise ValueError('Missing broad orientation branch')
            valid = [b for b in branches if b['estimated']]
            if valid:
                chosen = max(valid, key=lambda b: (b['proposal_info']['inliers'], -b['proposal_info']['median_proposal_residual'], -b['quarter_turn']))
                if chosen['quarter_turn'] != route['initial_proposal_quarter_turn']:
                    raise ValueError('Truth-free broad branch ranking differs')
                initial = np.asarray(route['initial_proposal_cam2world']); truth = np.asarray(tiles[row['tile_id']]['cameras'][1]['cam2world'])
                angle = float(np.degrees(np.arccos(np.clip((np.trace(initial[:3, :3].T@truth[:3, :3])-1)/2, -1, 1))))
                initial_errors.append(dict(pair_id=row['pair_id'], rotation_deg=angle,
                    translation_m=float(np.linalg.norm(initial[:3, 3]-truth[:3, 3])), final_correct=row['pose_correct']))
            success = any(h['estimated'] for h in row['geometry_history'])
            if route['recovered'] != success or (row['estimated_cam2world'] is not None) != success:
                raise ValueError('A broad-only estimate was returned as a final pose')
        key = row['tile_id']; tile = tiles[key]
        if key not in data:
            with np.load(tile['native_cache'], allow_pickle=False) as z:
                data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
        z = data[key]; replay_row(row, tile, z)
        p = np.asarray(row['reference_points_native'], float).reshape(-1, 2)
        q = np.asarray(row['query_points_native'], float).reshape(-1, 2); indices = np.rint(p).astype(int)
        if not ((indices >= 0) & (indices < np.array(z['depth_a'].shape[::-1]))).all():
            raise ValueError('Invalid native reference pixels')
        a, b = tile['cameras']; ca = np.asarray(a['cam2world'])
        ray = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(np.asarray(a['intrinsics'])).T
        world = (ray*z['depth_a'][indices[:, 1], indices[:, 0]][:, None])@ca[:3, :3].T+ca[:3, 3]
        final, _ = estimate(world, q, b['intrinsics'], config); saved = row['estimated_cam2world']
        if (final is None) != (saved is None) or (final is not None and not np.allclose(final, saved, rtol=0, atol=.001)):
            raise ValueError('Unchanged3px final estimator replay differs')
        history = row['geometry_history']
        if len(history) > 2 or any(not h['estimated'] for h in history[:-1]):
            raise ValueError('Surface iteration budget/failure behavior differs')
        if history and not history[-1]['estimated'] and saved is not None and not np.allclose(saved, history[-1]['input_estimated_pose']):
            raise ValueError('Failed later update did not retain prior strict pose')
    counts = dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows), strict=sum(r['strict_pose_correct'] for r in rows),
        estimated=sum(r['estimated_cam2world'] is not None for r in rows), wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows),
        attempted=sum(r['recovery']['attempted'] for r in rows), recovered=sum(r['recovery']['recovered'] for r in rows),
        newly_correct=sum(r['pose_correct'] and r['recovery']['attempted'] for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows))
    if any(summary[k] != v for k, v in counts.items()):
        raise ValueError('Recovery summary differs')
    result = dict(passed=True, results=counts, initial_proposal_errors=initial_errors,
        source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'), results_sha256=summary['results_sha256'],
        scope='Independent native metric and unchanged3px final-estimator replay, routing/source/checkpoint/control audit; no fresh test or confidence guarantee.',
        limits='Does not independently reimplement image matching or replay every12px proposal fit. A strict inlier fit can still be a wrong pose.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); audit(args.run, args.output)
