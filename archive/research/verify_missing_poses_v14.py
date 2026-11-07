"""Independent native metrics and broad-estimator replay of every missing branch."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from pose_recovery_v14 import broad_proposal
from verify_coarse_training_v5 import replay_row


def audit(root, output):
    root = Path(root); cv2.setNumThreads(1)
    protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    for name, digest in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(name) != digest:
            raise ValueError('Source/input changed')
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    if protocol['calibration_or_test_pixels_loaded'] or sha256(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('Scope/manifest changed')
    if sha256(Path(manifest['coarse_model'])/'selected.pt') != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    original = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    if sha256(original) != protocol['original_results_sha256']:
        raise ValueError('Original cases changed')
    with gzip.open(original, 'rt') as stream:
        expected = [r['pair_id'] for r in json.load(stream) if r['estimated_cam2world'] is None]
    with gzip.open(root/'results.json.gz', 'rt') as stream:
        rows = json.load(stream)
    if sha256(root/'results.json.gz') != summary['results_sha256'] or protocol['case_ids'] != expected:
        raise ValueError('Missing-case freeze/results changed')
    if [(r['pair_id'], r['quarter_turn']) for r in rows] != [(p, k) for p in expected for k in range(4)]:
        raise ValueError('Missing/repeated branches')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}; data = {}
    for row in rows:
        key = row['tile_id']; tile = tiles[key]
        if key not in data:
            with np.load(tile['native_cache'], allow_pickle=False) as z:
                data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
        z = data[key]; replay_row(row, tile, z)
        p = np.asarray(row['reference_points_native'], float).reshape(-1, 2)
        q = np.asarray(row['query_points_native'], float).reshape(-1, 2); index = np.rint(p).astype(int)
        a, b = tile['cameras']; ca = np.asarray(a['cam2world'])
        rays = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(np.asarray(a['intrinsics'])).T
        world = (rays*z['depth_a'][index[:, 1], index[:, 0]][:, None])@ca[:3, :3].T+ca[:3, 3]
        pose, info = broad_proposal(world, q, b['intrinsics']); saved = row['estimated_cam2world']
        if (pose is None) != (saved is None) or (pose is not None and not np.allclose(pose, saved, rtol=0, atol=.001)) or info != row['proposal_info']:
            raise ValueError('12px proposal estimator replay differs')
    chosen_correct = oracle_correct = 0; per_case = []
    for pair in expected:
        part = [r for r in rows if r['pair_id'] == pair]; valid = [r for r in part if r['estimated_cam2world'] is not None]
        chosen = max(valid, key=lambda r: (r['proposal_info']['inliers'], -r['proposal_info']['median_proposal_residual'], -r['quarter_turn'])) if valid else None
        if any(r['selected'] != (r is chosen) for r in part):
            raise ValueError('Image/reference-only branch choice differs')
        chosen_correct += bool(chosen and chosen['pose_correct']); oracle_correct += any(r['pose_correct'] for r in part)
        per_case.append(dict(pair_id=pair, max_true_12px_matches=max(r['correct_at_native_px']['12'] for r in part),
            max_true_2px_matches=max(r['correct_at_native_px']['2'] for r in part),
            max_true_12px_fraction=max(r['correct_at_native_px']['12']/max(r['returned'], 1) for r in part),
            best_broad_rotation_error_deg=min([r['pose_error']['rotation_deg'] for r in part if r['pose_error'] is not None], default=None),
            any_correct_broad_pose=any(r['pose_correct'] for r in part)))
    if (len(expected), len(rows), chosen_correct, oracle_correct, per_case) != (summary['cases'], summary['branches'],
            summary['image_selected_broad_correct'], summary['oracle_any_broad_correct'], summary['per_case']):
        raise ValueError('Diagnostic aggregate differs')
    result = dict(passed=True, cases=len(expected), branches=len(rows), image_selected_broad_correct=chosen_correct,
        oracle_any_broad_correct=oracle_correct, source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'),
        results_sha256=summary['results_sha256'], scope='Independent native projection/pose-error and all12px broad-estimator replay; oracle is diagnosis only, not deployed selection.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); audit(args.run, args.output)
