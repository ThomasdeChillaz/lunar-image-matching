"""Replay native coordinates, branch selection and pose of the affine control."""
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
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    controls = json.loads((root/'controls.json').read_text())
    for p, h in {**protocol['source_sha256'], **protocol['input_sha256']}.items():
        if sha256(p) != h:
            raise ValueError('Executed affine control source/input differs')
    manifest_path = Path('reports/localization_training_data_v5.json')
    if sha256(manifest_path) != protocol['manifest_sha256'] or protocol['calibration_or_test_pixels_loaded'] != 0:
        raise ValueError('Affine manifest/scope differs')
    if (sha256(root/'results.json.gz') != summary['results_sha256']
            or sha256(root/'protocol.json') != summary['protocol_sha256']):
        raise ValueError('Saved affine result changed')
    cp, cq = [np.asarray(controls[k]).reshape(-1, 2) for k in ('reference_points_native', 'query_points_native')]
    error = float(np.median(np.linalg.norm(cq-cp-[8., -4.], axis=1)))
    pose = np.array(controls['estimated_cam2world'])
    translation = float(np.linalg.norm(pose[:3, 3]-[-.2, .1, 0.]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3])-1)/2, -1, 1))))
    if (not controls['passed'] or not controls['blank_pose_null'] or error > 1 or translation > .05 or rotation > .5
            or not np.allclose([error, translation, rotation],
                [controls['median_native_error_px'], controls['translation_error_m'], controls['rotation_error_deg']])):
        raise ValueError('Actual affine model control differs')
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse checkpoint differs')
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection'
             and (not protocol['bank_only'] or c['pair_id'] in manifest['selection_bank'])]
    if protocol['case_ids'] != [c['pair_id'] for c in cases]:
        raise ValueError('Eligible affine cases changed')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    data = {}
    for key, t in tiles.items():
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('depth_a', 'depth_b')}
    with gzip.open(root/'results.json.gz', 'rt') as f:
        rows = json.load(f)
    if len(rows) != len(cases) or [r['pair_id'] for r in rows] != protocol['case_ids']:
        raise ValueError('Missing or repeated affine cases')
    cv2.setNumThreads(1)
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    for row, case in zip(rows, cases):
        if any(row[k] != v for k, v in case.items()):
            raise ValueError('Affine case identity differs')
        t = tiles[row['tile_id']]; z = data[row['tile_id']]
        replay_row(row, t, z)
        branches = row['all_branches']
        if len(branches) != 4 or {b['quarter_turn'] for b in branches} != set(range(4)):
            raise ValueError('Missing affine orientation')
        valid = [b for b in branches if b['estimated']]
        selected = max(valid, key=lambda b: (b['info']['inliers'], -b['median_inlier_residual'], -b['quarter_turn'])) if valid else branches[0]
        if row['quarter_turn'] != selected['quarter_turn'] or row['affine'] != selected['affine']:
            raise ValueError('Image/reference-geometry branch choice differs')
        p = np.array(row['reference_points_native']).reshape(-1, 2)
        q = np.array(row['query_points_native']).reshape(-1, 2)
        a, b = t['cameras']; indices = np.rint(p).astype(int)
        if not ((indices >= 0) & (indices < 512)).all():
            raise ValueError('Reference point outside native image')
        ray = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(np.array(a['intrinsics'])).T
        depth = z['depth_a'][indices[:, 1], indices[:, 0]]
        ca = np.array(a['cam2world']); world = (ray*depth[:, None])@ca[:3, :3].T+ca[:3, 3]
        pose, _ = estimate(world, q, b['intrinsics'], config)
        saved = row['estimated_cam2world']
        if (pose is None) != (saved is None) or (pose is not None and not np.allclose(pose, saved, atol=.001, rtol=0)):
            raise ValueError('Common reference-map pose estimator replay differs')
    result = dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows), strict=sum(r['strict_pose_correct'] for r in rows),
        estimated=sum(r['estimated_cam2world'] is not None for r in rows), correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows))
    if any(summary[k] != v for k, v in result.items()):
        raise ValueError('Affine saved aggregate differs')
    report = dict(passed=True, scope='Exposed-development deterministic affine-control replay; not a learned or final-test result.',
        results=result, actual_model_control_replayed=True, pose_estimator_reexecuted=True,
        frozen_checkpoint_sha256=sha256(checkpoint),
        source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'), results_sha256=summary['results_sha256'])
    write_new(output, report); print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
