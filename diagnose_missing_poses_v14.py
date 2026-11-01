"""Score every coarse/broad branch on the15 previously missing development poses.

Oracle branch counts are diagnostics only. Query truth never chooses a model
output, constructs a proposal, or changes the frozen broad initializer.
"""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer as initialize
from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_pose_recovery_v14 import SOURCES
from lunar_geometry_v5 import map_points_z
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score
from pose_recovery_v14 import broad_proposal


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve all diagnostics')
    root.mkdir(parents=True); torch.set_num_threads(2); cv2.setNumThreads(1)
    path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(path.read_text())
    original = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    with gzip.open(original, 'rt') as stream:
        missing = {r['pair_id'] for r in json.load(stream) if r['estimated_cam2world'] is None}
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection' and c['pair_id'] in missing]
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Coarse checkpoint differs')
    protocol = dict(scope='Exposed development diagnosis of all coarse/broad branches, not an oracle-deployed matcher or new test.',
        source_sha256={p: sha256(p) for p in SOURCES+['diagnose_missing_poses_v14.py']},
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        manifest_sha256=sha256(path), original_results_sha256=sha256(original),
        checkpoint_sha256=sha256(checkpoint), case_ids=[c['pair_id'] for c in cases],
        selection='Every previously missing pose, all four orientations, unchanged12px broad initializer. Query truth scores only after predictions.',
        calibration_or_test_pixels_loaded=0, cpu_threads=2)
    write_new(root/'protocol.json', protocol)
    coarse = CompactMatcher(manifest['coarse_model']); data = {}; features = {}; previous = None; rows = []
    for i, c in enumerate(cases):
        key = c['tile_id']; tile = tiles[key]
        if key not in data:
            if sha256(tile['native_cache']) != tile['native_cache_sha256']:
                raise ValueError('Input differs')
            with np.load(tile['native_cache'], allow_pickle=False) as z:
                data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
        if previous != key:
            features.clear(); previous = key
        z = data[key]; a, b = tile['cameras']; ia, ib = 2*c['light_a'], 2*c['light_b']+1
        selected, branches = initialize(coarse, coarse.model, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], features, ((key, ia), (key, ib)))
        if selected is not None:
            raise ValueError('Historical missing-pose definition differs')
        proposals = []
        for branch in branches:
            world, valid = map_points_z(branch['p'], z['depth_a'], a['intrinsics'], a['cam2world'])
            pose, info = broad_proposal(world, branch['q'][valid], b['intrinsics'])
            proposals.append(dict(branch, pose=pose, info=info))
        valid = [r for r in proposals if r['pose'] is not None]
        chosen = max(valid, key=lambda r: (r['info']['inliers'], -r['info']['median_proposal_residual'], -r['quarter_turn'])) if valid else None
        for branch in proposals:
            row = dict(c, quarter_turn=branch['quarter_turn'], selected=branch is chosen,
                proposal_info=branch['info'], **score(branch['p'], branch['q'], branch['pose'], z, tile['cameras']))
            rows.append(row)
        print('missing-pose branch diagnosis', i+1, '/', len(cases), flush=True)
    with gzip.open(root/'results.json.gz', 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)
    report = dict(scope=protocol['scope'], cases=len(cases), branches=len(rows),
        image_selected_broad_correct=sum(r['selected'] and r['pose_correct'] for r in rows),
        oracle_any_broad_correct=sum(any(r['pose_correct'] for r in rows if r['pair_id'] == c['pair_id']) for c in cases),
        per_case=[dict(pair_id=c['pair_id'],
            max_true_12px_matches=max(r['correct_at_native_px']['12'] for r in rows if r['pair_id'] == c['pair_id']),
            max_true_2px_matches=max(r['correct_at_native_px']['2'] for r in rows if r['pair_id'] == c['pair_id']),
            max_true_12px_fraction=max(r['correct_at_native_px']['12']/max(r['returned'], 1) for r in rows if r['pair_id'] == c['pair_id']),
            best_broad_rotation_error_deg=min([r['pose_error']['rotation_deg'] for r in rows if r['pair_id'] == c['pair_id'] and r['pose_error'] is not None], default=None),
            any_correct_broad_pose=any(r['pose_correct'] for r in rows if r['pair_id'] == c['pair_id'])) for c in cases],
        results_sha256=sha256(root/'results.json.gz'),
        note='Oracle scores do not choose outputs. Small true-inlier fractions and broad branch pose errors help distinguish global proposal failure from final precision.')
    write_new(root/'summary.json', report); print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
