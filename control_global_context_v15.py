"""Full historical-equivalence gate for the untrained context proposal module."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import SOURCES as SURFACE_SOURCES, control
from geometry_warp_native_v11 import infer
from global_context_v15 import PairContext, ContextMatcher
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score
from train_balanced_localization_v3 import save_torch


def run(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve all initial-control attempts')
    root.mkdir(parents=True); torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.manual_seed(20261101)
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    baseline_path = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    audit = json.loads(Path('reports/native_surface_audit_v11.json').read_text())
    if not audit['passed'] or audit['results_sha256'] != sha256(baseline_path):
        raise ValueError('Audited surface baseline required')
    with gzip.open(baseline_path, 'rt') as stream:
        baseline = {r['pair_id']: r for r in json.load(stream)}
    coarse = CompactMatcher(manifest['coarse_model']); context = PairContext('cross_context')
    matcher = ContextMatcher(coarse, context)
    sources = list(dict.fromkeys(SURFACE_SOURCES+['global_context_v15.py', 'control_global_context_v15.py']))
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection']
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    protocol = dict(scope='Untrained context software/control gate only; no new trained result.',
        source_sha256={p: sha256(p) for p in sources}, manifest_sha256=sha256(manifest_path),
        checkpoint_sha256=sha256(Path(manifest['coarse_model'])/'selected.pt'),
        baseline_results_sha256=sha256(baseline_path), case_ids=[c['pair_id'] for c in cases],
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        variant='cross_context', context_parameters=sum(p.numel() for p in context.parameters()),
        total_with_frozen_encoder=28272+sum(p.numel() for p in context.parameters()),
        total_with_separate_pair_fine=28272+35233+sum(p.numel() for p in context.parameters()),
        completed_optimizer_steps=0, query_truth='Scoring/equivalence check only; no query geometry in matching.',
        calibration_or_test_pixels_loaded=0, cpu_threads=2,
        timing='Concurrent local control; no isolated latency claim.')
    if protocol['checkpoint_sha256'] != manifest['coarse_checkpoint_sha256'] or protocol['total_with_separate_pair_fine'] >= 100000:
        raise ValueError('Checkpoint/cap differs')
    write_new(root/'protocol.json', protocol); save_torch(context.state_dict(), root/'initial.pt')
    actual = control(matcher); write_new(root/'controls.json', actual)
    if not actual['passed']:
        raise ValueError('Actual pretrained known-transform/blank control failed')
    data = {}; features = {}; previous = None; rows = []
    for i, c in enumerate(cases):
        key = c['tile_id']; tile = tiles[key]
        if key not in data:
            if sha256(tile['native_cache']) != tile['native_cache_sha256']:
                raise ValueError('Native input differs')
            with np.load(tile['native_cache'], allow_pickle=False) as z:
                data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
        if previous != key:
            features.clear(); previous = key
        z = data[key]; a, b = tile['cameras']; ia, ib = 2*c['light_a'], 2*c['light_b']+1
        selected, branches = infer(matcher, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], features, ((key, ia), (key, ib)))
        result = selected if selected is not None else branches[0]
        row = dict(c, **score(result['p'], result['q'], None if selected is None else result['pose'], z, tile['cameras']),
            quarter_turn=result['quarter_turn'], pose_info=result['info'], geometry_history=result.get('geometry_history', []))
        old = baseline[c['pair_id']]
        for name in ('pose_correct', 'strict_pose_correct', 'reference_points_native', 'query_points_native', 'estimated_cam2world'):
            if row[name] is None or old[name] is None:
                if row[name] != old[name]:
                    raise ValueError('Initial absent pose changed')
            elif isinstance(row[name], bool):
                if row[name] != old[name]:
                    raise ValueError('Initial case success changed')
            elif not np.array_equal(row[name], old[name]):
                raise ValueError('Initial native points/pose changed: '+name)
        rows.append(row); print('context initial exact replay', i+1, '/', len(cases), flush=True)
    with gzip.open(root/'results.json.gz', 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)
    summary = dict(passed=True, cases=len(rows), correct=sum(r['pose_correct'] for r in rows),
        strict=sum(r['strict_pose_correct'] for r in rows), estimated=sum(r['estimated_cam2world'] is not None for r in rows),
        results_sha256=sha256(root/'results.json.gz'), initial_checkpoint_sha256=sha256(root/'initial.pt'),
        exact_native_and_pose_equivalence=True, completed_optimizer_steps=0, scope=protocol['scope'])
    write_new(root/'summary.json', summary); print(json.dumps(summary, indent=2)); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
