"""Independent saved-score/pose, label, sampler and checkpoint audit for v13."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from localization_robust_pose_v4 import CONFIGS, estimate
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner
from surface_learning_v13 import prepare_training, training_batch
from train_balanced_localization_v3 import load_bundle
from verify_axial_results_v5 import projected
from verify_coarse_training_v5 import replay_row


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def aggregate(rows):
    return dict(cases=len(rows), correct_poses=sum(r['pose_correct'] for r in rows),
        strict_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows), scorable=sum(r['scorable'] for r in rows))


def replay_pose(row, tile, data):
    replay_row(row, tile, data)
    p = np.asarray(row['reference_points_native'], float).reshape(-1, 2)
    q = np.asarray(row['query_points_native'], float).reshape(-1, 2)
    indices = np.rint(p).astype(int)
    if not ((indices >= 0) & (indices < np.array(data['depth_a'].shape[::-1]))).all():
        raise ValueError('Invalid reference coordinate')
    a, b = tile['cameras']; ca = np.asarray(a['cam2world'])
    ray = np.column_stack([p, np.ones(len(p))])@np.linalg.inv(np.asarray(a['intrinsics'])).T
    world = (ray*data['depth_a'][indices[:, 1], indices[:, 0]][:, None])@ca[:3, :3].T+ca[:3, 3]
    config = next(c for c in CONFIGS if c['name'] == 'ap3p_reclassify')
    pose, _ = estimate(world, q, b['intrinsics'], config)
    saved = row['estimated_cam2world']
    if (pose is None) != (saved is None) or (pose is not None and not np.allclose(pose, saved, rtol=0, atol=.001)):
        raise ValueError('Common final pose estimator replay differs')
    history = row['geometry_history']
    if len(history) > 2 or [h['iteration'] for h in history] != list(range(1, len(history)+1)):
        raise ValueError('Surface iteration budget differs')
    if any(not h['estimated'] for h in history[:-1]):
        raise ValueError('Continued after a failed refinement')
    if history and not history[-1]['estimated'] and not np.allclose(saved, history[-1]['input_estimated_pose']):
        raise ValueError('Failed update did not retain previous pose')


def audit(root, output):
    root = Path(root); torch.set_num_threads(2); cv2.setNumThreads(1)
    protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    manifest, tiles, data = load_bundle('reports/localization_training_data_v5.json')
    if protocol['manifest_sha256'] != sha256('reports/localization_training_data_v5.json'):
        raise ValueError('Manifest differs')
    if protocol['calibration_or_test_pixels_loaded'] or protocol['fresh_data_gate_passed']:
        raise ValueError('Invalid evaluation scope')
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Executed source differs: '+source)
    train_groups = {t['component'] for t in tiles.values() if t['split'] == 'train'}
    selection_groups = {t['component'] for t in tiles.values() if t['split'] == 'model_selection'}
    if train_groups & selection_groups or any(t['split'] not in ('train', 'model_selection') for t in tiles.values()):
        raise ValueError('Split leakage')
    if prepare_training(tiles, data) != protocol['training_support']:
        raise ValueError('Common training support differs')
    label_points = 0
    for key, tile in tiles.items():
        if tile['split'] != 'train':
            continue
        z = data[key]; p = z['surface_reference']
        target, visible = projected(p, z['depth_a'], z['depth_b'], *tile['cameras'])
        if not visible.all() or not np.allclose(target, z['surface_target'], rtol=0, atol=4e-5):
            raise ValueError('Independent projected train labels differ')
        label_points += len(p)
    if sha256(root/'initial_proposals.json.gz') != protocol['initial_proposals_sha256']:
        raise ValueError('Frozen initializer cache differs')
    expected = [c['pair_id'] for c in manifest['cases'] if c['split'] == 'model_selection'
                and (not protocol['pilot'] or c['pair_id'] in manifest['selection_bank'])]
    coarse = CompactMatcher(manifest['coarse_model']); records = []; replayed = 0; checkpoint_rows = 0
    paths = sorted(root.glob('*/training.json'))
    if len(paths) != (3 if protocol['pilot'] else 5):
        raise ValueError('Missing ablation/seed')
    for path in paths:
        folder = path.parent; meta = json.loads(path.read_text())
        complete = json.loads((folder/'COMPLETE.json').read_text())
        for key, digest in complete.items():
            if sha256(folder/key.removesuffix('_sha256')) != digest:
                raise ValueError('Completed artifact changed')
        for source, digest in protocol['source_sha256'].items():
            if meta['source_sha256'][source] != digest or sha256(folder/'source'/source) != digest:
                raise ValueError('Source snapshot changed')
        torch.manual_seed(meta['seed']); model = NativeRefiner(meta['variant']); model.initialize_encoder(coarse)
        load = lambda name: torch.load(folder/name, map_location='cpu', weights_only=True)
        initial, selected, final, latest = [load(n) for n in ('initial.pt', 'selected.pt', 'final.pt', 'latest.pt')]
        if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
            raise ValueError('Seeded initialization differs')
        model.load_state_dict(selected, strict=True)
        number = sum(p.numel() for p in model.parameters())
        if number != meta['trainable_parameters'] or number+28272 != meta['total_parameters'] or number+28272 >= 100000:
            raise ValueError('Parameter cap differs')
        if latest['step'] != protocol['steps'] or meta['completed_steps'] != protocol['steps']:
            raise ValueError('Optimizer budget differs')
        if any(not torch.equal(v, latest['model'][k]) for k, v in final.items()):
            raise ValueError('Final and optimizer checkpoint differ')
        if {int(s['step']) for s in latest['optimizer']['state'].values()} != {protocol['steps']}:
            raise ValueError('AdamW step counts differ')
        change = float(sum((v-initial[k]).square().sum() for k, v in final.items()).sqrt())
        if change <= 0 or not np.isclose(change, meta['final_weight_change_l2']):
            raise ValueError('Actual learning/weight change differs')
        history = meta['history']; steps = sorted({0, protocol['steps'], *range(protocol['interval'], protocol['steps']+1, protocol['interval'])})
        if [h['step'] for h in history] != steps:
            raise ValueError('Missing validation checkpoint')
        for h in history:
            rows = read_rows(folder/f'selection_step_{h["step"]}.json.gz')
            if [r['pair_id'] for r in rows] != expected or aggregate(rows) != h['selection']:
                raise ValueError('Checkpoint denominator/aggregate differs')
            for row in rows:
                replay_pose(row, tiles[row['tile_id']], data[row['tile_id']])
            checkpoint_rows += len(rows)
        best = max((h['selection']['correct_poses'], h['selection']['correct_at_two_px'], -h['step']) for h in history)
        if list(best) != meta['selection_key'] or -best[2] != meta['selected_step']:
            raise ValueError('Checkpoint selection differs')
        if meta['selected_step'] == 0 and any(not torch.equal(v, selected[k]) for k, v in initial.items()):
            raise ValueError('Selected initialization differs')
        rng = np.random.default_rng(meta['seed']); components = Counter()
        for _ in range(protocol['steps']):
            _, metadata = training_batch(tiles, data, rng, 8, meta['experiment'] != 'raw_encoder')
            components.update(m['component'] for m in metadata)
        if rng.bit_generator.state != latest['numpy_rng'] or dict(components) != meta['sampled_components']:
            raise ValueError('Exact sampler replay differs')
        if meta['sampled_examples'] != 8*protocol['steps']:
            raise ValueError('Training example budget differs')
        rows = read_rows(folder/'development_results.json.gz')
        saved = read_rows(folder/f'selection_step_{meta["selected_step"]}.json.gz')
        if rows != saved or aggregate(rows) != meta['evaluation']:
            raise ValueError('Selected checkpoint readout differs')
        replayed += len(rows)
        records.append({k: meta[k] for k in ('experiment', 'seed', 'completed_steps', 'selected_step', 'selection_key', 'evaluation')})
    order = {e: i for i, e in enumerate(protocol['experiments'])}
    primary = [r for r in records if r['seed'] == 20261101]
    winner = max(primary, key=lambda r: (tuple(r['selection_key']), -order[r['experiment']]))['experiment']
    expected_runs = {(e, 20261101) for e in protocol['experiments']}
    if not protocol['pilot']:
        expected_runs |= {(winner, s) for s in (20261102, 20261103)}
    if {(r['experiment'], r['seed']) for r in records} != expected_runs or summary['winner'] != winner:
        raise ValueError('Configuration/seed selection differs')
    if sorted(records, key=lambda r: (r['experiment'], r['seed'])) != sorted(summary['results'], key=lambda r: (r['experiment'], r['seed'])):
        raise ValueError('Cohort summary differs')
    result = dict(passed=True, trained_runs=len(records), optimizer_steps=protocol['steps']*len(records),
        independent_training_label_points=label_points, selected_rows=replayed, checkpoint_rows=checkpoint_rows,
        winner=winner, runs=records, source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'),
        scope='Development replay of saved metrics/pose estimator, labels, initialization, optimizer and sampler; not independent retraining or fresh validation.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); audit(args.run, args.output)
