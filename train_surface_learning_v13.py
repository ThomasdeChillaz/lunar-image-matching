"""Matched raw/geometry-normalized supervision with a frozen pose initializer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer as initialize
from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import control as frozen_control
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score
from native_refinement_v7 import NativeRefiner, objective
from surface_learning_v13 import EXPERIMENTS, prepare_training, refine_surface, training_batch
from train_balanced_localization_v3 import load_bundle, save_torch

MANIFEST = 'reports/localization_training_data_v5.json'
SEEDS = (20261101, 20261102, 20261103)
SOURCES = ['train_surface_learning_v13.py', 'surface_learning_v13.py',
    'geometry_warp_native_v11.py', 'evaluate_geometry_warp_v11.py', 'coarse_pipeline_v5.py',
    'native_refinement_v7.py', 'localization_refiner_v3.py', 'native_baseline_pose_v7.py',
    'train_balanced_localization_v3.py', 'prepare_localization_training_v3.py',
    'compact_lunar.py', 'contrastive_terrain.py', 'localization_diagnostics_v3.py',
    'lunar_geometry_v5.py', 'lunar_geometry_v2.py', 'localization_robust_pose_v4.py',
    'localization_pose_refit_v3.py', 'rotation_bank_pose_v4.py']


def save_rows(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with gzip.open(path, 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)


def counts(rows):
    return dict(cases=len(rows), correct_poses=sum(r['pose_correct'] for r in rows),
        strict_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows), scorable=sum(r['scorable'] for r in rows))


def check_sources(protocol):
    if protocol['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Frozen development manifest changed')
    for name, digest in protocol['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Pinned training source changed: '+name)


def prepare(root, bundle, coarse):
    manifest, tiles, data = bundle
    controls = frozen_control(coarse)
    write_new(root/'initializer_controls.json', controls)
    if not controls['passed']:
        raise ValueError('Actual known-transform/blank initializer control failed')
    initial = []; features = {}; previous_tile = None
    for c in manifest['cases']:
        if c['split'] != 'model_selection':
            continue
        if c['tile_id'] != previous_tile:
            features.clear(); previous_tile = c['tile_id']
        tile = tiles[c['tile_id']]; z = data[c['tile_id']]; a, b = tile['cameras']
        ia, ib = 2*c['light_a'], 2*c['light_b']+1
        selected, branches = initialize(coarse, coarse.model, z['images'][ia], z['images'][ib],
            z['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics'], features,
            ((c['tile_id'], ia), (c['tile_id'], ib)))
        branch = selected if selected is not None else branches[0]
        pose = None if selected is None else branch['pose']
        initial.append(dict(c, p=branch['p'].tolist(), q=branch['q'].tolist(),
            pose=None if pose is None else pose.tolist(), info=branch['info'],
            quarter_turn=branch['quarter_turn'], median_inlier_residual=branch['median_inlier_residual']))
        print('freeze initializer', len(initial), '/72', flush=True)
    save_rows(root/'initial_proposals.json.gz', initial)
    return {r['pair_id']: r for r in initial}


def evaluate(model, bundle, proposals, bank_only=False):
    manifest, tiles, data = bundle; rows = []; features = {}; previous_tile = None
    model.eval()
    for c in manifest['cases']:
        if c['split'] != 'model_selection' or (bank_only and c['pair_id'] not in manifest['selection_bank']):
            continue
        if c['tile_id'] != previous_tile:
            features.clear(); previous_tile = c['tile_id']
        tile = tiles[c['tile_id']]; z = data[c['tile_id']]; a, b = tile['cameras']
        ia, ib = 2*c['light_a'], 2*c['light_b']+1; identity = (c['tile_id'], ia)
        initial = proposals[c['pair_id']].copy()
        for key in ('p', 'q', 'pose'):
            if initial[key] is not None:
                initial[key] = np.asarray(initial[key], dtype=np.float32 if key != 'pose' else float)
        if initial['pose'] is not None and identity not in features:
            with torch.inference_mode():
                features[identity] = model.encoder(torch.from_numpy(z['images'][ia].copy()).float()[None, None]/255)
        result = refine_surface(model, z['images'][ia], z['images'][ib], z['depth_a'],
            a['intrinsics'], a['cam2world'], b['intrinsics'], initial, features.get(identity))
        # Query depth and true query pose first enter the evaluation here.
        rows.append(dict(c, **score(result['p'], result['q'], result['pose'], z, tile['cameras']),
            quarter_turn=result['quarter_turn'], pose_info=result['info'],
            geometry_history=result.get('geometry_history', [])))
    return counts(rows), rows


def verify_initial(root, bundle, coarse, proposals):
    historical = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    with gzip.open(historical, 'rt') as stream:
        old = {r['pair_id']: r for r in json.load(stream)}
    results = {}
    for variant in ('encoder_3d', 'pair_refinement'):
        torch.manual_seed(SEEDS[0]); model = NativeRefiner(variant); model.initialize_encoder(coarse)
        summary, rows = evaluate(model, bundle, proposals)
        if summary['cases'] != 72 or summary['correct_poses'] != 54 or summary['strict_poses'] != 16:
            raise ValueError('Surface initialization changed historical pose counts')
        for row in rows:
            previous = old[row['pair_id']]
            for name in ('pose_correct', 'strict_pose_correct'):
                if row[name] != previous[name]:
                    raise ValueError('Initial case status differs from v11: '+row['pair_id'])
            if variant == 'encoder_3d':
                for name in ('reference_points_native', 'query_points_native', 'estimated_cam2world'):
                    if row[name] is None or previous[name] is None:
                        if row[name] != previous[name]:
                            raise ValueError('Initial missing pose differs')
                    elif not np.allclose(row[name], previous[name], rtol=0, atol=.001):
                        raise ValueError('Exact deterministic initial replay differs: '+name)
        save_rows(root/f'initial_{variant}_results.json.gz', rows)
        results[variant] = summary
    result = dict(passed=True, historical_sha256=sha256(historical), results=results,
        comparison='Both variants preserve every primary/strict status; encoder-only also replays native points and poses within0.001.')
    write_new(root/'initial_equivalence.json', result)
    return result


def train(experiment, seed, root, bundle, coarse, proposals, protocol):
    folder = root/f'{experiment}-{seed}'; folder.mkdir(); (folder/'source').mkdir()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    variant = 'pair_refinement' if experiment == 'normalized_pair' else 'encoder_3d'
    model = NativeRefiner(variant); model.initialize_encoder(coarse)
    total = 28272+sum(p.numel() for p in model.parameters())
    if total >= 100000:
        raise ValueError('Combined model exceeds parameter cap')
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_torch(initial, folder/'initial.pt')
    config = dict(protocol, experiment=experiment, variant=variant, seed=seed,
        trainable_parameters=total-28272, total_parameters=total,
        initial_proposals_sha256=sha256(root/'initial_proposals.json.gz'),
        started_utc=datetime.now(timezone.utc).isoformat())
    write_new(folder/'frozen_config.json', config)
    for source in SOURCES:
        shutil.copy2(source, folder/'source'/source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    history = []; components = Counter(); started = perf_counter(); best = None; selected = 0
    for step in range(protocol['steps']+1):
        loss = None
        if step:
            model.train(); batch, metadata = training_batch(bundle[1], bundle[2], rng, 8, experiment != 'raw_encoder')
            components.update(m['component'] for m in metadata)
            optimizer.zero_grad(set_to_none=True); local, no_match = model(*batch[:3])
            loss = objective(local, no_match, batch[3], batch[4], False)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fine-matching loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if step % protocol['interval'] == 0 or step == protocol['steps']:
            check_sources(protocol)
            summary, rows = evaluate(model, bundle, proposals, protocol['pilot'])
            save_rows(folder/f'selection_step_{step}.json.gz', rows)
            key = (summary['correct_poses'], summary['correct_at_two_px'], -step)
            history.append(dict(step=step, loss=None if loss is None else float(loss), selection=summary,
                seconds=perf_counter()-started))
            if best is None or key > best:
                best, selected = key, step; save_torch(model.state_dict(), folder/'selected.pt')
            write_new(folder/f'validation_step_{step}.json', history[-1])
            print(experiment, seed, 'step', step, summary, 'loss', None if loss is None else round(float(loss), 4), flush=True)
        if step and step % 100 == 0:
            save_torch(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), history=history,
                best=best, selected_step=selected, components=dict(components)), folder/'latest.pt')
            print(experiment, seed, 'gradient', step, 'loss', round(float(loss), 4), flush=True)
        if perf_counter()-started > protocol['per_run_wall_cap_seconds']:
            write_new(folder/'TIMEOUT.json', dict(step=step, elapsed_seconds=perf_counter()-started))
            raise TimeoutError('Per-run wall budget exhausted; retain incomplete artifacts')
    save_torch(model.state_dict(), folder/'final.pt')
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    model.load_state_dict(torch.load(folder/'selected.pt', map_location='cpu', weights_only=True))
    evaluation, rows = evaluate(model, bundle, proposals, protocol['pilot'])
    save_rows(folder/'development_results.json.gz', rows)
    result = dict(config, completed_steps=protocol['steps'], selected_step=selected, selection_key=list(best),
        history=history, evaluation=evaluation, final_weight_change_l2=change,
        sampled_components=dict(components), sampled_examples=8*protocol['steps'], elapsed_seconds=perf_counter()-started)
    write_new(folder/'training.json', result)
    write_new(folder/'COMPLETE.json', {name+'_sha256': sha256(folder/name) for name in
        ('initial.pt', 'selected.pt', 'final.pt', 'latest.pt', 'training.json', 'development_results.json.gz')})
    return result


def run(output, steps, interval, pilot=False, prepare_only=False):
    root = Path(output)
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    bundle = load_bundle(MANIFEST); manifest, tiles, data = bundle
    if (manifest.get('schema_version') != 5 or not manifest.get('train_selection_disjoint')
            or sha256('lunar_geometry_v5.py') != manifest['geometry_source_sha256']
            or sha256(manifest['split_audit']) != manifest['split_audit_sha256']):
        raise ValueError('Corrected, disjoint geometry/split audit required')
    gate = Path('runs/localization-v5-axial-pose-controls.json')
    gate_value = json.loads(gate.read_text())
    if not gate_value['passed'] or gate_value['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Exact-correspondence pose diagnostic gate failed')
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse checkpoint changed')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval()
    support = prepare_training(tiles, data)
    if root.exists():
        protocol = json.loads((root/'protocol.json').read_text()); check_sources(protocol)
        if (protocol['steps'], protocol['interval'], protocol['pilot']) != (steps, interval, pilot):
            raise ValueError('Prepared protocol differs; use a new run directory')
        proof = json.loads((root/'initial_equivalence.json').read_text())
        if not proof['passed'] or sha256(root/'initial_proposals.json.gz') != protocol['initial_proposals_sha256']:
            raise ValueError('Prepared initial control/cache changed')
        with gzip.open(root/'initial_proposals.json.gz', 'rt') as stream:
            proposals = {r['pair_id']: r for r in json.load(stream)}
    else:
        root.mkdir(parents=True)
        protocol = dict(scope='Exposed development-only geometry-normalized supervision; no fresh final, calibrated acceptance or SOTA claim.',
            source_sha256={p: sha256(p) for p in SOURCES}, manifest_sha256=sha256(MANIFEST),
            coarse_checkpoint_sha256=sha256(checkpoint), steps=steps, interval=interval, pilot=pilot,
            experiments=list(EXPERIMENTS), seeds=list(SEEDS), training_support=support,
            cpu_threads=2, batch_size=8, learning_rate=.0001, weight_decay=.0001,
            optimizer='AdamW', loss='Bilinear exact-coordinate categorical likelihood; eight visible positives per batch. No new rejection head.',
            selection='All72 exposed model-selection pairs, unfiltered pose success, then2px point count, then earlier step. Pilot uses fixed16. Initial checkpoint eligible.',
            matched_ablation='Raw/normalized encoder use identical component/point/light/jitter draws and sensor augmentations; common training support. Normalization changes geometry, resampling and effective search extent.',
            augmentation='Released rendered light variants; exposure, gamma, blur, noise and quantization are sensor perturbations only.',
            inference='Frozen four-orientation initializer; two reference-surface updates; unchanged3nativepx AP3P. Estimated query pose only. Preserve missing-initial-pose failures.',
            training_truth='True query pose/depth used only for train-group normalized images and correspondence visibility. Shadows remain geometrically visible.',
            calibration_or_test_pixels_loaded=0, fresh_data_gate_passed=False, per_run_wall_cap_seconds=6*3600,
            exact_pose_gate_sha256=sha256(gate), created_utc=datetime.now(timezone.utc).isoformat())
        # Freeze the protocol before either inference or fitting.
        write_new(root/'preregistered_protocol.json', protocol)
        proposals = prepare(root, bundle, coarse)
        protocol['initial_proposals_sha256'] = sha256(root/'initial_proposals.json.gz')
        write_new(root/'protocol.json', protocol)
        verify_initial(root, bundle, coarse, proposals)
    if prepare_only:
        return
    results = []
    for experiment in EXPERIMENTS:
        results.append(train(experiment, SEEDS[0], root, bundle, coarse, proposals, protocol))
    winner = max(results, key=lambda r: tuple(r['selection_key']))['experiment']
    if not pilot:
        for seed in SEEDS[1:]:
            results.append(train(winner, seed, root, bundle, coarse, proposals, protocol))
    write_new(root/'summary.json', dict(scope=protocol['scope'], winner=winner,
        results=[{k: r[k] for k in ('experiment', 'seed', 'completed_steps', 'selected_step', 'evaluation', 'selection_key')} for r in results]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--interval', type=int, default=500); parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    if args.steps < 100 or args.steps % 100 or args.interval < 1:
        raise ValueError('At least100 optimizer steps, in100-step multiples, required')
    run(args.output, args.steps, args.interval, args.pilot, args.prepare_only)
