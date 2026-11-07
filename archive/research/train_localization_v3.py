"""Finite CPU development training; calibration/test pixels are forbidden."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from localization_diagnostics_v3 import map_points, point_metrics
from localization_pose_refit_v3 import estimate_refined_pose
from localization_refiner_v3 import LocalRefiner, VARIANTS, local_loss, refine_dense, PATCH, REFERENCE_PATCH
from lunar_geometry_v2 import pose_error
from lunargeo_dataset import write_new
from prepare_localization_training_v3 import allowed_records

SOURCES = ['localization_refiner_v3.py', 'train_localization_v3.py',
           'prepare_localization_training_v3.py', 'localization_diagnostics_v3.py',
           'localization_pose_refit_v3.py', 'lunar_geometry_v2.py', 'contrastive_terrain.py', 'compact_lunar.py']
SEEDS = (20261101, 20261102, 20261103)


def load_bundle(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('calibration_or_test_pixels_loaded') != 0:
        raise ValueError('Development data must exclude calibration/test pixels')
    tiles = allowed_records(manifest)
    if len(tiles) != len(manifest['tiles']):
        raise ValueError('Forbidden pixels in prepared training manifest')
    if sha256(manifest['origin_manifest']) != manifest['origin_manifest_sha256']:
        raise ValueError('Original manifest changed')
    data = {}
    for tile in tiles:
        if sha256(tile['native_cache']) != tile['native_cache_sha256']:
            raise ValueError('Native supervised cache changed')
        with np.load(tile['native_cache'], allow_pickle=False) as z:
            data[tile['tile_id']] = {k:z[k].copy() for k in z.files}
    return manifest, {t['tile_id']:t for t in tiles}, data


def appearance(image, rng):
    image = image.astype(np.float32)/255
    image = np.clip(image*rng.uniform(.8, 1.2), 0, 1)**rng.uniform(.85, 1.15)
    if rng.random() < .25:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(.3, .8)))
    image = np.clip(image+rng.normal(0, rng.uniform(0, .015), image.shape), 0, 1)
    levels = int(rng.choice([31, 63, 127, 255]))
    return (np.round(image*levels)/levels).astype(np.float32)


def nonoverlap(a, b):
    aa, bb = np.asarray(a['world_bounds']), np.asarray(b['world_bounds'])
    return bool(np.any(np.minimum(aa[1], bb[1])+200 < np.maximum(aa[0], bb[0])))


def training_batch(manifest, tiles, data, rng, size):
    groups = defaultdict(list)
    for c in manifest['cases']:
        if c['split'] == 'train' and c['proposed_points']:
            groups[c['component']].append(c)
    names = sorted(groups); refs = []; queries = []; ref_xy = []; targets = []; labels = []; kinds = []
    for _ in range(size):
        cases = groups[names[int(rng.integers(len(names)))]]
        c = cases[int(rng.integers(len(cases)))]; tile = tiles[c['tile_id']]; z = data[c['tile_id']]
        k = c['key']; index = int(rng.integers(c['proposed_points']))
        p, q = z[f'p{k}'][index], z[f'q{k}'][index]
        target = z[f'target{k}'][index]
        visible = bool(z[f'visible{k}'][index])
        reference = z['images'][2*c['light_a']]
        query = z['images'][2*c['light_b']+1]
        center = np.rint(q).astype(int)
        kind = 'actual_coarse'
        choice = rng.random()
        if choice < .10:
            eligible = [t for t in tiles.values() if t['split'] == 'train' and nonoverlap(tile, t)]
            if not eligible:
                raise ValueError('No independently nonoverlapping train terrain for negatives')
            other = eligible[int(rng.integers(len(eligible)))]; query = data[other['tile_id']]['images'][int(rng.integers(6))]
            center = rng.integers(30, np.array(query.shape[::-1])-30)
            visible = False; kind = 'nonoverlap'
        elif choice < .25:
            center = rng.integers(30, np.array(query.shape[::-1])-30)
            kind = 'incorrect_location'
        elif choice < .50:
            center = np.clip(center+rng.integers(-4, 5, 2), 30, np.array(query.shape[::-1])-30)
            kind = 'jittered_coarse'
        ref_center = np.rint(p).astype(int)
        reference_patch = cv2.getRectSubPix(reference, (REFERENCE_PATCH, REFERENCE_PATCH), tuple(map(float, ref_center)))
        query_patch = cv2.getRectSubPix(query, (PATCH, PATCH), tuple(map(float, center)))
        # Exact projected coordinates define all labels. A visible dark point
        # remains geometrically visible; no brightness threshold changes labels.
        local = target-center+16
        positive = bool(visible and ((local >= 0) & (local <= 32)).all())
        refs.append(appearance(reference_patch, rng)); queries.append(appearance(query_patch, rng))
        ref_xy.append(p-ref_center+13); targets.append(local if positive else [0, 0])
        labels.append(positive); kinds.append(kind)
    arrays = [np.stack(refs)[:, None], np.stack(queries)[:, None], np.asarray(ref_xy, np.float32),
              np.asarray(targets, np.float32), np.asarray(labels, np.float32)]
    return [torch.from_numpy(a) for a in arrays], Counter(kinds), int(sum(labels))


def evaluate(model, manifest, tiles, data, bank_only=False, keep_records=True):
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection' and
             (not bank_only or c['pair_id'] in manifest['selection_bank'])]
    groups = defaultdict(list)
    for c in cases:
        groups[c['tile_id']].append(c)
    rows = []; model.eval(); feature_ms = 0.
    with torch.inference_mode():
        for tile_id, tile_cases in groups.items():
            tile, z = tiles[tile_id], data[tile_id]
            indices = sorted({2*c['light_a'] for c in tile_cases} | {2*c['light_b']+1 for c in tile_cases})
            tick = perf_counter()
            dense = {i:model.encoder(torch.from_numpy(z['images'][i].copy()).float()[None, None]/255) for i in indices}
            feature_ms += (perf_counter()-tick)*1000
            for c in tile_cases:
                k = c['key']; tick = perf_counter()
                p, q, confidence, original = refine_dense(model, z[f'p{k}'], z[f'q{k}'],
                                    dense[2*c['light_a']], dense[2*c['light_b']+1])
                if model.variant == 'reliability':
                    use = confidence >= .5
                    p, q, confidence, original = p[use], q[use], confidence[use], original[use]
                inference_ms = (perf_counter()-tick)*1000
                camera = tile['cameras'][0]
                world, valid = map_points(p, z['depth_a'], camera['intrinsics'], camera['cam2world'])
                pose, inliers = estimate_refined_pose(world, q[valid], tile['cameras'][1]['intrinsics'])
                # Ground-truth query pose is consumed only AFTER pose inference.
                error = None if pose is None else pose_error(pose, np.asarray(tile['cameras'][1]['cam2world']))
                ranges = z['depth_a'][np.isfinite(z['depth_a']) & (z['depth_a'] > 0)]
                distance = float(np.median(ranges))
                correct = bool(error and error['rotation_deg'] <= 2 and error['translation_m'] <= .01*distance)
                strict = bool(error and error['rotation_deg'] <= .5 and error['translation_m'] <= .001*distance)
                visible = z[f'visible{k}'][original]
                target = z[f'target{k}'][original][visible]
                row = dict(c, **point_metrics(q, target, visible), pose_correct=correct,
                           strict_pose_correct=strict, pose_estimated=pose is not None,
                           pose_error=error, reference_range_m=distance, inliers=inliers,
                           refinement_ms=inference_ms, point_rejection='Fixed0.5 local matchability score' if model.variant=='reliability' else 'None',
                           acceptance_policy=None)
                if keep_records:
                    row.update(reference_points_native=p.tolist(), query_points_native=q.tolist(),
                               source_proposal_indices=original.tolist(), target_points_native=target.tolist(),
                               visible=visible.tolist(), point_scores=confidence.tolist(),
                               estimated_cam2world=None if pose is None else pose.tolist())
                rows.append(row)
    counts = dict(cases=len(rows), geometries=len(groups),
        components=len({r['component'] for r in rows}), correct_poses=sum(r['pose_correct'] for r in rows),
        strict_correct_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['pose_estimated'] for r in rows), returned_matches=sum(r['returned'] for r in rows),
        scorable_matches=sum(r['scorable'] for r in rows),
        correct_correspondences_at_native_px={str(t):sum(r['correct_at_native_px'][str(t)] for r in rows) for t in (1,2,4,8,12)},
        feature_encoding_ms=feature_ms, bank_only=bank_only,
        scope='Development model-selection groups; no fresh test, pair-acceptance calibration or guaranteed error rate.')
    return counts, rows


def save_torch(value, path):
    path = Path(path); tmp = path.with_suffix('.temporary')
    torch.save(value, tmp); tmp.replace(path)


def train(manifest_path, bundle, output, variant, seed, steps, batch_size=8, interval=500,
          max_seconds=48*3600):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Training artifacts are immutable; choose new output')
    manifest, tiles, data = bundle
    output.mkdir(parents=True); (output/'source').mkdir()
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    coarse = CompactMatcher(manifest['coarse_model'])
    if sha256(Path(manifest['coarse_model'])/'selected.pt') != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse checkpoint changed')
    model = LocalRefiner(variant); model.initialize_encoder(coarse)
    config = dict(schema_version=3, scope='Explicitly user-authorized development-only training continuation.',
        seed=seed, variant=variant, steps=steps, batch_size=batch_size, checkpoint_interval=interval,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        frozen_coarse_parameters=28272, total_parameters=28272+sum(p.numel() for p in model.parameters()),
        manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path),
        source_sha256={p:sha256(p) for p in SOURCES},
        optimizer='AdamW', learning_rate=.0001, weight_decay=.0001, cpu_threads=2,
        selection='Maximum unfiltered model-selection-bank pose count, then2px-correct correspondence count, then earliest step; initial checkpoint eligible.',
        selection_bank=manifest['selection_bank'],
        loaded_groups=[dict(tile_id=t['tile_id'],component=t['component'],split=t['split']) for t in tiles.values()],
        calibration_or_test_pixels_loaded=0, fresh_data_gate_passed=False, final_test_enabled=False,
        local_matchability_threshold=.5 if variant=='reliability' else None,
        augmentations='Exposure/gamma/blur/noise/quantization; physical sunlight variation comes from the released renders.',
        max_wall_seconds=max_seconds, started_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
    write_new(output/'frozen_config.json', config)
    for p in SOURCES:
        shutil.copy2(p, output/'source'/p)
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    save_torch(initial, output/'initial.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    started = perf_counter(); history = []; kinds = Counter(); positive = 0; samples = 0
    def validate(step, loss=None):
        counts, _ = evaluate(model, manifest, tiles, data, bank_only=True, keep_records=False)
        history.append(dict(step=step, loss=loss, elapsed_seconds=perf_counter()-started, selection=counts))
        print(variant, seed, step, 'selection poses', counts['correct_poses'], '/', counts['cases'],
              'loss', loss, 'seconds', round(perf_counter()-started,1), flush=True)
        return (counts['correct_poses'], counts['correct_correspondences_at_native_px']['2'], -step)
    best = validate(0); selected_step = 0
    save_torch(model.state_dict(), output/'selected.pt')
    completed_steps = 0
    for step in range(1, steps+1):
        if perf_counter()-started >= max_seconds:
            break
        model.train(); batch, categories, n_positive = training_batch(manifest, tiles, data, rng, batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits, match = model(batch[0], batch[1], batch[2])
        loss = local_loss(logits, match, batch[3], batch[4], variant=='reliability')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite supervised objective')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        completed_steps = step; kinds.update(categories); positive += n_positive; samples += batch_size
        if step % 100 == 0:
            save_torch(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), output/'latest.pt')
            print(variant, seed, 'gradient step', step, 'loss', round(float(loss),4), flush=True)
        if step % interval == 0 or step == steps:
            score = validate(step, float(loss))
            if score > best:
                best = score; selected_step = step
                save_torch(model.state_dict(), output/'selected.pt')
    save_torch(model.state_dict(), output/'final.pt')
    delta = float(sum((v.detach()-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt())
    model.load_state_dict(torch.load(output/'selected.pt', map_location='cpu', weights_only=True))
    counts, rows = evaluate(model, manifest, tiles, data)
    with gzip.open(output/'development_results.json.gz', 'wt') as f:
        json.dump(rows, f, allow_nan=False)
    result = dict(config, completed_steps=completed_steps, selected_step=selected_step,
                  selection_key=list(best), history=history, evaluation=counts,
                  final_weight_change_l2=delta, sampled_examples=samples, positive_local_labels=positive,
                  sampled_kinds=dict(kinds), elapsed_seconds=perf_counter()-started,
                  completed_requested_steps=completed_steps==steps)
    write_new(output/'training.json', result)
    write_new(output/'COMPLETE.json', dict(checkpoint_sha256=sha256(output/'selected.pt'),
        final_sha256=sha256(output/'final.pt'), training_sha256=sha256(output/'training.json'),
        results_sha256=sha256(output/'development_results.json.gz'),
        requested_steps_completed=completed_steps==steps))
    return result


def cohort(manifest_path, output, steps, batch_size, interval, pilot=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_new(output/'protocol.json', dict(scope='Development only; no fresh final evaluation.',
        manifest_sha256=sha256(manifest_path), sources={p:sha256(p) for p in SOURCES},
        steps=steps, batch_size=batch_size, interval=interval, seeds=list(SEEDS),
        variants=['descriptor_3d'] if pilot else list(VARIANTS), total_training_wall_budget_seconds=48*3600,
        repeat_selection='Winner chosen by the predeclared16-case selection-bank key, not by72-case development outcomes.',
        final_test_enabled=False, fresh_data_gate_passed=False))
    bundle = load_bundle(manifest_path); results = []; started = perf_counter()
    for variant in (('descriptor_3d',) if pilot else VARIANTS):
        results.append(train(manifest_path, bundle, output/f'{variant}-{SEEDS[0]}', variant, SEEDS[0],
                             steps, batch_size, interval, 48*3600-(perf_counter()-started)))
    winner = max(results, key=lambda r:tuple(r['selection_key']))['variant']
    if not pilot:
        for seed in SEEDS[1:]:
            results.append(train(manifest_path, bundle, output/f'{winner}-{seed}', winner, seed,
                                 steps, batch_size, interval, 48*3600-(perf_counter()-started)))
    report = dict(scope='Development-only trained candidates; no fresh/generalization/acceptance claim.',
        winner=winner, pilot=pilot, elapsed_seconds=perf_counter()-started,
        results=[{k:r[k] for k in ('variant','seed','completed_steps','selected_step','selection_key',
                                  'total_parameters','final_weight_change_l2','evaluation','elapsed_seconds')} for r in results])
    write_new(output/'summary.json', report)
    print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-only', action='store_true', required=True)
    parser.add_argument('--manifest', default='reports/localization_training_data_v3.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.batch_size,args.interval) < 1:
        raise ValueError('Positive training settings required')
    cohort(args.manifest,args.output,args.steps,args.batch_size,args.interval,args.pilot)
