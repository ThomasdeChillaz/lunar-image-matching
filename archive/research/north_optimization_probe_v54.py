"""Completion-aware fixed-pair optimization diagnostic, not a candidate.

Four metadata-fixed fixtures isolate identity, physical relighting, viewpoint
and their combination. Original architecture/loss/optimizer, no sensor
augmentation, fresh identical initialization for each fixture. No pose result
or generalization claim follows from fitting these training examples.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

SEED = 20261101
STEPS = 100
SNAPSHOTS = (0, 25, 50, 75, 100)
MANIFEST = 'reports/north_training_data_v38.json'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid128():
    y, x = np.mgrid[12:116:4, 12:116:4]
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)


def fixed_specs(manifest):
    train = sorted((t for t in manifest['tiles'] if t['split'] == 'train'), key=lambda t: t['tile_id'])
    if not train or manifest['calibration_or_test_pixels_loaded']:
        raise ValueError('Original training-only fixtures required')
    tile = train[0]
    if tile['lighting_count'] != 6: raise ValueError('Six physical rendered lights required')
    return [dict(name=name, tile_id=tile['tile_id'], component=tile['component'],
                 reference_image=0, query_image=query, identity_geometry=identity,
                 query_quarter_turn=0, sensor_augmentation=False)
        for name, query, identity in [('identity', 0, True), ('relighting', 8, True),
            ('viewpoint', 1, False), ('relighting_and_viewpoint', 9, False)]]


def fixture_arrays(z, spec):
    import cv2
    images = np.asarray(z['images'])
    if images.shape != (12, 512, 512) or images.dtype != np.uint8:
        raise ValueError('Audited native training images required')
    a, b = [cv2.resize(images[i], (128, 128), interpolation=cv2.INTER_AREA)
            for i in (spec['reference_image'], spec['query_image'])]
    if spec['identity_geometry']:
        # Lighting never creates a geometric occlusion in the same static view.
        target = dict(forward=grid128(), reverse=grid128(),
            forward_visible=np.ones(676, bool), reverse_visible=np.ones(676, bool))
    else:
        target = dict(forward=((z['forward_target'] + .5) / 4 - .5).astype(np.float32),
            reverse=((z['reverse_target'][0] + .5) / 4 - .5).astype(np.float32),
            forward_visible=z['forward_visible'].copy(), reverse_visible=z['reverse_visible'][0].copy())
    return a, b, target


def verify_fixture(path, a, b, target):
    expected = dict(image_a=a, image_b=b, **target)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected): raise ValueError('Saved fixture members differ')
        for name, value in expected.items():
            actual = saved[name]
            if actual.dtype != value.dtype or not np.array_equal(actual, value, equal_nan=True):
                raise ValueError('Saved fixture differs from immutable training input')


def reachable(target128, visible):
    target = np.asarray(target128, np.float32); visible = np.asarray(visible, bool)
    if target.shape != (676, 2) or visible.shape != (676,) or not np.isfinite(target[visible]).all():
        raise ValueError('Exact full-grid training labels required')
    native = (target + np.float32(.5)) * np.float32(4) - np.float32(.5)
    centers = np.rint((grid128() + .5) * 4 - .5)
    return visible[:, None] & (np.abs(native[:, None] - centers[None]) <= 16).all(-1)


def score_metrics(scores, targets, reference_indices, query_indices):
    scores = np.asarray(scores, float)
    raw_indices = [np.asarray(v) for v in (reference_indices, query_indices)]
    if any(v.size and not np.issubdtype(v.dtype, np.integer) for v in raw_indices):
        raise ValueError('Integer decoded token indices required')
    ia, ib = [v.astype(int) for v in raw_indices]
    if scores.shape != (676, 676) or not np.isfinite(scores).all():
        raise ValueError('Finite full proposal-score matrix required')
    if (ia.shape != ib.shape or ia.ndim != 1 or len(ia) > 512
            or any(((v < 0) | (v >= 676)).any() or len(np.unique(v)) != len(v) for v in (ia, ib))):
        raise ValueError('Unique declared-budget returned correspondences required')
    good = reachable(targets['forward'], targets['forward_visible']); positive = good.any(1)
    best = scores.argmax(1); peak = scores.max(1)
    gap = peak[ia] - scores[ia, ib]
    if len(gap) and float(gap.max()) > 1e-6:
        raise ValueError('Training-score and actual inference decoding disagree beyond tied peaks')
    native_query = (grid128()[ib] + .5) * 4 - .5
    native_truth = (np.asarray(targets['forward'])[ia] + .5) * 4 - .5
    visible = np.asarray(targets['forward_visible'], bool)[ia]
    error = np.linalg.norm(native_query[visible] - native_truth[visible], axis=1)
    correct = {str(t): int((error <= t).sum()) for t in (1, 2, 4, 8, 12)}
    return dict(positive_rows=int(positive.sum()), geometrically_visible_rows=int(np.asarray(targets['forward_visible']).sum()),
        positive_row_top1_reachable=int(good[np.arange(676), best].sum()),
        positive_row_top1_reachable_fraction=None if not positive.any() else float(good[np.arange(676), best].sum() / positive.sum()),
        returned_mutual_matches=len(ia), returned_reachable=int(good[ia, ib].sum()),
        returned_reachable_precision=None if not len(ia) else float(good[ia, ib].mean()),
        returned_native_correct=correct,
        returned_native_precision={k: None if not len(ia) else v / len(ia) for k, v in correct.items()},
        returned_decode_max_score_gap=None if not len(gap) else float(gap.max()),
        returned_hard_argmax_disagreements=int((ib != best[ia]).sum()),
        score_mean=float(scores.mean()), score_standard_deviation=float(scores.std()),
        pose_accuracy_evaluated=False, training_fixture_only=True)


def decoded_indices(p, q):
    grid = {tuple(point): i for i, point in enumerate(grid128())}
    try: return [np.asarray([grid[tuple(point)] for point in np.asarray(v).reshape(-1, 2)], int) for v in (p, q)]
    except KeyError: raise ValueError('Inference points left the exact proposal grid') from None


def run(output, commit, resume=False):
    started = datetime.now(timezone.utc); tick = perf_counter(); cap = 3600
    from north_modern_comparison_v49 import write_once
    from north_coarse_reachability_v53 import verify_ended
    from north_completion_v52 import original_ended, INVENTORY
    verify_ended('runs/north-v53-coarse-state.json')
    original_ended(read(INVENTORY))
    amended=read('reports/north_amended_completion_audit_v52.json')
    diagnostic=read('reports/north_coarse_reachability_audit_v53.json')
    if (not amended['passed'] or amended['original_time_protocol_completed'] is not False
            or amended['recorded_original_fitting_updates']!=6000 or not diagnostic['passed']
            or diagnostic['replayed_branch_rows']!=768):
        raise ValueError('Audited completion and coarse diagnosis required before new fitting')
    gates={p:sha(p)for p in ['reports/north_amended_completion_audit_v52.json',
        'reports/north_coarse_reachability_audit_v53.json','runs/north-v53-coarse-state.json']}
    software_path = 'reports/north_optimization_software_checks_v54.json'
    software = read(software_path)
    required = {'north_optimization_probe_v54.py', 'test_north_optimization_probe_v54.py', 'verify_north_optimization_v54.py'}
    if (not software['passed'] or not software['full_repository_regression'] or not software['fresh_smoke']
            or not required.issubset(software['source_sha256'])):
        raise ValueError('Complete source-bound optimization-probe software checks required')
    for name, digest in software['source_sha256'].items():
        if sha(name) != digest: raise ValueError('Probe changed after software checks')
    import torch
    from compact_lunar import CompactMatcher
    from north_training_bundle_v38 import load_bundle
    from search_window_learning_v18 import SearchWindowMatcher, objective
    from train_balanced_localization_v3 import save_torch
    from train_north_control_v39 import SOURCES as OLD_SOURCES
    from train_search_window_v18 import configure, make_model, make_optimizer, verify_frozen
    sources = list(dict.fromkeys(OLD_SOURCES + sorted(required) + ['north_modern_comparison_v49.py','north_optimization_probe_v51.py','north_coarse_reachability_v53.py','north_completion_v52.py']))
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('Full public source commit required')
    for name in sources:
        if subprocess.check_output(['git', 'show', commit + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Probe source differs from public freeze')
    configure(); manifest, tiles, data = load_bundle(MANIFEST, train_only=True); specs = fixed_specs(manifest)
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    root = Path(output)
    initial_path = Path('runs/north-v39-expanded-control/coarse_adaptation-20261101/initial.pt')
    if sha(initial_path) != read(initial_path.parent / 'COMPLETE.json')['sha256']['initial.pt']:
        raise ValueError('Audited original initialization changed')
    static = dict(publication_commit=commit, source_sha256={n: sha(n) for n in sources},
        manifest_sha256=sha(MANIFEST), software_check_sha256=sha(software_path), upstream_check_sha256=gates,
        original_initial_checkpoint_sha256=sha(initial_path),
        fixtures=specs, seed=SEED, logical_updates_per_fixture=STEPS, snapshot_steps=list(SNAPSHOTS),
        total_parameters=81696, trainable_parameters=53424, cpu_threads=2, max_calendar_seconds=cap,
        loss='Unchanged original bidirectional reachable-window objective and0.5 frozen genuine anchors.',
        optimizer='Unchanged AdamW/context1e-4/coarse1e-5, clip global norm5.',
        scope='Fixed training-pair optimization diagnostic, sensor augmentation disabled; no candidate selection, pose/generalization/SOTA or calibration claim.')
    if root.exists():
        if not resume: raise FileExistsError('Explicit resume required; preserve existing diagnostic')
        protocol = read(root / 'protocol.json')
        if any(protocol[k] != value for k, value in static.items()): raise ValueError('Changed probe on resume')
        if (root / 'COMPLETE.json').exists():
            for n, h in read(root / 'COMPLETE.json')['sha256'].items():
                if sha(root / n) != h: raise ValueError('Completed probe evidence changed')
            return read(root / 'summary.json')
    else:
        if resume: raise FileNotFoundError('No existing diagnostic to resume')
        root.mkdir(parents=True)
        protocol = dict(static, created_utc=started.isoformat(), absolute_deadline_utc=(started + timedelta(seconds=cap)).isoformat())
        write_once(root / 'protocol.json', protocol)
    def clock():
        now = datetime.now(timezone.utc)
        if now < datetime.fromisoformat(protocol['created_utc']) or now >= datetime.fromisoformat(protocol['absolute_deadline_utc']) or perf_counter() - tick >= cap:
            raise TimeoutError('Original fixed optimization-probe allowance exhausted; do not reset')
    original_initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    outcomes = []
    try:
        for spec in specs:
            clock(); folder = root / spec['name']
            if (folder / 'COMPLETE.json').exists():
                for n, h in read(folder / 'COMPLETE.json')['sha256'].items():
                    if sha(folder / n) != h: raise ValueError('Completed fixture changed')
                outcomes.append(read(folder / 'summary.json')); continue
            a, b, arrays = fixture_arrays(data[spec['tile_id']], spec)
            torch.manual_seed(SEED); model = make_model(coarse, 'coarse_adaptation'); optimizer = make_optimizer(model)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            if (initial.keys() != original_initial.keys() or any(not torch.equal(v, original_initial[k]) for k, v in initial.items())
                    or sum(p.numel() for p in model.parameters()) != 81696
                    or sum(p.numel() for p in model.parameters() if p.requires_grad) != 53424):
                raise ValueError('Original controlled graph/initialization required')
            images = torch.from_numpy(np.stack([a, b])[:, None]).float() / 255
            target = {k: torch.from_numpy(v) for k, v in arrays.items()}
            if not folder.exists():
                folder.mkdir(); np.savez_compressed(folder / 'fixture.npz', image_a=a, image_b=b, **arrays)
                save_torch(initial, folder / 'initial.pt')
                save_torch(dict(step=0, model=initial, optimizer=optimizer.state_dict(), history=[], torch_rng=torch.get_rng_state()), folder / 'latest.pt')
            verify_fixture(folder / 'fixture.npz', a, b, arrays)
            state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer']); torch.set_rng_state(state['torch_rng'])
            history = state['history']; step = state['step']
            if len(history) != step: raise ValueError('Diagnostic checkpoint/history mismatch')
            def evaluate(at):
                clock(); model.eval()
                with torch.inference_mode():
                    scores, baseline = model(images); loss, labels = objective(scores, target, baseline, anchor_weight=.5)
                    p, q = SearchWindowMatcher(coarse, model).points(a, b)
                ia, ib = decoded_indices(p, q)
                metrics = score_metrics(scores[0].numpy(), arrays, ia, ib)
                record = dict(step=at, objective=float(loss), labels=labels, metrics=metrics,
                    fine_encoder_unchanged=True, source_checkpoint_scope='Fixed train-only example, not a selected model.')
                checkpoint_path=folder / f'checkpoint_{at}.pt'
                if not checkpoint_path.exists(): save_torch(model.state_dict(),checkpoint_path)
                record['checkpoint_sha256']=sha(checkpoint_path)
                scores_path = folder / f'scores_{at}.npz'
                if not scores_path.exists():
                    np.savez_compressed(scores_path, scores=scores[0].numpy(), baseline=baseline[0].numpy(), reference_indices=ia, query_indices=ib)
                record['scores_sha256'] = sha(scores_path)
                write_once(folder / f'evaluation_{at}.json', record); clock(); return record
            if step in SNAPSHOTS: evaluate(step)
            fitting_seconds = sum(r['fitting_seconds'] for r in history)
            for update in range(step + 1, STEPS + 1):
                clock(); before = perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
                scores, baseline = model(images); loss, labels = objective(scores, target, baseline, anchor_weight=.5)
                if not torch.isfinite(loss): raise ValueError('Nonfinite fixed-pair loss')
                loss.backward()
                group_norms = {prefix: float(sum(float(p.grad.square().sum()) for n, p in model.named_parameters()
                    if n.startswith(prefix + '.') and p.grad is not None) ** .5) for prefix in ('coarse', 'context', 'fine')}
                gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
                if not np.isfinite([float(gradient), *group_norms.values()]).all(): raise ValueError('Nonfinite probe gradient')
                optimizer.step(); verify_frozen(model, initial)
                seconds = perf_counter() - before; fitting_seconds += seconds
                history.append(dict(step=update, loss=float(loss), labels=labels, global_unclipped_gradient_norm=float(gradient),
                    parameter_group_unclipped_gradient_norm=group_norms, fitting_seconds=seconds))
                if update == 10:
                    projection = fitting_seconds / 10 * (4 * STEPS) * 2
                    cost = dict(measured_updates=10, fitting_seconds=fitting_seconds, projected400update_seconds_with_factor2=projection,
                        admitted=projection < cap - (datetime.now(timezone.utc) - datetime.fromisoformat(protocol['created_utc'])).total_seconds())
                    cost_path = folder / 'cost_pilot.json'
                    if cost_path.exists():
                        # Keep the original cost pilot on an explicit resume;
                        # replaying uncommitted updates does not reset its admission.
                        cost = read(cost_path)
                        if cost['measured_updates'] != 10: raise ValueError('Original cost pilot differs')
                    else: write_once(cost_path, cost)
                    if spec['name'] == specs[0]['name'] and not cost['admitted']: raise TimeoutError('Measured original-optimizer diagnostic cost exceeds allowance')
                if update in SNAPSHOTS:
                    save_torch(dict(step=update, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        history=history, torch_rng=torch.get_rng_state()), folder / 'latest.pt')
                    evaluate(update)
            delta = {prefix: float(sum(float((v - initial[n]).square().sum()) for n, v in model.state_dict().items()
                if n.startswith(prefix + '.')) ** .5) for prefix in ('coarse', 'context', 'fine')}
            outcome = dict(fixture=spec, logical_completed_updates=STEPS, history=history, parameter_group_weight_change_l2=delta,
                evaluation=[read(folder / f'evaluation_{n}.json') for n in SNAPSHOTS], input_fixture_sha256=sha(folder / 'fixture.npz'),
                final_checkpoint_sha256=sha(folder / 'latest.pt'), trained_candidate_promoted=False)
            write_once(folder / 'summary.json', outcome)
            write_once(folder / 'COMPLETE.json', dict(sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}))
            outcomes.append(outcome); print('FIXED PAIR COMPLETE', spec['name'], flush=True)
        clock(); summary = dict(complete=True, logical_completed_updates=400, fixtures=outcomes,
            protocol_sha256=sha(root / 'protocol.json'), scope=static['scope'], pose_accuracy_evaluated=False,
            calendar_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(protocol['created_utc'])).total_seconds(),
            calibration_or_final_pixels_loaded=0, model_promoted=False,
            resume_note='Latest25step boundaries are retained. Explicit in-deadline resume may replay uncommitted work;400 counts logical, not guaranteed physical update calls.')
        write_once(root / 'summary.json', summary)
        write_once(root / 'COMPLETE.json', dict(sha256={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
        return summary
    except BaseException as exc:
        write_once(root / ('FAILED-' + str(datetime.now(timezone.utc).timestamp()).replace('.', '-') + '.json'),
            dict(error_type=type(exc).__name__, error=str(exc), original_deadline_utc=protocol['absolute_deadline_utc']))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default='runs/north-v54-fixed-pair-optimization')
    p.add_argument('--publication-commit', required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); run(a.output, a.publication_commit, a.resume)
