"""Bounded train-only cost/gradient pilot, not the full context research cohort."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import make_context, objective, prepare_supervision, proposal_grid, sample_training
from contrastive_terrain import sample_descriptors
from dataset import sha256
from lunargeo_dataset import write_new
from prepare_localization_training_v3 import allowed_records
from train_balanced_localization_v3 import save_torch

SOURCES = ['train_context_pilot_v15.py', 'context_supervision_v15.py', 'global_context_v15.py',
    'compact_lunar.py', 'contrastive_terrain.py', 'lunar_geometry_v5.py', 'lunar_geometry_v2.py',
    'rotation_bank_pose_v4.py', 'localization_diagnostics_v3.py', 'localization_robust_pose_v4.py',
    'localization_pose_refit_v3.py', 'prepare_localization_training_v3.py', 'train_balanced_localization_v3.py']


def load_training(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text()); records = allowed_records(manifest)
    if (len(records) != len(manifest['tiles']) or manifest['schema_version'] != 5
            or not manifest['train_selection_disjoint'] or manifest['calibration_or_test_pixels_loaded']):
        raise ValueError('Audited development-only manifest required')
    if sha256(manifest['origin_manifest']) != manifest['origin_manifest_sha256']:
        raise ValueError('Origin metadata changed')
    if sha256('lunar_geometry_v5.py') != manifest['geometry_source_sha256'] or sha256(manifest['split_audit']) != manifest['split_audit_sha256']:
        raise ValueError('Axial geometry/split audit differs')
    tiles = {t['tile_id']: t for t in records if t['split'] == 'train'}; data = {}
    for key, tile in tiles.items():
        if sha256(tile['native_cache']) != tile['native_cache_sha256']:
            raise ValueError('Native training input changed')
        with np.load(tile['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
    return manifest, tiles, data


def run(output, variant, steps, commit, max_seconds):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve every pilot/checkpoint/failure')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Published source commit required')
    # The actor verifies this commit is public before launch. Bind every fitting
    # source byte to it locally so edits cannot slip into the measured pilot.
    for source in SOURCES:
        if subprocess.check_output(['git', 'show', commit+':'+source]) != Path(source).read_bytes():
            raise ValueError('Fitting source differs from the public freeze: '+source)
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261101); rng = np.random.default_rng(20261101)
    manifest_path = Path('reports/localization_training_data_v5.json')
    manifest, tiles, data = load_training(manifest_path); prepared, support = prepare_supervision(tiles, data)
    initial_root = Path('runs/localization-v15-context-initial-control')
    gate = json.loads((initial_root/'summary.json').read_text()); gate_protocol = json.loads((initial_root/'protocol.json').read_text())
    if not gate['passed'] or gate['cases'] != 72 or not gate['exact_native_and_pose_equivalence']:
        raise ValueError('Actual full initial-control gate required')
    if sha256(initial_root/'results.json.gz') != gate['results_sha256']:
        raise ValueError('Initial control changed')
    for source, digest in gate_protocol['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Executed initial-control source changed')
    supervision_gate = json.loads(Path('reports/native_context_supervision_gate_v15.json').read_text())
    if not supervision_gate['passed'] or support != supervision_gate['training_support']:
        raise ValueError('Training support differs from prepared gate')
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse checkpoint changed')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    frozen = {k: v.detach().clone() for k, v in coarse.model.state_dict().items()}
    model = make_context(variant); initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if variant == 'cross_context':
        controlled = torch.load(initial_root/'initial.pt', map_location='cpu', weights_only=True)
        if any(not torch.equal(v, controlled[k]) for k, v in initial.items()):
            raise ValueError('Seeded cross-context initial weights differ from controlled checkpoint')
    parameters = sum(p.numel() for p in model.parameters()); root.mkdir(parents=True); (root/'source').mkdir()
    protocol = dict(scope='Train-only optimizer/cost pilot, not a model-selection study or demonstrated localization gain.',
        variant=variant, steps=steps, seed=20261101, publication_commit=commit,
        source_sha256={p: sha256(p) for p in SOURCES}, manifest_sha256=sha256(manifest_path),
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        frozen_checkpoint_sha256=sha256(checkpoint), training_support=support,
        context_parameters=parameters, total_with_frozen_backbone=28272+parameters,
        total_with_separate_learned_fine=28272+35233+parameters, feedforward_width=model.feedforward_width,
        optimizer='AdamW', learning_rate=.0001, weight_decay=.0001, gradient_clip=5.,
        cpu_threads=2, pairs_per_update=1, tokens_per_image=676, negative_pair_every=4,
        fitting='Both directions projected3D candidate likelihood, positive score margin0.7, negative max-score margin0.5; ignore ambiguous candidate-edge bands.',
        augmentation='Actual128px exposure/gamma/blur/noise/quantization plus query quarter turns. Physical lighting from released renders.',
        selection_inference=0, selection_pixels_loaded=0, calibration_or_test_pixels_loaded=0,
        max_seconds=max_seconds, timing='Observed two-thread fitting cost under concurrent workload, not isolated inference latency.',
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
        started_utc=datetime.now(timezone.utc).isoformat())
    write_new(root/'protocol.json', protocol); save_torch(initial, root/'initial.pt')
    for source in SOURCES:
        shutil.copy2(source, root/'source'/source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    grid = torch.from_numpy(proposal_grid())[None]; history = []; started = perf_counter()
    for step in range(1, steps+1):
        model.train(); images, targets, metadata = sample_training(tiles, prepared, rng, negative_pair=step % 4 == 0)
        with torch.no_grad():
            dense, _ = coarse.model(images)
            a, b = sample_descriptors(dense[:1], grid), sample_descriptors(dense[1:], grid)
        optimizer.zero_grad(set_to_none=True); scores = model(a, b, grid, grid)
        loss, labels = objective(scores, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite context pilot loss')
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(gradient):
            raise ValueError('Nonfinite context pilot gradient')
        optimizer.step(); history.append(dict(step=step, loss=float(loss), gradient_norm=float(gradient), labels=labels, sample=metadata))
        if step % 10 == 0:
            save_torch(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), history=history), root/'latest.pt')
            print('context pilot', variant, 'step', step, 'loss', round(float(loss), 4), 'seconds', round(perf_counter()-started, 2), flush=True)
        if perf_counter()-started > max_seconds:
            write_new(root/'TIMEOUT.json', dict(completed_steps=step, elapsed_seconds=perf_counter()-started))
            raise TimeoutError('Bounded pilot exhausted; retain incomplete work')
    if any(not torch.equal(v, coarse.model.state_dict()[k]) for k, v in frozen.items()):
        raise ValueError('Frozen backbone changed during fitting')
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Pinned pilot source changed during fitting')
    save_torch(model.state_dict(), root/'final.pt')
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    summary = dict(scope=protocol['scope'], completed_steps=steps, elapsed_seconds=perf_counter()-started,
        final_weight_change_l2=change, history=history, negative_pairs=sum(h['sample']['verified_nonoverlap'] for h in history),
        no_selection_inference=True, no_selection_pixels_loaded=True, no_calibration_or_test_pixels_loaded=True,
        frozen_backbone_unchanged=True, context_parameters=parameters)
    write_new(root/'summary.json', summary)
    write_new(root/'COMPLETE.json', {p+'_sha256': sha256(root/p) for p in ('protocol.json','initial.pt','final.pt','latest.pt','summary.json')})
    print(json.dumps({k: v for k, v in summary.items() if k != 'history'}, indent=2)); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=['pointwise','self_context','cross_context'], default='cross_context')
    parser.add_argument('--steps', type=int, default=100); parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-seconds', type=int, default=600); args = parser.parse_args()
    if args.steps < 10 or args.steps > 100 or args.steps % 10 or not 1 <= args.max_seconds <= 3600:
        raise ValueError('A10–100step measured pilot and bounded wall budget are required')
    run(args.output, args.variant, args.steps, args.source_commit, args.max_seconds)
