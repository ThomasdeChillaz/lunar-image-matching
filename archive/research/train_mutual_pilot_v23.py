"""Train-only100-update dual-assignment pilot with unchanged native inference.

Source/publication, exact reused initialization and prior independent geometry
gates are required before fitting. It is not a selected localization model.
"""
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
from context_supervision_v15 import prepare_supervision, sample_training
from dataset import sha256
from lunargeo_dataset import write_new
from mutual_assignment_v23 import objective
from train_balanced_localization_v3 import save_torch
from train_context_pilot_v15 import load_training
from train_search_window_v18 import (SOURCES as ORIGINAL_SOURCES, checkpoint,
    configure, make_model, make_optimizer, verify_frozen)

MANIFEST = 'reports/localization_training_data_v5.json'
CONFIGURATION = 'coarse_adaptation'
SEED = 20261101
SOURCES = list(dict.fromkeys(ORIGINAL_SOURCES+['mutual_assignment_v23.py', 'train_mutual_pilot_v23.py']))


def published(commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):raise ValueError('Full public source freeze required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Fitting source differs from public freeze: '+name)


def fit_step(model, optimizer, tiles, prepared, rng, step):
    model.train(); images, targets, sample = sample_training(tiles, prepared, rng, negative_pair=step % 4 == 0)
    optimizer.zero_grad(set_to_none=True); scores, baseline = model(images)
    loss, labels = objective(scores, targets, baseline, anchor_weight=.5)
    if not torch.isfinite(loss):raise ValueError('Nonfinite assignment pilot loss')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
    if not torch.isfinite(gradient):raise ValueError('Nonfinite assignment pilot gradient')
    optimizer.step()
    return dict(step=step, loss=float(loss), gradient_norm=float(gradient), sample=sample, labels=labels)


def reusable_gates():
    initial_root = Path('runs/localization-v18-window-initial-control')
    gate = json.loads((initial_root/'summary.json').read_text())
    prior_root = Path('runs/localization-v18-window-cost-pilot')
    prior = json.loads(Path('reports/native_window_pilot_audit_v18.json').read_text())
    if not gate['passed'] or gate['cases_per_variant'] != 72 or not gate['exact_native_and_pose_equivalence']:
        raise ValueError('Actual unchanged two-graph initialization control required')
    if (not prior['passed'] or not prior['exact_full_retraining'] or prior['completed_steps'] != 100
            or prior['pilot_protocol_sha256'] != sha256(prior_root/'protocol.json')
            or prior['pilot_summary_sha256'] != sha256(prior_root/'summary.json')):
        raise ValueError('Prior exact pilot/independent geometry gate changed')
    for name, want in gate['source_sha256'].items():
        if sha256(name) != want:raise ValueError('Previously controlled model/inference source changed: '+name)
    for name, want in prior['fitting_source_sha256'].items():
        if sha256(name) != want:raise ValueError('Independently checked training geometry/source changed: '+name)
    previous_environment = json.loads(Path('runs/localization-v18-window-development/protocol.json').read_text())['environment']
    if previous_environment != dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__):
        raise ValueError('Previously controlled numerical runtime changed; actual initialization must be checked again')
    complete = json.loads((prior_root/'COMPLETE.json').read_text())
    for name, want in complete['sha256'].items():
        if sha256(prior_root/name) != want:raise ValueError('Matched historical pilot changed')
    return gate, prior


def initialized_model(coarse):
    torch.manual_seed(SEED); model = make_model(coarse, CONFIGURATION)
    prior = torch.load('runs/localization-v18-window-cost-pilot/initial.pt', map_location='cpu', weights_only=True)
    if model.state_dict().keys() != prior.keys() or any(not torch.equal(v, prior[k]) for k, v in model.state_dict().items()):
        raise ValueError('Fresh initialized weights differ from the actually controlled historical graph')
    # This is fresh initialization equality, never a warm start from fitted weights.
    return model


def run(output, source_commit):
    root = Path(output)
    if root.exists():raise FileExistsError('Preserve every pilot/failure; no overwrite or warm-start')
    published(source_commit); configure(); gate, prior = reusable_gates()
    manifest, tiles, data = load_training(MANIFEST); prepared, support = prepare_supervision(tiles, data)
    old_protocol = json.loads(Path('runs/localization-v18-window-cost-pilot/protocol.json').read_text())
    if sha256(MANIFEST) != old_protocol['manifest_sha256'] or support != old_protocol['training_support']:
        raise ValueError('Compared training manifest/support differs')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    if sha256(Path(manifest['coarse_model'])/'selected.pt') != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse source checkpoint changed')
    model = initialized_model(coarse); rng = np.random.default_rng(SEED); optimizer = make_optimizer(model)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    total = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (total, trainable) != (81696, 53424) or total >= 100000:raise ValueError('Declared parameter budget differs')
    root.mkdir(parents=True); (root/'source').mkdir()
    protocol = dict(scope='Train-only100-update dual-assignment cost/gradient pilot; no selection or localization outcome.',
        publication_commit=source_commit, source_sha256={p: sha256(p) for p in SOURCES},
        manifest_sha256=sha256(MANIFEST), input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        variant=CONFIGURATION, seed=SEED, steps=100, max_fitting_seconds=600, cpu_threads=2,
        total_parameters=total, trainable_parameters=trainable, training_support=support,
        initial_gate_sha256=sha256('runs/localization-v18-window-initial-control/summary.json'),
        reused_geometry_audit_sha256=sha256('reports/native_window_pilot_audit_v18.json'),
        matched_prior_protocol_sha256=sha256('runs/localization-v18-window-cost-pilot/protocol.json'),
        initialization='Fresh seeded model exactly equals historical actually controlled initial weights; never fitted pilot/cohort weights.',
        change='Only add bidirectional log(row reachable mass / dual-softmax reachable mass), temperature0.1, weight1. Existing v18 window masks, margins, anchor0.5, architectures, optimizer, augmentation and sampling unchanged.',
        novelty='Dual-softmax is standard LoFTR/EfficientLoFTR machinery; lunar localization gain/novelty unproven.',
        optimizer='Same AdamW/context1e-4/coarse1e-5/weight_decay1e-4/clip5; separately frozen original native fine CNN.',
        environment=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
        selection_inference=0, selection_pixels_loaded=0, calibration_or_test_pixels_loaded=0,
        recorded_utc=datetime.now(timezone.utc).isoformat())
    write_new(root/'protocol.json', protocol); save_torch(initial, root/'initial.pt')
    for name in SOURCES:shutil.copy2(name, root/'source'/name)
    started = perf_counter(); history = []
    for step in range(1, 101):
        history.append(fit_step(model, optimizer, tiles, prepared, rng, step))
        if step % 10 == 0:
            save_torch(checkpoint(model, optimizer, rng, step, history, perf_counter()-started), root/'latest.pt')
            print('mutual pilot', step, 'loss', history[-1]['loss'], 'seconds', perf_counter()-started, flush=True)
        if perf_counter()-started > 600:raise TimeoutError('Pilot fitting budget exhausted; keep partial artifacts')
    verify_frozen(model, initial)
    if any(p.grad is not None for p in model.fine.parameters()):raise ValueError('Protected fine encoder received gradients')
    for name, want in protocol['source_sha256'].items():
        if sha256(name) != want:raise ValueError('Frozen pilot source changed')
    save_torch(model.state_dict(), root/'final.pt')
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    result = dict(completed_steps=100, fitting_seconds=perf_counter()-started, history=history,
        fine_encoder_unchanged=True, fresh_initialization_matches_prior=True, final_weight_change_l2=change,
        selection_pixels_loaded=0, calibration_or_test_pixels_loaded=0)
    write_new(root/'summary.json', result)
    write_new(root/'COMPLETE.json', dict(sha256={p: sha256(root/p) for p in ('protocol.json', 'initial.pt', 'final.pt', 'latest.pt', 'summary.json')}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--source-commit', required=True)
    a = p.parse_args(); run(a.output, a.source_commit)
