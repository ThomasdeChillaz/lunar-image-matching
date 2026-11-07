"""Independent assignment scalar checks plus exact100-update pilot replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import prepare_supervision, proposal_grid, sample_training
from dataset import sha256
from lunargeo_dataset import write_new
from mutual_assignment_v23 import window_correction
from train_context_pilot_v15 import load_training
from train_mutual_pilot_v23 import (CONFIGURATION, MANIFEST, SEED, fit_step,
    initialized_model, published, reusable_gates)
from train_search_window_v18 import configure, make_optimizer, verify_frozen


def logsumexp(array, axis):
    maximum = np.max(array, axis=axis, keepdims=True)
    return maximum+np.log(np.exp(array-maximum).sum(axis=axis, keepdims=True))


def independent_correction(scores, target, visible, grid):
    """Float64 NumPy implementation; no Torch loss/window helper involved."""
    native = np.asarray(target, np.float32)*4+1.5
    centers = np.rint((np.asarray(grid, np.float32)+.5)*4-.5)
    allowed = np.asarray(visible, bool)[:, None] & (np.abs(native[:, None]-centers[None]) <= 16).all(-1)
    positive = allowed.any(1)
    if not positive.any():return 0., allowed
    scaled = np.asarray(scores, np.float64)/.1
    row = scaled-logsumexp(scaled, 1)
    column = scaled-logsumexp(scaled, 0)
    ordinary = np.where(allowed[positive], row[positive], -np.inf)
    coupled = np.where(allowed[positive], (row+column)[positive], -np.inf)
    return float(np.mean(logsumexp(ordinary, 1)-logsumexp(coupled, 1))), allowed


def audit(run, output):
    root = Path(run); configure(); protocol = json.loads((root/'protocol.json').read_text())
    summary = json.loads((root/'summary.json').read_text()); complete = json.loads((root/'COMPLETE.json').read_text())
    published(protocol['publication_commit']); reusable_gates()
    for name, want in protocol['source_sha256'].items():
        if sha256(name) != want or sha256(root/'source'/name) != want:raise ValueError('Pilot source changed')
    for name, want in complete['sha256'].items():
        if sha256(root/name) != want:raise ValueError('Completed pilot changed')
    if (protocol['steps'], protocol['seed'], protocol['variant'], protocol['total_parameters'], protocol['trainable_parameters']) != (
            100, SEED, CONFIGURATION, 81696, 53424):raise ValueError('Declared pilot configuration differs')
    if any(protocol[k] for k in ('selection_inference', 'selection_pixels_loaded', 'calibration_or_test_pixels_loaded')):
        raise ValueError('Train-only scope changed')
    manifest, tiles, data = load_training(MANIFEST); prepared, support = prepare_supervision(tiles, data)
    if sha256(MANIFEST) != protocol['manifest_sha256'] or support != protocol['training_support']:
        raise ValueError('Pilot data/support changed')
    if {t['native_cache']: sha256(t['native_cache']) for t in tiles.values()} != protocol['input_sha256']:
        raise ValueError('Native train inputs changed')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    model = initialized_model(coarse); optimizer = make_optimizer(model); rng = np.random.default_rng(SEED)
    initial = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
    final = torch.load(root/'final.pt', map_location='cpu', weights_only=True)
    latest = torch.load(root/'latest.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):raise ValueError('Fresh initialization changed')
    if len(summary['history']) != 100 or latest['step'] != 100 or latest['history'] != summary['history']:
        raise ValueError('Pilot budget/history differs')
    independent = []; old_history = json.loads(Path('runs/localization-v18-window-cost-pilot/summary.json').read_text())['history']
    grid = proposal_grid()
    for step, saved in enumerate(summary['history'], 1):
        if saved['sample'] != old_history[step-1]['sample']:raise ValueError('Compared augmentation/component draw differs')
        if step in (1, 4, 50, 100):
            # Peek with a copied generator. This changes neither the actual
            # sampler nor optimizer; the architecture has no stochastic layer.
            peek = np.random.default_rng(); peek.bit_generator.state = deepcopy(rng.bit_generator.state)
            model.train(); images, targets, sample = sample_training(tiles, prepared, peek, negative_pair=step % 4 == 0)
            scores, _ = model(images)
            for name, matrix in [('forward', scores[0]), ('reverse', scores[0].T)]:
                value, mask = independent_correction(matrix.detach().numpy(), targets[name].numpy(), targets[name+'_visible'].numpy(), grid)
                actual = float(window_correction(matrix, torch.from_numpy(mask)).detach())
                if not np.isclose(value, actual, atol=5e-5, rtol=0):raise ValueError('Independent assignment correction differs')
                if not np.isclose(value, saved['labels']['mutual_'+name+'_correction'], atol=5e-5, rtol=0):
                    raise ValueError('Recorded assignment scalar differs')
                independent.append(dict(step=step, direction=name, numpy_float64=value, torch_recorded=actual))
            del scores, images, targets, matrix
        actual = fit_step(model, optimizer, tiles, prepared, rng, step)
        if actual != saved:raise ValueError('Exact loss/gradient/sampler replay differs at step '+str(step))
    for name, value in model.state_dict().items():
        if not torch.equal(value, final[name]) or not torch.equal(value, latest['model'][name]):raise ValueError('Exactly retrained weights differ')
    verify_frozen(model, initial)
    if any(p.grad is not None for p in model.fine.parameters()):raise ValueError('Frozen fine encoder received gradients')
    if not any(not torch.equal(v, initial['coarse.'+k]) for k, v in model.coarse.state_dict().items()):
        raise ValueError('Declared trainable coarse weights never changed')
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    if change <= 0 or not np.isclose(change, summary['final_weight_change_l2']):raise ValueError('Actual learned weight change differs')
    if rng.bit_generator.state != latest['numpy_rng'] or not torch.equal(torch.get_rng_state(), latest['torch_rng']):
        raise ValueError('Exact RNG differs')
    historical = torch.load('runs/localization-v18-window-cost-pilot/latest.pt', map_location='cpu', weights_only=True)
    if rng.bit_generator.state != historical['numpy_rng']:raise ValueError('Compared historical image-augmentation draws differ')
    if optimizer.state_dict()['param_groups'] != latest['optimizer']['param_groups']:raise ValueError('Optimizer groups differ')
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            if not torch.equal(value, latest['optimizer']['state'][key][name]):raise ValueError('Exactly retrained optimizer differs')
    result = dict(passed=True, completed_steps=100, exact_full_retraining=True, exact_model_optimizer_rng_replay=True,
        independent_assignment_scalar_matrices=len(independent), independent_scalar_checks=independent,
        prior_independent_geometry_audit_reused=True, unchanged_initial_inference_graph_reused=True,
        fine_encoder_unchanged=True, total_parameters=81696, pilot_summary_sha256=sha256(root/'summary.json'),
        pilot_protocol_sha256=sha256(root/'protocol.json'), source_sha256=sha256(__file__),
        scope='Train-only pilot reproducibility/new-scalar numerical audit. Not a localization gain, selected model, fresh terrain test or independent full neural implementation.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
