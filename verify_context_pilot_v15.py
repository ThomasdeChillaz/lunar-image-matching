"""Independent label audit and exact100-step deterministic pilot retraining."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import make_context, objective, prepare_supervision, proposal_grid, sample_training
from contrastive_terrain import sample_descriptors
from dataset import sha256
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import quarter_turn
from train_context_pilot_v15 import load_training
from verify_axial_results_v5 import projected


def audit(root, output):
    root = Path(root); torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    completed = json.loads((root/'COMPLETE.json').read_text())
    for key, digest in completed.items():
        if sha256(root/key.removesuffix('_sha256')) != digest:
            raise ValueError('Completed pilot artifact changed')
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest or sha256(root/'source'/source) != digest:
            raise ValueError('Executed source/snapshot changed')
        if subprocess.check_output(['git', 'show', protocol['publication_commit']+':'+source]) != Path(source).read_bytes():
            raise ValueError('Executed source differs from the public freeze')
    if protocol['selection_inference'] or protocol['selection_pixels_loaded'] or protocol['calibration_or_test_pixels_loaded']:
        raise ValueError('Train-only pilot scope changed')
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest, tiles, data = load_training(manifest_path)
    if sha256(manifest_path) != protocol['manifest_sha256'] or any(t['split'] != 'train' for t in tiles.values()):
        raise ValueError('Training-only data changed')
    for source, digest in protocol['input_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Training input changed')
    prepared, support = prepare_supervision(tiles, data)
    if support != protocol['training_support']:
        raise ValueError('Prepared training support changed')
    native = (proposal_grid()+.5)*4-.5; label_points = 0
    for key, tile in tiles.items():
        z = data[key]; a, b = tile['cameras']
        forward, visible = projected(native, z['depth_a'], z['depth_b'], a, b)
        for k in range(4):
            labels = prepared[key]['rotations'][k]
            expect = quarter_turn((forward[visible]+.5)/4-.5, k, 128)
            if not np.array_equal(labels['forward_visible'], visible) or not np.allclose(labels['forward'][visible], expect, rtol=0, atol=4e-5):
                raise ValueError('Independent forward3D projection/visibility differs')
            source = quarter_turn(native, k, 512, inverse=True)
            reverse, reverse_visible = projected(source, z['depth_b'], z['depth_a'], b, a)
            if (not np.array_equal(labels['reverse_visible'], reverse_visible)
                    or not np.allclose(labels['reverse'][reverse_visible], (reverse[reverse_visible]+.5)/4-.5, rtol=0, atol=4e-5)):
                raise ValueError('Independent reverse3D projection/visibility differs')
            label_points += 2*len(native)
    torch.manual_seed(protocol['seed']); rng = np.random.default_rng(protocol['seed'])
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    model = make_context(protocol['variant'])
    initial = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
    final = torch.load(root/'final.pt', map_location='cpu', weights_only=True)
    latest = torch.load(root/'latest.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
        raise ValueError('Seeded initial context state differs')
    number = sum(p.numel() for p in model.parameters())
    if number != protocol['context_parameters'] or number+28272+35233 >= 100000:
        raise ValueError('Parameter cap differs')
    if latest['step'] != protocol['steps'] or summary['completed_steps'] != protocol['steps']:
        raise ValueError('Pilot update budget differs')
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    grid = torch.from_numpy(proposal_grid())[None]; negative = 0
    if len(summary['history']) != protocol['steps']:
        raise ValueError('Missing pilot fitting history')
    for step, saved in enumerate(summary['history'], 1):
        model.train(); images, targets, sample = sample_training(tiles, prepared, rng, negative_pair=step % 4 == 0)
        if sample != saved['sample'] or step != saved['step']:
            raise ValueError('Exact training sample replay differs')
        negative += sample['verified_nonoverlap']
        with torch.no_grad():
            dense, _ = coarse.model(images)
            a, b = sample_descriptors(dense[:1], grid), sample_descriptors(dense[1:], grid)
        optimizer.zero_grad(set_to_none=True); loss, labels = objective(model(a, b, grid, grid), targets)
        loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
        if labels != saved['labels'] or float(loss) != saved['loss'] or float(gradient) != saved['gradient_norm']:
            raise ValueError('Exact loss/label/gradient retraining differs at step'+str(step))
        optimizer.step()
    if any(not torch.equal(v, final[k]) or not torch.equal(v, latest['model'][k]) for k, v in model.state_dict().items()):
        raise ValueError('Exactly retrained final model differs')
    if rng.bit_generator.state != latest['numpy_rng'] or not torch.equal(torch.get_rng_state(), latest['torch_rng']):
        raise ValueError('Recorded sampler/model RNG differs')
    if {int(v['step']) for v in latest['optimizer']['state'].values()} != {protocol['steps']}:
        raise ValueError('Optimizer step counts differ')
    for key, state in optimizer.state_dict()['state'].items():
        for field, value in state.items():
            if not torch.equal(value, latest['optimizer']['state'][key][field]):
                raise ValueError('Exactly retrained AdamW state differs')
    change = float(sum((v-initial[k]).square().sum() for k, v in model.state_dict().items()).sqrt())
    if change <= 0 or change != summary['final_weight_change_l2'] or negative != summary['negative_pairs']:
        raise ValueError('Weight/sample summary differs')
    result = dict(passed=True, completed_steps=protocol['steps'], independently_projected_label_points=label_points,
        negative_pairs=negative, exact_full_retraining=True, exact_model_optimizer_rng_replay=True,
        context_parameters=number, final_weight_change_l2=change, source_sha256=sha256(__file__),
        summary_sha256=sha256(root/'summary.json'),
        scope='Train-only geometry/optimizer reproducibility and cost pilot, not localization performance, calibrated acceptance or fresh validation.')
    write_new(output, result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); audit(args.run, args.output)
