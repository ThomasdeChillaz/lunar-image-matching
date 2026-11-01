"""Independent saved-pose/sampler audit, with optional exact context retraining."""
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
from train_balanced_localization_v3 import load_bundle
from verify_axial_results_v5 import projected
from verify_surface_learning_v13 import aggregate, read_rows, replay_pose


def key(record):
    return (record['selection']['correct_poses'], record['selection']['correct_at_two_px'], -record['step'])


def labels_independently(tiles, data, prepared):
    native = (proposal_grid()+.5)*4-.5; checked = 0
    for tile_id, tile in tiles.items():
        if tile['split'] != 'train':
            continue
        z = data[tile_id]; a, b = tile['cameras']
        forward, visible = projected(native, z['depth_a'], z['depth_b'], a, b)
        for rotation in range(4):
            labels = prepared[tile_id]['rotations'][rotation]
            expected = quarter_turn((forward[visible]+.5)/4-.5, rotation, 128)
            if not np.array_equal(labels['forward_visible'], visible) or not np.allclose(labels['forward'][visible], expected, rtol=0, atol=4e-5):
                raise ValueError('Independent forward projection/visibility differs')
            source = quarter_turn(native, rotation, 512, inverse=True)
            reverse, valid = projected(source, z['depth_b'], z['depth_a'], b, a)
            if not np.array_equal(labels['reverse_visible'], valid) or not np.allclose(labels['reverse'][valid], (reverse[valid]+.5)/4-.5, rtol=0, atol=4e-5):
                raise ValueError('Independent reverse projection/visibility differs')
            checked += 2*len(native)
    return checked


def audit(run, output, partial=False, retrain=False):
    root = Path(run); torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    protocol = json.loads((root/'protocol.json').read_text())
    manifest_path = Path('reports/localization_training_data_v5.json')
    if protocol['calibration_or_test_pixels_loaded'] or protocol['fresh_data_gate_passed'] or sha256(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('Manifest/evaluation scope differs')
    for source, digest in protocol['source_sha256'].items():
        if sha256(source) != digest or subprocess.check_output(['git','show',protocol['publication_commit']+':'+source]) != Path(source).read_bytes():
            raise ValueError('Executed source/public freeze differs: '+source)
    for path, digest in protocol['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Input bytes differ')
    bundle = load_bundle(manifest_path); manifest, tiles, data = bundle
    groups = {split: {t['component'] for t in tiles.values() if t['split'] == split} for split in ('train','model_selection')}
    if groups['train'] & groups['model_selection'] or any(t['split'] not in groups for t in tiles.values()):
        raise ValueError('Train/selection leakage')
    prepared, support = prepare_supervision(tiles, data)
    if support != protocol['training_support']:
        raise ValueError('Training support differs')
    label_points = labels_independently(tiles, data, prepared)
    expected_cases = [c['pair_id'] for c in manifest['cases'] if c['split'] == 'model_selection'
        and (not protocol['pilot'] or c['pair_id'] in manifest['selection_bank'])]
    baseline = {r['pair_id']: r for r in read_rows('runs/localization-v11-surface-warp-development/results.json.gz')}
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    records = []; pose_rows = 0; retrained_steps = 0
    folders = sorted(p.parent for p in root.glob('*/COMPLETE.json'))
    if not folders:
        raise ValueError('No completed configuration to audit')
    for folder in folders:
        meta = json.loads((folder/'training.json').read_text()); config = json.loads((folder/'frozen_config.json').read_text())
        for path, digest in json.loads((folder/'COMPLETE.json').read_text())['sha256'].items():
            if sha256(folder/path) != digest:
                raise ValueError('Completed evidence changed: '+path)
        for source, digest in protocol['source_sha256'].items():
            if meta['source_sha256'][source] != digest or sha256(folder/'source'/source) != digest:
                raise ValueError('Source snapshot differs')
        if any(config[k] != v for k, v in protocol.items()):
            raise ValueError('Per-run frozen protocol differs')
        torch.manual_seed(meta['seed']); rng = np.random.default_rng(meta['seed']); model = make_context(meta['variant'])
        load = lambda p: torch.load(folder/p, map_location='cpu', weights_only=False)
        initial, final, latest, selected = [load(p) for p in ('initial.pt','final.pt','latest.pt','selected.pt')]
        if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
            raise ValueError('Predefined seeded initialization differs')
        number = sum(p.numel() for p in model.parameters())
        if number != meta['context_parameters'] or 28272+number != meta['total_parameters'] or 28272+35233+number >= 100000:
            raise ValueError('Capacity/cap differs')
        if meta['completed_steps'] != protocol['steps'] or latest['step'] != protocol['steps'] or len(meta['history']) != protocol['steps']:
            raise ValueError('Fitting budget/history differs')
        if any(not torch.equal(v, latest['model'][k]) for k,v in final.items()):
            raise ValueError('Final/checkpoint weights differ')
        if {int(s['step']) for s in latest['optimizer']['state'].values()} != {protocol['steps']}:
            raise ValueError('Optimizer step budget differs')
        expected_steps = sorted({0,protocol['steps'],*range(protocol['interval'],protocol['steps']+1,protocol['interval'])})
        if [v['step'] for v in meta['validation']] != expected_steps:
            raise ValueError('Missing selection checkpoint')
        for validation in meta['validation']:
            path = folder/f'selection_step_{validation["step"]}.json.gz'; rows = read_rows(path)
            if sha256(path) != validation['results_sha256'] or aggregate(rows) != validation['selection'] or [r['pair_id'] for r in rows] != expected_cases:
                raise ValueError('Selection denominators/aggregate differs')
            for row in rows:
                replay_pose(row, tiles[row['tile_id']], data[row['tile_id']])
                if validation['step'] == 0:
                    for name in ('pose_correct','strict_pose_correct','reference_points_native','query_points_native','estimated_cam2world'):
                        if row[name] != baseline[row['pair_id']][name]:
                            raise ValueError('Initial exact native equivalence differs')
            pose_rows += len(rows)
        winner = max(meta['validation'], key=key)
        if list(key(winner)) != meta['selection_key'] or winner['step'] != meta['selected_step'] or winner['selection'] != meta['evaluation']:
            raise ValueError('Checkpoint selection differs')
        chosen = load(f'checkpoint_step_{winner["step"]}.pt')
        if any(not torch.equal(v, chosen[k]) for k,v in selected.items()) or winner['results_sha256'] != meta['selected_rows_sha256']:
            raise ValueError('Selected weights/results differ')
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
        grid = torch.from_numpy(proposal_grid())[None]; negatives = 0
        for step, saved in enumerate(meta['history'], 1):
            images, targets, sample = sample_training(tiles, prepared, rng, negative_pair=step % 4 == 0)
            if saved['step'] != step or saved['sample'] != sample or not np.isfinite([saved['loss'],saved['gradient_norm']]).all():
                raise ValueError('Sampler/history replay differs')
            negatives += sample['verified_nonoverlap']
            if retrain:
                model.train()
                with torch.no_grad():
                    dense,_ = coarse.model(images)
                    a,b = sample_descriptors(dense[:1],grid),sample_descriptors(dense[1:],grid)
                optimizer.zero_grad(set_to_none=True); loss,labels = objective(model(a,b,grid,grid),targets)
                loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
                if float(loss) != saved['loss'] or float(gradient) != saved['gradient_norm'] or labels != saved['labels']:
                    raise ValueError('Exact loss/gradient retraining differs at'+str(step))
                optimizer.step(); retrained_steps += 1
        if rng.bit_generator.state != latest['numpy_rng'] or negatives != meta['negative_pairs']:
            raise ValueError('Sampler RNG/nonoverlap budget differs')
        if retrain:
            if any(not torch.equal(v,final[k]) for k,v in model.state_dict().items()) or not torch.equal(torch.get_rng_state(),latest['torch_rng']):
                raise ValueError('Exact final weights/model RNG differ')
            for index,state in optimizer.state_dict()['state'].items():
                for field,value in state.items():
                    if not torch.equal(value,latest['optimizer']['state'][index][field]):
                        raise ValueError('Exact optimizer replay differs')
        records.append({k:meta[k] for k in ('variant','seed','completed_steps','selected_step','evaluation','selection_key','fitting_seconds')})
        print('audited context',folder.name,'selected',winner['selection'],flush=True)
    order = {v:i for i,v in enumerate(protocol['variants'])}
    primary = [r for r in records if r['seed'] == 20261101]
    if not partial:
        summary = json.loads((root/'summary.json').read_text())
        winner = max(primary,key=lambda r:(tuple(r['selection_key']),-order[r['variant']]))['variant']
        expected = {(v,20261101) for v in protocol['variants']}
        if not protocol['pilot']:
            expected |= {(winner,s) for s in (20261102,20261103)}
        if {(r['variant'],r['seed']) for r in records} != expected or summary['winner'] != winner:
            raise ValueError('Full ablation/repeat cohort differs')
        sort = lambda rows: sorted(rows,key=lambda r:(r['variant'],r['seed']))
        if sort(summary['results']) != sort(records) or summary['completed_optimizer_steps'] != sum(r['completed_steps'] for r in records):
            raise ValueError('Cohort summary differs')
    result = dict(passed=True,partial=partial,completed_runs=len(records),recorded_updates=sum(r['completed_steps'] for r in records),
        replayed_pose_rows=pose_rows,independently_projected_label_points=label_points,
        exact_retrained_steps=retrained_steps,exact_retraining_requested=retrain,runs=records,
        source_sha256=sha256(__file__),protocol_sha256=sha256(root/'protocol.json'),
        scope='Exposed-development reproducibility audit, not fresh geographic validation or calibrated acceptance. Full retraining only if explicitly recorded.')
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--partial',action='store_true');parser.add_argument('--retrain',action='store_true')
    args=parser.parse_args();audit(args.run,args.output,args.partial,args.retrain)
