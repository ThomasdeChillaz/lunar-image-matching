"""Expanded, all-light-pair DISK adaptation; never load geographic reserves.

This is a candidate generator. Descriptor retrieval selects a checkpoint using
development terrain; a downstream development experiment is still necessary
before freezing any final model or claiming registration improvements.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from landmark_review import fingerprint
from learned import LearnedMatcher
from moon_dataset import load_image
from research_dataset import validate


def train(manifest, output, steps=2000, seed=20261029):
    validate(manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if steps <= 0:
        raise ValueError('Positive training steps required')
    output.mkdir(parents=True)
    selected = [t for t in manifest['tiles'] if t['split'] in ('train', 'dev')]
    model = LearnedMatcher(device='cpu')
    torch = model.torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source_files = ['train_research_adapter.py', 'research_dataset.py', 'learned.py', 'moon_dataset.py', 'model_lock.json']
    config = dict(seed=seed, steps=steps, manifest_sha256=fingerprint(manifest),
        model_lock_sha256=sha256('model_lock.json'), included_splits=['train', 'dev'],
        selected_tiles=[t['tile_id'] for t in selected], grid='8..112 every 8 native pixels',
        light_pairs='All unordered pairs, both descriptor retrieval directions; minimum 32 jointly non-dark grid locations',
        temperature=.1, learning_rate=.0003, identity_regularization=.25, orthogonality_regularization=.01,
        singular_value_projection=[.75, 1.25], validation='Four deterministic hash-selected eligible light pairs per development patch',
        checkpoint_steps=[0]+[s for s in range(200, steps+1, 200)]+([] if steps % 200 == 0 else [steps]),
        source_sha256={name: sha256(name) for name in source_files},
        warning='Descriptor surrogate only. No final reserve matching or model selection has occurred.')
    (output/'frozen_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    y, x = np.meshgrid(np.arange(8, 120, 8), np.arange(8, 120, 8), indexing='ij')
    y, x = y.ravel(), x.ravel()
    all_descriptors, all_valid, training, development, excluded = [], [], [], [], []
    for ti, tile in enumerate(selected):
        descriptors, valid = [], []
        # Only the already filtered train/dev tile list can reach load_image.
        for image in tile['images']:
            byte, mask, raw = load_image(image)
            tensor = torch.from_numpy(byte).float()[None, None].repeat(1, 3, 1, 1)/255
            with torch.inference_mode():
                heat, dense = model.extractor.model.heatmap_and_dense_descriptors(tensor)
                desc = torch.nn.functional.normalize(dense[0, :, y, x].T, dim=-1).numpy()
            descriptors.append(desc.astype(np.float16))
            valid.append(mask[y, x] & (byte[y, x] > 3))
        descriptors, valid = np.array(descriptors), np.array(valid)
        all_descriptors.append(descriptors)
        all_valid.append(valid)
        eligible = []
        for i, j in itertools.combinations(range(9), 2):
            count = int((valid[i] & valid[j]).sum())
            if count < 32:
                excluded.append(dict(tile_id=tile['tile_id'], split=tile['split'], i=i, j=j, jointly_visible_grid_points=count))
                continue
            eligible.append((ti, i, j))
        if tile['split'] == 'train':
            training.extend(eligible)
        else:
            eligible.sort(key=lambda p: hashlib.sha256(f'{seed}:{tile["tile_id"]}:{p[1]}:{p[2]}'.encode()).hexdigest())
            development.extend(eligible[:4])
        print(f'Extract {ti+1}/{len(selected)} {tile["split"]} {tile["tile_id"]} eligible={len(eligible)}', flush=True)
    if not training or not development:
        raise ValueError('Need co-visible training and development descriptor pairs')
    # Float16 cache reduces storage; fitting converts selected samples to float32.
    descriptors = np.array(all_descriptors)
    visible = np.array(all_valid)
    np.savez_compressed(output/'descriptor_grid.npz', descriptors=descriptors, visible=visible)
    (output/'training_pairs.json').write_text(json.dumps(dict(training=training, development=development,
        tile_ids=[t['tile_id'] for t in selected], excluded_from_descriptor_training=excluded), indent=2))
    identity = torch.eye(128)
    matrix = torch.nn.Parameter(identity.clone())
    optimizer = torch.optim.Adam([matrix], lr=config['learning_rate'])
    def sample(pair):
        tile, i, j = pair
        keep = visible[tile, i] & visible[tile, j]
        return torch.from_numpy(descriptors[tile, i, keep].astype(np.float32)), torch.from_numpy(descriptors[tile, j, keep].astype(np.float32))
    dev_tensors = [sample(pair) for pair in development]
    def loss(a, b):
        aa = torch.nn.functional.normalize(a@matrix.T, dim=-1)
        bb = torch.nn.functional.normalize(b@matrix.T, dim=-1)
        logits = aa@bb.T/config['temperature']
        target = torch.arange(len(a))
        return (torch.nn.functional.cross_entropy(logits, target)+torch.nn.functional.cross_entropy(logits.T, target))/2
    def validation():
        with torch.no_grad():
            return float(np.mean([loss(a, b).item() for a, b in dev_tensors]))
    best_loss, best_step, best = validation(), 0, matrix.detach().clone()
    history = [dict(step=0, development_descriptor_loss=best_loss)]
    for step in range(1, steps+1):
        a, b = sample(training[int(rng.integers(len(training)))])
        optimizer.zero_grad()
        contrastive = loss(a, b)
        objective = contrastive+config['identity_regularization']*((matrix-identity)**2).mean()
        objective += config['orthogonality_regularization']*((matrix.T@matrix-identity)**2).mean()
        objective.backward()
        optimizer.step()
        if step % 50 == 0:
            with torch.no_grad():
                u, singular, vh = torch.linalg.svd(matrix)
                matrix.copy_((u*singular.clamp(*config['singular_value_projection']))@vh)
        if step in config['checkpoint_steps']:
            value = validation()
            history.append(dict(step=step, training_descriptor_loss=float(contrastive.item()), development_descriptor_loss=value))
            if value < best_loss:
                best_loss, best_step, best = value, step, matrix.detach().clone()
            print(f'Step {step}/{steps}: development descriptor loss={value:.6f}; selected={best_step}', flush=True)
    adapter = dict(schema_version=1, method='Bounded near-identity 128x128 DISK projection, all physical Sun pairs',
        manifest_sha256=fingerprint(manifest), model_lock_sha256=sha256('model_lock.json'),
        training_config_sha256=sha256(output/'frozen_config.json'), descriptor_cache_sha256=sha256(output/'descriptor_grid.npz'),
        seed=seed, steps=steps, selected_step=best_step, training_pairs=len(training), development_pairs=len(development),
        training_tiles=sum(t['split'] == 'train' for t in selected), development_tiles=sum(t['split'] == 'dev' for t in selected),
        selection='Minimum development descriptor loss; identity eligible. Must still evaluate downstream registration on development only.',
        matrix=best.numpy().tolist(), singular_values=torch.linalg.svdvals(best).tolist(), history=history,
        reserve_images_loaded=0, warning=config['warning'])
    (output/'adapter.json').write_text(json.dumps(adapter, indent=2, allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(adapter_sha256=sha256(output/'adapter.json'),
        config_sha256=sha256(output/'frozen_config.json'), manifest_sha256=sha256(output/'frozen_manifest.json'),
        training_pairs_sha256=sha256(output/'training_pairs.json'), descriptor_cache_sha256=sha256(output/'descriptor_grid.npz'))))
    print(f'Training complete: selected step {best_step}; no reserve images loaded.', flush=True)
    return adapter


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='research_manifest_v1.json')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=2000)
    args = p.parse_args()
    train(json.loads(Path(args.manifest).read_text()), args.output, args.steps)
