"""Reconstruct all training inputs and the fixed descriptor validation bank.

This verifies actual pixel membership, eligible pairs, initialization, checkpoint
selection and selected-checkpoint descriptor loss. It does not certify a new
registration improvement or reproduce every stochastic optimization step.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from contrastive_terrain import TerrainDescriptor
from dataset import sha256
from landmark_review import fingerprint
from moon_dataset import load_image
from research_dataset import validate
from train_contrastive_terrain import crop_pair, pair_loss, training_tiles


def rebuild_cache(root):
    """Rebuild omitted image cache from acquired source rasters, without fitting."""
    root = Path(root)
    path = root/'training_arrays.npz'
    if path.exists():
        raise FileExistsError(path)
    metadata = json.loads((root/'training.json').read_text())
    manifest = validate(json.loads((root/'frozen_manifest.json').read_text()))
    if fingerprint(manifest) != metadata['manifest_sha256']:
        raise ValueError('Training manifest differs')
    records = json.loads((root/'pairs.json').read_text())
    images, masks = [], []
    for ti, tile in enumerate(training_tiles(manifest)):
        entry = records['source_arrays'][ti]
        if entry['tile_id'] != tile['tile_id'] or entry['split'] != tile['split']:
            raise ValueError('Rebuild allowlist differs')
        loaded = [load_image(image)[:2] for image in tile['images']]
        for index, (image, mask) in enumerate(loaded):
            if hashlib.sha256(image.tobytes()).hexdigest() != entry['image_sha256'][index] or hashlib.sha256(mask.tobytes()).hexdigest() != entry['mask_sha256'][index]:
                raise ValueError('Rebuilt native input bytes differ')
        images.append(np.stack([p[0] for p in loaded]))
        masks.append(np.stack([p[1] for p in loaded]))
    np.savez_compressed(path, images=np.stack(images), masks=np.stack(masks))
    if sha256(path) != metadata['training_arrays_sha256']:
        raise ValueError('Rebuilt cache encoding differs from the frozen cache')


def verify(root):
    root = Path(root)
    metadata = json.loads((root/'training.json').read_text())
    terminal = json.loads((root/'COMPLETE.json').read_text())
    if sha256(root/'training.json') != terminal['training_sha256']:
        raise ValueError('Training metadata changed')
    for name, digest in metadata['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Training source changed')
    for name in ['frozen_config', 'frozen_manifest', 'pairs']:
        if sha256(root/f'{name}.json') != metadata[f'{name}_sha256']:
            raise ValueError('Training design or pair bytes changed')
    if sha256(root/'training_arrays.npz') != metadata['training_arrays_sha256']:
        raise ValueError('Training cache changed')
    manifest = validate(json.loads((root/'frozen_manifest.json').read_text()))
    if fingerprint(manifest) != metadata['manifest_sha256']:
        raise ValueError('Training manifest differs')
    tiles = training_tiles(manifest)
    if metadata['selected_tiles'] != [dict(tile_id=t['tile_id'], split=t['split']) for t in tiles]:
        raise ValueError('Training tile allowlist differs')
    cache = np.load(root/'training_arrays.npz', allow_pickle=False)
    images, masks = cache['images'], cache['masks']
    if images.shape != (len(tiles), 9, 128, 128) or masks.shape != images.shape or images.dtype != np.uint8 or masks.dtype != bool:
        raise ValueError('Unexpected cache shape/dtype')
    records = json.loads((root/'pairs.json').read_text())
    training, development = [], []
    for ti, tile in enumerate(tiles):
        entry = records['source_arrays'][ti]
        if entry['tile_id'] != tile['tile_id'] or entry['split'] != tile['split']:
            raise ValueError('Source array identity differs')
        for j, source in enumerate(tile['images']):
            byte, mask, _ = load_image(source)
            if not np.array_equal(byte, images[ti, j]) or not np.array_equal(mask, masks[ti, j]):
                raise ValueError('Actual source pixels/masks differ from cached training inputs')
            if hashlib.sha256(byte.tobytes()).hexdigest() != entry['image_sha256'][j] or hashlib.sha256(mask.tobytes()).hexdigest() != entry['mask_sha256'][j]:
                raise ValueError('Source-array hashes differ')
        eligible = [(ti, i, j) for i, j in itertools.combinations(range(9), 2)
            if int((masks[ti, i] & masks[ti, j] & (images[ti, i] > 3) & (images[ti, j] > 3)).sum()) >= 32]
        if tile['split'] == 'train':
            training.extend(eligible)
        else:
            eligible.sort(key=lambda p: hashlib.sha256(f'{metadata["seed"]}:{tile["tile_id"]}:{p[1]}:{p[2]}'.encode()).hexdigest())
            development.extend(eligible[:1])
    if [list(p) for p in training] != records['training'] or [list(p) for p in development] != records['development']:
        raise ValueError('Eligibility or pair identities differ')
    rng = np.random.default_rng(metadata['seed']+1)
    dev = [crop_pair(images, masks, p, rng, True) for p in development]
    dev = [s for s in dev if s is not None]
    bank = hashlib.sha256()
    for sample in dev:
        for array in sample:
            bank.update(array.tobytes())
    if bank.hexdigest() != metadata['fixed_development_bank_sha256'] or len(dev) != metadata['development_samples']:
        raise ValueError('Fixed development descriptor bank differs')
    torch.set_num_threads(2)
    torch.manual_seed(metadata['seed'])
    model = TerrainDescriptor().eval()
    for checkpoint in ['initial', 'selected']:
        path = root/f'{checkpoint}.pt'
        if sha256(path) != terminal[f'{checkpoint}_sha256']:
            raise ValueError('Checkpoint bytes changed')
    initial = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
        raise ValueError('Saved initialization differs from declared seed')
    selected = torch.load(root/'selected.pt', map_location='cpu', weights_only=True)
    if any(not torch.isfinite(v).all() for v in selected.values()):
        raise ValueError('Nonfinite selected weights')
    model.load_state_dict(selected, strict=True)
    with torch.no_grad():
        value = float(np.mean([pair_loss(model, s, 'cpu').item() for s in dev]))
    best = min(metadata['history'], key=lambda r: r['development_descriptor_loss'])
    if best['step'] != metadata['selected_step'] or not np.isclose(value, best['development_descriptor_loss'], atol=1e-6):
        raise ValueError('Selected checkpoint or actual descriptor validation loss differs')
    return dict(training_tiles=sum(t['split']=='train' for t in tiles), development_tiles=sum(t['split']=='dev' for t in tiles),
        reconstructed_native_images=len(tiles)*9, reconstructed_native_masks=len(tiles)*9,
        training_pairs=len(training), development_samples=len(dev), development_bank_sha256=bank.hexdigest(),
        initial_seed_verified=True, selected_step=metadata['selected_step'], selected_descriptor_loss=value,
        reserve_images_loaded=0, training_sha256=terminal['training_sha256'],
        warning='Descriptor training/input audit only; not a registration or fresh-geography improvement claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('training')
    p.add_argument('--output', required=True)
    p.add_argument('--rebuild-cache', action='store_true', help='Recreate omitted train/dev cache from acquired rasters; never retrain')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.rebuild_cache:
        rebuild_cache(args.training)
    result = verify(args.training)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
