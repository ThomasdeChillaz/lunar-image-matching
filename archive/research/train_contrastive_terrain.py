"""Train an original local structural CNN on train/dev only; no reserve loader.

Motivation: local self-similarity from medical registration, contrastive common
representations (CoMIR), and dense correspondence learning. This is not a
reproduction of any of those methods. Descriptor loss selects a checkpoint;
independent downstream development registration is still required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from contrastive_terrain import TerrainDescriptor, contrastive_loss, sample_descriptors, valid_centers
from dataset import sha256
from geometry import transform
from landmark_review import fingerprint
from moon_dataset import load_image
from research_dataset import validate


def training_tiles(manifest):
    # Apply allowlist BEFORE any image loader, including cache construction.
    return [t for t in manifest['tiles'] if t['split'] in ('train', 'dev')]


def crop_pair(images, masks, pair, rng, augmented):
    ti, i, j = pair
    a, base = images[ti, i], images[ti, j]
    ma, mb = masks[ti, i], masks[ti, j]
    h, w = a.shape
    angle = float(rng.uniform(-45, 45)) if augmented else 0.
    scale = float(np.exp(rng.uniform(np.log(.7), np.log(1.4)))) if augmented else 1.
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, scale)
    if augmented:
        matrix[:, 2] += rng.uniform(-12, 12, 2)
    b = cv2.warpAffine(np.where(mb, base, 0), matrix, (w, h), flags=cv2.INTER_LINEAR)
    qb = cv2.warpAffine(mb.astype(np.float32), matrix, (w, h), flags=cv2.INTER_LINEAR) >= 1.-1e-6
    b[~qb] = 0
    size = 80
    ox, oy = (int(rng.integers(w-size+1)), int(rng.integers(h-size+1)))
    a, b = a[oy:oy+size, ox:ox+size], b[oy:oy+size, ox:ox+size]
    ma, qb = ma[oy:oy+size, ox:ox+size], qb[oy:oy+size, ox:ox+size]
    y, x = np.mgrid[16:size-15:8, 16:size-15:8]
    points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
    target = transform(points+[ox, oy], matrix)-[ox, oy]
    # Fractional query coordinates need all four bilinear descriptor centers.
    # Radius-13 erosion conservatively contains radius-12 descriptor support.
    va, vb = valid_centers(ma), valid_centers(qb, radius=13)
    inside = ((target >= 0) & (target <= size-1)).all(1)
    idx = np.rint(target).astype(int).clip(0, size-1)
    keep = inside & va[points[:, 1].astype(int), points[:, 0].astype(int)] & vb[idx[:, 1], idx[:, 0]]
    # Non-dark is an explicit proxy for training signal, not visibility truth.
    keep &= (a[points[:, 1].astype(int), points[:, 0].astype(int)] > 3) & (b[idx[:, 1], idx[:, 0]] > 3)
    points, target = points[keep], target[keep].astype(np.float32)
    if len(points) < 12:
        return None
    return np.stack([np.where(ma, a, 0), np.where(qb, b, 0)]), points, target


def pair_loss(model, sample, device):
    images, points, target = sample
    dense = model(torch.from_numpy(images.copy()).to(device).float()[:, None]/255)
    a = sample_descriptors(dense[:1], torch.from_numpy(points).to(device)[None])[0]
    b = sample_descriptors(dense[1:], torch.from_numpy(target).to(device)[None])[0]
    return contrastive_loss(a, b)


def train(manifest, output, steps=1200, augmented=True, device='cpu', seed=20261030):
    validate(manifest)
    if steps <= 0:
        raise ValueError('Positive training steps required')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    tiles = training_tiles(manifest)
    files = ['train_contrastive_terrain.py', 'contrastive_terrain.py', 'moon_dataset.py', 'research_dataset.py',
             'dataset.py', 'geometry.py', 'landmark_review.py']
    config = dict(schema_version=1, seed=seed, steps=steps, augmented=augmented, device=device,
        manifest_sha256=fingerprint(manifest), source_sha256={p: sha256(p) for p in files},
        included_splits=['train', 'dev'], selected_tiles=[dict(tile_id=t['tile_id'], split=t['split']) for t in tiles],
        learning_rate=.0005, temperature=.1, weight_decay=.0001, gradient_clip=5., crop_size=80,
        training_motion=dict(rotation_deg=[-45, 45], scale=[.7, 1.4], shift_px=[-12, 12]),
        checkpoint_interval=200, minimum_paired_points=12, training_pair_minimum_nondark=32,
        descriptor_radius_px=12, model='13 local input channels; 24/32/32/32 convolution, dilation1/2/4/1',
        selection='Minimum fixed development descriptor loss; initial weights eligible; not registration-based model selection',
        warning='Development-only representation hypothesis; no reserve pixels or outcomes read.')
    (output/'frozen_config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    images, masks, pairs, dev_pairs, source_arrays = [], [], [], [], []
    for ti, tile in enumerate(tiles):
        loaded = [load_image(image)[:2] for image in tile['images']]
        ii, mm = np.stack([p[0] for p in loaded]), np.stack([p[1] for p in loaded])
        images.append(ii); masks.append(mm)
        source_arrays.append(dict(tile_id=tile['tile_id'], split=tile['split'],
            image_sha256=[hashlib.sha256(p.tobytes()).hexdigest() for p in ii],
            mask_sha256=[hashlib.sha256(p.tobytes()).hexdigest() for p in mm]))
        eligible = [(ti, i, j) for i, j in itertools.combinations(range(len(ii)), 2)
                    if int((mm[i] & mm[j] & (ii[i] > 3) & (ii[j] > 3)).sum()) >= 32]
        if tile['split'] == 'train':
            pairs.extend(eligible)
        else:
            eligible.sort(key=lambda p: hashlib.sha256(f'{seed}:{tile["tile_id"]}:{p[1]}:{p[2]}'.encode()).hexdigest())
            dev_pairs.extend(eligible[:1])
        if (ti+1) % 32 == 0:
            print(f'Cache {ti+1}/{len(tiles)} train/dev tiles', flush=True)
    images, masks = np.stack(images), np.stack(masks)
    np.savez_compressed(output/'training_arrays.npz', images=images, masks=masks)
    records = dict(training=pairs, development=dev_pairs, source_arrays=source_arrays)
    (output/'pairs.json').write_text(json.dumps(records, indent=2)+'\n')
    if not pairs or not dev_pairs:
        raise ValueError('Need eligible train/dev pairs')
    # This fixed bank is constructed once and reused for every checkpoint.
    dev_rng = np.random.default_rng(seed+1)
    # Always the same geometric validation bank, including for a no-motion
    # training ablation. Otherwise two checkpoint selectors see different tasks.
    dev = [crop_pair(images, masks, p, dev_rng, True) for p in dev_pairs]
    dev = [s for s in dev if s is not None]
    if len(dev) < 16:
        raise ValueError('Insufficient fixed development samples')
    bank_hash = hashlib.sha256()
    for sample in dev:
        for array in sample:
            bank_hash.update(array.tobytes())
    model = TerrainDescriptor().to(device)
    torch.save(model.state_dict(), output/'initial.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    def validation():
        model.eval()
        with torch.no_grad():
            value = float(np.mean([pair_loss(model, sample, device).item() for sample in dev]))
        model.train()
        return value
    best_loss, best_step = validation(), 0
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [dict(step=0, development_descriptor_loss=best_loss)]
    print(f'Initial fixed dev loss={best_loss:.6f}; {len(pairs)} train pairs, {len(dev)} dev samples', flush=True)
    skipped, start = 0, perf_counter()
    for step in range(1, steps+1):
        sample = None
        for _ in range(100):
            pair = pairs[int(rng.integers(len(pairs)))]
            if rng.random() < .5:
                pair = (pair[0], pair[2], pair[1])
            sample = crop_pair(images, masks, pair, rng, augmented)
            if sample is not None:
                break
            skipped += 1
        if sample is None:
            raise ValueError('Unable to sample sufficient co-visible training structure')
        optimizer.zero_grad(set_to_none=True)
        loss = pair_loss(model, sample, device)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite descriptor objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()
        if step % config['checkpoint_interval'] == 0 or step == steps:
            value = validation()
            history.append(dict(step=step, training_descriptor_loss=float(loss.item()), development_descriptor_loss=value))
            if value < best_loss:
                best_loss, best_step = value, step
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f'Step {step}/{steps}; dev loss={value:.6f}; selected={best_step}; elapsed={perf_counter()-start:.1f}s', flush=True)
    torch.save(best, output/'selected.pt')
    metadata = dict(config, frozen_config_sha256=sha256(output/'frozen_config.json'),
        frozen_manifest_sha256=sha256(output/'frozen_manifest.json'), pairs_sha256=sha256(output/'pairs.json'),
        training_arrays_sha256=sha256(output/'training_arrays.npz'), fixed_development_bank_sha256=bank_hash.hexdigest(),
        training_pairs=len(pairs), development_samples=len(dev), selected_step=best_step, history=history,
        skipped_crop_proposals=skipped, reserve_images_loaded=0, elapsed_training_seconds=perf_counter()-start,
        environment=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__))
    (output/'training.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    completed = {f'{name}_sha256': sha256(output/f'{name}.pt') for name in ('initial', 'selected')}
    completed['training_sha256'] = sha256(output/'training.json')
    (output/'COMPLETE.json').write_text(json.dumps(completed, indent=2)+'\n')
    print(f'Completed training; selected={best_step}; no reserve images read', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='research_manifest_v1.json')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--without-motion-augmentation', action='store_true')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = p.parse_args()
    train(json.loads(Path(args.manifest).read_text()), args.output, args.steps, not args.without_motion_augmentation, args.device)
