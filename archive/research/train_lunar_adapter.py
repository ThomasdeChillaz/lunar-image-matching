"""Learn a small DISK descriptor projection from physically varied lunar light.

DISK detection/backbone and LightGlue remain frozen. Exact pixel correspondences
come only from co-rendered views of each terrain patch. Train/dev are geographically
and acquisition-separated; the test split is never loaded for fitting/selection.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset import sha256
from learned import LearnedMatcher
from landmark_review import fingerprint
from moon_dataset import load_image, validate


def descriptor_data(manifest, model, output):
    cache = Path(output)/'descriptor_samples.npz'
    signature = fingerprint(dict(manifest=manifest, model_lock=json.loads(Path('model_lock.json').read_text()),
        grid='8..112 every 8 native pixels', minimum_joint_intensity=3, feature='normalized frozen DISK dense descriptors'))
    identity = cache.with_suffix('.json')
    if cache.exists() and identity.exists():
        meta = json.loads(identity.read_text())
        if meta['signature'] == signature and meta['sha256'] == sha256(cache):
            arrays = np.load(cache, allow_pickle=False)
            return [(meta['items'][i]['split'], arrays[f'a{i}'], arrays[f'b{i}']) for i in range(len(meta['items']))]
        raise ValueError('Descriptor cache provenance mismatch')
    torch = model.torch
    y, x = np.meshgrid(np.arange(8, 120, 8), np.arange(8, 120, 8), indexing='ij')
    y, x = y.ravel(), x.ravel()
    items, arrays, logs = [], {}, []
    for tile in manifest['tiles']:
        if tile['split'] not in ('train', 'dev'):
            continue
        descriptors, valid = [], []
        for image in tile['images']:
            byte, mask, raw = load_image(image)
            tensor = torch.from_numpy(byte).float()[None, None].repeat(1, 3, 1, 1)/255
            with torch.inference_mode():
                heat, dense = model.extractor.model.heatmap_and_dense_descriptors(tensor)
                sample = torch.nn.functional.normalize(dense[0, :, y, x].T, dim=-1).numpy()
            descriptors.append(sample)
            valid.append(mask[y, x] & (byte[y, x] > 3))
        for query in range(1, len(tile['images'])):
            joint = valid[0] & valid[query]
            if joint.sum() < 32:
                continue
            i = len(items)
            a, b = descriptors[0][joint].copy(), descriptors[query][joint].copy()
            arrays[f'a{i}'], arrays[f'b{i}'] = a, b
            items.append((tile['split'], a, b))
            logs.append(dict(split=tile['split'], tile_id=tile['tile_id'], query_index=query, positive_correspondences=int(joint.sum())))
        print(tile['split'], tile['tile_id'], 'descriptor pairs', len(items), flush=True)
    if {item[0] for item in items} != {'train', 'dev'}:
        raise ValueError('Training and development need co-visible correspondences')
    np.savez_compressed(cache, **arrays)
    identity.write_text(json.dumps(dict(signature=signature, sha256=sha256(cache), items=logs), indent=2))
    return items


def train(manifest, output, steps=300, seed=20261026):
    validate(manifest, '.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
    model = LearnedMatcher(max_keypoints=512, device='cpu')
    torch = model.torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    samples = descriptor_data(manifest, model, output)
    training = [(torch.from_numpy(a), torch.from_numpy(b)) for split, a, b in samples if split == 'train']
    development = [(torch.from_numpy(a), torch.from_numpy(b)) for split, a, b in samples if split == 'dev']
    identity = torch.eye(128)
    matrix = torch.nn.Parameter(identity.clone())
    optimizer = torch.optim.Adam([matrix], lr=3e-4)
    def losses(a, b):
        aa = torch.nn.functional.normalize(a@matrix.T, dim=-1)
        bb = torch.nn.functional.normalize(b@matrix.T, dim=-1)
        logits = aa@bb.T/.1
        labels = torch.arange(len(a))
        loss = (torch.nn.functional.cross_entropy(logits, labels)+torch.nn.functional.cross_entropy(logits.T, labels))/2
        accuracy = ((logits.argmax(dim=1) == labels).float().mean()+(logits.argmax(dim=0) == labels).float().mean())/2
        return loss, accuracy
    def validation():
        with torch.no_grad():
            values = [losses(a, b) for a, b in development]
        return float(np.mean([v[0].item() for v in values])), float(np.mean([v[1].item() for v in values]))
    initial_loss, initial_accuracy = validation()
    best_loss, best_step, best = initial_loss, 0, matrix.detach().clone()
    history = [dict(step=0, development_loss=initial_loss, development_retrieval_accuracy=initial_accuracy)]
    rng = np.random.default_rng(seed)
    for step in range(1, steps+1):
        a, b = training[int(rng.integers(len(training)))]
        optimizer.zero_grad()
        loss, accuracy = losses(a, b)
        objective = loss+.1*((matrix-identity)**2).mean()
        objective.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps:
            dev_loss, dev_accuracy = validation()
            history.append(dict(step=step, training_loss=float(loss.item()), development_loss=dev_loss,
                development_retrieval_accuracy=dev_accuracy))
            if dev_loss < best_loss:
                best_loss, best_step, best = dev_loss, step, matrix.detach().clone()
            print('step', step, 'dev loss', dev_loss, 'retrieval', dev_accuracy, flush=True)
    result = dict(schema_version=1, method='128x128 linear projection of frozen DISK descriptors, normalized before frozen LightGlue',
        seed=seed, steps=steps, selected_step=best_step, selection='Minimum development symmetric contrastive loss; identity step zero eligible. No test outcomes used.',
        manifest_sha256=fingerprint(manifest), model_lock_sha256=sha256('model_lock.json'),
        training_source_sha256=sha256(__file__), descriptor_cache_sha256=sha256(output/'descriptor_samples.npz'),
        training_pairs=len(training), development_pairs=len(development), history=history,
        matrix=best.numpy().tolist(), singular_values=torch.linalg.svdvals(best).tolist(),
        warning='Descriptor adaptation only; no detector/backbone fine-tuning. Synthetic descriptor retrieval does not establish real registration improvement.')
    (output/'adapter.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='moon_manifest.json')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=300)
    args = p.parse_args()
    if args.steps <= 0:
        raise ValueError('Training steps must be positive')
    result = train(json.loads(Path(args.manifest).read_text()), args.output, args.steps)
    print('Selected checkpoint:', result['selected_step'])
