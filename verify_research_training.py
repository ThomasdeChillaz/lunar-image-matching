"""Audit expanded adapter provenance, split access and selected surrogate loss."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np

from dataset import sha256
from landmark_review import fingerprint
from research_dataset import validate


def verify(run, source_ref):
    import torch
    torch.set_num_threads(1)
    run = Path(run)
    complete = json.loads((run/'COMPLETE.json').read_text())
    for name, field in [('adapter.json', 'adapter_sha256'), ('frozen_config.json', 'config_sha256'),
                        ('frozen_manifest.json', 'manifest_sha256'), ('training_pairs.json', 'training_pairs_sha256'),
                        ('descriptor_grid.npz', 'descriptor_cache_sha256')]:
        if sha256(run/name) != complete[field]:
            raise ValueError('Training artifact hash differs')
    config = json.loads((run/'frozen_config.json').read_text())
    adapter = json.loads((run/'adapter.json').read_text())
    manifest = validate(json.loads((run/'frozen_manifest.json').read_text()))
    for name, digest in config['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f'{source_ref}:{name}'])
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Historical training source differs')
    if config['manifest_sha256'] != fingerprint(manifest) or adapter['manifest_sha256'] != fingerprint(manifest):
        raise ValueError('Training manifest differs')
    selected = [t for t in manifest['tiles'] if t['split'] in ('train', 'dev')]
    ids = [t['tile_id'] for t in selected]
    if config['selected_tiles'] != ids or config['included_splits'] != ['train', 'dev'] or adapter['reserve_images_loaded'] != 0:
        raise ValueError('Reserve terrain in training data access plan')
    logged = json.loads((run/'training_pairs.json').read_text())
    if logged['tile_ids'] != ids:
        raise ValueError('Descriptor tile identities differ')
    with np.load(run/'descriptor_grid.npz', allow_pickle=False) as cache:
        descriptors, visible = cache['descriptors'], cache['visible']
    if descriptors.shape != (len(selected), 9, 196, 128) or visible.shape != (len(selected), 9, 196):
        raise ValueError('Descriptor cache has unexpected shape')
    if not np.isfinite(descriptors).all():
        raise ValueError('Nonfinite descriptor')
    training, development = [], []
    for ti, tile in enumerate(selected):
        eligible = [(ti, i, j) for i, j in itertools.combinations(range(9), 2) if (visible[ti, i] & visible[ti, j]).sum() >= 32]
        if tile['split'] == 'train':
            training.extend(eligible)
        else:
            eligible.sort(key=lambda p: hashlib.sha256(f'{config["seed"]}:{tile["tile_id"]}:{p[1]}:{p[2]}'.encode()).hexdigest())
            development.extend(eligible[:4])
    if [list(p) for p in training] != logged['training'] or [list(p) for p in development] != logged['development']:
        raise ValueError('All-light pair selection differs')
    if len(training) != adapter['training_pairs'] or len(development) != adapter['development_pairs']:
        raise ValueError('Training/development count differs')
    matrix = torch.tensor(adapter['matrix'], dtype=torch.float32)
    singular = torch.linalg.svdvals(matrix)
    if matrix.shape != (128, 128) or not torch.isfinite(matrix).all() or float(singular.min()) < .75-1e-5 or float(singular.max()) > 1.25+1e-5:
        raise ValueError('Adapter exceeds frozen singular-value bounds')
    losses = []
    with torch.inference_mode():
        for ti, i, j in development:
            keep = visible[ti, i] & visible[ti, j]
            a = torch.from_numpy(descriptors[ti, i, keep].astype(np.float32))
            b = torch.from_numpy(descriptors[ti, j, keep].astype(np.float32))
            aa = torch.nn.functional.normalize(a@matrix.T, dim=-1)
            bb = torch.nn.functional.normalize(b@matrix.T, dim=-1)
            logits = aa@bb.T/config['temperature']
            labels = torch.arange(len(a))
            losses.append(float(((torch.nn.functional.cross_entropy(logits, labels)+torch.nn.functional.cross_entropy(logits.T, labels))/2).item()))
    selected_history = min(adapter['history'], key=lambda h: h['development_descriptor_loss'])
    measured = float(np.mean(losses))
    if selected_history['step'] != adapter['selected_step'] or not np.isclose(measured, selected_history['development_descriptor_loss'], atol=1e-6):
        raise ValueError('Selected checkpoint/surrogate loss differs')
    return dict(training_tiles=sum(t['split'] == 'train' for t in selected), development_tiles=sum(t['split'] == 'dev' for t in selected),
        training_pairs=len(training), development_pairs=len(development), selected_step=adapter['selected_step'],
        recomputed_development_descriptor_loss=measured, reserve_images_loaded=0,
        adapter_sha256=complete['adapter_sha256'], source_ref=source_ref, completion_sha256=sha256(run/'COMPLETE.json'),
        warning='Audits descriptor training only; no fresh registration gain or real-image improvement established.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--source-ref', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = verify(args.run, args.source_ref)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
