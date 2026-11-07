"""Prepare true-3D development supervision, refusing calibration/test pixels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import native_points
from lunar_geometry_v2 import correspondences
from lunargeo_dataset import write_new


def allowed_records(manifest):
    rows = [t for t in manifest['tiles'] if t['split'] in ('train', 'model_selection')]
    train = {t['component'] for t in rows if t['split'] == 'train'}
    dev = {t['component'] for t in rows if t['split'] == 'model_selection'}
    if not train or not dev or train & dev:
        raise ValueError('Disjoint nonempty train/model-selection components required')
    return rows


def prepare(manifest_path, coarse_path, output, cache_root):
    import torch
    from compact_lunar import CompactMatcher
    torch.set_num_threads(2); cv2.setNumThreads(1)
    output, cache_root = Path(output), Path(cache_root)
    if output.exists() or cache_root.exists():
        raise FileExistsError('Use new immutable preparation paths')
    origin = json.loads(Path(manifest_path).read_text())
    records = allowed_records(origin)
    cache_root.mkdir(parents=True); archives = {}; rows = []; cases = []; selection = []
    try:
        coarse = CompactMatcher(coarse_path)
        for t in records:
            mode = t['mode']
            if mode not in archives:
                path = Path('data/lunargeo')/(mode+'.zip')
                if sha256(path) != json.loads(path.with_suffix('.provenance.json').read_text())['sha256']:
                    raise ValueError('Released archive changed')
                archives[mode] = zipfile.ZipFile(path)
            images = []
            for frame, expected in zip(t['frames'], t['source_hashes']):
                data = archives[mode].read(frame['image'])
                if hashlib.sha256(data).hexdigest() != expected['image']:
                    raise ValueError('Native training image differs from source provenance')
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None or list(image.shape) != t['source_size']:
                    raise ValueError('Invalid source image dimensions')
                images.append(image)
            if sha256(t['cache']) != t['cache_sha256']:
                raise ValueError('Original depth cache changed')
            with np.load(t['cache'], allow_pickle=False) as z:
                da, db = z['depth_a'].copy(), z['depth_b'].copy()
            arrays = dict(images=np.stack(images), depth_a=da, depth_b=db)
            a, b = t['cameras']; phase = int(hashlib.sha256(t['tile_id'].encode()).hexdigest()[:8], 16) % 3
            tile_cases = []
            for i in range(3):
                for j in range(3):
                    k = 3*i+j
                    ia, ib = 2*i, 2*j+1
                    pa, pb = coarse.points(*[cv2.resize(images[v], (128, 128), interpolation=cv2.INTER_AREA) for v in (ia, ib)])
                    p, q = native_points(pa, da.shape), native_points(pb, db.shape)
                    _, target, _, visible = correspondences(da, db, a['intrinsics'], a['cam2world'],
                                                           b['intrinsics'], b['cam2world'], points=p)
                    truth = np.zeros_like(p); truth[visible] = target
                    arrays.update({f'p{k}':p, f'q{k}':q, f'target{k}':truth, f'visible{k}':visible})
                    case = dict(pair_id=t['tile_id']+f'/lights-{i}-{j}', tile_id=t['tile_id'],
                                component=t['component'], split=t['split'], mode=mode,
                                light_a=i, light_b=j, key=k, proposed_points=len(p))
                    cases.append(case); tile_cases.append(case)
                    if t['split'] == 'model_selection' and i == phase and j in (phase, (phase+1)%3):
                        selection.append(case['pair_id'])
            path = cache_root/(hashlib.sha256(t['tile_id'].encode()).hexdigest()[:16]+'.npz')
            np.savez_compressed(path, **arrays)
            rows.append(dict(t, native_cache=str(path), native_cache_sha256=sha256(path)))
            print('prepared', t['split'], t['tile_id'], 'cases', len(tile_cases), flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    result = dict(schema_version=3, scope='User-authorized development-only training; no fresh/final validation.',
        prepared_utc=datetime.now(timezone.utc).isoformat(), origin_manifest=str(manifest_path),
        origin_manifest_sha256=sha256(manifest_path), coarse_model=str(coarse_path),
        coarse_checkpoint_sha256=sha256(Path(coarse_path)/'selected.pt'),
        preparation_source_sha256=sha256(__file__), tiles=rows, cases=cases,
        selection_bank=selection, train_components=8, selection_components=4,
        calibration_or_test_pixels_loaded=0, trajectory_independence_verified=False,
        geometric_labels='Reference ray-depth unprojection, exact camera projection and query-depth visibility. No brightness-based occlusion labels.',
        inference_inputs='Image pixels/coarse image matches only. Cameras/depth are labels or pose-map/scoring inputs, never refiner inputs.')
    write_new(output, result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='reports/compact_dataset_v2.json')
    p.add_argument('--coarse', default='runs/compact-v2-expanded-20261030')
    p.add_argument('--output', default='reports/localization_training_data_v3.json')
    p.add_argument('--cache', default='data/localization-v3-development')
    a = p.parse_args(); prepare(a.manifest, a.coarse, a.output, a.cache)
