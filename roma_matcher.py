"""Pinned RoMa outdoor dense baseline in a separate Linux CPU runtime.

The author forward/default resolutions are retained. CPU arithmetic explicitly
uses float32 DINO features and the author's KDE half=False option for
consistent CPU density arithmetic. This is a declared CPU profile,
not a reproduction of a GPU paper timing or another benchmark's pose score.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import urllib.request

import cv2
import numpy as np

from dataset import sha256
from loftr_matcher import valid_correspondences
from matchers import Match, fit_geometry


LOCK = Path(__file__).with_name('roma_model_lock.json')


def normalized_to_native_centers(points, shape):
    """Author normalized grids represent edge coordinates; subtract half a pixel."""
    h, w = shape
    points = np.asarray(points, np.float64)
    if points.shape[-1] != 2:
        raise ValueError('Expected normalized x,y coordinates')
    return (points+1)*np.array([w, h])/2-.5


def verify_source(root, lock):
    if not root.is_dir():
        raise RuntimeError('Missing RoMa source; run roma_matcher.py --prepare')
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['source_revision']:
        raise ValueError('RoMa source revision differs')
    for name, expected in lock['source_sha256'].items():
        if not (root/name).is_file() or sha256(root/name) != expected:
            raise ValueError(f'RoMa author source differs: {name}')


def prepare(root='.'):
    root, lock = Path(root), json.loads(LOCK.read_text())
    source = root/lock['source_directory']
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', lock['source_url'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', lock['source_revision']], check=True)
    verify_source(source, lock)
    for weight in lock['weights'].values():
        path = root/weight['path']
        if path.exists():
            if sha256(path) != weight['sha256']:
                raise ValueError('Existing RoMa/DINOv2 weights differ; no automatic replacement')
            continue
        temporary = path.with_suffix('.download')
        if temporary.exists():
            raise FileExistsError(temporary)
        with urllib.request.urlopen(weight['url'], timeout=120) as response, temporary.open('wb') as out:
            while block := response.read(1024*1024):
                out.write(block)
        if temporary.stat().st_size != weight['bytes'] or sha256(temporary) != weight['sha256']:
            raise ValueError('Downloaded RoMa/DINOv2 weights differ')
        temporary.replace(path)
    return source


@contextmanager
def cpu_density_precision():
    import romatch.models.matcher as implementation
    original = implementation.kde
    implementation.kde = partial(original, half=False)
    try:
        yield
    finally:
        implementation.kde = original


class RoMaMatcher:
    def __init__(self, root='.'):
        lock, root = json.loads(LOCK.read_text()), Path(root)
        for package, version in lock['runtime_versions'].items():
            if importlib.metadata.version(package).split('+')[0] != version:
                raise RuntimeError(f'RoMa Linux runtime differs: {package}; keep core environment unchanged')
        import torch
        self.torch = torch
        torch.set_num_threads(min(os.cpu_count() or 1, 4))
        torch.set_float32_matmul_precision('highest')
        source = root/lock['source_directory']
        verify_source(source, lock)
        if 'romatch' in sys.modules and Path(sys.modules['romatch'].__file__).resolve().parent != (source/'romatch').resolve():
            raise RuntimeError('Another romatch source is already imported')
        sys.path.insert(0, str(source.resolve()))
        weights = {}
        for name, weight in lock['weights'].items():
            path = root/weight['path']
            if not path.exists() or sha256(path) != weight['sha256']:
                raise RuntimeError('Missing/corrupt RoMa weights; preparation must be explicit')
            weights[name] = torch.load(path, map_location='cpu', weights_only=True)
        from romatch import roma_outdoor
        started = perf_counter()
        self.model = roma_outdoor(device='cpu', weights=weights['roma'], dinov2_weights=weights['dinov2'],
            amp_dtype=torch.float32, use_custom_corr=False, do_compile=False).eval()
        self.initialization_ms = (perf_counter()-started)*1000
        self.sample_count = lock['sampling']['count']

    def __call__(self, reference, query, reference_mask=None, query_mask=None):
        if reference.ndim != 2 or query.shape != reference.shape:
            raise ValueError('RoMa benchmark needs equal grayscale native images')
        masks = [np.ones_like(reference, bool) if m is None else np.asarray(m, bool) for m in [reference_mask, query_mask]]
        if any(m.shape != reference.shape for m in masks):
            raise ValueError('Native validity shape differs')
        if any(not m.any() for m in masks):
            return Match(reason='no_valid_support')
        if min(reference[masks[0]].std(), query[masks[1]].std()) < 1:
            return Match(reason='low_texture')
        from PIL import Image
        inputs = [Image.fromarray(np.repeat(np.where(m, a, 0)[..., None], 3, axis=2).astype(np.uint8))
                  for a, m in zip([reference, query], masks)]
        self.torch.manual_seed(20261029)
        with self.torch.inference_mode(), cpu_density_precision():
            warp, certainty = self.model.match(*inputs, device='cpu')
            points, confidence = self.model.sample(warp, certainty, num=self.sample_count)
        points = points.float().cpu().numpy()
        a = normalized_to_native_centers(points[:, :2], reference.shape)
        b = normalized_to_native_centers(points[:, 2:], query.shape)
        a, b = valid_correspondences(a, b, *masks)
        cv2.setRNGSeed(20261029)
        return fit_geometry(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    args = parser.parse_args()
    print(prepare())
