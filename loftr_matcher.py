"""Pinned, offline LoFTR outdoor inference on the benchmark's native pixels."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import urllib.request

import numpy as np

from dataset import sha256
from matchers import Match, fit_geometry


LOCK = Path(__file__).with_name('loftr_model_lock.json')


def implementation_hashes(module):
    directory = Path(inspect.getfile(module)).parent
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*.py'))}


def fetch_weights(root='.'):
    lock = json.loads(LOCK.read_text())
    path = Path(root)/lock['weight_path']
    if path.exists():
        if sha256(path) != lock['weight_sha256']:
            raise ValueError('Existing LoFTR weights are corrupt; no automatic replacement')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with urllib.request.urlopen(lock['weight_url'], timeout=120) as source, temporary.open('wb') as out:
            while block := source.read(1024*1024):
                out.write(block)
        if sha256(temporary) != lock['weight_sha256']:
            raise ValueError('LoFTR weight checksum mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def valid_correspondences(a, b, reference_mask, query_mask):
    """Fine refinement can move a coarse correspondence outside valid support."""
    a, b = np.asarray(a), np.asarray(b)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    for points, mask in [(a, reference_mask), (b, query_mask)]:
        h, w = mask.shape
        inside = (points[:, 0] >= 0) & (points[:, 0] <= w-1) & (points[:, 1] >= 0) & (points[:, 1] <= h-1)
        supported = np.zeros(len(points), bool)
        eligible = np.flatnonzero(inside & np.isfinite(points).all(axis=1))
        xy = np.rint(points[eligible]).astype(int)
        supported[eligible] = mask[xy[:, 1], xy[:, 0]]
        keep &= supported
    return a[keep], b[keep]


class LoFTRMatcher:
    def __init__(self, root='.', device='cpu'):
        import kornia
        import torch
        import kornia.feature.loftr.loftr as implementation
        lock = json.loads(LOCK.read_text())
        if kornia.__version__ != lock['kornia_version']:
            raise RuntimeError('LoFTR requires the pinned Kornia version in requirements-lock.txt')
        files = implementation_hashes(implementation)
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if files != lock['implementation_files_sha256'] or digest != lock['implementation_tree_sha256']:
            raise RuntimeError('LoFTR implementation differs from its source lock')
        path = Path(root)/lock['weight_path']
        if not path.exists() or sha256(path) != lock['weight_sha256']:
            raise RuntimeError('Missing/corrupt LoFTR weights. Run python loftr_matcher.py --fetch')
        if device not in ['cpu', 'mps', 'cuda']:
            raise ValueError('Unsupported device')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.torch, self.device = torch, device
        torch.set_num_threads(1)
        started = perf_counter()
        self.model = implementation.LoFTR(pretrained=None)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval().to(device)
        self.sync()
        self.initialization_ms = (perf_counter()-started)*1000

    def sync(self):
        if self.device == 'mps':
            self.torch.mps.synchronize()
        elif self.device == 'cuda':
            self.torch.cuda.synchronize()

    def __call__(self, reference, query, reference_mask=None, query_mask=None):
        if reference.ndim != 2 or query.shape != reference.shape or any(n % 8 for n in reference.shape):
            raise ValueError('LoFTR native grayscale inputs must have equal dimensions divisible by eight')
        masks = [np.ones_like(im, dtype=bool) if mask is None else np.asarray(mask, bool)
                 for im, mask in [(reference, reference_mask), (query, query_mask)]]
        if any(m.shape != reference.shape for m in masks):
            raise ValueError('LoFTR image/mask shape mismatch')
        if any(not m.any() for m in masks):
            return Match(reason='no_valid_support')
        if reference[masks[0]].std() < 1 or query[masks[1]].std() < 1:
            return Match(reason='low_texture')
        batch = {}
        for i, (image, mask) in enumerate(zip([reference, query], masks)):
            image = np.where(mask, image, 0).astype(np.float32)/255
            batch[f'image{i}'] = self.torch.from_numpy(np.ascontiguousarray(image))[None, None].to(self.device)
            batch[f'mask{i}'] = self.torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None].to(self.device)
        with self.torch.inference_mode():
            output = self.model(batch)
            a = output['keypoints0'].cpu().numpy()
            b = output['keypoints1'].cpu().numpy()
        a, b = valid_correspondences(a, b, *masks)
        return fit_geometry(a, b)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch', action='store_true', required=True)
    args = p.parse_args()
    print(fetch_weights())
