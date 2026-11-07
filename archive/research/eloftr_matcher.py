"""Pinned original EfficientLoFTR with native-grid geometry and offline weights."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from loftr_matcher import valid_correspondences
from matchers import Match, fit_geometry


LOCK = Path(__file__).with_name('eloftr_model_lock.json')


def install_dependency_path(path):
    if path is not None:
        directory = Path(path).resolve()
        if not directory.is_dir():
            raise RuntimeError('Optional EfficientLoFTR dependency directory is absent')
        sys.path.insert(0, str(directory))


def verify_source(root, lock):
    if not root.exists():
        raise RuntimeError('Missing EfficientLoFTR source. Run python eloftr_matcher.py --prepare')
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    files = {str(p.relative_to(root)): sha256(p) for p in sorted((root/'src').rglob('*.py'))}
    files['LICENSE'] = sha256(root/'LICENSE')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if revision != lock['source_revision'] or files != lock['source_files_sha256'] or digest != lock['source_tree_sha256']:
        raise RuntimeError('EfficientLoFTR source/revision differs from lock')


def prepare(root='.', dependency_dir=None):
    install_dependency_path(dependency_dir)
    root = Path(root)
    lock = json.loads(LOCK.read_text())
    upstream = root/lock['upstream_directory']
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', lock['source_url'], str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', lock['source_revision']], check=True)
    verify_source(upstream, lock)
    original, safe = root/lock['original_weight_path'], root/lock['safe_weight_path']
    if safe.exists():
        if sha256(safe) != lock['safe_weight_sha256']:
            raise ValueError('Corrupt safe weights; no automatic replacement')
        return safe
    if not original.exists():
        import gdown
        if not gdown.download(id=lock['original_weight_google_drive_id'], output=str(original), quiet=False, use_cookies=False):
            raise RuntimeError('Original author weights unavailable')
    if sha256(original) != lock['original_weight_sha256']:
        raise ValueError('Original checkpoint checksum differs')
    import torch
    from safetensors.torch import save_file
    if torch.__version__.split('+')[0] != lock['torch_version']:
        raise RuntimeError('Restricted metadata extraction requires pinned PyTorch')
    # PyTorch 2.2's restricted loader lacks public safe_globals. The author file
    # contains a callback class as a metadata dictionary key, never as executable
    # training code. Permit an inert tag and Counter; all other globals remain
    # restricted. Restore the loader immediately after tensor extraction.
    allowed = torch._weights_only_unpickler._get_allowed_globals()
    previous = allowed.copy()
    class TrainingCallbackTag:
        pass
    allowed.update({'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint': TrainingCallbackTag,
                    'collections.Counter': Counter})
    try:
        checkpoint = torch.load(original, map_location='cpu', weights_only=True)
    finally:
        allowed.clear()
        allowed.update(previous)
    tensors = {key: value.contiguous().clone() for key, value in checkpoint['state_dict'].items()}
    temporary = safe.with_suffix('.part')
    save_file(tensors, str(temporary))
    if sha256(temporary) != lock['safe_weight_sha256']:
        raise ValueError('Safe tensor conversion differs from pinned artifact')
    temporary.replace(safe)
    return safe


def native_pixel_centers(points, factor):
    return (np.asarray(points)+.5)/factor-.5


class EfficientLoFTRMatcher:
    def __init__(self, root='.', dependency_dir=None, device='cpu', factor=1):
        install_dependency_path(dependency_dir)
        lock = json.loads(LOCK.read_text())
        for package, version in lock['optional_dependencies'].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f'Optional dependency differs: {package}')
        import torch
        import kornia
        if torch.__version__.split('+')[0] != lock['torch_version'] or kornia.__version__ != lock['kornia_version']:
            raise RuntimeError('Core runtime differs from EfficientLoFTR lock')
        if factor not in (1, 2):
            raise ValueError('Only separately declared native/2x variants are supported')
        if device not in ['cpu', 'mps', 'cuda']:
            raise ValueError('Unsupported device')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        root = Path(root)
        upstream = root/lock['upstream_directory']
        verify_source(upstream, lock)
        weights = root/lock['safe_weight_path']
        if not weights.exists() or sha256(weights) != lock['safe_weight_sha256']:
            raise RuntimeError('Missing/corrupt safe weights; run eloftr_matcher.py --prepare')
        namespace = 'lunar_original_eloftr'
        if namespace not in sys.modules:
            source = (upstream/'src').resolve()
            spec = importlib.util.spec_from_file_location(namespace, source/'__init__.py', submodule_search_locations=[str(source)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
        implementation = __import__(namespace+'.loftr', fromlist=['LoFTR'])
        from safetensors.torch import load_file
        torch.set_num_threads(1)
        self.torch, self.device, self.factor = torch, device, factor
        started = perf_counter()
        self.model = implementation.LoFTR(config=deepcopy(implementation.full_default_cfg))
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        self.model = implementation.reparameter(self.model).eval().to(device)
        self.sync()
        self.initialization_ms = (perf_counter()-started)*1000

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        elif self.device == 'mps':
            self.torch.mps.synchronize()

    def __call__(self, reference, query, reference_mask=None, query_mask=None):
        if reference.ndim != 2 or query.shape != reference.shape or any(n % 32 for n in reference.shape):
            raise ValueError('Original EfficientLoFTR needs equal grayscale dimensions divisible by 32')
        masks = [np.ones_like(reference, bool) if m is None else np.asarray(m, bool) for m in [reference_mask, query_mask]]
        if any(m.shape != reference.shape for m in masks):
            raise ValueError('Mask shape differs')
        if any(not m.any() for m in masks):
            return Match(reason='no_valid_support')
        if min(reference[masks[0]].std(), query[masks[1]].std()) < 1:
            return Match(reason='low_texture')
        batch = {}
        for i, (image, mask) in enumerate(zip([reference, query], masks)):
            values = np.where(mask, image, 0)
            if self.factor != 1:
                values = cv2.resize(values, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LINEAR)
            batch[f'image{i}'] = self.torch.from_numpy(np.ascontiguousarray(values)).float()[None, None].to(self.device)/255
        with self.torch.inference_mode():
            self.model(batch)
            a = native_pixel_centers(batch['mkpts0_f'].cpu().numpy(), self.factor)
            b = native_pixel_centers(batch['mkpts1_f'].cpu().numpy(), self.factor)
        a, b = valid_correspondences(a, b, *masks)
        return fit_geometry(a, b)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true', required=True)
    p.add_argument('--dependency-dir')
    args = p.parse_args()
    print(prepare(dependency_dir=args.dependency_dir))
