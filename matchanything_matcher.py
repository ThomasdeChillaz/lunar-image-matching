"""Pinned author MatchAnything ELoFTR core, with explicit native-grid variants.

Uses the author's released architecture and configuration, not a weight swap
into the later EfficientLoFTR code. Shared input conversion and geometry fitting
come from our audited EfficientLoFTR wrapper. This is a benchmark integration,
not a reproduction of the paper's unrelated task-specific evaluation protocols.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
import urllib.request
import zipfile

from dataset import sha256
from eloftr_matcher import EfficientLoFTRMatcher, install_dependency_path


LOCK = Path(__file__).with_name('matchanything_model_lock.json')


def verify_source(root, lock):
    for name, expected in lock['source_files'].items():
        path = root/name
        if not path.is_file() or sha256(path) != expected['sha256']:
            raise RuntimeError(f'Missing/changed pinned MatchAnything source: {name}')


def prepare(root='.', dependency_dir=None):
    install_dependency_path(dependency_dir)
    root, lock = Path(root), json.loads(LOCK.read_text())
    source = root/lock['source_directory']
    def acquire(item):
        name, expected = item
        target = source/name
        if target.exists():
            if sha256(target) != expected['sha256']:
                raise ValueError(f'Existing MatchAnything source differs: {name}')
            return
        url = f"https://huggingface.co/spaces/{lock['source_repository']}/resolve/{lock['source_revision']}/{lock['source_prefix']}{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        blob = b'blob '+str(len(body)).encode()+b'\0'+body
        if hashlib.sha256(body).hexdigest() != expected['sha256'] or hashlib.sha1(blob).hexdigest() != expected['git_blob_sha1']:
            raise ValueError(f'Author source checksum differs: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, lock['source_files'].items()))
    verify_source(source, lock)
    safe, original = root/lock['safe_weight_path'], root/lock['original_weight_path']
    if safe.exists():
        if sha256(safe) != lock['safe_weight_sha256']:
            raise ValueError('Existing MatchAnything safe weights differ')
        return safe
    if not original.exists():
        archive = root/lock['weight_archive_path']
        if not archive.exists():
            import gdown
            if not gdown.download(id=lock['weight_archive_drive_id'], output=str(archive), quiet=False, use_cookies=False):
                raise RuntimeError('Author MatchAnything weight archive unavailable')
        if sha256(archive) != lock['weight_archive_sha256']:
            raise ValueError('MatchAnything archive checksum differs')
        with zipfile.ZipFile(archive) as bundle:
            original.write_bytes(bundle.read('weights/matchanything_eloftr.ckpt'))
    if sha256(original) != lock['original_weight_sha256']:
        raise ValueError('Original MatchAnything checkpoint differs')
    import torch
    from safetensors.torch import save_file
    checkpoint = torch.load(original, map_location='cpu', weights_only=True)
    tensors = {k: v.contiguous().clone() for k, v in checkpoint['state_dict'].items()}
    temporary = safe.with_suffix('.part')
    save_file(tensors, str(temporary))
    if sha256(temporary) != lock['safe_weight_sha256']:
        raise ValueError('MatchAnything tensor extraction differs from lock')
    temporary.replace(safe)
    return safe


def lower_config(value):
    return {k.lower(): lower_config(v) for k, v in value.items()} if hasattr(value, 'items') else value


def author_config(source, namespace, internal_size):
    defaults = importlib.import_module(namespace+'.config.default')
    config = defaults.get_cfg_defaults()
    path = source/'configs/models/eloftr_model.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    first = tree.body[0]
    if not (isinstance(first, ast.ImportFrom) and first.module == 'src.config.default'
            and len(first.names) == 1 and first.names[0].name == '_CN' and first.names[0].asname == 'cfg'):
        raise RuntimeError('Unexpected author configuration import')
    # Execute the exact author assignments against its cloned defaults. Removing
    # only the absolute import avoids changing another model's global src package.
    tree.body = tree.body[1:]
    exec(compile(tree, str(path), 'exec'), {'cfg': config})
    config.METHOD = 'matchanything_eloftr'
    config.LOFTR.COARSE.NPE = [832, 832, internal_size, internal_size]
    return config


class MatchAnythingMatcher(EfficientLoFTRMatcher):
    def __init__(self, root='.', dependency_dir=None, device='cpu', factor=2):
        install_dependency_path(dependency_dir)
        lock = json.loads(LOCK.read_text())
        for package, version in lock['optional_dependencies'].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f'MatchAnything dependency differs: {package}')
        import torch
        import kornia
        if torch.__version__.split('+')[0] != lock['torch_version'] or kornia.__version__ != lock['kornia_version']:
            raise RuntimeError('Core MatchAnything runtime differs')
        if factor not in (1, 2, 4) or device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Declared variants support factors 1/2/4 and cpu/mps/cuda')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        root, source = Path(root), Path(root)/lock['source_directory']
        verify_source(source, lock)
        weights = root/lock['safe_weight_path']
        if not weights.is_file() or sha256(weights) != lock['safe_weight_sha256']:
            raise RuntimeError('Missing/corrupt MatchAnything weights; run --prepare')
        namespace = 'lunar_author_matchanything'
        if namespace not in sys.modules:
            package = (source/'src').resolve()
            spec = importlib.util.spec_from_file_location(namespace, package/'__init__.py', submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
        implementation = importlib.import_module(namespace+'.loftr')
        config = author_config(source, namespace, 128*factor)
        from safetensors.torch import load_file
        torch.set_num_threads(1)
        self.torch, self.device, self.factor = torch, device, factor
        started = perf_counter()
        self.model = implementation.LoFTR(lower_config(config)['loftr'])
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        # The released PL wrapper does not reparameterize matchanything_eloftr;
        # its condition explicitly requires METHOD == 'loftr'. Keep that behavior.
        self.model.eval().to(device)
        self.sync()
        self.initialization_ms = (perf_counter()-started)*1000

    def __call__(self, reference, query, reference_mask=None, query_mask=None):
        if reference.shape != (128, 128):
            raise ValueError('Pinned NPE profile is explicitly for native 128px crops')
        return super().__call__(reference, query, reference_mask, query_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.add_argument('--dependency-dir')
    args = parser.parse_args()
    print(prepare(dependency_dir=args.dependency_dir))
