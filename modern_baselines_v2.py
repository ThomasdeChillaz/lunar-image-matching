"""Pinned author LoMa/RoMa-v2 inference; no fallback or replacement weights."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np

from dataset import sha256
from geometry import endpoint_error, evaluation_points
from matchers import Match, fit_geometry

SOURCES = {
    'loma': dict(url='https://github.com/davnords/loma', revision='415caedde23a9ed479172ed174fe4ff7b4514d4b',
                 license='MIT; matcher Apache-2.0'),
    'romav2': dict(url='https://github.com/Parskatt/RoMaV2', revision='95c9968145c8906b7b59383258e9f73b02853d89',
                   license='MIT except DINOv3 custom license; inspect upstream notices'),
}


def pin_source(directory, family):
    directory = Path(directory)
    revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCES[family]['revision']:
        raise ValueError('Author source revision differs from pinned baseline')
    if subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Modified upstream source is not the pinned author implementation')
    return dict(SOURCES[family], tree_sha256={str(p.relative_to(directory)): sha256(p)
                for p in sorted((directory/'src').rglob('*.py'))})


def validate_pixels(image, mask):
    if image.ndim != 2 or image.dtype != np.uint8 or mask.shape != image.shape:
        raise ValueError('Grayscale uint8 pixels and same-size validity mask required')
    return bool(mask.any() and np.std(image[mask]) >= 1)


def native_points(points, source_shape, internal_shape):
    scale = np.array(internal_shape[::-1])/np.array(source_shape[::-1])
    return (np.asarray(points, float)+.5)/scale-.5


class ModernMatcher:
    def __init__(self, method, source, profile='common', device='cpu'):
        if method not in ('loma_b', 'loma_r', 'romav2') or profile not in ('common', 'author'):
            raise ValueError('Explicit supported baseline and profile required')
        if device not in ('cpu', 'mps'):
            raise ValueError('This no-paid-compute experiment supports CPU or local MPS')
        self.family = 'romav2' if method == 'romav2' else 'loma'
        self.provenance = pin_source(source, self.family)
        sys.path.insert(0, str(Path(source)/'src'))
        import torch
        torch.set_num_threads(2)
        self.torch, self.device, self.method, self.profile = torch, device, method, profile
        with patch('torch.backends.mps.is_available', return_value=device == 'mps'):
            package = importlib.import_module(self.family)
            if method == 'romav2':
                self.model = package.RoMaV2()
                if profile == 'common':
                    self.model.H_lr = self.model.W_lr = 512
                    self.model.H_hr = self.model.W_hr = None
                    self.model.bidirectional = False
            else:
                cfg = package.LoMaR() if method == 'loma_r' else package.LoMaB()
                self.model = package.LoMa(cfg)
        self.model.eval()
        cache = Path(torch.hub.get_dir())/'checkpoints'
        self.provenance.update(method=method, profile=profile, device=device,
            torch=torch.__version__, opencv=cv2.__version__,
            weights={p.name: sha256(p) for p in sorted(cache.glob('*')) if p.is_file()},
            settings={k: getattr(self.model, k) for k in ('H_lr','W_lr','H_hr','W_hr','bidirectional')
                      if hasattr(self.model, k)})

    def sync(self):
        if self.device == 'mps':
            self.torch.mps.synchronize()

    def points(self, a, b, ma=None, mb=None):
        ma = np.ones(a.shape, bool) if ma is None else np.asarray(ma, bool)
        mb = np.ones(b.shape, bool) if mb is None else np.asarray(mb, bool)
        if not validate_pixels(a, ma) or not validate_pixels(b, mb):
            return np.empty((0, 2)), np.empty((0, 2))
        # No extra source information is introduced. LoMa needs dimensions
        # divisible by both8 and14;504 is the closest compatible size to512.
        aa, bb = [cv2.cvtColor(np.where(m, im, 0), cv2.COLOR_GRAY2RGB)
                  for im, m in ((a, ma), (b, mb))]
        internal_a, internal_b = a.shape, b.shape
        if self.profile == 'common':
            size=512 if self.method=='romav2' else 504
            internal_a = internal_b = (size,size)
            aa, bb = [cv2.resize(im,(size,size),interpolation=cv2.INTER_LINEAR) for im in (aa,bb)]
        with self.torch.inference_mode():
            if self.method == 'romav2':
                preds = self.model.match(aa, bb)
                matches, *_ = self.model.sample(preds, 2000)
                pa, pb = self.model.to_pixel_coordinates(matches, *internal_a, *internal_b)
                pa, pb = pa.cpu().numpy(), pb.cpu().numpy()
            else:
                if self.profile == 'author':
                    # Author path loader uses separate detector/descriptor resize
                    # settings; retain those rather than silently approximating it.
                    import tempfile
                    with tempfile.TemporaryDirectory() as folder:
                        paths=[str(Path(folder)/name) for name in ('a.png','b.png')]
                        for path, im in zip(paths,(aa,bb)):
                            cv2.imwrite(path,cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
                        pa,pb=self.model.match(*paths)
                else:
                    tensors=[self.torch.from_numpy(im.copy()).permute(2,0,1)[None].float()/255 for im in (aa,bb)]
                    pa, pb = self.model.match(*tensors)
                pa, pb = np.asarray(pa), np.asarray(pb)
        pa, pb = native_points(pa, a.shape, internal_a), native_points(pb, b.shape, internal_b)
        good = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
        for points, mask in ((pa, ma), (pb, mb)):
            good &= ((points >= 0) & (points <= np.array(mask.shape[::-1])-1)).all(1)
            index = np.rint(np.where(np.isfinite(points), points, 0)).astype(int).clip(0, np.array(mask.shape[::-1])-1)
            good &= mask[index[:, 1], index[:, 0]]
        return pa[good].astype(np.float32), pb[good].astype(np.float32)

    def __call__(self, a, b, ma=None, mb=None):
        pa, pb = self.points(a, b, ma, mb)
        return fit_geometry(pa, pb)


def probe(method, source, output, profile='common', device='cpu'):
    from synthetic import texture
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = perf_counter()
    try:
        matcher = ModernMatcher(method, source, profile, device)
        a = texture(20261030, 128)
        truth = np.array([[1.,0,8.],[0,1.,-4.]])
        b = cv2.warpAffine(a, truth, (128,128))
        matcher(a,b)
        matcher.sync()
        t = perf_counter()
        found = matcher(a,b)
        matcher.sync()
        elapsed = (perf_counter()-t)*1000
        error = endpoint_error(found.matrix, truth, evaluation_points(a.shape,truth))
        blank = matcher(np.zeros_like(a), np.zeros_like(a))
        invalid = matcher(a,b,np.zeros_like(a,bool),np.zeros_like(b,bool))
        result = dict(status='passed' if error is not None and error<=3 and blank.matrix is None and invalid.matrix is None else 'failed',
            software_control_only=True, method=method, endpoint_error_px=error, support=found.support,
            warm_inference_ms=elapsed, peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            provenance=matcher.provenance, source_sha256=sha256(__file__))
    except Exception as exc:
        result = dict(status='blocked',method=method,error_type=type(exc).__name__,error=str(exc),
            elapsed_seconds=perf_counter()-start,source_sha256=sha256(__file__),
            warning='No inferred lunar result; missing runtime/weights are never replaced.')
    (output/'probe.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['loma_b','loma_r','romav2'],required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--profile',choices=['common','author'],default='common')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=p.parse_args()
    probe(args.method,args.source,args.output,args.profile,args.device)
