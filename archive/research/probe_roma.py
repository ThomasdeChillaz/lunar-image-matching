"""Actual pretrained RoMa control in the isolated Linux CPU environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from geometry import endpoint_error, evaluation_points
from roma_matcher import RoMaMatcher
from synthetic import texture, make_pair


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = ['probe_roma.py', 'roma_matcher.py', 'roma_model_lock.json', 'requirements-roma-linux.txt',
        'synthetic.py', 'geometry.py', 'matchers.py', 'loftr_matcher.py', 'dataset.py']
    contract = dict(source_sha256={p: sha256(p) for p in sources}, seed=20261029,
        model_lock=json.loads(Path('roma_model_lock.json').read_text()),
        started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        execution=dict(github_run_id=os.environ.get('GITHUB_RUN_ID'), github_sha=os.environ.get('GITHUB_SHA')),
        runtime={name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow', 'einops', 'loguru']},
        status='Procedural software control only, not lunar illumination or fresh geographic evaluation.')
    (output/'frozen_inputs.json').write_text(json.dumps(contract, indent=2))
    image = texture(42, 128)
    query, truth = make_pair(image, dict(angle=12, dx=6, dy=-4, gain=.7, gamma=1.2, noise=.005), 43)
    ma = np.ones_like(image, bool)
    mask_float = cv2.warpAffine(ma.astype(np.float32), truth, (128, 128), flags=cv2.INTER_LINEAR)
    mb = mask_float >= 1.-1e-6
    model = RoMaMatcher()
    cv2.setNumThreads(1)
    print('Author RoMa initialized; warming the complete declared CPU profile.', flush=True)
    model(image, image, ma, ma)
    started = perf_counter()
    result = model(image, query, ma, mb)
    elapsed = (perf_counter()-started)*1000
    error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
    blank = model(np.zeros_like(image), np.zeros_like(image))
    invalid = model(image, image, np.zeros_like(ma), ma)
    report = dict(input_array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=image, query=query, reference_mask=ma, query_mask=mb).items()},
        transform=None if result.matrix is None else result.matrix.tolist(), truth=truth.tolist(),
        error_px=error, returned=result.matrix is not None, support=result.support, reason=result.reason,
        forward_ms=elapsed, initialization_ms=model.initialization_ms,
        blank_reason=blank.reason, invalid_reason=invalid.reason,
        passed=bool(error is not None and error <= 3 and blank.reason=='low_texture' and invalid.reason=='no_valid_support'),
        warning='A passing known-transform forward is software evidence, not a lunar benchmark result or SOTA claim.')
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('Actual pretrained dense matcher failed the declared control')
    (output/'COMPLETE.json').write_text(json.dumps(dict(result_sha256=sha256(output/'result.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.output)
