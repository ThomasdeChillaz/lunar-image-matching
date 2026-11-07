"""Run reproducible controlled perturbations. No real-repeat validation implied."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from geometry import endpoint_error, evaluation_points
from synthetic import texture, make_pair
from matchers import ncc, sift, preprocess


def summarize(rows):
    result = []
    for method in sorted({r['method'] for r in rows}):
        for case in sorted({r['case'] for r in rows}):
            subset = [r for r in rows if r['method'] == method and r['case'] == case]
            errors = [r['error_px'] for r in subset if r['error_px'] is not None]
            result.append(dict(method=method, case=case, pairs=len(subset),
                               success_rate=sum(r['success'] for r in subset)/len(subset),
                               failed_registrations=sum(r['error_px'] is None for r in subset),
                               median_error_registered_px=float(np.median(errors)) if errors else None,
                               median_runtime_ms=float(np.median([r['runtime_ms'] for r in subset]))))
    return result


def run(config, output, image_path=None):
    if config['seeds'] < 1 or config['size'] < 96 or config['success_threshold_px'] <= 0 or not config['cases']:
        raise ValueError('Invalid seed count, image size, threshold or empty cases')
    if len({c['name'] for c in config['cases']}) != len(config['cases']):
        raise ValueError('Case names must be unique')
    for c in config['cases']:
        if not all(np.isfinite(c[k]) for k in ['dx', 'dy', 'angle', 'gain', 'gamma', 'noise']) or c['gamma'] <= 0 or c['gain'] <= 0 or c['noise'] < 0:
            raise ValueError('Invalid perturbation parameters')
    source = None
    if image_path:
        source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if source is None or min(source.shape) < 96:
            raise ValueError('Image must be readable grayscale-compatible data, at least 96x96')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rows, gallery = [], []
    methods = {'ncc': ncc, 'sift': sift}
    for i in range(config['seeds']):
        seed = config['seed']+i
        reference = texture(seed, config['size']) if source is None else source
        for j, case in enumerate(config['cases']):
            query, truth = make_pair(reference, case, seed*1000+j)
            points = evaluation_points(reference.shape, truth)
            for name, matcher in methods.items():
                for use_clahe in [False, True]:
                    cv2.setRNGSeed(seed)
                    start = perf_counter()
                    result = matcher(preprocess(reference, use_clahe), preprocess(query, use_clahe))
                    elapsed = (perf_counter()-start)*1000
                    error = endpoint_error(result.matrix, truth, points)
                    method = name + ('+clahe' if use_clahe else '')
                    rows.append(dict(seed=seed, case=case['name'], method=method,
                                     dataset_kind='procedural' if source is None else 'real_image_synthetic_transform',
                                     applicability='out_of_model_rotation' if name=='ncc' and case['angle'] else 'in_model',
                                     error_px=error, success=error is not None and error <= config['success_threshold_px'],
                                     reason=result.reason or ('no_common_support' if not len(points) else ''),
                                     runtime_ms=elapsed, support=result.support,
                                     transform=None if result.matrix is None else result.matrix.tolist()))
                    if i == 0 and name == 'sift' and not use_clahe:
                        gallery.append((case['name'], reference, query, result.matrix, error))
    with (output/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output/'config.json').write_text(json.dumps(config, indent=2))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source_hashes = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ['benchmark.py', 'geometry.py', 'synthetic.py', 'matchers.py', 'reporting.py']}
    metadata = dict(source_sha256=source_hashes, python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                    opencv=cv2.__version__, git_revision=revision, opencv_threads=1,
                    image_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest() if image_path else None,
                    timing='CPU wall time including preprocessing and detector initialization; no accelerator',
                    limitation='Controlled perturbations only; no physical shadow model or real-repeat validation.')
    (output/'environment.json').write_text(json.dumps(metadata, indent=2))
    from reporting import report
    report(summary, gallery, output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('smoke.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    summary = run(json.loads(args.config.read_text()), args.output, args.image)
    print(f'Wrote {len(summary)} method/case summaries to {args.output}; controlled tests only.')
