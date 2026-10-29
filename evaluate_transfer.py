"""Measure the frozen lunar adapter on real images without inventing truth.

Model selection ended on physical development data. Real map disagreement and
transported-policy acceptance remain diagnostics, not correctness labels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from confidence import accepts, diagnostics
from dataset import common_points, load_pair as load_cnsfm, validate_manifest, sha256
from evaluate import endpoint_statistics
from evaluate_extended import CachedLearnedMatcher, illumination_metrics
from evaluate_moon import AdaptedMatcher
from landmark_review import fingerprint
from matchers import preprocess, sift
from tycho_dataset import load_pair as load_tycho, validate as validate_tycho


def run(manifest, physical, adapter, policies, output, dataset):
    if adapter['manifest_sha256'] != fingerprint(physical) or adapter['model_lock_sha256'] != sha256('model_lock.json'):
        raise ValueError('Adapter provenance mismatch')
    assets = validate_tycho(manifest, physical, '.') if dataset == 'tycho' else validate_manifest(manifest, '.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    methods = ['sift', 'disk_lightglue', 'disk_lightglue_clahe', 'disk_lightglue_lunar'] if dataset == 'tycho' else ['disk_lightglue_lunar']
    sources = ['evaluate_transfer.py', 'tycho_dataset.py', 'evaluate_moon.py', 'evaluate_extended.py',
        'moon_dataset.py', 'learned.py', 'matchers.py', 'confidence.py', 'dataset.py', 'geometry.py', 'evaluate.py']
    frozen = dict(manifest=manifest, physical_manifest_sha256=fingerprint(physical), adapter=adapter,
        physical_development_policies=policies, dataset=dataset, methods=methods, seed=20261026,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in sources},
        labels='Unknown; real map disagreement is not true accuracy. No real outcomes used for adapter selection or threshold calibration.',
        physical_policy_transport='Only on 128px Tycho common grids, explicitly out-of-domain and without correctness claims. No cycle-pixel rescaling or transfer to 512px grids.')
    (output/'frozen_inputs.json').write_text(json.dumps(frozen, indent=2, allow_nan=False))
    cv2.setNumThreads(1)
    cached = CachedLearnedMatcher()
    adapted = AdaptedMatcher(cached, adapter)
    rows = []
    for index, pair in enumerate(manifest['pairs'], 1):
        if dataset == 'tycho':
            a, b, ma, mb = load_tycho(pair, assets)
        else:
            a, b, ma, mb, _ = load_cnsfm(pair, assets, '.')
        size = a.shape[0]
        points = common_points(ma, mb, spacing=size//16, margin=size//16)
        for method in methods:
            aa, bb = preprocess(a, method.endswith('_clahe')), preprocess(b, method.endswith('_clahe'))
            aa[~ma], bb[~mb] = 0, 0
            def match(x, y, mx, my):
                cv2.setRNGSeed(20261026)
                if method == 'sift':
                    return sift(x, y, mx, my)
                if method == 'disk_lightglue_lunar':
                    return adapted(x, y, mx, my)
                return cached(x, y, mx, my)[0]
            forward, reverse = match(aa, bb, ma, mb), match(bb, aa, mb, ma)
            error, rmse = endpoint_statistics(forward.matrix, points, points)
            row = dict(pair_id=pair['pair_id'], region_id=pair['region_id'], split=pair['split'],
                reference=pair['reference'], query=pair['query'], method=method,
                preprocessing='clahe' if method.endswith('_clahe') else 'none', image_size=size,
                dataset_kind='real_repeat_observation', error_kind='map_reference_disagreement',
                label_source=None, correct=None, reference_uncertainty_px=None,
                estimated=forward.matrix is not None, error_px=error, rmse_px=rmse, scored_points=len(points),
                support=forward.support, reason=forward.reason,
                transform=None if forward.matrix is None else forward.matrix.tolist(),
                reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
                reference_points=None if forward.reference_points is None else forward.reference_points.tolist(),
                query_points=None if forward.query_points is None else forward.query_points.tolist(),
                input_array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()},
                **diagnostics(forward, reverse, a.shape), **illumination_metrics(a, b, ma, mb))
            if dataset == 'tycho' and 'policy' in policies.get(method, {}):
                row['accepted_by_transported_physical_policy'] = accepts(row, policies[method]['policy'])
            with (output/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)
        print(index, '/', len(manifest['pairs']), pair['pair_id'], flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['region_id'], row['method']].append(row)
    summary = []
    for (region, method), subset in sorted(groups.items()):
        errors = [r['error_px'] for r in subset if r['error_px'] is not None]
        summary.append(dict(region_id=region, method=method, pairs=len(subset),
            returned=sum(r['estimated'] for r in subset),
            median_map_disagreement_returned_px=float(np.median(errors)) if errors else None,
            accepted_by_transported_physical_policy=sum(r.get('accepted_by_transported_physical_policy', False) for r in subset) if dataset == 'tycho' else None))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(dict(measurements=summary,
        warning='Returned or accepted alignments are not independently verified correct alignments. Unknown real map-reference uncertainty; additional human landmarks required.'), indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['tycho', 'cnsfm'], required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--physical-manifest', default='moon_manifest.json')
    p.add_argument('--adapter', default='reports/lunar_adapter.json')
    p.add_argument('--policies', default='reports/frozen_policies.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    read = lambda name:json.loads(Path(name).read_text())
    run(read(args.manifest), read(args.physical_manifest), read(args.adapter), read(args.policies), args.output, args.dataset)
