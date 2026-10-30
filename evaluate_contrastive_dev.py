"""Exact common development plan for initial/learned structural CNN weights.

The source and completed training are frozen before any registration outcomes.
No reserve option exists. Test counts are not silently reduced on failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep

import cv2
import numpy as np
import torch

from confidence import diagnostics
from contrastive_terrain import ContrastiveTerrainMatcher
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_research_dev import development_plan
from geometry import endpoint_error, evaluation_points
from landmark_review import fingerprint
from moon_dataset import load_image
from research_dataset import validate
from research_geometry import prepare
from synthetic import texture, make_pair


def control(model):
    image = texture(20261030, 128)
    query, truth = make_pair(image, dict(angle=0, dx=8, dy=-4, gain=.8, gamma=1., noise=0.), 43)
    ma = np.ones_like(image, bool)
    mb = cv2.warpAffine(ma.astype(np.float32), truth, (128, 128), flags=cv2.INTER_LINEAR) >= 1.-1e-6
    model.warmup(image)
    cv2.setRNGSeed(20261029)
    result = model(image, query, ma, mb)
    error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
    blank = model(np.zeros_like(image), np.zeros_like(image))
    invalid = model(image, image, np.zeros_like(ma), ma)
    return dict(error_px=error, support=result.support, transform=None if result.matrix is None else result.matrix.tolist(),
        blank_reason=blank.reason, invalid_reason=invalid.reason,
        passed=bool(error is not None and error <= 3 and blank.reason == 'low_texture' and invalid.reason == 'no_valid_support'),
        status='Procedural translation software control only; not lunar matching performance.')


def run(training, output, tile_limit=8):
    training, output = Path(training), Path(output)
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((training/'training.json').read_text())
    terminal = json.loads((training/'COMPLETE.json').read_text())
    for name, digest in metadata['source_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Training source changed')
    manifest = validate(json.loads((training/'frozen_manifest.json').read_text()))
    if fingerprint(manifest) != metadata['manifest_sha256']:
        raise ValueError('Training manifest changed')
    cases = development_plan(manifest, tile_limit)
    output.mkdir(parents=True)
    methods = ['contrastive_initial', 'contrastive_selected']
    files = ['evaluate_contrastive_dev.py', 'contrastive_terrain.py', 'train_contrastive_terrain.py',
        'confidence.py', 'dataset.py', 'evaluate.py', 'evaluate_research_dev.py', 'geometry.py',
        'landmark_review.py', 'matchers.py', 'moon_dataset.py', 'research_dataset.py', 'research_geometry.py', 'synthetic.py']
    frozen = dict(schema_version=1, split='dev', tile_limit=tile_limit, cases=cases, manifest=manifest,
        manifest_sha256=fingerprint(manifest), methods=methods, seed=20261029, threshold_px=3,
        training=dict(path=str(training), completed=terminal, metadata=metadata),
        source_sha256={p: sha256(p) for p in files}, device='cpu',
        status='Original trained dense structural descriptor hypothesis; no geographic reserve evaluated. Initial weights are a required control.')
    (output/'frozen_inputs.json').write_text(json.dumps(frozen, sort_keys=True, allow_nan=False))
    (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    models = {m: ContrastiveTerrainMatcher(training, c) for m, c in zip(methods, ['initial', 'selected'])}
    controls = {m: control(model) for m, model in models.items()}
    (output/'software_controls.json').write_text(json.dumps(controls, indent=2, allow_nan=False))
    if not all(c['passed'] for c in controls.values()):
        raise RuntimeError('Initial/selected actual matcher software control failed')
    images, rows = {}, []
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    def image(tile_id, index):
        key = tile_id, index
        if key not in images:
            images[key] = load_image(tiles[tile_id]['images'][index])[:2]
        return images[key]
    coords = lambda x: None if x is None else np.asarray(x).tolist()
    for number, case in enumerate(cases, 1):
        a, ma = image(case['tile_id'], case['reference_light_index'])
        qtile = case.get('query_tile_id', case['tile_id'])
        base, mb = image(qtile, case['query_light_index'])
        negative = case['geometry_family'] == 'nonoverlap_negative'
        if negative:
            b, qm, points, targets = base, mb, None, None
        else:
            b, qm, _, points, targets, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
        for method, model in models.items():
            cv2.setRNGSeed(20261029)
            start = perf_counter()
            forward = model(a, b, ma, qm)
            model.sync()
            elapsed = (perf_counter()-start)*1000
            cv2.setRNGSeed(20261029)
            reverse = model(b, a, qm, ma)
            error, rmse = (None, None) if negative else endpoint_statistics(forward.matrix, points, targets)
            row = dict(**case, method=method, observation_group=tiles[case['tile_id']]['observation_group'],
                query_observation_group=tiles[qtile]['observation_group'],
                dataset_kind='physical_render_research_development', label_source='known_nonoverlap' if negative else 'exact_geometry',
                estimated=forward.matrix is not None, correct=bool(not negative and error is not None and error <= 3),
                error_px=error, rmse_px=rmse, reason=forward.reason, support=forward.support,
                transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
                reference_points=coords(forward.reference_points), query_points=coords(forward.query_points),
                forward_ms=elapsed, input_array_sha256=hashes, applicability='within_model',
                **diagnostics(forward, reverse, a.shape))
            rows.append(row)
            with (output/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        if number % 24 == 0:
            print(f'{number}/{len(cases)} {case["pair_id"]}', flush=True)
    summary = []
    for family in sorted({c['geometry_family'] for c in cases}):
        for method in methods:
            selected = [r for r in rows if r['geometry_family'] == family and r['method'] == method]
            summary.append(dict(geometry_family=family, method=method, cases=len(selected),
                returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'), software_controls_sha256=sha256(output/'software_controls.json'))))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tiles', type=int, default=8)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    if args.wait:
        while not (Path(args.training)/'COMPLETE.json').exists():
            print('Waiting for the explicit training completion artifact; no evaluation yet.', flush=True)
            sleep(20)
    run(args.training, args.output, args.tiles)
