"""Add unmodified LoFTR to exactly the frozen all-Sun-pairs benchmark.

This is a baseline comparison on already inspected terrain, not a fresh test.
No model, confidence threshold or image resizing is selected from its outcomes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from confidence import diagnostics
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_crosslighting import plan, prepare_pair
from loftr_matcher import LoFTRMatcher
from moon_dataset import load_image, validate


def array_hashes(**arrays):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}


def run(source, output, resume=False, device='cpu', limit=None):
    source, output = Path(source), Path(output)
    complete = json.loads((source/'COMPLETE.json').read_text())
    if sha256(source/'results.json') != complete['results_sha256'] or sha256(source/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
        raise ValueError('Source benchmark differs from completion record')
    original = json.loads((source/'frozen_inputs.json').read_text())
    if original['warp'] != dict(angle=0, dx=6, dy=-4, gain=1, gamma=1, noise=0) or original['threshold_px'] != 3:
        raise ValueError('Expected the original exact-translation benchmark')
    manifest = original['manifest']
    validate(manifest, '.')
    planned = plan(manifest, original['split'])
    if limit is not None:
        if limit <= 0:
            raise ValueError('Limit must be positive')
        planned = planned[:limit]
    sources = ['evaluate_loftr.py', 'loftr_matcher.py', 'loftr_model_lock.json', 'evaluate_crosslighting.py',
               'moon_dataset.py', 'synthetic.py', 'matchers.py', 'geometry.py', 'confidence.py', 'evaluate.py', 'dataset.py']
    frozen = dict(schema_version=1, source_run=str(source), original_results_sha256=complete['results_sha256'],
        original_inputs_sha256=complete['frozen_inputs_sha256'], model_lock=json.loads(Path('loftr_model_lock.json').read_text()),
        manifest=manifest, planned=planned, method='loftr_outdoor', device=device, seed=20261026,
        threshold_px=3, warp=original['warp'], image_size=128,
        source_sha256={name: sha256(name) for name in sources},
        subset_limit=limit, status='Unmodified pretrained baseline; previously inspected terrain; no development or test tuning.')
    canonical = json.dumps(frozen, sort_keys=True, allow_nan=False)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != canonical:
            raise ValueError('Resume requires identical design, sources, model and device')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(canonical)
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    target = output/'results.jsonl'
    rows = [json.loads(line) for line in target.read_text().splitlines()] if target.exists() else []
    done = {r['pair_id'] for r in rows}
    if len(done) != len(rows) or not done <= {p['pair_id'] for p in planned}:
        raise ValueError('Duplicate or unplanned saved outcomes')
    originals = {}
    for row in json.loads((source/'results.json').read_text()):
        previous = originals.setdefault(row['pair_id'], row['input_array_sha256'])
        if previous != row['input_array_sha256']:
            raise ValueError('Original methods used different inputs')
    cv2.setNumThreads(1)
    matcher = LoFTRMatcher(device=device)
    tiles = {t['tile_id']: t for t in manifest['tiles']}
    images = {}
    warm = False
    for number, case in enumerate(planned, 1):
        if case['pair_id'] in done:
            continue
        tile = tiles[case['tile_id']]
        def image(index):
            key = (tile['tile_id'], index)
            if key not in images:
                images[key] = load_image(tile['images'][index])[:2]
            return images[key]
        a, ma = image(case['reference_light_index'])
        base, mb = image(case['query_light_index'])
        b, qm, truth, points, targets = prepare_pair(a, base, ma, mb)
        hashes = array_hashes(reference=a, query=b, reference_mask=ma, query_mask=qm)
        if originals.get(case['pair_id']) != hashes:
            raise ValueError('LoFTR inputs differ from the original common benchmark')
        if not warm:
            matcher(a, a, ma, ma)
            matcher.sync()
            warm = True
        cv2.setRNGSeed(20261026)
        matcher.sync()
        started = perf_counter()
        forward = matcher(a, b, ma, qm)
        matcher.sync()
        elapsed = (perf_counter()-started)*1000
        cv2.setRNGSeed(20261026)
        reverse = matcher(b, a, qm, ma)
        error, rmse = endpoint_statistics(forward.matrix, points, targets)
        def coords(value):
            return None if value is None else np.asarray(value).tolist()
        row = dict(**case, method='loftr_outdoor', observation_group=tile['observation_group'],
            dataset_kind='physical_render_fixed_translation', label_source='exact_geometry',
            estimated=forward.matrix is not None, correct=bool(error is not None and error <= 3),
            error_px=error, rmse_px=rmse, reason=forward.reason, support=forward.support,
            forward_inference_ms=elapsed, scored_points=len(points), input_array_sha256=hashes,
            transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
            reference_points=coords(forward.reference_points), query_points=coords(forward.query_points),
            **diagnostics(forward, reverse, a.shape))
        with target.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
        rows.append(row)
        print(f'{number}/{len(planned)} {case["pair_id"]} correct={row["correct"]} {elapsed:.1f}ms', flush=True)
    errors = [r['error_px'] for r in rows if r['estimated']]
    summary = dict(method='loftr_outdoor', pairs=len(rows), correct=sum(r['correct'] for r in rows),
        returned=sum(r['estimated'] for r in rows), median_error_returned_px=float(np.median(errors)) if errors else None,
        median_forward_inference_ms=float(np.median([r['forward_inference_ms'] for r in rows])),
        initialization_ms=matcher.initialization_ms, rejection_policy='Not calibrated; no test-selected policy',
        subset_limit=limit, status=frozen['status'])
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    print(json.dumps(summary), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='runs/physical-crosslighting-v1')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    run(args.source, args.output, args.resume, args.device, args.limit)
