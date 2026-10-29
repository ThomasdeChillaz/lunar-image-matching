"""Separate original EfficientLoFTR 2x/4x MPS development comparison; no reserves.

Uses the same case construction and scoring as evaluate_research_dev.py, with
separate pinned author source, weights and optional dependency environment.
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
from landmark_review import fingerprint
from matchers import ncc, sift
from moon_dataset import load_image
from research_dataset import overlap, validate
from research_geometry import plan, prepare
from structural_matchers import structural_ncc


METHODS = ['matchanything_2x_mps', 'matchanything_4x_mps']


def development_plan(manifest, tile_limit):
    selected = [t for t in manifest['tiles'] if t['split'] == 'dev'][:tile_limit]
    if len(selected) != tile_limit or tile_limit <= 0:
        raise ValueError('Invalid development tile limit')
    subset = dict(tiles=selected)
    cases = plan(subset, 'dev')
    for tile in selected:
        partners = [t for t in manifest['tiles'] if t['split'] == 'dev' and t['observation_group'] != tile['observation_group']
                    and not overlap(t['bounds_lonlat_deg'], tile['bounds_lonlat_deg'], .1)]
        if not partners:
            raise ValueError('No independently disjoint development negative')
        partners.sort(key=lambda t: hashlib.sha256((tile['tile_id']+':'+t['tile_id']).encode()).hexdigest())
        for index in range(9):
            cases.append(dict(pair_id=f'{tile["tile_id"]}_negative_{index}', tile_id=tile['tile_id'],
                query_tile_id=partners[0]['tile_id'], split='dev', reference_light_index=index,
                query_light_index=(index+3) % 9, geometry_family='nonoverlap_negative'))
    return cases


def run(manifest, output, methods, tile_limit=8, resume=False, dependency_dir=None):
    validate(manifest)
    if len(set(methods)) != len(methods) or not methods or not set(methods) <= set(METHODS):
        raise ValueError('Unknown or duplicate method')
    output = Path(output)
    cases = development_plan(manifest, tile_limit)
    source_files = ['evaluate_matchanything_dev.py', 'matchanything_matcher.py', 'matchanything_model_lock.json', 'eloftr_matcher.py', 'requirements-eloftr.txt',
                    'research_geometry.py', 'research_dataset.py', 'structural_matchers.py', 'loftr_matcher.py',
                    'matchers.py', 'confidence.py', 'geometry.py', 'moon_dataset.py', 'evaluate.py', 'dataset.py']
    frozen = dict(schema_version=1, manifest_sha256=fingerprint(manifest), manifest=manifest, cases=cases,
        methods=methods, tile_limit=tile_limit, split='dev', seed=20261029, threshold_px=3,
        source_sha256={f: sha256(f) for f in source_files},
        modern_model_lock=json.loads(Path('matchanything_model_lock.json').read_text()), device='mps',
        resizing='Separate 2x/4x internal bilinear variants on MPS; pixel-center coordinates returned to native grid. No new source information; not a cross-device speed comparison.',
        status='Author MatchAnything core and released eloftr configuration, with declared NPE resize profile. No reserve access. Internal 2x/4x resizing on MPS reported separately; float32 GPU results need not be bitwise identical to CPU.')
    canonical = json.dumps(frozen, sort_keys=True, allow_nan=False)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != canonical:
            raise ValueError('Resume requires identical full design and methods')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(canonical)
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    path = output/'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected = {(c['pair_id'], m) for c in cases for m in methods}
    done = {(r['pair_id'], r['method']) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Duplicate or unplanned outcomes')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    from matchanything_matcher import MatchAnythingMatcher
    models = {method: MatchAnythingMatcher(dependency_dir=dependency_dir, factor=4 if '_4x_' in method else 2, device='mps') for method in methods}
    warmed = set()
    cv2.setNumThreads(1)
    images = {}
    def image(tile_id, index):
        key = (tile_id, index)
        if key not in images:
            if tiles[tile_id]['split'] != 'dev':
                raise ValueError('Development evaluator cannot read reserve images')
            images[key] = load_image(tiles[tile_id]['images'][index])[:2]
        return images[key]
    for number, case in enumerate(cases, 1):
        if all((case['pair_id'], m) in done for m in methods):
            continue
        a, ma = image(case['tile_id'], case['reference_light_index'])
        base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
        negative = case['geometry_family'] == 'nonoverlap_negative'
        if negative:
            b, qm, points, targets = base, mb, None, None
        else:
            b, qm, truth, points, targets, parameters = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
        for method in methods:
            if (case['pair_id'], method) in done:
                continue
            if method not in warmed and ma.any() and a[ma].std() >= 1:
                models[method](a, a, ma, ma)
                models[method].sync()
                warmed.add(method)
            cv2.setRNGSeed(20261029)
            models[method].sync()
            started = perf_counter()
            forward = models[method](a, b, ma, qm)
            models[method].sync()
            elapsed = (perf_counter()-started)*1000
            cv2.setRNGSeed(20261029)
            reverse = models[method](b, a, qm, ma)
            error, rmse = (None, None) if negative else endpoint_statistics(forward.matrix, points, targets)
            coords = lambda value: None if value is None else np.asarray(value).tolist()
            row = dict(**case, method=method, observation_group=tiles[case['tile_id']]['observation_group'],
                query_observation_group=tiles[case.get('query_tile_id', case['tile_id'])]['observation_group'],
                dataset_kind='physical_render_research_development', label_source='known_nonoverlap' if negative else 'exact_geometry',
                estimated=forward.matrix is not None, correct=bool(not negative and error is not None and error <= 3),
                error_px=error, rmse_px=rmse, reason=forward.reason, support=forward.support,
                applicability='out_of_model_stress' if method in ['ncc', 'self_similarity_ncc', 'orientation_ncc'] and case['geometry_family'] in ['rotation', 'scale', 'combined'] else 'within_model',
                transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
                reference_points=coords(forward.reference_points), query_points=coords(forward.query_points),
                correlation_score=getattr(forward, 'score', None), peak_margin=getattr(forward, 'peak_margin', None),
                forward_ms=elapsed, input_array_sha256=hashes, **diagnostics(forward, reverse, a.shape))
            with path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)
        print(f'{number}/{len(cases)} {case["pair_id"]}', flush=True)
    if {(r['pair_id'], r['method']) for r in rows} != expected:
        raise ValueError('Missing planned outcome')
    summary = []
    for family in sorted({c['geometry_family'] for c in cases}):
        for method in methods:
            selected = [r for r in rows if r['geometry_family'] == family and r['method'] == method]
            summary.append(dict(geometry_family=family, method=method, cases=len(selected), returned=sum(r['estimated'] for r in selected),
                correct=sum(r['correct'] for r in selected), status='Development only; negatives have no correct alignment.'))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    print(json.dumps(summary), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='research_manifest_v1.json')
    p.add_argument('--output', required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--tiles', type=int, default=8)
    p.add_argument('--dependency-dir')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(json.loads(Path(args.manifest).read_text()), args.output, args.methods, args.tiles, args.resume, args.dependency_dir)
