"""Frozen development-only test of image-selected local structural refinement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from candidate_selection import Candidate, choose, transform_distance
from confidence import diagnostics
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_blend_dev import FAMILIES
from evaluate_research_dev import development_plan
from landmark_review import fingerprint
from matchers import Match
from moon_dataset import load_image
from research_geometry import prepare
from structural_refinement import AlignmentObjective, refine


METHODS = ['blend_global_support_v2', 'refined_global_v1']


def image_scores(objective, candidates):
    values = {}
    for c in candidates:
        q = objective.quality(c.matrix)
        if q is not None:
            values[c.name] = agreement(q, c, candidates, objective.shape)
    return values


def agreement(quality, candidate, candidates, shape):
    families = {c.family for c in candidates if c.family != candidate.family and c.matrix is not None}
    agree = {c.family for c in candidates if c.family in families and c.matrix is not None
             and transform_distance(candidate.matrix, c.matrix, shape) <= 3.}
    return dict(quality, agreement_fraction=len(agree)/max(len(families), 1), agreeing_other_families=sorted(agree))


def infer(objective, candidates):
    scores = image_scores(objective, candidates)
    selected, quality = choose(candidates, scores, 'global')
    refined = refine(objective, selected, 'global', max_evaluations=120)
    rq = refined.quality if refined.candidate is None else agreement(refined.quality, refined.candidate, candidates, objective.shape)
    return [(selected, quality), (refined.candidate, rq)]


def run(sources, output, tile_limit=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    predictions, bindings, first = {}, {}, None
    for root in map(Path, sources):
        frozen = json.loads((root/'frozen_inputs.json').read_text())
        terminal = json.loads((root/'COMPLETE.json').read_text())
        if terminal['results_sha256'] != sha256(root/'results.json') or terminal['frozen_inputs_sha256'] != sha256(root/'frozen_inputs.json'):
            raise ValueError('Completed upstream artifact changed')
        if first is None:
            first = frozen
        if frozen['manifest'] != first['manifest'] or frozen['cases'] != first['cases'] or frozen['split'] != 'dev':
            raise ValueError('Upstream plans differ or include non-development terrain')
        values = json.loads((root/'results.json').read_text())
        index = {(r['pair_id'], r['method']): r for r in values}
        if len(index) != len(values):
            raise ValueError('Duplicate upstream outcome')
        for method in frozen['methods']:
            if method not in FAMILIES or method in predictions:
                raise ValueError('Unknown or duplicate predictor')
            predictions[method] = index
            bindings[method] = dict(source_run=str(root), family=FAMILIES[method], **terminal)
    manifest = first['manifest']
    cases = development_plan(manifest, tile_limit)
    if any(c not in first['cases'] for c in cases):
        raise ValueError('Unplanned development subset')
    files = ['evaluate_refinement_dev.py', 'structural_refinement.py', 'candidate_selection.py',
        'evaluate_blend_dev.py', 'evaluate_research_dev.py', 'structural_matchers.py', 'research_geometry.py',
        'research_dataset.py', 'moon_dataset.py', 'geometry.py', 'confidence.py', 'evaluate.py', 'dataset.py']
    frozen = dict(schema_version=1, split='dev', manifest=manifest, manifest_sha256=fingerprint(manifest), cases=cases,
        tile_limit=tile_limit, methods=METHODS, predictors=bindings, source_sha256={p: sha256(p) for p in files},
        seed=20261029, threshold_px=3, max_refinement_evaluations=120, valid_feature_radius=7,
        status='Development hypothesis; same cached predictors, larger complete feature-support mask. Image-only similarity selection/refinement; no reserve access. Runtime excludes cached predictor inference.')
    output.mkdir(parents=True)
    (output/'frozen_inputs.json').write_text(json.dumps(frozen, sort_keys=True, allow_nan=False))
    (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    images, rows = {}, []
    def image(tile, index):
        key = (tile, index)
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    cv2.setNumThreads(1)
    for number, case in enumerate(cases, 1):
        a, ma = image(case['tile_id'], case['reference_light_index'])
        base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
        negative = case['geometry_family'] == 'nonoverlap_negative'
        if negative:
            b, qm, points, targets = base, mb, None, None
        else:
            b, qm, _, points, targets, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
        upstream = {m: index[(case['pair_id'], m)] for m, index in predictions.items()}
        if any(r['input_array_sha256'] != hashes for r in upstream.values()):
            raise ValueError('Cached predictor input bytes differ')
        candidates = [Candidate(m, FAMILIES[m], r['transform']) for m, r in upstream.items()]
        reverses = [Candidate(m, FAMILIES[m], r['reverse_transform']) for m, r in upstream.items()]
        started = perf_counter()
        forwards = infer(AlignmentObjective(a, b, ma, qm), candidates)
        backwards = infer(AlignmentObjective(b, a, qm, ma), reverses)
        elapsed = (perf_counter()-started)*1000
        for method, (selected, quality), (back, back_quality) in zip(METHODS, forwards, backwards):
            forward = Match(reason='no_image_supported_candidate') if selected is None else Match(np.asarray(selected.matrix))
            reverse = Match() if back is None else Match(np.asarray(back.matrix))
            error, rmse = (None, None) if negative else endpoint_statistics(forward.matrix, points, targets)
            coords = lambda v: None if v is None else np.asarray(v).tolist()
            row = dict(**case, method=method, observation_group=tiles[case['tile_id']]['observation_group'],
                query_observation_group=tiles[case.get('query_tile_id', case['tile_id'])]['observation_group'],
                dataset_kind='physical_render_research_development', label_source='known_nonoverlap' if negative else 'exact_geometry',
                estimated=selected is not None, correct=bool(not negative and error is not None and error <= 3),
                error_px=error, rmse_px=rmse, reason=forward.reason, support=0, support_kind='image_region',
                transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix), reference_points=None, query_points=None,
                selection_quality=quality, reverse_selection_quality=back_quality, input_array_sha256=hashes,
                selection_and_refinement_ms=elapsed, applicability='within_model', **diagnostics(forward, reverse, a.shape))
            rows.append(row)
            with (output/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(f'{number}/{len(cases)} {case["pair_id"]}', flush=True)
    summary = []
    for family in sorted({c['geometry_family'] for c in cases}):
        for method in METHODS:
            selected = [r for r in rows if r['geometry_family'] == family and r['method'] == method]
            summary.append(dict(geometry_family=family, method=method, cases=len(selected),
                returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=['runs/structural-dev-v1', 'runs/geometry-dev-v1', 'runs/adapter-dev-v1'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--tiles', type=int, default=2)
    args = parser.parse_args()
    run(args.sources, args.output, args.tiles)
