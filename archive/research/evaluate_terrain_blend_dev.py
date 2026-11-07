"""Stream controlled stronger-model pools with/without the trained structural CNN.

Only native-image evidence reaches selection. All upstream methods must have an
outcome before a case is processed, and terminal output requires equality with
the final upstream JSON records, not merely an append-log prefix.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from candidate_selection import Candidate
from confidence import diagnostics
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_blend_dev import Rows
from landmark_review import fingerprint
from matchers import Match
from terrain_selection import FAMILIES, METHODS, POOLS, MODES, infer_pools
from moon_dataset import load_image
from research_dataset import validate
from research_geometry import prepare


def bind_terminal(source):
    terminal = source.completion()
    if terminal is None:
        return None
    source.refresh()
    values = json.loads((source.run/'results.json').read_text())
    index = {(r['pair_id'], r['method']): r for r in values}
    if len(index) != len(values) or index != source.rows or len(values) != terminal['rows']:
        raise ValueError('Streamed predictions differ from final upstream measurements')
    return terminal


def run(sources, output, follow=False, resume=False):
    sources = [Rows(p) for p in sources]
    first = sources[0].frozen
    manifest, cases = first['manifest'], first['cases']
    validate(manifest)
    predictors = {}
    for source in sources:
        if source.frozen['manifest'] != manifest or source.frozen['cases'] != cases or source.frozen['split'] != 'dev':
            raise ValueError('Every source must use the identical development plan')
        for method in source.frozen['methods']:
            if method not in FAMILIES or method in predictors:
                raise ValueError('Duplicate or unsupported predictor')
            predictors[method] = source
    if set(predictors) != set(FAMILIES):
        raise ValueError('All fourteen declared predictors are required')
    files = ['evaluate_terrain_blend_dev.py', 'terrain_selection.py', 'modern_selection.py', 'candidate_selection.py', 'structural_refinement.py',
        'structural_matchers.py', 'evaluate_blend_dev.py', 'confidence.py', 'geometry.py', 'evaluate.py',
        'matchers.py', 'dataset.py', 'research_dataset.py', 'research_geometry.py', 'moon_dataset.py', 'landmark_review.py']
    frozen = dict(schema_version=1, split='dev', manifest=manifest, cases=cases, tile_limit=first['tile_limit'],
        manifest_sha256=fingerprint(manifest), methods=METHODS, pools=POOLS, selection_modes=MODES,
        predictors={m: dict(source_run=str(s.run), frozen_inputs_sha256=s.design_sha256, family=FAMILIES[m]) for m, s in predictors.items()},
        source_sha256={f: sha256(f) for f in files}, seed=20261029, threshold_px=3, valid_feature_radius=7,
        status='Development-only controlled addition of trained structural CNN. Strong4 and expanded12 pools each have controls with and without selected CNN weights; initial weights have a separate pool only. All use common radius7 evidence and global scoring; no refinement. Runtime excludes cached inference.')
    canonical = json.dumps(frozen, sort_keys=True, allow_nan=False)
    output = Path(output)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != canonical:
            raise ValueError('Resume requires identical sources, pools and full case plan')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(canonical)
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    path = output/'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected = {(c['pair_id'], m) for c in cases for m in METHODS}
    done = {(r['pair_id'], r['method']) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Duplicate or unplanned pool outcome')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    images = {}
    def image(tile, index):
        key = tile, index
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    cv2.setNumThreads(1)
    while len(done) < len(expected):
        for source in sources:
            source.refresh()
        progress = 0
        for case in cases:
            if all((case['pair_id'], m) in done for m in METHODS):
                continue
            if not all((case['pair_id'], m) in s.rows for m, s in predictors.items()):
                continue
            a, ma = image(case['tile_id'], case['reference_light_index'])
            base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
            negative = case['geometry_family'] == 'nonoverlap_negative'
            if negative:
                b, qm, points, targets = base, mb, None, None
            else:
                b, qm, _, points, targets, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
            upstream = {m: s.rows[(case['pair_id'], m)] for m, s in predictors.items()}
            if any(r['input_array_sha256'] != hashes or r['split'] != 'dev' for r in upstream.values()):
                raise ValueError('Prediction inputs differ from reconstructed native arrays')
            candidates = [Candidate(m, FAMILIES[m], upstream[m]['transform']) for m in FAMILIES]
            reverses = [Candidate(m, FAMILIES[m], upstream[m]['reverse_transform']) for m in FAMILIES]
            started = time.perf_counter()
            forwards = infer_pools(a, b, ma, qm, candidates)
            backwards = infer_pools(b, a, qm, ma, reverses)
            elapsed = (time.perf_counter()-started)*1000
            for method in METHODS:
                if (case['pair_id'], method) in done:
                    continue
                forward_value, reverse_value = forwards[method], backwards[method]
                chosen, back = forward_value['candidate'], reverse_value['candidate']
                forward = Match(reason='no_image_supported_candidate') if chosen is None else Match(np.asarray(chosen.matrix))
                reverse = Match() if back is None else Match(np.asarray(back.matrix))
                error, rmse = (None, None) if negative else endpoint_statistics(forward.matrix, points, targets)
                if chosen is not None and not negative and not np.isclose(error, upstream[chosen.name]['error_px']):
                    raise ValueError('Unchanged chosen transform has a different geometric error')
                coords = lambda v: None if v is None else np.asarray(v).tolist()
                row = dict(**case, method=method, observation_group=tiles[case['tile_id']]['observation_group'],
                    query_observation_group=tiles[case.get('query_tile_id', case['tile_id'])]['observation_group'],
                    dataset_kind='physical_render_research_development', label_source='known_nonoverlap' if negative else 'exact_geometry',
                    estimated=chosen is not None, correct=bool(not negative and error is not None and error <= 3),
                    error_px=error, rmse_px=rmse, reason=forward.reason, support=0, support_kind='image_region',
                    selected_predictor_support=None if chosen is None else upstream[chosen.name]['support'],
                    transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
                    reference_points=None, query_points=None, selection_quality=forward_value['quality'],
                    reverse_selection_quality=reverse_value['quality'], candidate_quality=forward_value['candidate_quality'],
                    reverse_candidate_quality=reverse_value['candidate_quality'], input_array_sha256=hashes,
                    all_pools_bidirectional_selection_ms=elapsed, applicability='within_model',
                    **diagnostics(forward, reverse, a.shape))
                with path.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                done.add((case['pair_id'], method))
            progress += 1
        print(f'Pool cases={len(done)//len(METHODS)}/{len(cases)}; new={progress}', flush=True)
        if len(done) == len(expected):
            break
        if not follow:
            raise RuntimeError('Upstream predictions incomplete; --follow or identical resume required')
        time.sleep(2)
    terminals = {str(s.run): bind_terminal(s) for s in sources}
    while follow and any(v is None for v in terminals.values()):
        time.sleep(2)
        terminals = {str(s.run): bind_terminal(s) for s in sources}
    if any(v is None for v in terminals.values()):
        raise RuntimeError('Every predictor requires verified terminal completion')
    (output/'upstream_completion.json').write_text(json.dumps(terminals, indent=2))
    summary = []
    for family in sorted({c['geometry_family'] for c in cases}):
        for method in METHODS:
            selected = [r for r in rows if r['method']==method and r['geometry_family']==family]
            summary.append(dict(method=method, geometry_family=family, cases=len(selected),
                returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'), upstream_completion_sha256=sha256(output/'upstream_completion.json'))))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', default=['runs/structural-dev-v1', 'runs/geometry-dev-v1', 'runs/adapter-dev-v1',
        'runs/eloftr-scaled-dev-v1', 'runs/matchanything-dev-v1', 'runs/contrastive-terrain-dev-v1'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.sources, args.output, args.follow, args.resume)
