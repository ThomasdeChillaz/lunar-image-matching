"""Stream a truth-blind image-agreement blend from frozen development runs.

Upstream runs may still be live: only cases with every planned predictor are
processed. Completion requires every case and the final upstream completion
hashes. This never evaluates geographic reserves or chooses per-case oracles.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from candidate_selection import Candidate, MODES, choose, score_candidates
from confidence import diagnostics
from dataset import sha256
from evaluate import endpoint_statistics
from landmark_review import fingerprint
from matchers import Match
from moon_dataset import load_image
from research_geometry import prepare


FAMILIES = dict(ncc='intensity', sift='sift', self_similarity_ncc='self_similarity',
    orientation_ncc='orientation', self_similarity_search='self_similarity',
    loftr_outdoor='loftr', disk_lightglue='disk', research_adapter='disk')
FEATURE_METHODS = {'sift', 'loftr_outdoor', 'disk_lightglue', 'research_adapter'}


class Rows:
    def __init__(self, run):
        self.run = Path(run)
        self.frozen = json.loads((self.run/'frozen_inputs.json').read_text())
        self.design_sha256 = sha256(self.run/'frozen_inputs.json')
        self.offset, self.inode = 0, None
        self.rows = {}

    def refresh(self):
        if sha256(self.run/'frozen_inputs.json') != self.design_sha256:
            raise ValueError('Live upstream design changed')
        path = self.run/'results.jsonl'
        if not path.exists():
            return
        stat = path.stat()
        if self.inode is not None and (stat.st_ino != self.inode or stat.st_size < self.offset):
            raise ValueError('Upstream results were replaced/truncated')
        self.inode = stat.st_ino
        with path.open('rb') as stream:
            stream.seek(self.offset)
            for line in stream.read().splitlines(keepends=True):
                if not line.endswith(b'\n'):
                    break
                row = json.loads(line)
                key = (row['pair_id'], row['method'])
                if key in self.rows:
                    raise ValueError('Duplicate upstream prediction')
                self.rows[key] = row
                self.offset += len(line)

    def completion(self):
        path = self.run/'COMPLETE.json'
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value['results_sha256'] != sha256(self.run/'results.json') or value['frozen_inputs_sha256'] != self.design_sha256:
            raise ValueError('Upstream completion fingerprint differs')
        return value


def run(sources, output, follow=False, resume=False):
    sources = [Rows(p) for p in sources]
    first = sources[0].frozen
    manifest, cases = first['manifest'], first['cases']
    predictors = {}
    for source in sources:
        if fingerprint(source.frozen['manifest']) != fingerprint(manifest) or source.frozen['cases'] != cases or source.frozen['split'] != 'dev':
            raise ValueError('Every development source must use identical terrain and cases')
        for method in source.frozen['methods']:
            if method in predictors or method not in FAMILIES:
                raise ValueError('Duplicate or unsupported predictor')
            predictors[method] = source
    source_files = ['evaluate_blend_dev.py', 'candidate_selection.py', 'structural_matchers.py', 'research_geometry.py',
                    'moon_dataset.py', 'geometry.py', 'confidence.py', 'evaluate.py', 'dataset.py']
    frozen = dict(schema_version=1, split='dev', manifest=manifest, cases=cases, tile_limit=first['tile_limit'],
        manifest_sha256=fingerprint(manifest), methods=[f'blend_{mode}' for mode in MODES], selection_modes=MODES,
        predictors={m: dict(source_run=str(s.run), frozen_inputs_sha256=s.design_sha256, family=FAMILIES[m]) for m, s in predictors.items()},
        source_sha256={p: sha256(p) for p in source_files},
        status='Development-only image-agreement selection; no truth labels, motion family or ground-truth motion enter the selector. Runtime excludes cached predictor inference.')
    canonical = json.dumps(frozen, sort_keys=True, allow_nan=False)
    output = Path(output)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != canonical:
            raise ValueError('Resume requires identical predictors, selector and complete plan')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(canonical)
        (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    path = output/'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['pair_id'], r['method']) for r in rows}
    expected = {(c['pair_id'], method) for c in cases for method in frozen['methods']}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Unplanned/duplicate saved blend outcome')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    images = {}
    def image(tile, index):
        key = (tile, index)
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    while len(done) < len(expected):
        for source in sources:
            source.refresh()
        progress = 0
        for case in cases:
            if all((case['pair_id'], m) in done for m in frozen['methods']):
                continue
            if not all((case['pair_id'], m) in s.rows for m, s in predictors.items()):
                continue
            upstream = {m: s.rows[(case['pair_id'], m)] for m, s in predictors.items()}
            a, ma = image(case['tile_id'], case['reference_light_index'])
            base, mb = image(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
            negative = case['geometry_family'] == 'nonoverlap_negative'
            if negative:
                b, qm, points, targets = base, mb, None, None
            else:
                b, qm, _, points, targets, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in dict(reference=a, query=b, reference_mask=ma, query_mask=qm).items()}
            if any(r['input_array_sha256'] != hashes or r['split'] != 'dev' for r in upstream.values()):
                raise ValueError('Predictors received different native input bytes')
            # The strict Candidate type excludes correctness/errors/known motion.
            candidates = [Candidate(m, FAMILIES[m], r['transform']) for m, r in upstream.items()]
            reverses = [Candidate(m, FAMILIES[m], r['reverse_transform']) for m, r in upstream.items()]
            started = time.perf_counter()
            forward_scores = score_candidates(a, b, ma, qm, candidates)
            reverse_scores = score_candidates(b, a, qm, ma, reverses)
            for mode in MODES:
                method = f'blend_{mode}'
                if (case['pair_id'], method) in done:
                    continue
                selected, quality = choose(candidates, forward_scores, mode)
                back, back_quality = choose(reverses, reverse_scores, mode)
                if selected is None:
                    forward, origin, support_kind = Match(reason='no_image_supported_candidate'), None, 'none'
                else:
                    origin = upstream[selected.name]
                    forward = Match(np.asarray(selected.matrix), support=origin['support'],
                        reference_points=None if origin['reference_points'] is None else np.asarray(origin['reference_points']),
                        query_points=None if origin['query_points'] is None else np.asarray(origin['query_points']))
                    support_kind = 'feature_inliers' if selected.name in FEATURE_METHODS else 'correlation_template_probes_or_area'
                reverse = Match() if back is None else Match(np.asarray(back.matrix))
                diagnostic = diagnostics(forward, reverse, a.shape)
                if support_kind != 'feature_inliers':
                    diagnostic['inlier_residual_p95_px'] = None
                error, rmse = (None, None) if negative else endpoint_statistics(forward.matrix, points, targets)
                if origin is not None and not negative and (origin['error_px'] is None or not np.isclose(origin['error_px'], error)):
                    raise ValueError('Selected unchanged prediction has different geometric error')
                coords = lambda v: None if v is None else np.asarray(v).tolist()
                row = dict(**case, method=method, observation_group=tiles[case['tile_id']]['observation_group'],
                    query_observation_group=tiles[case.get('query_tile_id', case['tile_id'])]['observation_group'],
                    dataset_kind='physical_render_research_development', label_source='known_nonoverlap' if negative else 'exact_geometry',
                    estimated=selected is not None, correct=bool(not negative and error is not None and error <= 3),
                    error_px=error, rmse_px=rmse, reason=forward.reason, support=forward.support, support_kind=support_kind,
                    transform=coords(forward.matrix), reverse_transform=coords(reverse.matrix),
                    reference_points=coords(forward.reference_points), query_points=coords(forward.query_points),
                    correlation_score=quality.get('global_correlation'), peak_margin=None,
                    selection_quality=quality, reverse_selection_quality=back_quality,
                    candidate_quality=forward_scores, reverse_candidate_quality=reverse_scores,
                    input_array_sha256=hashes, selection_ms=(time.perf_counter()-started)*1000,
                    applicability='within_model', **diagnostic)
                with path.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                done.add((case['pair_id'], method))
            progress += 1
        print(f'Blend complete cases={len(done)//len(MODES)}/{len(cases)}; new={progress}; upstream final={[s.completion() is not None for s in sources]}', flush=True)
        if len(done) == len(expected):
            break
        if not follow:
            raise RuntimeError('Upstream predictions incomplete; use --follow or resume after completion')
        time.sleep(2)
    completions = {str(s.run): s.completion() for s in sources}
    if any(v is None for v in completions.values()):
        while follow and any(s.completion() is None for s in sources):
            time.sleep(2)
        completions = {str(s.run): s.completion() for s in sources}
        if any(v is None for v in completions.values()):
            raise RuntimeError('Every upstream run must have verified completion')
    (output/'upstream_completion.json').write_text(json.dumps(completions, indent=2))
    summary = []
    for family in sorted({c['geometry_family'] for c in cases}):
        for method in frozen['methods']:
            selected = [r for r in rows if r['geometry_family'] == family and r['method'] == method]
            summary.append(dict(geometry_family=family, method=method, cases=len(selected),
                returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'), upstream_completion_sha256=sha256(output/'upstream_completion.json'))))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', default=['runs/structural-dev-v1', 'runs/geometry-dev-v1', 'runs/adapter-dev-v1'])
    p.add_argument('--output', required=True)
    p.add_argument('--follow', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.sources, args.output, args.follow, args.resume)
