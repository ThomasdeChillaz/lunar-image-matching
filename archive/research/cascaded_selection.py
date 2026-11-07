"""Development-calibrated confidence cascade over image-only predictions.

This is a decision fallback, not a computational early-exit claim: the current
benchmark caches every predictor first. InferenceEvidence excludes labels and
known geometry. Conditional fallback calibration is confined to development.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from candidate_selection import Candidate
from confidence import diagnostics
from dataset import sha256
from matchers import Match
from research_quality_gate import accepts, calibrate


@dataclass(frozen=True)
class InferenceEvidence:
    candidate: Candidate
    reverse_matrix: object
    quality: dict


def inference_row(evidence, shape=(128, 128)):
    forward, reverse = Match(evidence.candidate.matrix), Match(evidence.reverse_matrix)
    return dict(estimated=forward.matrix is not None, selection_quality=evidence.quality,
        **diagnostics(forward, reverse, shape))


def select(primary, fallback, policy, shape=(128, 128)):
    if accepts(inference_row(primary, shape), policy['primary']):
        return primary, 'primary'
    if accepts(inference_row(fallback, shape), policy['fallback']):
        return fallback, 'fallback'
    return None, 'rejected'


def run(blend_run, output):
    source, output = Path(blend_run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    terminal = json.loads((source/'COMPLETE.json').read_text())
    design = json.loads((source/'frozen_inputs.json').read_text())
    if sha256(source/'results.json') != terminal['results_sha256'] or sha256(source/'frozen_inputs.json') != terminal['frozen_inputs_sha256']:
        raise ValueError('Completed blend artifacts changed')
    if design['split'] != 'dev':
        raise ValueError('Cascade fitting accepts development only')
    rows = [r for r in json.loads((source/'results.json').read_text()) if r['method'] == 'blend_global']
    fallback_method = 'self_similarity_search'
    binding = design['predictors'][fallback_method]
    parent = Path(binding['source_run'])
    parent_terminal = json.loads((parent/'COMPLETE.json').read_text())
    if sha256(parent/'frozen_inputs.json') != binding['frozen_inputs_sha256'] or sha256(parent/'results.json') != parent_terminal['results_sha256']:
        raise ValueError('Completed fallback artifacts changed')
    fallback_rows = {r['pair_id']: r for r in json.loads((parent/'results.json').read_text()) if r['method'] == fallback_method}
    if {r['pair_id'] for r in rows} != set(fallback_rows):
        raise ValueError('Cascade methods have different cases')
    primary_calibration = calibrate(rows, design['manifest'])
    policies, outcomes = {}, []
    # Explicitly declared risk settings; residual fallback gets zero empirical
    # errors. The final combined counts are checked against the declared budget.
    for budget in (0., .01):
        method = f'cascade_{budget}'
        first_policy = primary_calibration['policies']['blend_global']['empirical_accepted_error_budgets'][str(budget)]['policy']
        residual = []
        enhanced_fallback = {}
        for row in rows:
            fallback = fallback_rows[row['pair_id']]
            if row['input_array_sha256'] != fallback['input_array_sha256']:
                raise ValueError('Fallback input pixels differ')
            q = row['candidate_quality'].get(fallback_method, {})
            enhanced = dict(fallback, selection_quality=q)
            enhanced_fallback[row['pair_id']] = enhanced
            if not accepts(row, first_policy):
                residual.append(enhanced)
        if residual:
            fitted = calibrate(residual, design['manifest'])
            second_policy = fitted['policies'][fallback_method]['empirical_accepted_error_budgets']['0.0']['policy']
        else:
            second_policy = dict(reject_all=True)
        policy = dict(primary=first_policy, fallback=second_policy)
        selected_rows = []
        for row in rows:
            fallback = enhanced_fallback[row['pair_id']]
            first = InferenceEvidence(Candidate('blend_global', 'blend', row['transform']), row['reverse_transform'], row['selection_quality'])
            second = InferenceEvidence(Candidate(fallback_method, 'structural', fallback['transform']), fallback['reverse_transform'], fallback['selection_quality'])
            chosen, stage = select(first, second, policy)
            origin = row if stage == 'primary' else fallback if stage == 'fallback' else None
            result = dict(row, method=method, cascade_stage=stage, estimated=chosen is not None,
                correct=False if origin is None else origin['correct'], transform=None if chosen is None else chosen.candidate.matrix,
                reverse_transform=None if chosen is None else chosen.reverse_matrix,
                error_px=None if origin is None else origin['error_px'], rmse_px=None if origin is None else origin['rmse_px'],
                selection_quality={} if chosen is None else chosen.quality,
                reason='rejected_by_development_policy' if chosen is None else '',
                support=0, support_kind='image_region', reference_points=None, query_points=None)
            # Rejected known-geometry pairs keep their exact-geometry label even
            # when they have no estimate; negatives never acquire a true warp.
            result.update(inference_row(InferenceEvidence(Candidate(method, 'cascade', result['transform']), result['reverse_transform'], result['selection_quality'])))
            selected_rows.append(result)
        good = sum(r['correct'] for r in selected_rows)
        wrong = sum(r['estimated'] and not r['correct'] for r in selected_rows)
        negative = sum(r['estimated'] and r['label_source'] == 'known_nonoverlap' for r in selected_rows)
        if wrong > budget*max(good+wrong, 1) or negative:
            raise ValueError('Combined empirical cascade constraint failed')
        policies[method] = dict(policy=policy, accepted_correct=good, accepted_incorrect=wrong,
            accepted_nonoverlap=negative, primary_count=sum(r['cascade_stage']=='primary' for r in selected_rows),
            fallback_count=sum(r['cascade_stage']=='fallback' for r in selected_rows),
            residual_development_cases=len(residual), empirical_budget=budget)
        outcomes.extend(selected_rows)
    files = ['cascaded_selection.py', 'research_quality_gate.py', 'candidate_selection.py', 'confidence.py', 'geometry.py', 'matchers.py', 'dataset.py']
    frozen = dict(design, methods=list(policies), source_sha256={p: sha256(p) for p in files},
        parent_blend_run=str(source), parent_completion=terminal, fallback_completion=parent_terminal,
        cascade_policies=policies, frozen_utc=datetime.now(timezone.utc).isoformat(),
        status='Development-only conditional confidence fallback. Primary rules fitted on development; fallback fitted on rejected development cases. Not a fresh-test error guarantee or speed claim.')
    output.mkdir(parents=True)
    (output/'frozen_inputs.json').write_text(json.dumps(frozen, sort_keys=True, allow_nan=False))
    summary = []
    for family in sorted({r['geometry_family'] for r in outcomes}):
        for method in policies:
            selected = [r for r in outcomes if r['method']==method and r['geometry_family']==family]
            summary.append(dict(method=method, geometry_family=family, cases=len(selected), returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'results.json').write_text(json.dumps(outcomes, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'policies.json').write_text(json.dumps(policies, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(outcomes), results_sha256=sha256(output/'results.json'), frozen_inputs_sha256=sha256(output/'frozen_inputs.json'))))
    print(json.dumps(policies), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend-run', default='runs/blend-dev-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.blend_run, args.output)
