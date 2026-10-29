"""Development-only rejection without the old hard displacement cutoff.

Select transparent diagnostic thresholds on known-geometry development outcomes.
This is empirical policy selection, not a probability or safety guarantee. In
particular, repeated light pairs and overlapping terrain are not independent.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

from dataset import sha256
from uncertainty_groups import geographic_groups


def finite(value, fallback):
    return fallback if value is None or not np.isfinite(value) else float(value)


def accepts(row, policy):
    if not row['estimated'] or policy.get('reject_all'):
        return False
    coverage = min(finite(row.get('coverage_reference'), 0), finite(row.get('coverage_query'), 0))
    cycle = finite(row.get('cycle_p95_px'), float('inf'))
    score = finite(row.get('correlation_score'), 0)
    margin = finite(row.get('peak_margin'), 0)
    return bool(row.get('reverse_estimated') and row['support'] >= policy['min_support']
        and coverage >= policy['min_coverage'] and cycle <= policy['max_cycle_p95_px']
        and score >= policy['min_correlation_score'] and margin >= policy['min_peak_margin'])


def counts(rows, accepted):
    correct = np.array([r['correct'] for r in rows], bool)
    negative = np.array([r['label_source'] == 'known_nonoverlap' for r in rows], bool)
    return dict(cases=len(rows), returned=sum(r['estimated'] for r in rows),
        correct_available=int(correct.sum()), accepted_correct=int((accepted & correct).sum()),
        accepted_incorrect=int((accepted & ~correct).sum()), accepted_nonoverlap=int((accepted & negative).sum()),
        known_nonoverlap=int(negative.sum()), accepted_total=int(accepted.sum()))


def calibrate(rows, manifest, max_accepted_error_fraction=0.):
    if not 0 <= max_accepted_error_fraction < 1:
        raise ValueError('Invalid empirical risk budget')
    if not rows or any(r['split'] != 'dev' or r['label_source'] not in ['exact_geometry', 'known_nonoverlap'] or type(r['correct']) is not bool for r in rows):
        raise ValueError('Only known-geometry/nonoverlap development labels are allowed')
    identities = [(r['pair_id'], r['method']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate development outcome')
    dev = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    if any(r['tile_id'] not in dev or r.get('query_tile_id', r['tile_id']) not in dev for r in rows):
        raise ValueError('Non-development terrain in calibration')
    lookup, components, overlaps = geographic_groups(list(dev.values()))
    policies = {}
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method'] == method]
        true = np.array([r['correct'] for r in subset], bool)
        negative = np.array([r['label_source'] == 'known_nonoverlap' for r in subset], bool)
        coverage = np.array([min(finite(r.get('coverage_reference'), 0), finite(r.get('coverage_query'), 0)) for r in subset])
        cycles = np.array([finite(r.get('cycle_p95_px'), float('inf')) for r in subset])
        support = np.array([r['support'] for r in subset])
        score = np.array([finite(r.get('correlation_score'), 0) for r in subset])
        margin = np.array([finite(r.get('peak_margin'), 0) for r in subset])
        estimated = np.array([r['estimated'] and r.get('reverse_estimated', False) for r in subset], bool)
        correlation = any(r.get('correlation_score') is not None for r in subset)
        best = dict(reject_all=True)
        best_counts = counts(subset, np.zeros(len(subset), bool))
        frontier = []
        for cov, cycle, supp, corr, peak in itertools.product([0., .01, .05, .15], [.5, 1., 2., 4., 8.],
                [6, 12, 24], [0., .3, .5, .7] if correlation else [0.], [0., .01, .05, .1] if correlation else [0.]):
            accepted = estimated & (coverage >= cov) & (cycles <= cycle) & (support >= supp) & (score >= corr) & (margin >= peak)
            correct, wrong, bad_negative = int((accepted & true).sum()), int((accepted & ~true).sum()), int((accepted & negative).sum())
            # Known disjoint terrain must never be accepted on development.
            feasible = bad_negative == 0 and wrong <= max_accepted_error_fraction*max(correct+wrong, 1)
            if not feasible:
                continue
            policy = dict(min_coverage=cov, max_cycle_p95_px=cycle, min_support=supp, min_correlation_score=corr, min_peak_margin=peak)
            candidate_counts = counts(subset, accepted)
            frontier.append(dict(policy=policy, counts=candidate_counts))
            if correct > best_counts['accepted_correct'] or (correct == best_counts['accepted_correct'] and wrong < best_counts['accepted_incorrect']):
                best, best_counts = policy, candidate_counts
        frontier.sort(key=lambda p: (-p['counts']['accepted_correct'], p['counts']['accepted_incorrect']))
        best_mask = np.array([accepts(r, best) for r in subset])
        if counts(subset, best_mask) != best_counts:
            raise ValueError('Scalar/vector policy implementation disagreement')
        policies[method] = dict(policy=best, development_counts=best_counts, top_feasible_candidates=frontier[:30])
    return dict(schema_version=1, selected_on_split='dev', empirical_accepted_error_budget=max_accepted_error_fraction,
        independent_development_components=len(components), development_component_members=components,
        overlapping_development_footprint_pairs=len(overlaps), policies=policies,
        excluded_features=['ground_truth_transform', 'reference_error', 'correctness_at_inference', 'maximum_translation_fraction'],
        selection='Maximize accepted correct development alignments subject to empirical accepted-error budget and zero accepted development nonoverlap pairs. Ties retain the first declared candidate.',
        warning='No held-out evaluation or safety guarantee. Small geographic component count and repeated development hypothesis selection limit generalization; report fresh-test accepted errors and rejected correct fits.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('development_run')
    p.add_argument('--output', required=True)
    p.add_argument('--max-accepted-error', type=float, default=0.)
    args = p.parse_args()
    source, output = Path(args.development_run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    complete = json.loads((source/'COMPLETE.json').read_text())
    if sha256(source/'results.json') != complete['results_sha256'] or sha256(source/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
        raise ValueError('Development artifacts changed')
    frozen = json.loads((source/'frozen_inputs.json').read_text())
    result = calibrate(json.loads((source/'results.json').read_text()), frozen['manifest'], args.max_accepted_error)
    result.update(development_results_sha256=complete['results_sha256'], development_inputs_sha256=complete['frozen_inputs_sha256'],
        gate_source_sha256=sha256(__file__), frozen_before_reserve_evaluation=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print({method: value['development_counts'] for method, value in result['policies'].items()})
