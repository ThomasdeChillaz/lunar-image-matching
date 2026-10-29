"""Choose rejection rules from development image evidence, then freeze them.

Uses actual post-alignment image scores, common valid/lit support and independent
reverse-cycle diagnostics. Never use template-probe residuals or known motion at
inference. Empirical development constraints are not deployment guarantees.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np

from dataset import sha256
from uncertainty_groups import geographic_groups


GRID = dict(min_global=[-.5, 0., .2, .4, .6, .8], min_patch=[-1., 0., .2, .4, .6, .8],
    min_lit=[0., .05, .1, .25, .5], min_support=[.2, .4, .6, .8],
    min_agreement=[0., .25, .5, .75], max_cycle=[.5, 1., 2., 4., 8.])
BUDGETS = [0., .01, .05]


def features(row):
    q = row.get('selection_quality', {})
    values = [q.get('global_correlation', -1.), q.get('patch_median_correlation', -1.),
              q.get('joint_nondark_fraction', 0.), q.get('geometric_support_fraction', 0.),
              q.get('agreement_fraction', 0.), row.get('cycle_p95_px')]
    return np.array([float('inf') if i == 5 and v is None else (-1. if v is None else v) for i, v in enumerate(values)], float)


def accepts(row, policy):
    if policy.get('reject_all') or not row['estimated'] or not row.get('reverse_estimated'):
        return False
    g, p, lit, support, agreement, cycle = features(row)
    return bool(np.isfinite([g, p, lit, support, agreement, cycle]).all() and
        g >= policy['min_global'] and p >= policy['min_patch'] and lit >= policy['min_lit'] and
        support >= policy['min_support'] and agreement >= policy['min_agreement'] and cycle <= policy['max_cycle'])


def counts(rows, mask):
    truth = np.array([r['correct'] for r in rows], bool)
    negative = np.array([r['label_source'] == 'known_nonoverlap' for r in rows], bool)
    wrong = int((mask & ~truth).sum())
    accepted = int(mask.sum())
    return dict(cases=len(rows), correct_available=int(truth.sum()), returned=sum(r['estimated'] for r in rows),
        accepted_correct=int((mask & truth).sum()), rejected_correct=int((~mask & truth).sum()),
        accepted_incorrect=wrong, accepted_total=accepted,
        accepted_error_fraction=wrong/accepted if accepted else None,
        negative_cases=int(negative.sum()), accepted_nonoverlap=int((mask & negative).sum()))


def calibrate(rows, manifest):
    if not rows or any(r['split'] != 'dev' or r['label_source'] not in ['exact_geometry', 'known_nonoverlap'] or type(r['correct']) is not bool for r in rows):
        raise ValueError('Only known-geometry development evidence may calibrate a quality gate')
    keys = [(r['pair_id'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Repeated development outcome')
    dev = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'dev'}
    if any(r['tile_id'] not in dev or r.get('query_tile_id', r['tile_id']) not in dev for r in rows):
        raise ValueError('Non-development terrain in gate fitting')
    lookup, components, overlaps = geographic_groups(list(dev.values()))
    policies = {}
    grid_names = list(GRID)
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        x = np.array([features(r) for r in selected])
        base = np.array([r['estimated'] and r.get('reverse_estimated', False) for r in selected], bool)
        base &= np.isfinite(x).all(axis=1)
        truth = np.array([r['correct'] for r in selected], bool)
        negative = np.array([r['label_source'] == 'known_nonoverlap' for r in selected], bool)
        best = {risk: dict(policy=dict(reject_all=True), development_counts=counts(selected, np.zeros(len(selected), bool))) for risk in BUDGETS}
        examined = 0
        for values in itertools.product(*(GRID[name] for name in grid_names)):
            examined += 1
            g, p, lit, support, agreement, cycle = values
            mask = base & (x[:, 0] >= g) & (x[:, 1] >= p) & (x[:, 2] >= lit) & (x[:, 3] >= support) & (x[:, 4] >= agreement) & (x[:, 5] <= cycle)
            if (mask & negative).any():
                continue
            good, wrong = int((mask & truth).sum()), int((mask & ~truth).sum())
            for risk in BUDGETS:
                if wrong > risk*max(good+wrong, 1):
                    continue
                previous = best[risk]['development_counts']
                if good > previous['accepted_correct'] or (good == previous['accepted_correct'] and wrong < previous['accepted_incorrect']):
                    best[risk] = dict(policy=dict(zip(grid_names, values)), development_counts=counts(selected, mask))
        for value in best.values():
            scalar_mask = np.array([accepts(r, value['policy']) for r in selected])
            if counts(selected, scalar_mask) != value['development_counts']:
                raise ValueError('Scalar/vector confidence policy differs')
        policies[method] = dict(grid_candidates_evaluated=examined,
            empirical_accepted_error_budgets={str(risk): value for risk, value in best.items()})
    return dict(schema_version=1, selected_on_split='dev', policies=policies, grid=GRID,
        empirical_budgets=BUDGETS, zero_nonoverlap_acceptance_required_on_development=True,
        independent_development_components=len(components), development_component_members=components,
        overlapping_development_footprint_pairs=len(overlaps),
        selection='Maximize retained correct development alignments at each declared empirical accepted-error budget, with zero accepted development non-overlap pairs. First declared candidate wins remaining ties.',
        inference_evidence=['image-only structural correlation', 'local patch correlation', 'joint non-dark support proxy',
            'common valid-pixel support', 'independent predictor-family agreement', 'separately selected reverse-cycle error'],
        excluded_inference_inputs=['correctness', 'ground-truth motion', 'reference errors', 'template-probe residuals', 'maximum displacement'],
        warning='Development selection is not a held-out error bound or safety guarantee. The small number of geographic components limits confidence. Freeze final model/policy before fresh evaluation.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('development_run')
    p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    root, output = Path(args.development_run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.wait:
        print('Waiting for verified terminal development artifacts; no reserve evaluation is scheduled.', flush=True)
        while not (root/'COMPLETE.json').exists():
            time.sleep(2)
    complete = json.loads((root/'COMPLETE.json').read_text())
    if sha256(root/'results.json') != complete['results_sha256'] or sha256(root/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
        raise ValueError('Completed development artifacts changed')
    frozen = json.loads((root/'frozen_inputs.json').read_text())
    result = calibrate(json.loads((root/'results.json').read_text()), frozen['manifest'])
    result.update(frozen_utc=datetime.now(timezone.utc).isoformat(), development_results_sha256=complete['results_sha256'],
        development_inputs_sha256=complete['frozen_inputs_sha256'], gate_source_sha256=sha256(__file__),
        declared_before_fresh_geographic_evaluation=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({method: value['empirical_accepted_error_budgets'] for method, value in result['policies'].items()}), flush=True)
