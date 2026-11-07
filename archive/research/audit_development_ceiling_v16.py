"""Exact arithmetic limit of the current four-component raw-pose bootstrap."""
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def audit():
    surface_path = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    sift_path = Path('runs/localization-v5-axial-correction/results.json.gz')
    for path, proof in [(surface_path, 'reports/native_surface_audit_v11.json'),
                        (sift_path, 'reports/localization_v5_rescore_audit.json')]:
        evidence = json.loads(Path(proof).read_text())
        if not evidence['passed'] or evidence['results_sha256'] != sha256(path):
            raise ValueError('Audited frozen raw-pose sources required')
    with gzip.open(surface_path, 'rt') as stream:
        base = {r['pair_id']: r for r in json.load(stream)}
    with gzip.open(sift_path, 'rt') as stream:
        sift = {r['pair_id']: r for r in json.load(stream) if r['method'] == 'sift_native' and r['solver'] == 'ap3p_reclassify'}
    mixed = {i: sift[i] if sift[i]['pose'] is not None else base[i] for i in base}
    groups = sorted({r['component'] for r in base.values()}); stages = {}
    for name, rows in [('surface_raw', base), ('sift_surface_raw', mixed)]:
        counts = [sum(r['component'] == g for r in base.values()) for g in groups]
        correct = [sum(rows[i]['pose_correct'] for i in base if base[i]['component'] == g) for g in groups]
        draws = list(itertools.product(range(len(groups)), repeat=len(groups)))
        # Perfect raw poses form an arithmetic upper bound, not model output.
        gains = np.array([100*sum(counts[i]-correct[i] for i in draw)/sum(counts[i] for i in draw) for draw in draws])
        saturated = sum(n == c for n, c in zip(counts, correct))
        stages[name] = dict(components=[dict(component=g, cases=n, baseline_correct=c) for g, n, c in zip(groups, counts, correct)],
            saturated_components=saturated, exact_bootstrap_draws=len(draws),
            zero_gain_draws_for_perfect_oracle=int((gains == 0).sum()),
            zero_gain_probability_for_perfect_oracle=float((gains == 0).mean()),
            perfect_oracle_95_percentile_interval_pp=np.percentile(gains, [2.5, 97.5]).tolist(),
            strictly_positive_95_percentile_lower_bound_possible=False)
        if saturated != 2 or (gains == 0).sum() != 16 or len(draws) != 256 or np.percentile(gains, 2.5) != 0:
            raise ValueError('Expected arithmetic ceiling differs; inspect before claiming it')
    result = dict(passed=True, source_sha256=sha256(__file__), surface_source_sha256=sha256(surface_path),
        sift_source_sha256=sha256(sift_path), stages=stages,
        scope='Arithmetic diagnosis of the existing four-component unfiltered raw-pose percentile bootstrap only. Perfect predictions are hypothetical, not measured.',
        conclusion='Two of four components already have perfect raw baselines. Sixteen of256 component resamples use only those components, forcing even a perfect raw model to have a zero lower95percentile bound. Any imperfect model cannot exceed that bound.',
        limits='This is not an impossibility result for calibrated correct acceptance, different assumptions/intervals, new data, or lunar localization. Acceptance policies can change baseline component counts. Do not weaken uncertainty criteria or count lighting variants as independent terrain to evade this limit.',
        action='Continue development work as declared, and obtain a broader untouched evaluation for the full research claim. The fresh-data gate remains unmet.')
    write_new('reports/development_raw_ceiling_v16.json', result)
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    audit()
