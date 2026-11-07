"""Give every cached predictor the same development image-evidence gate grid.

No new matching, rendering or reserve reads. Quality measurements are the
image-only candidate scores already recorded by the completed blend experiment.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dataset import sha256
from research_quality_gate import calibrate


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    design = json.loads((source/'frozen_inputs.json').read_text())
    terminal = json.loads((source/'COMPLETE.json').read_text())
    if design['split'] != 'dev' or terminal['results_sha256'] != sha256(source/'results.json') or terminal['frozen_inputs_sha256'] != sha256(source/'frozen_inputs.json'):
        raise ValueError('Non-development or changed blend artifacts')
    scores = {r['pair_id']: r for r in json.loads((source/'results.json').read_text()) if r['method']=='blend_global'}
    rows, parents, cache = [], {}, {}
    for method, binding in design['predictors'].items():
        root = Path(binding['source_run'])
        if str(root) not in cache:
            complete = json.loads((root/'COMPLETE.json').read_text())
            if complete['results_sha256'] != sha256(root/'results.json') or complete['frozen_inputs_sha256'] != sha256(root/'frozen_inputs.json'):
                raise ValueError('Upstream results changed')
            cache[str(root)] = json.loads((root/'results.json').read_text())
            parents[str(root)] = complete
        if parents[str(root)]['frozen_inputs_sha256'] != binding['frozen_inputs_sha256']:
            raise ValueError('Upstream design binding differs')
        for row in cache[str(root)]:
            if row['method'] != method:
                continue
            scored = scores[row['pair_id']]
            if row['input_array_sha256'] != scored['input_array_sha256']:
                raise ValueError('Native pixels differ')
            rows.append(dict(row, selection_quality=scored['candidate_quality'].get(method, {}),
                reverse_selection_quality=scored['reverse_candidate_quality'].get(method, {})))
    if len(rows) != len(scores)*len(design['predictors']):
        raise ValueError('Incomplete candidate quality coverage')
    policies = calibrate(rows, design['manifest'])
    files = ['calibrate_predictor_quality.py', 'research_quality_gate.py', 'candidate_selection.py',
        'uncertainty_groups.py', 'confidence.py', 'geometry.py', 'dataset.py']
    frozen = dict(design, methods=list(design['predictors']), parent_blend_run=str(source), parent_completion=terminal,
        parent_predictor_completions=parents, source_sha256={p: sha256(p) for p in files},
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        status='Development-only identical image-evidence grid for every predictor. Missing image support is rejection evidence. No fresh-test guarantee or reserve access.')
    output.mkdir(parents=True)
    (output/'frozen_inputs.json').write_text(json.dumps(frozen, sort_keys=True, allow_nan=False))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'quality_gates.json').write_text(json.dumps(policies, indent=2))
    summary = []
    for family in sorted({r['geometry_family'] for r in rows}):
        for method in frozen['methods']:
            selected = [r for r in rows if r['method']==method and r['geometry_family']==family]
            summary.append(dict(geometry_family=family, method=method, cases=len(selected), returned=sum(r['estimated'] for r in selected), correct=sum(r['correct'] for r in selected)))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        frozen_inputs_sha256=sha256(output/'frozen_inputs.json'), quality_gates_sha256=sha256(output/'quality_gates.json'))))
    print(json.dumps({m: v['empirical_accepted_error_budgets'] for m, v in policies['policies'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend-run', default='runs/blend-dev-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.blend_run, args.output)
