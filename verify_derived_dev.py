"""Audit cached quality-gating/cascade provenance, choices and exact geometry."""
import argparse
import json
from pathlib import Path
import numpy as np

from candidate_selection import Candidate
from cascaded_selection import InferenceEvidence, select
from dataset import sha256
from verify_research_dev import verify as verify_geometry


def same_matrix(a, b):
    return (a is None and b is None) or (a is not None and b is not None and np.allclose(a, b, atol=1e-12, rtol=0))


def verify(run):
    run = Path(run)
    audit = verify_geometry(run)
    frozen = json.loads((run/'frozen_inputs.json').read_text())
    parent = Path(frozen['parent_blend_run'])
    terminal = json.loads((parent/'COMPLETE.json').read_text())
    if terminal != frozen['parent_completion'] or sha256(parent/'results.json') != terminal['results_sha256'] or sha256(parent/'frozen_inputs.json') != terminal['frozen_inputs_sha256']:
        raise ValueError('Parent blend identity differs')
    parent_design = json.loads((parent/'frozen_inputs.json').read_text())
    if parent_design['manifest'] != frozen['manifest'] or parent_design['cases'] != frozen['cases']:
        raise ValueError('Derived case plan differs')
    primary = {r['pair_id']: r for r in json.loads((parent/'results.json').read_text()) if r['method']=='blend_global'}
    predictions, cache = {}, {}
    for method, binding in parent_design['predictors'].items():
        source = Path(binding['source_run'])
        if str(source) not in cache:
            complete = json.loads((source/'COMPLETE.json').read_text())
            if sha256(source/'results.json') != complete['results_sha256'] or sha256(source/'frozen_inputs.json') != complete['frozen_inputs_sha256']:
                raise ValueError('Parent predictor changed')
            values = json.loads((source/'results.json').read_text())
            cache[str(source)] = {(r['pair_id'], r['method']): r for r in values}
        predictions[method] = cache[str(source)]
        if sha256(source/'frozen_inputs.json') != binding['frozen_inputs_sha256']:
            raise ValueError('Predictor source binding differs')
    rows = json.loads((run/'results.json').read_text())
    cascade = 'cascade_policies' in frozen
    for row in rows:
        p = primary[row['pair_id']]
        if cascade:
            fallback = predictions['self_similarity_search'][(row['pair_id'], 'self_similarity_search')]
            first = InferenceEvidence(Candidate('blend_global', 'blend', p['transform']), p['reverse_transform'], p['selection_quality'])
            second = InferenceEvidence(Candidate('self_similarity_search', 'structural', fallback['transform']),
                fallback['reverse_transform'], p['candidate_quality'].get('self_similarity_search', {}))
            chosen, stage = select(first, second, frozen['cascade_policies'][row['method']]['policy'])
            if stage != row['cascade_stage'] or not same_matrix(None if chosen is None else chosen.candidate.matrix, row['transform']):
                raise ValueError('Cascade choice differs from image-only evidence')
            if not same_matrix(None if chosen is None else chosen.reverse_matrix, row['reverse_transform']):
                raise ValueError('Cascade reverse differs')
        else:
            origin = predictions[row['method']][(row['pair_id'], row['method'])]
            if not same_matrix(origin['transform'], row['transform']) or origin['input_array_sha256'] != row['input_array_sha256']:
                raise ValueError('Cached predictor was changed')
            if row['selection_quality'] != p['candidate_quality'].get(row['method'], {}):
                raise ValueError('Candidate image quality differs')
    if cascade:
        if json.loads((run/'policies.json').read_text()) != frozen['cascade_policies']:
            raise ValueError('Cascade policy copies differ')
        for method, policy in frozen['cascade_policies'].items():
            selected = [r for r in rows if r['method']==method]
            actual = dict(accepted_correct=sum(r['correct'] for r in selected),
                accepted_incorrect=sum(r['estimated'] and not r['correct'] for r in selected),
                accepted_nonoverlap=sum(r['estimated'] and r['label_source']=='known_nonoverlap' for r in selected),
                primary_count=sum(r['cascade_stage']=='primary' for r in selected), fallback_count=sum(r['cascade_stage']=='fallback' for r in selected))
            if any(policy[k] != v for k, v in actual.items()):
                raise ValueError('Incorrect cascade counts')
    else:
        from research_quality_gate import calibrate
        complete = json.loads((run/'COMPLETE.json').read_text())
        if sha256(run/'quality_gates.json') != complete['quality_gates_sha256']:
            raise ValueError('Quality policies changed')
        if calibrate(rows, frozen['manifest']) != json.loads((run/'quality_gates.json').read_text()):
            raise ValueError('Development calibration cannot be reproduced')
    audit.update(derived_prediction_choices_replayed=len(rows), parent_blend_identity=True,
        image_feature_replay_dependency=str(parent/'audit.json'),
        note='Pixel/geometry audit and cached-choice replay. Full image-score replay is recorded separately by verify_blend_dev.py.')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = verify(args.run)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
