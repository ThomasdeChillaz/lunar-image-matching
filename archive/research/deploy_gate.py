"""Publish only frozen-gate accepted transforms from a completed prediction run.

Raw proposals stay in the research run. Correctness/error labels are not copied
to or consulted by this deployment-facing output.
"""
import argparse
import json
from pathlib import Path

from dataset import sha256
from research_quality_gate import accepts


def public_alignment(row,policy):
    accepted=accepts(row,policy)
    return dict(pair_id=row['pair_id'],accepted=accepted,
        transform=row['transform'] if accepted else None,
        reason='accepted_by_frozen_gate' if accepted else 'frozen_gate_rejected')


def export(run,gate,method,output):
    run,gate,output=Path(run),Path(gate),Path(output)
    if output.exists():
        raise FileExistsError(output)
    terminal=json.loads((run/'COMPLETE.json').read_text())
    rules=json.loads(gate.read_text())
    if sha256(run/'results.json')!=terminal['results_sha256'] or sha256(run/'frozen_inputs.json')!=terminal['frozen_inputs_sha256']:
        raise ValueError('Completed prediction bytes changed')
    if rules['development_results_sha256']!=terminal['results_sha256'] or rules['selected_on_split']!='dev':
        raise ValueError('This export requires the completed development run bound to this policy; it is not a fresh-test calibration path')
    policy=rules['policies'][method]['empirical_accepted_error_budgets']['0.0']['policy']
    rows=[r for r in json.loads((run/'results.json').read_text()) if r['method']==method]
    if not rows or len({r['pair_id'] for r in rows})!=len(rows):
        raise ValueError('Missing/duplicate method outcomes')
    public=[public_alignment(r,policy) for r in rows]
    output.mkdir(parents=True)
    (output/'alignments.json').write_text(json.dumps(public,indent=2)+'\n')
    provenance=dict(prediction_run=str(run),prediction_results_sha256=terminal['results_sha256'],
        gate_sha256=sha256(gate),method=method,policy=policy,cases=len(public),
        accepted=sum(r['accepted'] for r in public),alignments_sha256=sha256(output/'alignments.json'),
        warning='Development export demonstrating enforced rejection. This does not supply held-out reliability or verified navigation accuracy.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/terrain-blend-dev-v1')
    p.add_argument('--gate',default='runs/terrain-blend-dev-v1/quality_gates.json')
    p.add_argument('--method',default='expanded_terrain_global')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    export(args.run,args.gate,args.method,args.output)
