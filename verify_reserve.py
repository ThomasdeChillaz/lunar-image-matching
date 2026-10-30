"""Independent replay of reserve pixels, analytic geometry and frozen decisions."""
import argparse
import json
from pathlib import Path

import numpy as np

from confidence import diagnostics
from dataset import sha256
from matchers import Match
from reserve_evaluation import array_hashes, inputs, summarize, validate_freeze
from research_geometry import motion
from research_quality_gate import accepts
from structural_refinement import AlignmentObjective


def verify(run):
    run = Path(run)
    protocol = run/'frozen_protocol.json'
    doc = validate_freeze(protocol)
    terminal = json.loads((run/'COMPLETE.json').read_text())
    if terminal['protocol_sha256'] != sha256(protocol) or terminal['results_sha256'] != sha256(run/'results.json') or terminal['summary_sha256'] != sha256(run/'summary.json'):
        raise ValueError('Completed reserve bytes changed')
    rows = json.loads((run/'results.json').read_text())
    cases = {c['pair_id']:c for c in doc['cases']}
    expected = {(p,m) for p in cases for m in doc['methods']}
    if len(rows)!=len(expected) or {(r['pair_id'],r['method']) for r in rows}!=expected or terminal['rows']!=len(rows):
        raise ValueError('Incomplete/duplicate reserve outcomes')
    tiles = {t['tile_id']:t for t in doc['manifest']['tiles'] if t['split'].startswith('reserve_')}
    images, current_pair, saved = {}, None, None
    for row in rows:
        case = cases[row['pair_id']]
        if any(row[k]!=v for k,v in case.items()) or row['freeze_sha256']!=sha256(protocol):
            raise ValueError('Changed reserve case/freeze binding')
        if current_pair!=row['pair_id']:
            a,b,ma,mb,points,targets = inputs(case,tiles,images)
            saved = a,b,ma,mb,points,targets
            current_pair = row['pair_id']
        a,b,ma,mb,points,targets = saved
        if array_hashes(a,b,ma,mb)!=row['input_array_sha256']:
            raise ValueError('Native reserve inputs changed')
        negative = case['geometry_family']=='nonoverlap_negative'
        if row['label_source']!=('known_nonoverlap' if negative else 'exact_geometry'):
            raise ValueError('Incorrect reserve label provenance')
        matrix = row['transform']
        if negative or matrix is None:
            if row['correct'] or row['error_px'] is not None or row['rmse_px'] is not None:
                raise ValueError('Failed/negative alignment has an accuracy label')
        else:
            _, parameters = motion(case['pair_id'],case['geometry_family'])
            theta = np.deg2rad(parameters['angle_deg'])
            linear = parameters['scale']*np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            center = np.array([63.5,63.5])
            targets_analytic = (points-center)@linear.T+center+[parameters['dx'],parameters['dy']]
            if not np.allclose(targets,targets_analytic):
                raise ValueError('Analytic geometry differs from case generation')
            residual = np.linalg.norm(points@np.asarray(matrix)[:,:2].T+np.asarray(matrix)[:,2]-targets_analytic,axis=1)
            if not np.isclose(np.median(residual),row['error_px']) or not np.isclose(np.sqrt(np.mean(residual**2)),row['rmse_px']) or row['correct']!=bool(np.median(residual)<=doc['threshold_px']):
                raise ValueError('Incorrect analytic reserve error')
        if row['estimated']!=(matrix is not None):
            raise ValueError('Wrong fit availability')
        if row['method']!='sift':
            raw, reverse = row['raw_transform'],row['reverse_transform']
            q = AlignmentObjective(a,b,ma,mb).quality(raw)
            quality = {} if q is None else dict(q,agreement_fraction=0.,agreeing_other_families=[])
            evidence = row['gate_evidence']
            if quality!=evidence['selection_quality'] or evidence['estimated']!=(raw is not None):
                raise ValueError('Image evidence differs from frozen inference')
            replay = diagnostics(Match(None if raw is None else np.asarray(raw)),Match(None if reverse is None else np.asarray(reverse)),a.shape)
            for key in ('cycle_p95_px','reverse_estimated','cycle_median_px'):
                if replay[key]!=evidence[key]:
                    raise ValueError('Independent reverse-cycle evidence differs')
            accepted = accepts(evidence,doc['policy'])
            if accepted!=row['gate_accepted']:
                raise ValueError('Changed frozen gate decision')
            expected_transform = raw if row['method']=='terrain_raw' or accepted else None
            if matrix!=expected_transform:
                raise ValueError('Rejected proposal leaked into public transform')
    summary = summarize(rows,doc)
    if summary!=json.loads((run/'summary.json').read_text()):
        raise ValueError('Reserve summary differs from exhaustive replay')
    return dict(rows=len(rows),cases=len(cases),native_input_reconstruction=True,
        analytic_geometry_recomputed=True,frozen_gate_replayed=True,public_rejection_enforced=True,
        calibration_or_model_selection_on_reserves=False,components=len(doc['components']),
        results_sha256=terminal['results_sha256'],freeze_sha256=sha256(protocol),
        warning='Known-geometry within-Tycho experiment; does not verify real repeat-observation navigation accuracy.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result=verify(args.run)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
