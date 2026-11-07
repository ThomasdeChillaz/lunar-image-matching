"""Replay real-photo control pixels/errors; preserve unknown repeat accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from confidence import diagnostics
from dataset import load_pair,sha256
from matchers import Match
from real_sensor_controls import real_image
from reserve_evaluation import array_hashes,metrics,validate_freeze
from research_geometry import motion,prepare
from research_quality_gate import accepts
from structural_refinement import AlignmentObjective


def verify(run):
    import torch
    from contrastive_terrain import ContrastiveTerrainMatcher
    run=Path(run)
    doc=json.loads((run/'frozen_protocol.json').read_text())
    terrain=validate_freeze(doc['terrain_protocol_path'])
    for name,digest in doc['source_sha256'].items():
        if sha256(name)!=digest:
            raise ValueError('Changed real-control source')
    terminal=json.loads((run/'COMPLETE.json').read_text())
    for name,key in [('results.json','results_sha256'),('repeat_diagnostics.json','diagnostics_sha256'),('summary.json','summary_sha256'),('frozen_protocol.json','freeze_sha256')]:
        if sha256(run/name)!=terminal[key]:
            raise ValueError('Completed real-sensor bytes changed')
    rows=json.loads((run/'results.json').read_text())
    expected={(c['pair_id'],m) for c in doc['cases'] for m in doc['methods']}
    index={(r['pair_id'],r['method']):r for r in rows}
    if len(index)!=len(rows) or set(index)!=expected or terminal['rows']!=len(rows):
        raise ValueError('Incomplete real-photo control coverage')
    assets={a['asset_id']:a for a in doc['manifest']['assets']}
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model=ContrastiveTerrainMatcher(doc['training'])
    gate_replay=[]
    images={}
    def image(name):
        if name not in images:
            images[name]=real_image(assets[name])
        return images[name]
    for case in doc['cases']:
        a,ma=image(case['asset_id'])
        negative=case['label_source']=='known_nonoverlap'
        if negative:
            if assets[case['query_asset_id']]['region_id']==assets[case['asset_id']]['region_id']:
                raise ValueError('Real negative is not a different scene')
            b,mb=image(case['query_asset_id'])
            points,targets=None,None
        else:
            b,mb,_,points,targets,_=prepare(a,a,ma,ma,case['pair_id'],case['geometry_family'])
            _,p=motion(case['pair_id'],case['geometry_family'])
            theta=np.deg2rad(p['angle_deg'])
            linear=p['scale']*np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            targets_analytic=(points-63.5)@linear.T+63.5+[p['dx'],p['dy']]
            if not np.allclose(targets,targets_analytic):
                raise ValueError('Known real-photo warp geometry differs')
        raw=index[(case['pair_id'],'terrain_raw')]
        public=index[(case['pair_id'],'terrain_deployed')]
        if public['transform']!=(raw['transform'] if public['gate_accepted'] else None):
            raise ValueError('Rejected real-photo proposal escaped the public gate')
        # Replay the separately inferred reverse direction; no truth is passed.
        cv2.setRNGSeed(20261030)
        reverse=model(b,a,mb,ma)
        raw_matrix=None if raw['transform'] is None else np.asarray(raw['transform'])
        q=AlignmentObjective(a,b,ma,mb).quality(raw_matrix)
        quality={} if q is None else dict(q,agreement_fraction=0.,agreeing_other_families=[])
        evidence=dict(estimated=raw_matrix is not None,selection_quality=quality,
            **diagnostics(Match(raw_matrix),reverse,a.shape))
        accepted=accepts(evidence,doc['policy'])
        if accepted!=public['gate_accepted'] or accepted!=raw['gate_accepted']:
            raise ValueError('Independently replayed real-control gate differs')
        gate_replay.append(dict(pair_id=case['pair_id'],accepted=accepted,evidence=evidence,
            reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist()))
        for method in doc['methods']:
            row=index[(case['pair_id'],method)]
            if any(row[k]!=v for k,v in case.items()) or row['input_array_sha256']!=array_hashes(a,b,ma,mb):
                raise ValueError('Changed real-photo case or pixels')
            matrix=row['transform']
            if negative or matrix is None:
                if row['correct'] or row['error_px'] is not None or row['rmse_px'] is not None:
                    raise ValueError('Invented error for a failed/negative real-photo control')
            else:
                m=np.asarray(matrix)
                residual=np.linalg.norm(points@m[:,:2].T+m[:,2]-targets_analytic,axis=1)
                if not np.isclose(np.median(residual),row['error_px']) or not np.isclose(np.sqrt(np.mean(residual**2)),row['rmse_px']) or row['correct']!=bool(np.median(residual)<=3):
                    raise ValueError('Incorrect real-photo known-warp metric')
    repeat=json.loads((run/'repeat_diagnostics.json').read_text())
    pairs={p['pair_id']:p for p in doc['repeat_pairs']}
    if len(repeat)!=len(pairs) or {r['pair_id'] for r in repeat}!=set(pairs) or terminal['repeat_diagnostics']!=len(repeat):
        raise ValueError('Incomplete real repeat diagnostic coverage')
    for row in repeat:
        if row['correct'] is not None or row['independent_error_px'] is not None or row['reference_uncertainty_px'] is not None or row['label_source']!='unverified_map':
            raise ValueError('Unverified map diagnostic became independent ground truth')
        pair=pairs[row['pair_id']]
        a,b,ma,mb,_=load_pair(pair,assets,'.',size=128)
        if row['input_array_sha256']!=array_hashes(a,b,ma,mb):
            raise ValueError('Real repeat inputs changed')
        q=AlignmentObjective(a,b,ma,mb).quality(row['transform'])
        quality={} if q is None else dict(q,agreement_fraction=0.,agreeing_other_families=[])
        if row['selection_quality']!=quality or row['gate_accepted']!=accepts(row,doc['policy']):
            raise ValueError('Real repeat image evidence or frozen gate differs')
    summary=json.loads((run/'summary.json').read_text())
    for method,value in summary['by_method'].items():
        if value!=metrics([r for r in rows if r['method']==method]):
            raise ValueError('Incorrect real-photo control summary')
    replay_path=run/'independent_gate_replay.json'
    if replay_path.exists():
        if json.loads(replay_path.read_text())!=gate_replay:
            raise ValueError('Saved gate replay differs')
    else:
        replay_path.write_text(json.dumps(gate_replay,indent=2)+'\n')
    return dict(control_cases=len(doc['cases']),control_rows=len(rows),repeat_diagnostics=len(repeat),
        native_pixels_reconstructed=True,known_warp_errors_recomputed=True,public_rejection_enforced=True,
        repeat_image_evidence_and_gate_replayed=True,repeat_accuracy_labels_remain_unknown=True,
        control_gate_evidence_replayed=True,separate_reverse_inference_repeated=True,
        independent_gate_replay_sha256=sha256(replay_path),
        source_regions=3,source_acquisitions=len(assets),calibration_on_real_data=False,
        results_sha256=terminal['results_sha256'],freeze_sha256=terminal['freeze_sha256'],
        warning=doc['accuracy_boundary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    target=Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result=verify(args.run)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
