"""Independent pose-error/aggregate replay for the v4 solver diagnosis."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def audit(root,output):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if protocol['calibration_or_test_pixels_loaded']!=0:raise ValueError('Forbidden split access')
    for p,d in {**protocol['input_hashes'],**protocol['sources']}.items():
        if sha256(p)!=d:raise ValueError('Frozen input/source differs: '+p)
    if sha256(root/'results.json.gz')!=summary['results_sha256']:raise ValueError('Saved rows differ')
    manifest=json.loads(Path('reports/localization_training_data_v3.json').read_text())
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    with gzip.open(root/'results.json.gz','rt') as f:rows=json.load(f)
    groups=defaultdict(list)
    for r in rows:
        t=tiles[r['tile_id']];truth=np.asarray(t['cameras'][1]['cam2world'])
        correct=strict=False
        if r['pose'] is not None:
            pose=np.asarray(r['pose']);rotation=np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3].T@truth[:3,:3])-1)/2,-1,1)))
            translation=np.linalg.norm(pose[:3,3]-truth[:3,3])
            if not np.allclose([rotation,translation],[r['pose_error']['rotation_deg'],r['pose_error']['translation_m']],atol=1e-6):raise ValueError('Independent pose error differs')
            correct=rotation<=2 and translation<=.01*r['reference_range_m']
            strict=rotation<=.5 and translation<=.001*r['reference_range_m']
        if bool(correct)!=r['pose_correct'] or bool(strict)!=r['strict_correct']:raise ValueError('Pose label differs')
        groups[(r['stage'],r['model'],r['solver'],r['mode'])].append(r)
    for (stage,model,solver,mode),part in groups.items():
        ids={r['pair_id'] for r in part}
        if len(ids)!=len(part) or ids!=(set(protocol['selection_bank']) if stage=='selection' else expected):raise ValueError('Missing/repeated cases')
        if stage=='selection':
            original=next(s for s in summary['selection'] if s['config']==solver)['results'][model]
        elif stage=='expanded_development':original=summary['expanded_development'][model+'/'+solver]
        else:original=summary['oracle_controls'][model+'/'+mode]
        counts=dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_correct'] for r in part),estimated=sum(r['pose'] is not None for r in part))
        if any(original[k]!=v for k,v in counts.items()):raise ValueError('Aggregate differs')
    order={c['name']:i for i,c in enumerate(protocol['configs'])}
    winner=max(summary['selection'],key=lambda s:(sum(v['correct'] for v in s['results'].values()),s['results']['frozen_native']['correct'],-order[s['config']]))['config']
    if winner!=summary['selected_solver']:raise ValueError('Selection differs')
    regressions={'frozen_native':26,'balanced_20261101':27,'balanced_20261102':31,'balanced_20261103':36}
    for name,count in regressions.items():
        if summary['expanded_development'][name+'/frozen_epnp']['correct']!=count:raise ValueError('Frozen regression differs')
    result=dict(passed=True,scope='Development solver diagnosis only; oracle results are GT-assisted controls.',records=len(rows),groups=len(groups),
        selected_solver=winner,source_sha256=sha256(__file__),results_sha256=summary['results_sha256'],historical_counts_reproduced=regressions,
        note='Independent error/denominator/selection replay; solver behavior also covered by analytic 3D/outlier controls. No new fresh-test claim.')
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v4-solver-diagnosis')
    p.add_argument('--output',default='reports/localization_v4_solver_audit.json');a=p.parse_args();audit(a.run,a.output)
