"""Independent native geometry metrics and common-estimator replay for MatchAnything."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose
from verify_coarse_training_v5 import replay_row


def audit(run,output):
    root=Path(run);cv2.setNumThreads(1)
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text());controls=json.loads((root/'controls.json').read_text())
    lock=json.loads(Path('matchanything_model_lock.json').read_text())
    if provenance['source_lock_sha256']!=sha256('matchanything_model_lock.json'):raise ValueError('Pretrained lock differs')
    if provenance['source_revision']!='6a7bcb589ec8da3a9e861e799122beaa5eba2193' or provenance['weight_sha256']!='867ed7a097f2224b167efe79966cf0a2c1c77a9b0a4a103bc3ff74c302b6017b':
        raise ValueError('Pinned pretrained source/weights differ')
    if provenance['profile']!=protocol['profile'] or provenance['native_input_size']!=512:raise ValueError('Native profile differs')
    for p,h in protocol['source_sha256'].items():
        if sha256(p)!=h or sha256(root/'source'/p)!=h:raise ValueError('Comparison source differs')
        if subprocess.check_output(['git','show',protocol['publication_commit']+':'+p])!=Path(p).read_bytes():raise ValueError('Published source differs')
    for p,h in protocol['input_sha256'].items():
        if sha256(p)!=h:raise ValueError('Native input differs')
    if sha256(root/'protocol.json')!=summary['protocol_sha256']:raise ValueError('Protocol changed')
    actual={p.name:sha256(p) for p in sorted((root/'cases').glob('*.json'))}
    if actual!=summary['case_sha256']:raise ValueError('Case fingerprints differ')
    p,q=[np.asarray(controls[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
    errors=np.linalg.norm(q-p-controls['known_translation_native'],axis=1)
    if not controls['passed'] or len(p)!=controls['matches'] or not np.isclose(np.median(errors),controls['median_native_error_px']):raise ValueError('Actual pretrained control differs')
    if np.median(errors)>1 or not controls['blank_empty'] or not controls['invalid_mask_empty']:raise ValueError('Pretrained control failed')
    pose=np.asarray(controls['estimated_cam2world']);translation=float(np.linalg.norm(pose[:3,3]-[-.2,.1,0]))
    angle=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1))))
    if not np.allclose([translation,angle],[controls['translation_error_m'],controls['rotation_error_deg']]) or translation>.05 or angle>.5:raise ValueError('Control pose differs')
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    if sha256(manifest_path)!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    cases={c['pair_id']:c for c in manifest['cases'] if c['split']=='model_selection' and (not protocol['bank_only'] or c['pair_id'] in manifest['selection_bank'])}
    if set(protocol['case_ids'])!=set(cases) or len(protocol['case_ids'])!=len(cases):raise ValueError('Eligibility differs')
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'};data={}
    for key,t in tiles.items():
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('depth_a','depth_b')}
    rows=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))];replayed=0
    for row in rows:
        if row['pair_id'] not in cases or any(row[k]!=v for k,v in cases[row['pair_id']].items()):raise ValueError('Case metadata differs')
        tile=tiles[row['tile_id']];z=data[row['tile_id']];replay_row(row,tile,z)
        if not np.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<=0:raise ValueError('Invalid elapsed time')
        p,q=[np.asarray(row[k],float).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
        a,b=tile['cameras'];pose,info=propose_pose(p,q,z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
        saved=row['estimated_cam2world']
        if (pose is None)!=(saved is None) or (pose is not None and not np.allclose(pose,saved,atol=1e-8,rtol=0)):raise ValueError('Common pose-estimator replay differs')
        if info!=row['pose_info']:raise ValueError('Common pose-info replay differs')
        replayed+=1
    for method in protocol['methods']:
        part=[r for r in rows if r['method']==method]
        if len(part)!=len(cases) or {r['pair_id'] for r in part}!=set(cases):raise ValueError('Incomplete or duplicated method denominator')
        result=dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),
            wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in part),
            median_full_inference_seconds=float(np.median([r['elapsed_seconds'] for r in part])),inference_errors=sum(r['inference_error'] is not None for r in part))
        if result!=summary['results'][method]:raise ValueError('Method aggregate differs')
    proof=dict(passed=True,records=replayed,eligible_pairs=len(cases),native_metric_and_common_pose_replay=True,
        actual_pretrained_control_replayed=True,profile=protocol['profile'],source_sha256=sha256(__file__),summary_sha256=sha256(root/'summary.json'),
        scope='Exposed development common-native comparison. Replay verifies metrics, source and poses, not independent pretrained implementation or fresh test accuracy.')
    write_new(output,proof);print(json.dumps(proof,indent=2));return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.run,args.output)
