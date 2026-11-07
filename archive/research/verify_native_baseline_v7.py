"""Replay modern native controls and every common-reference pose measurement."""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from verify_coarse_training_v5 import replay_row


def audit(root,output):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text());control=json.loads((root/'controls.json').read_text())
    from modern_baselines_v2 import SOURCES
    family='romav2' if protocol['method']=='romav2' else 'loma'
    if provenance['revision']!=SOURCES[family]['revision'] or provenance['method']!=protocol['method'] or provenance['profile']!=protocol['profile']:
        raise ValueError('Pretrained author provenance differs')
    expected_weights=json.loads(Path('reports/compact_baseline_lock_v2.json').read_text())['modern'][protocol['method']]['weights']
    if any(provenance['weights'].get(p)!=h for p,h in expected_weights.items()):raise ValueError('Official weight fingerprint differs')
    for p,h in {**protocol['source_sha256'],**protocol['input_sha256']}.items():
        if sha256(p)!=h:raise ValueError('Native baseline source/input differs')
    p,q=[np.array(control[k]).reshape(-1,2) for k in ('reference_points_native','query_points_native')]
    error=np.linalg.norm(q-p-np.array(control['known_translation_native']),axis=1)
    if not control['passed'] or len(p)<6 or not np.isclose(np.median(error),control['median_native_error_px']) or np.median(error)>1:
        raise ValueError('Actual pretrained native-pixel control failed')
    pose=np.array(control['estimated_cam2world']);translation=float(np.linalg.norm(pose[:3,3]-[-.2,.1,0.]));rotation=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1))))
    if not np.allclose([translation,rotation],[control['translation_error_m'],control['rotation_error_deg']]) or translation>.05 or rotation>.5:
        raise ValueError('Native control camera pose differs')
    if not control['blank_empty'] or not control['invalid_mask_empty']:raise ValueError('Invalid control failed')
    manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text());tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'};data={}
    if set(protocol['case_ids'])!=expected or len(protocol['case_ids'])!=len(expected):raise ValueError('Protocol eligible cases differ')
    for key,t in tiles.items():
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('depth_a','depth_b')}
    rows=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    for row in rows:
        if row['tile_id'] not in tiles or row['component']!=tiles[row['tile_id']]['component']:raise ValueError('Development groups differ')
        replay_row(row,tiles[row['tile_id']],data[row['tile_id']])
        if not np.isfinite(row['elapsed_ms']) or row['elapsed_ms']<0:raise ValueError('Invalid timing')
    for method in (protocol['method'],'compact_orientation'):
        part=[r for r in rows if r['method']==method]
        if len(part)!=len(expected) or {r['pair_id'] for r in part}!=expected:raise ValueError('Incomplete common native comparison')
        replay=dict(cases=len(part),correct=sum(r['pose_correct'] for r in part),strict=sum(r['strict_pose_correct'] for r in part),
            estimated=sum(r['estimated_cam2world'] is not None for r in part),median_full_inference_ms=float(np.median([r['elapsed_ms'] for r in part])),
            inference_errors=sum(r['inference_error'] is not None for r in part))
        if replay!=summary['results'][method]:raise ValueError('Native baseline aggregate differs')
    result=dict(passed=True,scope='Common native development comparison; no final, calibrated acceptance or global SOTA claim.',
        records=len(rows),eligible_pairs=len(expected),method=protocol['method'],profile=protocol['profile'],
        actual_pretrained_native_control_replayed=True,source_sha256=sha256(__file__),summary_sha256=sha256(root/'summary.json'))
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.output)
