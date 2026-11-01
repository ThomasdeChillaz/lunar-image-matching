"""Independent native geometry and aggregate replay for proposal diagnosis."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunar_geometry_v5 import correspondences_z
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import quarter_turn


def audit(root,output):
    root=Path(root);cv2.setNumThreads(1)
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    for path,digest in {**protocol['source_sha256'],**protocol['input_sha256']}.items():
        if sha256(path)!=digest:raise ValueError('Pinned diagnostic source/input differs')
    for source,digest in protocol['source_sha256'].items():
        if sha256(root/'source'/source)!=digest:raise ValueError('Executed snapshot differs')
    manifest_path=Path('reports/localization_training_data_v5.json')
    if sha256(manifest_path)!=protocol['manifest_sha256'] or protocol['calibration_or_test_pixels_loaded']:
        raise ValueError('Diagnostic manifest/scope differs')
    manifest=json.loads(manifest_path.read_text());tiles={t['tile_id']:t for t in manifest['tiles']};data={}
    for key,tile in tiles.items():
        if tile['split'] not in ('train','model_selection'):raise ValueError('Forbidden diagnostic pixels')
        with np.load(tile['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in ('depth_a','depth_b')}
    expected=set(protocol['case_ids']);records=[];replayed=0
    for step in protocol['steps']:
        path=root/f'proposals_step_{step}.json.gz'
        if sha256(path)!=summary['results_sha256'][str(step)]:raise ValueError('Proposal records changed')
        with gzip.open(path,'rt') as stream:rows=json.load(stream)
        if len(rows)!=4*len(expected) or {(r['pair_id'],r['quarter_turn']) for r in rows}!={(i,k) for i in expected for k in range(4)}:
            raise ValueError('Missing cases/orientation branches')
        for row in rows:
            tile=tiles[row['tile_id']];z=data[row['tile_id']];a,b=tile['cameras'];c=row['coordinates']
            p=np.asarray(c['reference_points_native']).reshape(-1,2);q=np.asarray(c['query_points_rotated_native']).reshape(-1,2)
            _,truth,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
            target=quarter_turn(truth,row['quarter_turn'],512)
            saved=np.asarray(c['projected_points_rotated_native']).reshape(-1,2)
            if visible.tolist()!=c['geometrically_visible'] or not np.allclose(saved[visible],target,rtol=0,atol=1e-5):
                raise ValueError('Independent proposal projection/visibility differs')
            error=np.linalg.norm(q[visible]-target,axis=1);center=np.rint(q)
            reachable=np.zeros(len(p),bool);reachable[visible]=(np.abs(target-center[visible])<=16).all(1)
            cells=np.floor(p[reachable]/512*4).astype(int).clip(0,3)
            correct={str(t):int((error<=t).sum()) for t in (1,2,4,8,12,16,24)}
            metrics=dict(returned=len(p),geometrically_visible=int(visible.sum()),correct_at_native_px=correct,
                geometrically_reachable_33px=int(reachable.sum()),reachable_4x4_cells=len(set(map(tuple,cells))),
                median_visible_error_px=None if not len(error) else float(np.median(error)))
            if reachable.tolist()!=c['reachable_33px'] or metrics.keys()!=row['metrics'].keys():raise ValueError('Reachability/schema differs')
            for name,value in metrics.items():
                if name=='median_visible_error_px' and value is not None:
                    if not np.isclose(value,row['metrics'][name],atol=1e-5,rtol=0):raise ValueError('Native median differs')
                elif row['metrics'][name]!=value:raise ValueError('Native proposal metric differs: '+name)
            replayed+=1
        records+=rows
    for saved in summary['summary']:
        part=[r for r in records if r['step']==saved['step'] and r['split']==saved['split']]
        ids={r['pair_id'] for r in part};likelihood=[r['bilinear_target_likelihood'] for r in part]
        n=sum(r['points'] for r in likelihood)
        oracle=[max(r['metrics']['geometrically_reachable_33px'] for r in part if r['pair_id']==i) for i in ids]
        expected_values=dict(pairs=len(ids),branches=len(part),returned=sum(r['metrics']['returned'] for r in part),
            reachable=sum(r['metrics']['geometrically_reachable_33px'] for r in part),
            correct_at12px=sum(r['metrics']['correct_at_native_px']['12'] for r in part),target_points=n,
            mean_target_nll=None if not n else sum(r['points']*r['mean_nll'] for r in likelihood if r['points'])/n,
            oracle_pairs_with_at_least6_reachable=sum(v>=6 for v in oracle),oracle_median_best_reachable=float(np.median(oracle)))
        if any(saved[k]!=v for k,v in expected_values.items()):raise ValueError('Proposal aggregate differs')
    result=dict(passed=True,records=replayed,independent_projection_visibility_and_window_replay=True,
        likelihood_verification='Recomputed weighted aggregates of recorded forward readouts; not an independent neural likelihood replay.',
        actual_matcher_parity=summary['actual_matcher_exact_replay'],summary_sha256=sha256(root/'summary.json'),
        source_sha256=sha256(__file__),scope='Diagnostic reproducibility only; oracle branch counts are not camera-pose performance.')
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.run,args.output)
