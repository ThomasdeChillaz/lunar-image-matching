"""Independent geometry/filter/fixed-estimator replay of truth-only probes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunar_geometry_v5 import correspondences_z
from lunargeo_dataset import write_new
from verify_surface_learning_v13 import read_rows,replay_pose


def audit(run,output):
    root=Path(run);cv2.setNumThreads(1)
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    for source,digest in protocol['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Oracle probe source changed')
    baseline_path=Path('runs/localization-v11-surface-warp-development/results.json.gz')
    branch_path=Path('runs/localization-v14-missing-branch-diagnosis/results.json.gz')
    if sha256(baseline_path)!=protocol['baseline_sha256'] or sha256(branch_path)!=protocol['missing_branch_sha256']:
        raise ValueError('Source point sets changed')
    if sha256(root/'oracle_results.json.gz')!=summary['results_sha256']:raise ValueError('Oracle records changed')
    base=read_rows(baseline_path);branches=read_rows(branch_path);candidates={(r['pair_id'],r['quarter_turn']):r for r in branches}
    for r in base:
        if r['estimated_cam2world'] is not None and not r['pose_correct']:candidates[(r['pair_id'],r['quarter_turn'])]=r
    manifest_path=Path('reports/localization_training_data_v5.json')
    if sha256(manifest_path)!=protocol['manifest_sha256'] or protocol['calibration_or_test_pixels_loaded']:
        raise ValueError('Oracle scope changed')
    manifest=json.loads(manifest_path.read_text());tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'};data={}
    for tile_id,tile in tiles.items():
        if sha256(tile['native_cache'])!=tile['native_cache_sha256']:raise ValueError('Native input changed')
        with np.load(tile['native_cache'],allow_pickle=False) as z:data[tile_id]={k:z[k].copy() for k in ('depth_a','depth_b')}
    rows=read_rows(root/'oracle_results.json.gz');seen=set()
    for row in rows:
        identity=(row['pair_id'],row['quarter_turn']);candidate=candidates[identity]
        tile=tiles[row['tile_id']];a,b=tile['cameras'];z=data[row['tile_id']]
        p=np.asarray(candidate['reference_points_native']).reshape(-1,2);q=np.asarray(candidate['query_points_native']).reshape(-1,2)
        _,target,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
        use=np.zeros(len(p),bool);use[visible]=np.linalg.norm(q[visible]-target,axis=1)<=row['oracle_threshold_native_px']
        if (not np.array_equal(p[use],np.asarray(row['reference_points_native']).reshape(-1,2))
                or not np.array_equal(q[use],np.asarray(row['query_points_native']).reshape(-1,2)) or int(use.sum())!=row['oracle_kept']):
            raise ValueError('Independent oracle filter differs')
        replay_pose(dict(row,geometry_history=[]),tile,z)
        key=(*identity,row['oracle_threshold_native_px'])
        if key in seen:raise ValueError('Duplicate oracle fit')
        seen.add(key)
    if seen!={(*key,t) for key in candidates for t in (3,4,8,12)}:raise ValueError('Incomplete oracle probe denominator')
    base_by_id={r['pair_id']:r for r in base}
    for saved in summary['summary']:
        part=[r for r in rows if r['oracle_threshold_native_px']==saved['oracle_threshold_native_px']]
        ids={r['pair_id'] for r in part if r['pose_correct']}
        if (ids!=set(saved['recovered_pair_ids']) or len(ids)!=saved['additional_cases_with_any_correct_oracle_fit']
                or 54+len(ids)!=saved['truth_preserved_plus_oracle_total']
                or sum(base_by_id[i]['estimated_cam2world'] is None for i in ids)!=saved['missing_cases_recovered']):
            raise ValueError('Oracle summary differs')
    result=dict(passed=True,oracle_candidate_sets=len(candidates),oracle_fits=len(rows),
        summary_sha256=sha256(root/'summary.json'),source_sha256=sha256(__file__),
        scope='Independent truth-dependent filter/pose replay only. These counts are not a deployable matcher result, acceptance policy or universal upper bound.')
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.run,args.output)
