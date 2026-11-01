"""Truth-dependent match-filter probes; never a deployable acceptance/model result."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose,score
from verify_axial_results_v5 import projected


def read(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve every oracle diagnostic')
    root.mkdir(parents=True);cv2.setNumThreads(1)
    original=Path('runs/localization-v11-surface-warp-development/results.json.gz')
    branches=Path('runs/localization-v14-missing-branch-diagnosis/results.json.gz')
    base=read(original);missing=read(branches)
    audit=json.loads(Path('reports/native_missing_branch_audit_v14.json').read_text())
    if not audit['passed']:raise ValueError('Audited complete missing-branch outputs required')
    manifest_path=Path('reports/localization_training_data_v5.json');manifest=json.loads(manifest_path.read_text())
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'};data={}
    for tile_id,tile in tiles.items():
        if sha256(tile['native_cache'])!=tile['native_cache_sha256']:raise ValueError('Native development data changed')
        with np.load(tile['native_cache'],allow_pickle=False) as z:data[tile_id]={k:z[k].copy() for k in ('depth_a','depth_b')}
    source={r['pair_id']:r for r in base};cases=[r for r in base if not r['pose_correct']]
    if len(base)!=72 or len(cases)!=18 or len(missing)!=60:raise ValueError('Historical baseline denominator differs')
    protocol=dict(scope='Query-truth-dependent match-filter and oracle-branch diagnosis. Not model performance, acceptance, a mathematical upper bound, or new validation.',
        baseline_sha256=sha256(original),missing_branch_sha256=sha256(branches),
        missing_audit_sha256=sha256('reports/native_missing_branch_audit_v14.json'),
        source_sha256={p:sha256(p) for p in ['diagnose_oracle_filter_v17.py','native_baseline_pose_v7.py',
            'verify_axial_results_v5.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py',
            'lunar_geometry_v5.py','lunar_geometry_v2.py','localization_diagnostics_v3.py']},
        manifest_sha256=sha256(manifest_path),thresholds_native_px=[3,4,8,12],
        candidates='All60 audited native-refined branches of15 missing poses plus final frozen surface coordinates for3 wrong returned poses.',
        inference_changes='None. Reuse exact saved image correspondences. Oracle filters use query truth, then refit unchanged3px AP3P.',
        oracle_preservation='Preserving54 known-correct baseline cases and selecting any correct filtered branch uses truth. Never report the resulting total as a measured model.',
        calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json',protocol);rows=[]
    for case in cases:
        tile=tiles[case['tile_id']];z=data[case['tile_id']];a,b=tile['cameras']
        candidates=[r for r in missing if r['pair_id']==case['pair_id']]
        if not candidates:candidates=[case]
        for candidate in candidates:
            p=np.asarray(candidate['reference_points_native']).reshape(-1,2)
            q=np.asarray(candidate['query_points_native']).reshape(-1,2)
            target,visible=projected(p,z['depth_a'],z['depth_b'],a,b)
            error=np.full(len(p),np.inf);error[visible]=np.linalg.norm(q[visible]-target[visible],axis=1)
            for threshold in protocol['thresholds_native_px']:
                use=error<=threshold;pose,info=propose_pose(p[use],q[use],z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'])
                rows.append(dict(pair_id=case['pair_id'],tile_id=case['tile_id'],quarter_turn=candidate['quarter_turn'],
                    oracle_threshold_native_px=threshold,originally_missing=case['estimated_cam2world'] is None,
                    oracle_kept=int(use.sum()),pose_info=info,**score(p[use],q[use],pose,z,tile['cameras'])))
    summary=[]
    for threshold in protocol['thresholds_native_px']:
        part=[r for r in rows if r['oracle_threshold_native_px']==threshold]
        gains=[r['pair_id'] for r in cases if any(v['pair_id']==r['pair_id'] and v['pose_correct'] for v in part)]
        summary.append(dict(oracle_threshold_native_px=threshold,additional_cases_with_any_correct_oracle_fit=len(gains),
            missing_cases_recovered=sum(source[i]['estimated_cam2world'] is None for i in gains),
            wrong_estimated_cases_recovered=sum(source[i]['estimated_cam2world'] is not None for i in gains),
            truth_preserved_plus_oracle_total=54+len(gains),recovered_pair_ids=gains))
    with gzip.open(root/'oracle_results.json.gz','wt') as stream:json.dump(rows,stream,allow_nan=False)
    result=dict(scope=protocol['scope'],eligible_baseline_pairs=72,frozen_correct=54,diagnosed_failed_pairs=18,
        candidate_sets=63,oracle_fits=len(rows),summary=summary,
        protocol_sha256=sha256(root/'protocol.json'),results_sha256=sha256(root/'oracle_results.json.gz'),
        limitation='A negative outcome only limits these particular filters/point sets. Positive oracle fits require unknowable query truth and oracle branch choice; neither is a proposed model or calibrated acceptance result.')
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();run(args.output)
