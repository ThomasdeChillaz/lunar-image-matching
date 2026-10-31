"""Orientation-regime ablation: same corrected geometry and fixed v5 inference."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch

from coarse_training_v5 import build_supervision,objective
from coarse_training_v6 import sample_training
from compact_lunar import CompactMatcher
from dataset import sha256
from localization_diagnostics_v3 import point_metrics
from lunar_geometry_v2 import pose_error
from lunar_geometry_v5 import correspondences_z as correspondences,median_reference_range_z
from coarse_pipeline_v5 import infer
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle,save_torch

SOURCES=['train_coarse_localization_v6.py','coarse_training_v6.py','coarse_training_v5.py','coarse_pipeline_v5.py','lunar_geometry_v5.py','rotation_bank_pose_v4.py','localization_robust_pose_v4.py',
    'localization_pose_refit_v3.py','localization_diagnostics_v3.py','lunar_geometry_v2.py',
    'compact_lunar.py','contrastive_terrain.py','train_balanced_localization_v3.py','prepare_localization_training_v3.py']
SEEDS=(20261101,20261102,20261103)
VARIANTS=('true_3d','canonical_3d')


def evaluate(matcher,manifest,tiles,data,fine_model,features,bank_only=True):
    cases=[c for c in manifest['cases'] if c['split']=='model_selection' and
           (not bank_only or c['pair_id'] in manifest['selection_bank'])]
    rows=[];matcher.model.eval()
    for c in cases:
        t=tiles[c['tile_id']];z=data[c['tile_id']];a,b=t['cameras'];ia,ib=2*c['light_a'],2*c['light_b']+1
        selected,attempts=infer(matcher,fine_model,z['images'][ia],z['images'][ib],z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'],features,((c['tile_id'],ia),(c['tile_id'],ib)))
        if selected is None:
            p,q=attempts[0]['p'],attempts[0]['q'];pose=None;info=dict(inliers=0,reason='no_valid_orientation');rotation=0
        else:
            p,q,pose,info=selected['p'],selected['q'],selected['pose'],selected['info'];rotation=selected['quarter_turn']
        # Scoring below is the first query-pose/depth access during inference.
        error=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
        _,target,_,visible=correspondences(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
        distance=median_reference_range_z(z['depth_a'],a['intrinsics'])
        rows.append(dict(c,**point_metrics(q,target,visible),pose_correct=bool(error and error['rotation_deg']<=2 and error['translation_m']<=.01*distance),
            strict_pose_correct=bool(error and error['rotation_deg']<=.5 and error['translation_m']<=.001*distance),
            estimated_cam2world=None if pose is None else pose.tolist(),pose_error=error,reference_range_m=distance,
            reference_points_native=p.tolist(),query_points_native=q.tolist(),quarter_turn=rotation,
            target_points_native=target.tolist(),visible=visible.tolist(),inlier_info=info,
            acceptance_policy=None,raw_branch_fallback=selected is None))
    counts=dict(cases=len(rows),correct_poses=sum(r['pose_correct'] for r in rows),strict_poses=sum(r['strict_pose_correct'] for r in rows),
                estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),
                correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
                returned=sum(r['returned'] for r in rows),scorable=sum(r['scorable'] for r in rows),bank_only=bank_only)
    return counts,rows


def train(variant,seed,steps,interval,root,bundle,supervision,supervision_summary,fine_model,features):
    output=Path(root)/f'{variant}-{seed}';output.mkdir();(output/'source').mkdir()
    manifest,tiles,data=bundle;torch.manual_seed(seed);rng=np.random.default_rng(seed)
    matcher=CompactMatcher(manifest['coarse_model']);initial={k:v.detach().clone() for k,v in matcher.model.state_dict().items()}
    config=dict(scope='Orientation-regime development ablation with corrected axial-depth geometry; no fresh final test.',variant=variant,seed=seed,
        steps=steps,interval=interval,architecture='Original CompactDescriptor(expanded),128px; fixed frozen native descriptor refinement and four-orientation bank.',
        trainable_parameters=sum(p.numel() for p in matcher.model.parameters()),
        total_parameters=sum(p.numel() for p in matcher.model.parameters())+28272,
        frozen_fine_checkpoint_sha256=manifest['coarse_checkpoint_sha256'],
        source_sha256={p:sha256(p) for p in SOURCES},manifest_sha256=sha256('reports/localization_training_data_v5.json'),
        training_supervision=supervision_summary,calibration_or_test_pixels_loaded=0,
        optimizer='AdamW',lr=.0001,weight_decay=.0001,cpu_threads=2,selection='Fixed16-case bank raw pose success; then2px point count; earliest step.',
        query_augmentation=('True3D control: uncanonicalized, uniform +/-90deg.' if variant=='true_3d' else 'Canonical3D: training-only GT quarter-turn canonicalization plus uniform +/-15deg.')+' Both: log-uniform0.7..1.4scale, +/-8px shift and sensor appearance augmentation.',
        geometry_supervision='Both use identical uniform audited axial-depth/camera 3D correspondence targets. Only the training orientation regime changes.',
        started_utc=datetime.now(timezone.utc).isoformat())
    write_new(output/'frozen_config.json',config)
    for p in SOURCES:shutil.copy2(p,output/'source'/p)
    save_torch(initial,output/'initial.pt');optimizer=torch.optim.AdamW(matcher.model.parameters(),lr=.0001,weight_decay=.0001)
    history=[];counts_by_tile=Counter();orientation_counts=Counter();point_labels=0;started=perf_counter()
    def validate(step,loss=None):
        counts,_=evaluate(matcher,manifest,tiles,data,fine_model,features,True)
        history.append(dict(step=step,loss=loss,selection=counts,seconds=perf_counter()-started))
        print(variant,seed,'step',step,counts,'loss',loss,'seconds',round(perf_counter()-started,1),flush=True)
        return counts['correct_poses'],counts['correct_at_two_px'],-step
    best=validate(0);selected=0;save_torch(matcher.model.state_dict(),output/'selected.pt')
    for step in range(1,steps+1):
        matcher.model.train();images,p,q,sample=sample_training(variant,manifest,tiles,supervision,rng)
        optimizer.zero_grad(set_to_none=True);loss=objective(matcher.model,images,p,q)
        if not torch.isfinite(loss):raise ValueError('Nonfinite 3D contrastive loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(matcher.model.parameters(),5.);optimizer.step()
        counts_by_tile[sample['tile_id']]+=1;point_labels+=sample['points']
        orientation_counts[str(sample.get('training_quarter_turn','uncanonicalized'))]+=1
        if step%100==0:
            save_torch(dict(step=step,model=matcher.model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),output/'latest.pt')
            print(variant,seed,'gradient',step,'loss',round(float(loss),4),flush=True)
        if step%interval==0 or step==steps:
            key=validate(step,float(loss))
            if key>best:best=key;selected=step;save_torch(matcher.model.state_dict(),output/'selected.pt')
    save_torch(matcher.model.state_dict(),output/'final.pt')
    change=float(sum((v-initial[k]).square().sum() for k,v in matcher.model.state_dict().items()).sqrt())
    matcher.model.load_state_dict(torch.load(output/'selected.pt',map_location='cpu',weights_only=True))
    counts,rows=evaluate(matcher,manifest,tiles,data,fine_model,features,False)
    with gzip.open(output/'development_results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    meta=dict(config,completed_steps=steps,selected_step=selected,selection_key=list(best),history=history,
              evaluation=counts,sampled_tiles=dict(counts_by_tile),supervised_points=point_labels,
              training_orientation_counts=dict(orientation_counts),
              final_weight_change_l2=change,elapsed_seconds=perf_counter()-started)
    write_new(output/'training.json',meta)
    write_new(output/'COMPLETE.json',dict(checkpoint_sha256=sha256(output/'selected.pt'),final_sha256=sha256(output/'final.pt'),
        training_sha256=sha256(output/'training.json'),results_sha256=sha256(output/'development_results.json.gz')))
    return meta


def cohort(output,steps,interval,pilot=False):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    bundle=load_bundle('reports/localization_training_data_v5.json')
    manifest,tiles,data=bundle
    if manifest.get('schema_version')!=5 or manifest.get('train_selection_disjoint') is not True:
        raise ValueError('Audited corrected axial-depth manifest and disjoint split required')
    if manifest['geometry_source_sha256']!=sha256('lunar_geometry_v5.py') or sha256(manifest['split_audit'])!=manifest['split_audit_sha256']:
        raise ValueError('Corrected geometry/split audit differs')
    diagnostic_path=Path('runs/localization-v5-axial-pose-controls.json')
    diagnostic=json.loads(diagnostic_path.read_text())
    if not diagnostic['passed'] or diagnostic['manifest_sha256']!=sha256('reports/localization_training_data_v5.json'):
        raise ValueError('Corrected exact-pose diagnostic gate required')
    if diagnostic['results']['exact']['correct']/diagnostic['results']['exact']['cases']<.95:
        raise ValueError('Exact-pose diagnostic success below95%')
    supervision,stats=build_supervision(manifest,tiles,data)
    variants=('canonical_3d',) if pilot else VARIANTS
    protocol=dict(scope='Orientation-regime development tuning; original fresh-final gate still applies.',steps=steps,interval=interval,
        variants=list(variants),seeds=list(SEEDS),pilot=pilot,source_sha256={p:sha256(p) for p in SOURCES},
        manifest_sha256=sha256('reports/localization_training_data_v5.json'),training_supervision=stats,
        selection_rule='Fixed bank raw pose count, then2px point count, then earlier step; ablation-order tie break.',
        selected_solver='ap3p_reclassify',depth_convention='optical_axis_z',rotation_bank=[0,1,2,3],calibration_or_test_pixels_loaded=0,
        exact_pose_diagnostic=str(diagnostic_path),exact_pose_diagnostic_sha256=sha256(diagnostic_path))
    write_new(root/'protocol.json',protocol)
    fine=CompactMatcher(manifest['coarse_model']).model.eval();features={};results=[];started=perf_counter()
    for variant in variants:results.append(train(variant,SEEDS[0],steps,interval,root,bundle,supervision,stats,fine,features))
    winner=max(results,key=lambda r:tuple(r['selection_key']))['variant']
    if not pilot:
        for seed in SEEDS[1:]:results.append(train(winner,seed,steps,interval,root,bundle,supervision,stats,fine,features))
    result=dict(scope=protocol['scope'],winner=winner,pilot=pilot,elapsed_seconds=perf_counter()-started,
        results=[{k:r[k] for k in ('variant','seed','completed_steps','selected_step','selection_key','evaluation','final_weight_change_l2')} for r in results])
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=4000);p.add_argument('--interval',type=int,default=1000);p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.interval)<1:raise ValueError('Positive training budget required')
    cohort(a.output,a.steps,a.interval,a.pilot)
