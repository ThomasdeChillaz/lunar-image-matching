"""Uniform exact native positives; architecture, negatives and v7 inference unchanged."""
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

from compact_lunar import CompactMatcher
from dataset import sha256
from localization_diagnostics_v3 import point_metrics
from lunar_geometry_v2 import pose_error
from lunar_geometry_v5 import correspondences_z,median_reference_range_z
from lunargeo_dataset import write_new
from native_pipeline_v7 import infer
from native_refinement_v7 import NativeRefiner,VARIANTS,objective
from train_balanced_localization_v3 import load_bundle,save_torch
from native_sampling_v8 import prepare_uniform,training_batch
from verify_axial_results_v5 import projected

SOURCES=['train_native_refinement_v8.py','native_sampling_v8.py','native_refinement_v7.py','native_pipeline_v7.py',
    'train_balanced_localization_v3.py','localization_refiner_v3.py','compact_lunar.py',
    'contrastive_terrain.py','localization_diagnostics_v3.py','lunar_geometry_v5.py',
    'lunar_geometry_v2.py','localization_pose_refit_v3.py','localization_robust_pose_v4.py',
    'rotation_bank_pose_v4.py','prepare_localization_training_v3.py','verify_axial_results_v5.py']
SEEDS=(20261101,20261102,20261103)


def audit_labels(manifest,tiles,data):
    count=points=positive=0
    for tile in tiles.values():
        z=data[tile['tile_id']]
        if any((~np.isfinite(z[k])|(z[k]<=0)).any() for k in ('depth_a','depth_b')):
            raise ValueError('This frozen patch sampler requires full valid depth support')
    for c in manifest['cases']:
        t=tiles[c['tile_id']];z=data[c['tile_id']];key=c['key'];p=z[f'p{key}'];q=z[f'q{key}']
        target,visible=projected(p,z['depth_a'],z['depth_b'],*t['cameras'])
        if not np.array_equal(visible,z[f'visible{key}']) or not np.allclose(target[visible],z[f'target{key}'][visible],rtol=0,atol=4e-5):
            raise ValueError('Independent corrected training-label replay differs')
        if len(p) and ((np.rint(p)<13)|(np.rint(p)>=499)).any():raise ValueError('Reference patch boundary unsupported')
        if len(q) and ((np.rint(q)<32)|(np.rint(q)>=480)).any():raise ValueError('Query proposal/jitter patch boundary unsupported')
        count+=1;points+=len(p);positive+=int(visible.sum())
    return dict(cases=count,proposed_points=points,geometrically_visible=positive,depth_support='All26 native depth pairs finite/positive; fixed crop bounds verified.',
        labels='Independent axial projection/visibility replay; predictions choose sampling locations, never ground truth.')


def evaluate(model,coarse,manifest,tiles,data,bank_only):
    cases=[c for c in manifest['cases'] if c['split']=='model_selection' and (not bank_only or c['pair_id'] in manifest['selection_bank'])]
    rows=[];model.eval();features={}
    for c in cases:
        t=tiles[c['tile_id']];z=data[c['tile_id']];a,b=t['cameras'];ia,ib=2*c['light_a'],2*c['light_b']+1
        selected,attempts=infer(coarse,model,z['images'][ia],z['images'][ib],z['depth_a'],a['intrinsics'],a['cam2world'],b['intrinsics'],features,((c['tile_id'],ia),(c['tile_id'],ib)))
        branch=selected if selected is not None else attempts[0];p,q=branch['p'],branch['q'];pose=None if selected is None else selected['pose']
        error=None if pose is None else pose_error(pose,np.array(b['cam2world']))
        _,target,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
        distance=median_reference_range_z(z['depth_a'],a['intrinsics'])
        rows.append(dict(c,**point_metrics(q,target,visible),
            pose_correct=bool(error and error['rotation_deg']<=2 and error['translation_m']<=.01*distance),
            strict_pose_correct=bool(error and error['rotation_deg']<=.5 and error['translation_m']<=.001*distance),
            estimated_cam2world=None if pose is None else pose.tolist(),pose_error=error,reference_range_m=distance,
            reference_points_native=p.tolist(),query_points_native=q.tolist(),target_points_native=target.tolist(),visible=visible.tolist(),
            quarter_turn=branch['quarter_turn'],inlier_info=branch['info'],raw_branch_fallback=selected is None,
            supported_matches=branch['supported_matches'],no_match_count=branch['no_match_count'],point_reliability=branch['point_reliability'].tolist(),
            all_branches=[dict(quarter_turn=a['quarter_turn'],supported_matches=a['supported_matches'],no_match_count=a['no_match_count'],
                inliers=a['info']['inliers'],median_inlier_residual=a['median_inlier_residual'],estimated=a['pose'] is not None) for a in attempts],
            acceptance_policy=None,point_rejection='Categorical no-match bin wins against local peaks' if model.experiment_variant=='dustbin' else None))
    return dict(cases=len(rows),correct_poses=sum(r['pose_correct'] for r in rows),strict_poses=sum(r['strict_pose_correct'] for r in rows),
        estimated_poses=sum(r['estimated_cam2world'] is not None for r in rows),correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows),
        returned=sum(r['returned'] for r in rows),scorable=sum(r['scorable'] for r in rows),bank_only=bank_only),rows


def train(variant,seed,root,bundle,coarse,steps,interval):
    manifest,tiles,data=bundle;folder=root/f'{variant}-{seed}';folder.mkdir();(folder/'source').mkdir()
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=NativeRefiner(variant);model.initialize_encoder(coarse)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};save_torch(initial,folder/'initial.pt')
    config=dict(scope='Corrected axial-depth development native refinement; no final or calibrated acceptance claim.',variant=variant,seed=seed,steps=steps,interval=interval,
        trainable_parameters=sum(p.numel() for p in model.parameters()),total_parameters=28272+sum(p.numel() for p in model.parameters()),
        frozen_coarse_sha256=manifest['coarse_checkpoint_sha256'],source_sha256={p:sha256(p) for p in SOURCES},
        manifest_sha256=sha256('reports/localization_training_data_v5.json'),calibration_or_test_pixels_loaded=0,
        cpu_threads=2,optimizer='AdamW',lr=.0001,weight_decay=.0001,batch_size=8,
        sampling='Replace v7 positive slots with uniform training-component/reference-grid projected positives, exact-query-centered +/-12px jitter. Keep identical v7 negative distribution,8-example batches and sensor augmentation. No matcher-filtered positive requirement.',
        augmentations='Exposure/gamma/blur/noise/quantization on native patches. Physical relighting from released renders; no extra artificial orientation warp.',
        inference='Frozen coarse CNN plus four orientations, trained native encoder/head and unchanged reference-geometry AP3P estimator.',
        loss='Bilinear native-target categorical likelihood; dustbin variant adds an explicit no-match class for geometrically invalid local proposals.',
        no_match_initial_log_odds=-7.5 if variant=='dustbin' else None,
        selection='Fixed16-case unfiltered raw pose success; then2px point count; earlier step. Initial checkpoint eligible.',started_utc=datetime.now(timezone.utc).isoformat())
    if config['total_parameters']>=100000:raise ValueError('Combined parameter cap exceeded')
    write_new(folder/'frozen_config.json',config)
    for p in SOURCES:shutil.copy2(p,folder/'source'/p)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);history=[];kinds=Counter();positives=0;started=perf_counter()
    def validate(step,loss=None):
        counts,_=evaluate(model,coarse,manifest,tiles,data,True)
        history.append(dict(step=step,loss=loss,selection=counts,seconds=perf_counter()-started))
        print(variant,seed,'step',step,counts,'loss',loss,'seconds',round(perf_counter()-started,1),flush=True)
        return counts['correct_poses'],counts['correct_at_two_px'],-step
    best=validate(0);selected=0;save_torch(model.state_dict(),folder/'selected.pt')
    for step in range(1,steps+1):
        model.train();batch,categories,n=training_batch(manifest,tiles,data,rng,8);kinds.update(categories);positives+=n
        optimizer.zero_grad(set_to_none=True);local,no_match=model(batch[0],batch[1],batch[2])
        loss=objective(local,no_match,batch[3],batch[4],variant=='dustbin')
        if not torch.isfinite(loss):raise ValueError('Nonfinite native refinement objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0:
            save_torch(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),folder/'latest.pt')
            print(variant,seed,'gradient',step,'loss',round(float(loss),4),flush=True)
        if step%interval==0 or step==steps:
            key=validate(step,float(loss))
            if key>best:best=key;selected=step;save_torch(model.state_dict(),folder/'selected.pt')
    save_torch(model.state_dict(),folder/'final.pt');change=float(sum((v-initial[k]).square().sum() for k,v in model.state_dict().items()).sqrt())
    model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True));counts,rows=evaluate(model,coarse,manifest,tiles,data,False)
    with gzip.open(folder/'development_results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
    meta=dict(config,completed_steps=steps,selected_step=selected,selection_key=list(best),history=history,evaluation=counts,
        sampled_kinds=dict(kinds),sampled_examples=steps*8,positive_labels=positives,final_weight_change_l2=change,elapsed_seconds=perf_counter()-started)
    write_new(folder/'training.json',meta);write_new(folder/'COMPLETE.json',dict(checkpoint_sha256=sha256(folder/'selected.pt'),final_sha256=sha256(folder/'final.pt'),training_sha256=sha256(folder/'training.json'),results_sha256=sha256(folder/'development_results.json.gz')))
    return meta


def cohort(output,steps,interval,pilot=False,initial_control=False):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    bundle=load_bundle('reports/localization_training_data_v5.json');manifest,tiles,data=bundle
    if manifest.get('schema_version')!=5 or not manifest.get('train_selection_disjoint'):raise ValueError('Corrected disjoint development data required')
    if sha256('lunar_geometry_v5.py')!=manifest['geometry_source_sha256'] or sha256(manifest['split_audit'])!=manifest['split_audit_sha256']:raise ValueError('Geometry/split audit changed')
    diagnostic=Path('runs/localization-v5-axial-pose-controls.json');gate=json.loads(diagnostic.read_text())
    if not gate['passed'] or gate['manifest_sha256']!=sha256('reports/localization_training_data_v5.json'):raise ValueError('Corrected exact-pose gate required')
    uniform=prepare_uniform(tiles,data);labels=audit_labels(manifest,tiles,data);coarse=CompactMatcher(manifest['coarse_model']);variants=('dustbin',) if pilot else VARIANTS
    protocol=dict(scope='Uniform-positive native refinement development data ablation; original fresh-final gate remains unmet.',variants=list(variants),seeds=list(SEEDS),steps=steps,interval=interval,pilot=pilot,
        source_sha256={p:sha256(p) for p in SOURCES},manifest_sha256=sha256('reports/localization_training_data_v5.json'),
        uniform_supervision=uniform,label_audit=labels,exact_pose_gate_sha256=sha256(diagnostic),calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json',protocol)
    if initial_control:
        model=NativeRefiner('encoder_3d');model.initialize_encoder(coarse);counts,rows=evaluate(model,coarse,manifest,tiles,data,False)
        with gzip.open(root/'initial_results.json.gz','wt') as f:json.dump(rows,f,allow_nan=False)
        write_new(root/'initial_control.json',dict(scope='Initial native implementation equivalence control, not training.',evaluation=counts));print(json.dumps(counts));return
    results=[];started=perf_counter()
    for variant in variants:results.append(train(variant,SEEDS[0],root,bundle,coarse,steps,interval))
    winner=max(results,key=lambda r:tuple(r['selection_key']))['variant']
    if not pilot:
        for seed in SEEDS[1:]:results.append(train(winner,seed,root,bundle,coarse,steps,interval))
    summary=dict(scope=protocol['scope'],pilot=pilot,winner=winner,elapsed_seconds=perf_counter()-started,
        results=[{k:r[k] for k in ('variant','seed','completed_steps','selected_step','selection_key','evaluation','final_weight_change_l2')} for r in results])
    write_new(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--interval',type=int,default=500)
    p.add_argument('--pilot',action='store_true');p.add_argument('--initial-control',action='store_true');a=p.parse_args()
    if min(a.steps,a.interval)<1:raise ValueError('Positive optimizer budget required')
    cohort(a.output,a.steps,a.interval,a.pilot,a.initial_control)
