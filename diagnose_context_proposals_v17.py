"""Diagnose global proposal changes without adapting any model or pose rule.

Images/descriptors alone enter inference. Audited camera/depth truth enters
the separate score_proposals function, never proposal selection. Ground-truth
best branches are explicitly oracle diagnostics, not returned camera poses.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import make_context, proposal_grid
from contrastive_terrain import sample_descriptors
from dataset import sha256
from global_context_v15 import ContextMatcher
from localization_diagnostics_v3 import native_points
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import quarter_turn
from train_balanced_localization_v3 import load_bundle
from verify_axial_results_v5 import projected

SOURCES = ['diagnose_context_proposals_v17.py','compact_lunar.py','contrastive_terrain.py',
    'context_supervision_v15.py','global_context_v15.py','localization_diagnostics_v3.py',
    'rotation_bank_pose_v4.py','verify_axial_results_v5.py','train_balanced_localization_v3.py',
    'prepare_localization_training_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py']


def infer_proposals(coarse, model, reference128, query128):
    """Actual matching control plus full score matrix; no geometry arguments."""
    grid = proposal_grid(); points = torch.from_numpy(grid)[None]
    images = torch.from_numpy(np.stack([reference128,query128])).float()[:,None]/255
    with torch.inference_mode():
        dense,_ = coarse.model(images)
        a,b = sample_descriptors(dense[:1],points),sample_descriptors(dense[1:],points)
        scores = model(a,b,points,points)[0]
        top,index = scores.max(1); reverse = scores.argmax(0)
        keep = (reverse[index] == torch.arange(len(grid))) & (top >= .5)
        if min(reference128.std(),query128.std())<1:
            keep[:]=False
        ia = torch.where(keep)[0];ia=ia[torch.argsort(top[ia],descending=True)[:512]];ib=index[ia]
        p,q = grid[ia.numpy()],grid[ib.numpy()]
    actual_p,actual_q=ContextMatcher(coarse,model).points(reference128,query128)
    if not np.array_equal(p,actual_p) or not np.array_equal(q,actual_q):
        raise ValueError('Diagnostic proposals differ from the actual matcher')
    return p,q,scores,ia.numpy(),top[ia].numpy()


def score_proposals(p128,q128,truth_all_native,visible_all,indices,rotation):
    p=native_points(p128,(512,512));q=native_points(q128,(512,512))
    visible=visible_all[indices]
    truth=quarter_turn(truth_all_native[indices],rotation,512)
    errors=np.full(len(p),np.inf);errors[visible]=np.linalg.norm(q[visible]-truth[visible],axis=1)
    # refine_native searches integer centers round(q), then a33x33 stencil.
    # Darkness is not occlusion. Reachability is a geometric window property,
    # not a guarantee that the descriptor refinement will find the point.
    center=np.rint(q);reachable=visible & (np.abs(truth-center)<=16).all(1)
    cells=np.floor(p[reachable]/512*4).astype(int).clip(0,3)
    metrics=dict(returned=len(p),geometrically_visible=int(visible.sum()),
        correct_at_native_px={str(t):int((errors<=t).sum()) for t in (1,2,4,8,12,16,24)},
        geometrically_reachable_33px=int(reachable.sum()),reachable_4x4_cells=len(set(map(tuple,cells))),
        median_visible_error_px=None if not visible.any() else float(np.median(errors[visible])))
    return metrics,dict(reference_points_native=p.tolist(),query_points_rotated_native=q.tolist(),
        projected_points_rotated_native=np.where(visible[:,None],truth,0).tolist(),
        geometrically_visible=visible.tolist(),reachable_33px=reachable.tolist())


def target_likelihood(scores,truth_native,visible,rotation):
    target=(quarter_turn(truth_native,rotation,512)+.5)/4-.5
    use=visible & ((target>=12)&(target<=112)).all(1)
    if not use.any():
        return dict(points=0,mean_nll=None,mean_true_score=None,mean_best_other_score=None)
    target=torch.from_numpy(target[use].astype(np.float32));xy=(target-12)/4
    low=xy.floor().long();high=(low+1).clamp(max=25);fraction=xy-low
    s=scores[torch.from_numpy(use)];logp=(s/.1).log_softmax(-1)
    nll=torch.zeros(len(s));true_score=torch.zeros(len(s))
    for x,y,w in [(low[:,0],low[:,1],(1-fraction[:,0])*(1-fraction[:,1])),
        (high[:,0],low[:,1],fraction[:,0]*(1-fraction[:,1])),
        (low[:,0],high[:,1],(1-fraction[:,0])*fraction[:,1]),
        (high[:,0],high[:,1],fraction[:,0]*fraction[:,1])]:
        index=(y*26+x)[:,None];nll-=w*logp.gather(1,index)[:,0];true_score+=w*s.gather(1,index)[:,0]
    # Exclude all four nearest target candidates when describing distractors.
    other=s.clone();rows=torch.arange(len(s))
    for x,y in [(low[:,0],low[:,1]),(high[:,0],low[:,1]),(low[:,0],high[:,1]),(high[:,0],high[:,1])]:
        other[rows,y*26+x]=-torch.inf
    return dict(points=len(s),mean_nll=float(nll.mean()),mean_true_score=float(true_score.mean()),mean_best_other_score=float(other.max(1).values.mean()))


def fixtures(manifest,tiles):
    # Every exposed selection pair; training uses the first tile per component
    # and all nine released light pairs, chosen by metadata before inference.
    selected=[c.copy() for c in manifest['cases'] if c['split']=='model_selection']
    components={}
    for tile_id,tile in sorted(tiles.items()):
        if tile['split']=='train':components.setdefault(tile['component'],tile_id)
    training=[]
    for component,tile_id in sorted(components.items()):
        for a in range(3):
            for b in range(3):
                training.append(dict(pair_id=tile_id+f'/lights-{a}-{b}',tile_id=tile_id,component=component,
                    light_a=a,light_b=b,split='train'))
    return training+selected


def run(output,cohort,variant,steps):
    root,cohort=Path(output),Path(cohort)
    if root.exists():raise FileExistsError('Preserve every diagnostic attempt')
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((cohort/'protocol.json').read_text());folder=cohort/f'{variant}-20261101'
    complete=json.loads((folder/'COMPLETE.json').read_text())
    for path,digest in complete['sha256'].items():
        if sha256(folder/path)!=digest:raise ValueError('Completed cohort evidence changed')
    for source,digest in protocol['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Pinned fitting source changed')
    manifest_path=Path('reports/localization_training_data_v5.json');bundle=load_bundle(manifest_path)
    manifest,tiles,data=bundle
    if sha256(manifest_path)!=protocol['manifest_sha256'] or manifest['calibration_or_test_pixels_loaded']:
        raise ValueError('Development manifest changed')
    cases=fixtures(manifest,tiles);coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    root.mkdir(parents=True);(root/'source').mkdir()
    frozen=dict(scope='Development-only proposal failure diagnosis; no fitting, pose selection change, calibrated acceptance or fresh validation.',
        variant=variant,seed=20261101,steps=steps,case_ids=[c['pair_id'] for c in cases],
        train_fixture_geometries=len({c['tile_id'] for c in cases if c['split']=='train'}),
        train_fixture_components=len({c['component'] for c in cases if c['split']=='train'}),
        selection_pairs=sum(c['split']=='model_selection' for c in cases),
        source_sha256={p:sha256(p) for p in SOURCES},manifest_sha256=sha256(manifest_path),
        cohort_protocol_sha256=sha256(cohort/'protocol.json'),
        checkpoint_sha256={str(s):sha256(folder/f'checkpoint_step_{s}.pt') for s in steps},
        input_sha256={t['native_cache']:t['native_cache_sha256'] for t in tiles.values()},
        geometry_truth='Only the separate proposal scoring/likelihood diagnosis; never infer_proposals.',
        quarter_turns=[0,1,2,3],cpu_threads=1,calibration_or_test_pixels_loaded=0,
        oracle='Max genuine matches across orientations uses scoring truth and is a labelled oracle diagnostic, never an actual pose result.',
        created_utc=datetime.now(timezone.utc).isoformat())
    write_new(root/'protocol.json',frozen)
    for source in SOURCES:shutil.copy2(source,root/'source'/source)
    native=native_points(proposal_grid(),(512,512));truth={};rows=[]
    for tile_id,tile in tiles.items():
        if any(c['tile_id']==tile_id for c in cases):
            z=data[tile_id];truth[tile_id]=projected(native,z['depth_a'],z['depth_b'],*tile['cameras'])
    for step in steps:
        model=make_context(variant);model.load_state_dict(torch.load(folder/f'checkpoint_step_{step}.pt',map_location='cpu',weights_only=True));model.eval()
        step_rows=[]
        for i,case in enumerate(cases):
            z=data[case['tile_id']];reference=z['images'][2*case['light_a']];query=z['images'][2*case['light_b']+1]
            target,visible=truth[case['tile_id']]
            for rotation in range(4):
                a=cv2.resize(reference,(128,128),interpolation=cv2.INTER_AREA)
                b=cv2.resize(np.rot90(query,rotation).copy(),(128,128),interpolation=cv2.INTER_AREA)
                p,q,scores,indices,confidence=infer_proposals(coarse,model,a,b)
                metrics,coordinates=score_proposals(p,q,target,visible,indices,rotation)
                step_rows.append(dict(case,step=step,quarter_turn=rotation,metrics=metrics,coordinates=coordinates,
                    bilinear_target_likelihood=target_likelihood(scores,target,visible,rotation),proposal_scores=confidence.tolist()))
            if (i+1)%18==0:print('proposal diagnosis',variant,step,i+1,'/',len(cases),flush=True)
        with gzip.open(root/f'proposals_step_{step}.json.gz','wt') as stream:json.dump(step_rows,stream,allow_nan=False)
        rows+=step_rows
    summary=[]
    for step in steps:
        for split in ('train','model_selection'):
            part=[r for r in rows if r['step']==step and r['split']==split]
            ids=sorted({r['pair_id'] for r in part});oracle=[]
            for pair_id in ids:
                branches=[r for r in part if r['pair_id']==pair_id]
                oracle.append(max(r['metrics']['geometrically_reachable_33px'] for r in branches))
            nll=[r['bilinear_target_likelihood'] for r in part];n=sum(v['points'] for v in nll)
            summary.append(dict(step=step,split=split,pairs=len(ids),branches=len(part),
                returned=sum(r['metrics']['returned'] for r in part),reachable=sum(r['metrics']['geometrically_reachable_33px'] for r in part),
                correct_at12px=sum(r['metrics']['correct_at_native_px']['12'] for r in part),
                target_points=n,mean_target_nll=None if not n else sum(v['points']*v['mean_nll'] for v in nll if v['points'])/n,
                oracle_pairs_with_at_least6_reachable=sum(v>=6 for v in oracle),oracle_median_best_reachable=float(np.median(oracle))))
    result=dict(scope=frozen['scope'],summary=summary,actual_matcher_exact_replay=True,records=len(rows),
        source_sha256=sha256(__file__),protocol_sha256=sha256(root/'protocol.json'),
        results_sha256={str(s):sha256(root/f'proposals_step_{s}.json.gz') for s in steps},
        limits='These are coarse proposal diagnostics. Geometric local-window reachability is not correct refinement or pose recovery. Ground-truth best orientations are oracle-only. Consumed selection groups, no fresh validation.')
    for source,digest in frozen['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Diagnostic source changed during execution')
    write_new(root/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--cohort',default='runs/localization-v16-global-context-development')
    parser.add_argument('--variant',choices=['pointwise','self_context','cross_context'],required=True)
    parser.add_argument('--steps',type=int,nargs='+',default=[0,500,2000]);args=parser.parse_args()
    if len(set(args.steps))!=len(args.steps) or any(s not in (0,500,1000,1500,2000) for s in args.steps):raise ValueError('Existing frozen checkpoints required')
    run(args.output,args.cohort,args.variant,args.steps)
