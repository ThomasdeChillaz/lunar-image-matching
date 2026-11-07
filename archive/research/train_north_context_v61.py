"""Train-only cross-image context pilot on the unchanged northern training data.

Only the coarse context architecture changes: two attention blocks replace
pointwise blocks at nearly matched capacity (+512 parameters). The original
coarse/fine encoders, sampler, loss and AdamW rates are retained. Attention is
an existing architectural control, not a novelty claim. No selection/test fit.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from geometry_warp_native_v11 import infer
from benchmark_north_control_v38 import inference_inputs
from evaluate_geometry_warp_v11 import control
from lunargeo_dataset import write_new
from north_training_bundle_v38 import load_bundle,prepare_training,sample_training,sha
from search_window_learning_v18 import SearchWindowMatcher,objective
from train_balanced_localization_v3 import save_torch
from train_search_window_v18 import (SOURCES as OLD_SOURCES,configure,make_model,
                                     make_optimizer,verify_frozen)

SEED=20261101
SOURCES=list(dict.fromkeys(OLD_SOURCES+['north_development_plan_v35.py',
    'north_training_bundle_v38.py','train_north_pilot_v38.py','benchmark_north_control_v38.py',
    'train_north_context_v61.py','verify_north_context_v61.py','test_north_context_v61.py']))


def published(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full public source freeze required')
    for p in SOURCES:
        if subprocess.check_output(['git','show',commit+':'+p])!=Path(p).read_bytes():
            raise ValueError('Fitting source differs from public freeze: '+p)


def initialize(coarse):
    torch.manual_seed(SEED);model=make_model(coarse,'cross_adaptation')
    original=torch.load('runs/north-v38-training-cost-pilot/initial.pt',map_location='cpu',weights_only=True)
    for name,value in model.state_dict().items():
        if name.startswith(('coarse.','fine.')) and not torch.equal(value,original[name]):
            raise ValueError('Original coarse/fine encoder initialization differs: '+name)
    if (sum(p.numel()for p in model.parameters())!=82208
            or sum(p.numel()for p in model.parameters()if p.requires_grad)!=53936
            or model.variant!='cross_context' or not model.train_coarse):
        raise ValueError('Declared cross-context capacity/trainability changed')
    return model


def fitting_step(model,optimizer,tiles,prepared,rng,step):
    model.train();images,target,sample=sample_training(tiles,prepared,rng,negative_pair=step%4==0)
    images=torch.from_numpy(images);target={k:torch.from_numpy(v)for k,v in target.items()}
    optimizer.zero_grad(set_to_none=True);scores,baseline=model(images)
    loss,labels=objective(scores,target,baseline,anchor_weight=.5)
    if not torch.isfinite(loss):raise ValueError('Nonfinite expanded-data loss')
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],5.)
    if not torch.isfinite(gradient):raise ValueError('Nonfinite expanded-data gradient')
    optimizer.step()
    return dict(step=step,loss=float(loss),gradient_norm=float(gradient),sample=sample,labels=labels)


def run(manifest_path,output,commit):
    published(commit);configure();root=Path(output)
    if root.exists():raise FileExistsError('Preserve every pilot/checkpoint/failure')
    manifest,tiles,data=load_bundle(manifest_path,train_only=True);prepared=prepare_training(tiles,data)
    if len(tiles)!=48 or len({t['component']for t in tiles.values()})!=16:
        raise ValueError('All16training components/48geometries required')
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    model=initialize(coarse);optimizer=make_optimizer(model);rng=np.random.default_rng(SEED)
    initial={k:v.clone()for k,v in model.state_dict().items()}
    root.mkdir(parents=True);start=datetime.now(timezone.utc);tick=perf_counter()
    protocol=dict(scope='Train-only100update cross-image context pilot; not selected pose performance or novelty.',
        publication_commit=commit,source_sha256={p:sha(p)for p in SOURCES},manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path),steps=100,seed=SEED,total_parameters=82208,
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        objective='Originalv18bidirectional reachable-window mass plus .5frozen-genuine-anchor term; no v23dual-softmax correction.',
        training_groups=16,training_geometries=48,lighting_count=6,context_variant='cross_context',
        pointwise_control_parameters=81696,parameter_difference=512,
        original_sampler_loss_optimizer_rates=True,geometric_occlusion_training_supported=False,negative_every_fourth_step=True,
        exact_fractional_3d_targets=True,model_selection_pixels_loaded=0,calibration_or_test_pixels_loaded=0,
        max_calendar_seconds=900,started_utc=start.isoformat(),
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
    write_new(root/'protocol.json',protocol);save_torch(initial,root/'initial.pt')
    history=[];fitting_seconds=0.
    try:
        initial_control=control(SearchWindowMatcher(coarse,model));write_new(root/'initial_controls.json',initial_control)
        if not initial_control['passed']:raise ValueError('Actual starting-model control failed')
        cost_rows=[]
        for component in sorted({t['component']for t in tiles.values()})[:4]:
            key=min(k for k,t in tiles.items()if t['component']==component)
            case=dict(tile_id=key,light_a=0,light_b=5)
            before=perf_counter()
            chosen,branches=infer(SearchWindowMatcher(coarse,model),*inference_inputs((manifest,tiles,data),case),{},((key,0),(key,11)))
            cost_rows.append(dict(component=component,tile_id=key,light_a=0,light_b=5,
                elapsed_seconds=perf_counter()-before,pose_returned=chosen is not None))
        write_new(root/'train_cost_bank.json',cost_rows)
        save_torch(dict(torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),root/'initial_rng.pt')
        for step in range(1,101):
            wall=(datetime.now(timezone.utc)-start).total_seconds()
            if wall<0 or wall>900 or perf_counter()-tick>900:raise TimeoutError('Original pilot allowance exceeded')
            before=perf_counter();record=fitting_step(model,optimizer,tiles,prepared,rng,step)
            fitting_seconds+=perf_counter()-before;history.append(record);verify_frozen(model,initial)
            if step%50==0:
                save_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,step=step),root/f'step_{step}.pt')
                write_new(root/f'history_{step}.json',history)
                print('NORTH CROSS CONTEXT PILOT UPDATE',step,'loss',record['loss'],flush=True)
        lights_a=sorted({r['sample']['light_a']for r in history});lights_b=sorted({r['sample']['light_b']for r in history})
        if lights_a!=list(range(6))or lights_b!=list(range(6)):raise ValueError('Pilot did not exercise all6Sun settings')
        result=dict(complete=True,completed_steps=100,fitting_seconds=fitting_seconds,
            calendar_seconds=(datetime.now(timezone.utc)-start).total_seconds(),
            fixed_train_cost_cases=4,median_native_case_seconds=float(np.median([r['elapsed_seconds']for r in cost_rows])),
            train_cost_bank_sha256=sha(root/'train_cost_bank.json'),
            total_parameters=82208,trainable_parameters=protocol['trainable_parameters'],
            lights_a=lights_a,lights_b=lights_b,protocol_sha256=sha(root/'protocol.json'),
            final_checkpoint_sha256=sha(root/'step_100.pt'),fine_encoder_unchanged=True,
            model_selection_evaluated=False,scope=protocol['scope'])
        write_new(root/'summary.json',result)
        write_new(root/'COMPLETE.json',dict(sha256={p.name:sha(p)for p in root.iterdir()if p.is_file()}))
        print(json.dumps(result,indent=2),flush=True);return result
    except BaseException as exc:
        write_new(root/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),completed_steps=len(history)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='reports/north_training_data_v38.json')
    p.add_argument('--output',default='runs/north-v61-context-cost-pilot');p.add_argument('--publication-commit',required=True)
    a=p.parse_args();run(a.manifest,a.output,a.publication_commit)
