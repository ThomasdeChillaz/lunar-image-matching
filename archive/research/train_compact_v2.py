"""Finite CPU training ablations; final-test and calibration pixels never loaded."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from compact_lunar import CompactDescriptor,VARIANTS
from contrastive_terrain import sample_descriptors,contrastive_loss,valid_centers
from dataset import sha256


def appearance(image,rng):
    image=image.astype(np.float32)/255
    image=np.clip(image*rng.uniform(.6,1.4),0,1)**rng.uniform(.7,1.5)
    if rng.random()<.5:image=cv2.GaussianBlur(image,(3,3),float(rng.uniform(.2,1.2)))
    image=np.clip(image+rng.normal(0,rng.uniform(0,.025),image.shape),0,1)
    levels=int(rng.choice([31,63,127,255]))
    return (np.round(image*levels)/levels).astype(np.float32)


def sample_pair(tile,rng,augment=True,radius=13):
    images,masks=tile['images'],tile['masks']
    i,j=rng.choice(len(images),2,replace=False)
    a,b=images[i],images[j]
    size=a.shape[0]
    angle=float(rng.uniform(-45,45));scale=float(rng.uniform(.7,1.4))
    matrix=cv2.getRotationMatrix2D(((size-1)/2,(size-1)/2),angle,scale)
    matrix[:,2]+=rng.uniform(-12,12,2)
    b=cv2.warpAffine(b,matrix,(size,size),flags=cv2.INTER_LINEAR)
    mb=cv2.warpAffine(masks[j].astype(np.uint8),matrix,(size,size),flags=cv2.INTER_NEAREST)>0
    ma=valid_centers(masks[i],radius);vb=valid_centers(mb,radius)
    yy,xx=np.where(ma)
    if len(xx)<24:return None
    chosen=rng.choice(len(xx),min(128,len(xx)),replace=False)
    p=np.column_stack([xx[chosen],yy[chosen]]).astype(np.float32)
    target=p@matrix[:,:2].T+matrix[:,2]
    inside=((target>=0)&(target<=size-1)).all(1)
    index=np.rint(target).astype(int).clip(0,size-1)
    label=inside & vb[index[:,1],index[:,0]]
    # Descriptor supervision uses only exact valid correspondences. Dark support
    # is excluded from contrastive loss, but NOT redefined as geometric occlusion.
    usable=label & (a[p[:,1].astype(int),p[:,0].astype(int)]>3) & (b[index[:,1],index[:,0]]>3)
    if usable.sum()<12:return None
    if augment:a,b=appearance(a,rng),appearance(b,rng)
    else:a,b=a.astype(np.float32)/255,b.astype(np.float32)/255
    return np.stack([a,b]),p,target.astype(np.float32),label.astype(np.float32),usable


def objective(model,sample,negative=None,hard=False,multiscale=False):
    images,p,target,label,usable=sample
    device=next(model.parameters()).device
    inputs=torch.from_numpy(images.copy()).to(device)[:,None]
    dense,logits=model(inputs)
    if multiscale:
        half,_=model(F.interpolate(inputs,scale_factor=.5,mode='bilinear',align_corners=False))
        dense=F.normalize(dense+F.interpolate(half,size=images.shape[-2:],mode='bilinear',align_corners=False),dim=1)
    pp,qq=[torch.from_numpy(v).to(device)[None] for v in (p,target)]
    da=sample_descriptors(dense[:1],pp)[0]
    db=sample_descriptors(dense[1:],qq)[0]
    good=torch.from_numpy(usable).to(device)
    loss=contrastive_loss(da[good],db[good])
    if model.reliability:
        grid=(2*(pp+.5)/pp.new_tensor(images.shape[-2:][::-1])-1)[:,None]
        predicted=F.grid_sample(logits[:1],grid,align_corners=False)[0,0,0]
        loss+=.25*F.binary_cross_entropy_with_logits(predicted,torch.from_numpy(label).to(device))
    if hard and negative is not None:
        other,_=model(torch.from_numpy(negative.copy()).to(device)[None,None].float()/255)
        other=sample_descriptors(other,pp)[0]
        # Nonoverlapping terrain provides no positive correspondence. Penalize
        # its hardest descriptor impostors instead of inventing paired labels.
        impostors=(da[good]@other.T).max(1).values
        loss+=.1*F.relu(impostors-.5).mean()
    return loss


def load_allowed(manifest):
    tiles=[]
    for record in manifest['tiles']:
        if record['split'] not in ('train','model_selection'):continue
        path=Path(record['cache'])
        if sha256(path)!=record['cache_sha256']:raise ValueError('Training cache changed')
        with np.load(path,allow_pickle=False) as data:
            tiles.append(dict(record,images=data['images'].copy(),masks=data['masks'].copy()))
    if not any(t['split']=='train' for t in tiles) or not any(t['split']=='model_selection' for t in tiles):
        raise ValueError('Separate train and model-selection groups are required')
    train_components={t['component'] for t in tiles if t['split']=='train'}
    if train_components & {t['component'] for t in tiles if t['split']=='model_selection'}:
        raise ValueError('Training/model-selection component leakage')
    return tiles


def train(manifest_path,output,variant='expanded',steps=2000,seed=20261030,device='cpu'):
    if variant not in VARIANTS or steps<1:raise ValueError('Declared ablation and positive step count required')
    manifest=json.loads(Path(manifest_path).read_text())
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(device=='cpu')
    rng=np.random.default_rng(seed)
    tiles=load_allowed(manifest)
    train_tiles=[t for t in tiles if t['split']=='train']
    dev_tiles=[t for t in tiles if t['split']=='model_selection']
    if variant in ('hard_negatives','multiscale') and len({t['component'] for t in train_tiles})<2:
        raise ValueError('Hard-negative training requires separated terrain components')
    dev_rng=np.random.default_rng(20261031)
    bank=[sample_pair(t,dev_rng,False,27 if variant=='multiscale' else 13) for t in dev_tiles]
    bank=[s for s in bank if s is not None]
    if not bank:raise ValueError('No exact valid model-selection samples')
    model=CompactDescriptor(variant!='expanded').to(device)
    config=dict(schema_version=2,variant=variant,steps=steps,seed=seed,device=device,
        parameters=sum(p.numel() for p in model.parameters()),manifest_sha256=sha256(manifest_path),
        source_sha256={p:sha256(p) for p in ['compact_lunar.py','train_compact_v2.py','contrastive_terrain.py']},
        loaded_groups=[dict(id=t['tile_id'],split=t['split'],component=t['component']) for t in tiles],
        learning_rate=.0005,weight_decay=.0001,checkpoint_interval=200,
        selection='Minimum fixed model-selection descriptor/matchability objective; initial weights eligible',
        calibration_or_test_images_loaded=0,started_utc=datetime.now(timezone.utc).isoformat(),
        augmentations='Exposure/gamma/blur/noise/quantization; these are NOT simulated sunlight',
        physical_lighting_source=manifest['dataset'],
        environment=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
    (output/'frozen_config.json').write_text(json.dumps(config,indent=2)+'\n')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    def validation():
        model.eval()
        with torch.no_grad():value=float(np.mean([objective(model,s,multiscale=variant=='multiscale').item() for s in bank]))
        model.train();return value
    best_loss=validation();best_step=0
    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[dict(step=0,selection_loss=best_loss)]
    started=perf_counter();skipped=0
    for step in range(1,steps+1):
        sample=None
        for _ in range(100):
            tile=train_tiles[int(rng.integers(len(train_tiles)))]
            sample=sample_pair(tile,rng,radius=27 if variant=='multiscale' else 13)
            if sample is not None:break
            skipped+=1
        if sample is None:raise ValueError('Cannot sample enough co-visible training correspondences')
        negative=None
        if variant in ('hard_negatives','multiscale'):
            candidates=[t for t in train_tiles if t['component']!=tile['component']]
            other=candidates[int(rng.integers(len(candidates)))]
            negative=other['images'][int(rng.integers(len(other['images'])))]
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model,sample,negative,variant in ('hard_negatives','multiscale'),variant=='multiscale')
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%200==0 or step==steps:
            value=validation()
            history.append(dict(step=step,training_loss=float(loss.item()),selection_loss=value))
            if value<best_loss:
                best_loss,best_step=value,step
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(variant,seed,step,'selection_loss',round(value,6),'seconds',round(perf_counter()-started),flush=True)
    torch.save(best,output/'selected.pt')
    result=dict(config,history=history,selected_step=best_step,skipped_samples=skipped,
                elapsed_seconds=perf_counter()-started)
    (output/'training.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(output/'selected.pt'),
        training_sha256=sha256(output/'training.json')),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=VARIANTS,default='expanded')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    a=p.parse_args();train(a.manifest,a.output,a.variant,a.steps,a.seed,a.device)
