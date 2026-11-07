"""Training-only lighting-response patch bank with hard heldout-light isolation.

Banks are produced from 3D canonical patches and frozen teacher features. This
loader refuses missing source provenance, odd lighting targets, selection or
reserve groups. Query pose/depth are legal bank-construction labels only.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from solar_response_v63 import counterfactual_loss

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EVEN=tuple(range(0,24,2))


class TrainingBank:
    def __init__(self,manifest_path):
        self.manifest=json.loads(Path(manifest_path).read_text());m=self.manifest
        if m['schema_version']!=63 or m['split']!='training'or m['lighting_indices']!=list(EVEN)or m['heldout_image_reads']!=0:raise ValueError('Even-light training-only bank required')
        if m['teacher_descriptor_sha256']!=m['expected_frozen_descriptor_sha256']or not m['canonicalization_identical_to_inference']:raise ValueError('Frozen teacher and shared physical canonicalization required')
        if len(m['terrain_groups'])!=16 or len(set(m['terrain_groups']))!=16 or not m['geometry_gate_passed']:raise ValueError('Original16training groups and passed exact geometry gate required')
        self.records=[]
        for r in m['records']:
            if r['component']not in m['terrain_groups']or r['split']!='training'or r['source_lights']!=list(EVEN):raise ValueError('No selection/reserve/odd target images allowed')
            if sha(r['path'])!=r['sha256']:raise ValueError('Patch bank changed after source freeze')
            with np.load(r['path'],allow_pickle=False)as z:
                expected={'patches','conditions','target_suns','teacher','available','observed_lights'}
                if set(z.files)!=expected:raise ValueError('Explicit canonical patch/physical condition/teacher schema required')
                b={n:z[n].copy()for n in z.files}
            n=len(b['patches'])
            if b['patches'].shape!=(n,2,33,33)or b['patches'].dtype!=np.uint8 or b['conditions'].shape!=(n,2,9)or b['target_suns'].shape!=(n,2,24,3)or b['teacher'].shape!=(n,2,24,32)or b['available'].shape!=(n,2,24)or b['observed_lights'].shape!=(n,2):raise ValueError('Paired canonical response samples required')
            if b['available'].dtype!=bool or b['available'][:,:,1::2].any()or np.isfinite(b['teacher'][:,:,1::2]).any()or not np.isin(b['observed_lights'],EVEN).all():raise ValueError('Heldout illumination leakage into training bank')
            if not np.isfinite(b['conditions']).all()or not np.isfinite(b['target_suns']).all()or not np.isfinite(b['teacher'][b['available']]).all()or np.isfinite(b['teacher'][~b['available']]).any():raise ValueError('Finite available targets, absent unavailable targets required')
            if not b['available'].any(-1).all():raise ValueError('Each input requires available teacher targets')
            self.records.append((r,b))
        if not self.records or {r['component']for r,b in self.records}!=set(m['terrain_groups']):raise ValueError('Every declared training group must contribute a bank')

    def sample(self,rng,count=8,augment=True):
        # Terrain-uniform, then source-record/sample-uniform. No teacher/patch
        # targets are synthesized from heldout lighting or brightness masks.
        groups=self.manifest['terrain_groups'];group=groups[int(rng.integers(len(groups)))]
        options=[x for x in self.records if x[0]['component']==group];record,b=options[int(rng.integers(len(options)))]
        if type(count)is not int or count<1:raise ValueError('Positive physical-pair sample count required')
        # Repeating a physical point would label its identical duplicate as a
        # negative in the diagonal correspondence-discrimination objective.
        # Small usable records therefore yield smaller, unique minibatches.
        count=min(count,len(b['patches']))
        if not count:raise ValueError('No usable physical correspondence in selected record')
        ids=rng.choice(len(b['patches']),size=count,replace=False)
        patches=b['patches'][ids].astype(np.float32)/255
        if augment:
            import cv2
            for i in range(len(patches)):
                for j in range(2):
                    a=patches[i,j];gain=float(rng.uniform(.65,1.35));sigma=float(rng.uniform(0,.025))
                    if rng.random()<.3:a=cv2.GaussianBlur(a,(3,3),float(rng.uniform(.3,1.)))
                    a=np.clip(a*gain+rng.normal(0,sigma,a.shape),0,1);levels=int(rng.choice([32,64,128,256]));patches[i,j]=np.rint(a*(levels-1))/(levels-1)
        result=dict(patches=torch.from_numpy(patches.reshape(-1,1,33,33)),condition=torch.from_numpy(b['conditions'][ids].reshape(-1,9)).float(),target_suns=torch.from_numpy(b['target_suns'][ids].reshape(-1,24,3)).float(),teacher_features=torch.from_numpy(b['teacher'][ids].reshape(-1,24,32)).float(),available=torch.from_numpy(b['available'][ids].reshape(-1,24)),positive_pairs=np.arange(2*count).reshape(-1,2))
        return result,dict(component=group,record_sha256=record['sha256'],sample_indices=ids.tolist(),observed_lights=b['observed_lights'][ids].tolist(),augmentation_is_sensor_only=True)


def fitting_step(model,optimizer,bank,rng,pooled=False):
    model.train();batch,sample=bank.sample(rng);optimizer.zero_grad(set_to_none=True)
    loss,parts=counterfactual_loss(model,**batch,pooled=pooled);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters()if p.requires_grad],5.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite response gradient')
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),parts=parts,sample=sample)
