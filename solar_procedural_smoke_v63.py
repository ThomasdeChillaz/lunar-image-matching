"""Fresh procedural image-to-pose smoke; untrained software, not lunar evidence."""
import itertools,json,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import cv2
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel
from solar_frontend_v63 import ReferenceGeometry,SolarMatcher,reference_anchors
from localize_terrain import public_result,jsonable
from lunar_geometry_v5 import world_points_z


def run(output,pose_initialization='strict3',proposal_bank=4,rotation_deg=0.):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261101)
    from compact_lunar import load_checkpoint
    frozen,metadata=load_checkpoint('runs/compact-v2-expanded-20261030')
    model=LightingResponseModel(frozen).eval()
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    p=reference_anchors();world=world_points_z(p,np.full(256,100.),k,np.eye(4));variants=np.repeat(world[:,None],7,axis=1)
    geometry=ReferenceGeometry(np.full((512,512),100.),k,np.eye(4),np.tile([0.,0.,-1.],(256,1)),np.array([[0.,0.,-1.]]),np.ones((256,1),np.uint8),np.ones((256,1,7),np.uint8),np.array([0.,0.,-1.]),variants)
    from synthetic import texture
    reference=texture(20261101,512);affine=cv2.getRotationMatrix2D((255.5,255.5),rotation_deg,1.);affine[:,2]+=[8.,-4.]
    query=cv2.warpAffine(reference,affine,(512,512))
    angle=np.deg2rad(rotation_deg);r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    truth=np.eye(4);truth[:3,:3]=r;truth[:3,3]=-.25*r@np.array([8.,-4.,0.])
    if proposal_bank not in(4,16):raise ValueError('Declared four/sixteen-angle procedural control required')
    angles=None if proposal_bank==4 else(np.arange(16)*22.5).tolist()
    matcher=SolarMatcher(model.coarse,model.fine,model,proposal_angles=angles,pose_initialization=pose_initialization);rows=[]
    for a,b,c in itertools.product((False,True),repeat=3):
        tick=perf_counter();result=matcher.localize(reference,query,geometry,k,solar_mode='shared'if a else'off',response=b,active=c,budget=32)
        error=None if result['pose']is None else float(np.linalg.norm(result['pose'][:3,3]-truth[:3,3]))
        correspondence=None if not len(result.get('query_xy',[]))else float(np.median(np.linalg.norm(result['query_xy']-np.column_stack([result['reference_xy'],np.ones(len(result['reference_xy']))])@affine.T,axis=1)))
        passed=result['pose']is not None and error<1 and correspondence<1
        row=dict(solar=a,response=b,active=c,passed=bool(passed),translation_error_m=error,median_native_correspondence_error_px=correspondence,measurements=len(result['measurements']),response_measurements=result.get('response_measurements',0),elapsed_seconds=perf_counter()-tick)
        if row['measurements']>32:raise ValueError('Native patch budget exceeded')
        rows.append(row)
        with(root/f'{int(a)}{int(b)}{int(c)}.json').open('x')as f:json.dump(public_result(result,True),f,indent=2,allow_nan=False)
        print(json.dumps(row),flush=True)
    blank=matcher.localize(np.zeros_like(reference),np.zeros_like(query),geometry,k)
    result=dict(passed=all(r['passed']for r in rows)and blank['pose']is None,rows=rows,blank_rejected=blank['pose']is None,total_parameters=matcher.total_parameters,
        configuration=dict(pose_initialization=pose_initialization,proposal_bank=proposal_bank,rotation_deg=rotation_deg,affine_reference_to_query=affine.tolist(),scoring_query_cam2world=truth.tolist()),
        scope='Historical pretrained encoders with an UNTRAINED constant-response head, procedural known-transform imagery and planar map; checks wiring for all eight toggles at32searches. Not actual lunar performance or learning improvement.',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path('.').glob('solar*.py')})
    with(root/'summary.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--pose-initialization',choices=('strict3','quantization_aware'),default='strict3');p.add_argument('--proposal-bank',type=int,choices=(4,16),default=4);p.add_argument('--rotation-deg',type=float,default=0.)
    a=p.parse_args();run(a.output,a.pose_initialization,a.proposal_bank,a.rotation_deg)
