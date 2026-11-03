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


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(20261101)
    from compact_lunar import load_checkpoint
    frozen,metadata=load_checkpoint('runs/compact-v2-expanded-20261030')
    model=LightingResponseModel(frozen).eval()
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    p=reference_anchors();world=world_points_z(p,np.full(256,100.),k,np.eye(4));variants=np.repeat(world[:,None],7,axis=1)
    geometry=ReferenceGeometry(np.full((512,512),100.),k,np.eye(4),np.tile([0.,0.,-1.],(256,1)),np.array([[0.,0.,-1.]]),np.ones((256,1),np.uint8),np.ones((256,1,7),np.uint8),np.array([0.,0.,-1.]),variants)
    from synthetic import texture
    reference=texture(20261101,512);query=cv2.warpAffine(reference,np.array([[1.,0,8.],[0,1.,-4.]]),(512,512))
    truth=np.eye(4);truth[:3,3]=[-2.,1.,0.]
    matcher=SolarMatcher(model.coarse,model.fine,model);rows=[]
    for a,b,c in itertools.product((False,True),repeat=3):
        tick=perf_counter();result=matcher.localize(reference,query,geometry,k,solar_mode='shared'if a else'off',response=b,active=c,budget=32)
        error=None if result['pose']is None else float(np.linalg.norm(result['pose'][:3,3]-truth[:3,3]))
        correspondence=None if not len(result.get('query_xy',[]))else float(np.median(np.linalg.norm(result['query_xy']-result['reference_xy']-[8,-4],axis=1)))
        passed=result['pose']is not None and error<1 and correspondence<1
        row=dict(solar=a,response=b,active=c,passed=bool(passed),translation_error_m=error,median_native_correspondence_error_px=correspondence,measurements=len(result['measurements']),response_measurements=result.get('response_measurements',0),elapsed_seconds=perf_counter()-tick)
        if row['measurements']>32:raise ValueError('Native patch budget exceeded')
        rows.append(row)
        with(root/f'{int(a)}{int(b)}{int(c)}.json').open('x')as f:json.dump(public_result(result,True),f,indent=2,allow_nan=False)
        print(json.dumps(row),flush=True)
    blank=matcher.localize(np.zeros_like(reference),np.zeros_like(query),geometry,k)
    result=dict(passed=all(r['passed']for r in rows)and blank['pose']is None,rows=rows,blank_rejected=blank['pose']is None,total_parameters=matcher.total_parameters,scope='Historical pretrained encoders with an UNTRAINED constant-response head, procedural translation and planar map; checks wiring for all eight toggles at32searches. Not actual lunar performance or learning improvement.',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path('.').glob('solar*.py')})
    with(root/'summary.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
