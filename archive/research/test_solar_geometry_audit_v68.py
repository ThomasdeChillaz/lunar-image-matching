"""The pose oracle cannot silently remove its own failed eligible cases."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import verify_solar_development_v68 as audit
from solar_completion_v68 import sha


def make_fixture(folder):
    folder.mkdir();size=512;k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[0,0,100.]
    y,x=np.mgrid[:size,:size];world=np.stack([(x-255.5)/4,-(y-255.5)/4,np.zeros_like(x)],-1).astype(float);images=[];views={};labels=[]
    yy,xx=np.mgrid[:26,:26];p=np.column_stack([49.5+16*xx.ravel(),49.5+16*yy.ravel()]);w=np.column_stack([(p[:,0]-255.5)/4,-(p[:,1]-255.5)/4,np.zeros(len(p))]);distance=np.linalg.norm(w-pose[:3,3],axis=1)
    for role in('reference','query'):
        d=folder/role;d.mkdir();np.savez(d/'geometry.npz',position=world,valid=np.ones((size,size),bool),axial_depth=np.full((size,size),100.),cam2world=pose,intrinsics=k,body_origin=np.zeros(3),enu_rows=np.eye(3))
        views[role]=dict(path=str((d/'geometry.npz').relative_to(folder)),sha256=sha(d/'geometry.npz'))
        for light in range(24):
            path=d/f'light-{light}.png';Image.fromarray(np.zeros((size,size),np.uint8)).save(path);images.append(dict(role=role,light=light,path=str(path.relative_to(folder)),sha256=sha(path)))
        path=folder/(role+'-labels.npz');np.savez(path,source_points=p,target_points=p,world=w,source_valid=np.ones(len(p),bool),in_frame=np.ones(len(p),bool),visible=np.ones(len(p),bool),geometrically_occluded=np.zeros(len(p),bool),visibility_uncertain=np.zeros(len(p),bool),target_hit_distance_m=distance)
        labels.append(dict(source=role,path=path.name,sha256=sha(path),visible=len(p)))
    summary=dict(complete=True,rendered_images=48,images=images,views=views,labels=labels,matcher_forwards=0,optimizer_updates=0,training_allowed=False,calibration_or_final_access=False)
    (folder/'summary.json').write_text(json.dumps(summary));(folder/'COMPLETE.json').write_text(json.dumps(dict(sha256={str(p.relative_to(folder)):sha(p)for p in folder.rglob('*')if p.is_file()})))
    record=dict(output=str(folder),summary_sha256=sha(folder/'summary.json'),seal_sha256=sha(folder/'COMPLETE.json'))
    camera=dict(cam2world=pose.tolist(),intrinsics=k.tolist());spec=dict(mesh='software-fixture.npz',mesh_sha256='a'*64,site=dict(split='training',component='software'),camera_pairs=[dict(reference=camera,query=camera,family='training_motion',pair_index=0)])
    mesh=(np.array([[-500,-500,0],[500,-500,0],[500,500,0],[-500,500,0]],float),np.array([[0,1,2],[0,2,3]],int))
    return record,spec,mesh


def test_oracle_failure_stays_in_eligible_denominator(tmp_path,monkeypatch):
    record,spec,mesh=make_fixture(tmp_path/'unit')
    monkeypatch.setattr(audit,'propose_pose',lambda *a,**k:(None,dict(reason='deliberate_solver_failure')))
    row=audit.audit_unit(0,record,spec,mesh,lambda:None)
    assert row['eligible']and row['qualification_reasons']==[]
    assert len(row['oracle_outcomes'])==2 and not any(x['correct']for x in row['oracle_outcomes'])


def test_locally_correct_projection_cannot_hide_wrong_body_frame(tmp_path):
    record,spec,mesh=make_fixture(tmp_path/'unit')
    with pytest.raises(ValueError,match='frame metadata differs'):
        audit.audit_unit(0,record,spec,mesh,lambda:None,(np.ones(3),np.eye(3)))
