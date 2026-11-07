"""Procedural training-bank controls; never real lunar teacher extraction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pytest
import torch
from solar_training_bank_v68 import (EVEN,EvenImageReader,local_facets,exact_surface_normals,
    sample_geometry,build_unit_arrays,verify_geometry_gate,sha)
from build_solar_training_bank_v68 import cache_key,validate_record_arrays,save_record,completed_cache
from north_development_plan_v35 import proposal_grid_native
from north_render_plan_v31 import plan
from lunar_geometry_v5 import world_points_z,project_z


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


class ProceduralTeacher(torch.nn.Module):
    def forward(self,x):
        # Descriptor orientation varies with observed patch mean; no learned or
        # pretrained network, real image or optimizer is used in this fixture.
        mean=x.mean((1,2,3));frequency=torch.arange(1,33,dtype=x.dtype,device=x.device)
        f=torch.cos(mean[:,None]*frequency[None]);f=torch.nn.functional.normalize(f,dim=-1)
        return f[:,:,None,None].expand(-1,-1,33,33),torch.zeros_like(x)


def geometries_and_labels():
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    ref=np.diag([1.,-1.,-1.,1.]);ref[:3,3]=[0.,0.,100.];query=ref.copy();query[:3,3]+=[-2.,-1.,0.]
    p=proposal_grid_native();world=world_points_z(p,np.full(len(p),100.),k,ref);q,z,front=project_z(world,k,query)
    g={role:dict(intrinsics=k.copy(),cam2world=pose.copy(),axial_depth=np.full((512,512),100.))for role,pose in(('reference',ref),('query',query))}
    labels=dict(source_points=p,target_points=q,world=world,source_valid=np.ones(676,bool),in_frame=np.ones(676,bool),visible=np.ones(676,bool),geometrically_occluded=np.zeros(676,bool),visibility_uncertain=np.zeros(676,bool))
    return g,labels


def images(tmp_path,black=False):
    import cv2
    from PIL import Image
    rng=np.random.default_rng(4);base=np.zeros((512,512),np.uint8)if black else rng.integers(0,160,(512,512),dtype=np.uint8)
    records=[]
    for light in range(24):
        # ODD files deliberately DO NOT EXIST. Any unintended access fails.
        for role in('reference','query'):
            path=tmp_path/role/f'light-{light}.png'
            if light in EVEN:
                path.parent.mkdir(parents=True,exist_ok=True);a=base.copy()if black else(base+light).astype(np.uint8)
                if role=='query':a=cv2.warpAffine(a,np.array([[1.,0,8.],[0,1.,-4.]]),(512,512))
                Image.fromarray(a).save(path);digest=sha(path)
            else:digest='not-present-heldout-software-fixture'
            records.append(dict(role=role,light=light,path=str(path.relative_to(tmp_path)),sha256=digest))
    return records


def selected_points():
    g,labels=geometries_and_labels()
    points,counts=sample_geometry(g,labels,8,lambda w:(np.tile([0.,0.,1.],(len(w),1)),[dict(plane_error_m=0.,minimum_barycentric=.2)for _ in w]))
    return g,labels,points,counts


def test_even_reader_checks_parity_before_any_file_access_and_never_hashes_odd(tmp_path):
    records=images(tmp_path);opened=[]
    def guarded(path):
        assert int(re.search(r'light-(\d+)',path.name)[1])%2==0
        opened.append(path);return path.read_bytes()
    reader=EvenImageReader(tmp_path,records,guarded)
    with pytest.raises(ValueError,match='Heldout'):reader.get('reference',1)
    assert not opened
    reader.get('reference',0);reader.get('reference',0)
    assert len(opened)==1 and reader.decoded[0]['light']==0


def test_supported_facet_normals_and_plane_barycentric_failures():
    yy,xx=np.mgrid[-2:3,-2:3];a,b=.2,-.4;grid=np.stack([xx,yy,a*xx+b*yy],-1).astype(float)
    point=np.array([[.25,.2,a*.25+b*.2]])
    normal,provenance=local_facets(point,grid,[2],[2]);expected=np.array([-a,-b,1.]);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(normal[0],expected,rtol=0,atol=1e-12)
    assert provenance[0]['plane_error_m']<1e-12
    with pytest.raises(ValueError,match='supporting'):local_facets(point+[0,0,.2],grid,[2],[2])
    with pytest.raises(ValueError,match='supporting'):local_facets([[10,0,2]],grid,[2],[2])


def test_body_frame_inverse_lookup_recovers_exact_synthetic_sphere_facet():
    from north_terrain_pilot_v32 import local_vertices,grid_faces
    site=next(s for s in plan()['sites']if s['split']=='training');offsets=np.arange(-4,5)
    # A procedural constant-height sphere, no released terrain/raster read.
    vertices,origin,enu=local_vertices(np.zeros((2049,2049),np.float32),site,offsets);flat=vertices.reshape(-1,3);faces=grid_faces(9)
    mesh=dict(vertices=flat,faces=faces,body_origin=origin,enu_rows=enu);tri=flat[faces[3*8+3]];point=.2*tri[0]+.3*tri[1]+.5*tri[2]
    normal,meta=exact_surface_normals(point[None],mesh,site,offsets)
    expected=np.cross(tri[1]-tri[0],tri[2]-tri[0]);expected/=np.linalg.norm(expected)
    if expected[2]<0:expected=-expected
    np.testing.assert_allclose(normal[0],expected,atol=1e-6,rtol=0);assert meta[0]['plane_error_m']<1e-4


def test_native_geometry_and_teacher_targets_use_same_physical_point_and_frame(tmp_path):
    g,labels,points,counts=selected_points();records=images(tmp_path);opened=[]
    reader=EvenImageReader(tmp_path,records,lambda p:(opened.append(p)or p.read_bytes()))
    arrays,details=build_unit_arrays(g,points,reader,86.,ProceduralTeacher(),20261101)
    assert validate_record_arrays(arrays)==8 and len(opened)==24
    assert {r['light']for r in reader.decoded}==set(EVEN)and details['heldout_image_reads']==0
    np.testing.assert_allclose(arrays['teacher'][:,0,::2],arrays['teacher'][:,1,::2],rtol=0,atol=1e-7)
    assert np.isnan(arrays['teacher'][:,:,1::2]).all()and not arrays['available'][:,:,1::2].any()
    np.testing.assert_array_equal(arrays['target_suns'][:,0],arrays['target_suns'][:,1])
    ref=g['reference'];world=points[0]['world'];normal=np.array([0.,0.,1.])@ref['cam2world'][:3,:3]
    np.testing.assert_allclose(arrays['conditions'][0,0,3:6],normal,rtol=0,atol=1e-7)
    expected=(ref['cam2world'][:3,3]-world);expected/=np.linalg.norm(expected);expected=expected@ref['cam2world'][:3,:3]
    np.testing.assert_allclose(arrays['conditions'][0,0,6:9],expected,rtol=0,atol=1e-7)
    assert all(r['label_index']in range(676)for r in points)
    np.testing.assert_allclose(np.stack([r['query_xy']-r['reference_xy']for r in points]),np.tile([8.,-4.],(8,1)),atol=1e-10)


def test_darkness_does_not_create_camera_occlusion_labels(tmp_path):
    g,labels,points,counts=selected_points();reader=EvenImageReader(tmp_path,images(tmp_path,black=True))
    arrays,details=build_unit_arrays(g,points,reader,86.,ProceduralTeacher(),20261101)
    assert counts['camera_occluded_points']==0 and len(arrays['patches'])==8
    assert details['geometry_labels_independent_of_brightness']and details['teacher_availability_is_not_geometric_visibility']
    assert arrays['available'][:,:,::2].all()


def test_actual_geometric_occlusion_is_excluded_independently_of_pixels():
    g,labels=geometries_and_labels();index=0;labels['visible'][index]=False;labels['geometrically_occluded'][index]=True
    points,counts=sample_geometry(g,labels,8,lambda w:(np.tile([0.,0.,1.],(len(w),1)),[{}for _ in w]))
    assert index not in {p['label_index']for p in points}and counts['camera_occluded_points']==1
    corrupted=deepcopy(labels);corrupted['target_points'][1]+=[2,0]
    with pytest.raises(ValueError,match='query world/pixel'):sample_geometry(g,corrupted,8,lambda w:None)


def gate_fixture(tmp_path):
    sites=[s for s in plan()['sites']if s['split']in('training','model_selection')];units=[];data=[];audit=[]
    for site in sites:
        for j in range(4):
            i=len(units);root=tmp_path/'units'/str(i);root.mkdir(parents=True)
            summary=dict(complete=True,rendered_images=48,matcher_forwards=0,optimizer_updates=0,images=[dict(role=role,light=l,path=f'{role}/light-{l}.png')for role in('reference','query')for l in range(24)])
            write_json(root/'summary.json',summary);write_json(root/'COMPLETE.json',dict(sha256={}))
            units.append(dict(site=site));r=dict(unit_index=i,output=str(root),summary_sha256=sha(root/'summary.json'),seal_sha256=sha(root/'COMPLETE.json'));data.append(r)
            audit.append(dict(unit_index=i,component=site['component'],split=site['split'],eligible=True,source_summary_sha256=r['summary_sha256'],source_seal_sha256=r['seal_sha256']))
    master=write_json(tmp_path/'master.json',dict(units=units,calibration_or_final_access=False));ds=write_json(tmp_path/'data.json',dict(complete=True,original_master_sha256=sha(master),units=data));a=write_json(tmp_path/'audit.json',dict(passed=True,complete_dataset=True,expected_units=96,completed_units=96,terrain_counts=dict(training=16,model_selection=8),original_master_sha256=sha(master),completion_summary_sha256=sha(ds),oracle_success=1.,oracle_pose_directions=192,correct_oracle_poses=192,units=audit))
    return master,ds,a


def test_full_gate_requires_all_eligible_pose_directions_before_any_png_access(tmp_path,monkeypatch):
    paths=gate_fixture(tmp_path);original=Path.read_bytes
    def guarded(path):
        assert path.suffix!='.png'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    gate=verify_geometry_gate(*paths);assert len(gate.eligible_training_indices)==64 and len(gate.training_groups)==16
    a=Path(paths[2]);payload=json.loads(a.read_text());payload['oracle_pose_directions']=191;payload['correct_oracle_poses']=191;a.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='denominator'):verify_geometry_gate(*paths)
    payload['passed']=False;a.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='Independent full96unit'):verify_geometry_gate(*paths)


def test_cache_changes_with_checkpoint_and_rejects_finite_heldout_targets(tmp_path):
    config=dict(maximum_points_per_unit=8,seed=20261101);spec=dict(site=dict(component='fixture',split='training'),mesh_sha256='mesh');unit=dict(unit_index=0,summary_sha256='summary',seal_sha256='seal')
    key,meta=cache_key(spec,unit,'audit','checkpointA','training','descriptor',config,{},{});other,_=cache_key(spec,unit,'audit','checkpointB','training','descriptor',config,{},{});assert key!=other
    g,labels,points,counts=selected_points();arrays,details=build_unit_arrays(g,points,EvenImageReader(tmp_path,images(tmp_path)),86.,ProceduralTeacher(),20261101)
    root=tmp_path/'cache';root.mkdir();save_record(root,key,arrays,dict(input_key=key,input=meta,samples=8))
    assert completed_cache(root,key,meta)is not None
    broken=deepcopy(arrays);broken['teacher'][0,0,1]=0.
    with pytest.raises(ValueError,match='odd teacher'):validate_record_arrays(broken)
    payload=root/'records'/(key+'.npz');payload.write_bytes(payload.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='cache changed'):completed_cache(root,key,meta)


def test_produced_records_load_through_existing16group_fourcamera_training_interface(tmp_path):
    from solar_training_v63 import TrainingBank
    g,labels,points,counts=selected_points();arrays,details=build_unit_arrays(g,points,EvenImageReader(tmp_path/'images',images(tmp_path/'images')),86.,ProceduralTeacher(),20261101)
    groups=[f'procedural-terrain-{i:02}'for i in range(16)];records=[]
    for group in groups:
        for camera in range(4):
            folder=tmp_path/group/str(camera);folder.mkdir(parents=True);path=folder/'samples.npz'
            # A camera-specific ordering keeps identical physical pairs aligned
            # while giving the source-input sampler four distinguishable records.
            ordered={k:np.roll(v,camera,axis=0)for k,v in arrays.items()}
            with path.open('xb')as stream:np.savez_compressed(stream,**ordered)
            records.append(dict(component=group,split='training',source_lights=list(EVEN),path=str(path),sha256=sha(path),family='motion'if camera<2 else'roll',pair_index=camera%2))
    m=dict(schema_version=63,split='training',lighting_indices=list(EVEN),heldout_image_reads=0,teacher_descriptor_sha256='procedural-stub-only',expected_frozen_descriptor_sha256='procedural-stub-only',canonicalization_identical_to_inference=True,terrain_groups=groups,geometry_gate_passed=True,records=records)
    manifest=tmp_path/'manifest.json';write_json(manifest,m);bank=TrainingBank(manifest)
    rng=np.random.default_rng(20261101);seen_groups=set();seen_records=set()
    for _ in range(1024):
        batch,sample=bank.sample(rng,augment=False);seen_groups.add(sample['component']);seen_records.add((sample['component'],sample['record_sha256']))
        assert batch['patches'].shape==(16,1,33,33)and not batch['available'][:,1::2].any()
        assert torch.isnan(batch['teacher_features'][:,1::2]).all()
        assert len(set(sample['sample_indices']))==8
    # The producer retains four records per group and the unchanged loader can
    # consume every group/camera record without inspecting unavailable odd
    # targets. This is ONLY a procedural format/sampler software control.
    assert seen_groups==set(groups)and len(bank.records)==64 and len(seen_records)==64


def test_frozen_lighting_metadata_mismatch_is_rejected_before_image_access(tmp_path):
    from solar_physics_v63 import lighting_schedule
    g,labels,points,counts=selected_points();reader=EvenImageReader(tmp_path,images(tmp_path));lights=lighting_schedule(86.)
    lights[0]['sun_enu'][0]*=-1
    with pytest.raises(ValueError,match='lighting metadata differs'):build_unit_arrays(g,points,reader,86.,ProceduralTeacher(),20261101,lighting_records=lights)
    assert not reader.decoded
