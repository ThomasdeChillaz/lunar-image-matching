"""Toy source-adapter controls. No lunar files, checkpoints, model or forwards."""
from copy import deepcopy
import hashlib
import inspect
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from solar_coarse_training_adapter_v86 import (EVEN,native_to_coarse,coarse_support,
    reference_normal_field,training_view_basis,validate_metadata_inventory,
    load_training_metadata_index,build_coarse_training_unit,fit_balanced_training_view_prior)


class EvenToyReader:
    def __init__(self):self.calls=[]
    def get(self,role,light):
        assert role in ('reference','query')and light in EVEN
        self.calls.append((role,light))
        # Deliberately different observed camera content: native-view teachers
        # must remain different, without a geometry warp into reference view.
        yy,xx=np.mgrid[:512,:512]
        return ((xx+3*yy+light+(83 if role=='query'else 0))%256).astype(np.uint8)


class Full128ToyTeacher:
    def __init__(self):self.calls=[]
    def __call__(self,image,points):
        assert image.shape==(128,128)and image.dtype==np.uint8
        self.calls.append((image.copy(),points.copy()))
        p=np.rint(points).astype(int);v=image[p[:,1],p[:,0]].astype(float)
        out=np.zeros((len(points),32),np.float32)
        out[:,0]=1.;out[:,1]=v/255
        return out/np.linalg.norm(out,axis=1,keepdims=True)


def unit_fixture(points=None):
    from lunar_geometry_v5 import world_points_z,project_z
    from solar_physics_v63 import lighting_schedule
    k=np.array([[300.,0.,255.5],[0.,300.,255.5],[0.,0.,1.]])
    ref=np.eye(4);ref[:3,:3]=np.diag([1.,-1.,-1.]);ref[2,3]=100.
    query=ref.copy();query[:3,3]+=[8.,-4.,0.]
    geometry={role:dict(intrinsics=k.copy(),cam2world=c.copy(),axial_depth=np.full((512,512),100.),valid=np.ones((512,512),bool))for role,c in [('reference',ref),('query',query)]}
    p=np.array([[129.5,129.5],[257.5,257.5],[401.5,401.5]])if points is None else np.asarray(points,float)
    world=world_points_z(p,np.full(len(p),100.),k,ref);q,_,_=project_z(world,k,query)
    physical=[dict(label_index=22+22*i,reference_xy=a.tolist(),query_xy=b.tolist(),world=x.tolist(),normal=[0.,0.,1.])for i,(a,b,x)in enumerate(zip(p,q,world))]
    m=dict(unit_index=0,component='trainA',split='training',source_lights=list(EVEN),samples=len(p),
           physical_points=physical,reference_cam2world=ref.tolist(),query_training_truth_cam2world=query.tolist(),
           geometry_counts=dict(selected_physical_points=len(p)+1,unknown_patch_interior_visibility=True),
           teacher_extraction=dict(heldout_image_reads=0,omitted_trial_support_samples=1,
                canonical_trial_support_failures=[dict(sample_index=1,label_index=999,reason='historical_trial_crop')],
                selected_indices=[0]+list(range(2,len(p)+1)),
                even_images_decoded=[dict(role=role,light=i)for role in ('reference','query')for i in EVEN]))
    return m,geometry,lighting_schedule(86.01352704811886)


def build_fixture(points=None,teacher=None):
    m,g,l=unit_fixture(points);reader=EvenToyReader();teacher=teacher or Full128ToyTeacher()
    return build_coarse_training_unit(m,g,reader,l,teacher),m,g,l,reader,teacher


def test_halfpixel_coordinates_and_full_descriptor_support():
    p=native_to_coarse(np.array([[49.5,49.5],[449.5,449.5],[1.5,1.5]]))
    np.testing.assert_array_equal(p,np.array([[12.,12.],[112.,112.],[0.,0.]],np.float32))
    valid=np.ones((512,512),bool)
    np.testing.assert_array_equal(coarse_support(valid,p),[True,True,False])
    # A fractional descriptor requires BOTH adjacent support samples.
    np.testing.assert_array_equal(coarse_support(valid,np.array([[115.,64.],[115.1,64.]])),[True,False])
    valid[0,0]=False
    assert not coarse_support(valid,p[:1])[0]


def test_reference_normal_map_is_reference_only_estimate_and_tracks_missing_stencil():
    _,g,_=unit_fixture();a=reference_normal_field(g['reference'])
    assert a['valid'].all()and a['normal_xy'].shape==(2,128,128)
    np.testing.assert_allclose(a['normal_xy'],0.,atol=1e-12,rtol=0)
    assert not a['exact_facet_normals']and not a['uses_query_geometry']
    assert list(inspect.signature(reference_normal_field).parameters)==['geometry']
    bad=deepcopy(g['reference']);bad['valid'][254:258,254:258]=False
    b=reference_normal_field(bad)
    assert not b['valid'][63,63]and b['valid'][20,20]
    assert np.isfinite(b['normal_xy']).all()
    np.testing.assert_array_equal(b['normal_xy'][:,~b['valid']],0.)


def test_normal_field_uses_camera_frame_not_global_normal_components():
    _,g,_=unit_fixture();ref=g['reference'];angle=.4
    # Rotation about worldY, with a plane depth computed from each camera ray.
    r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1.,0],[-np.sin(angle),0,np.cos(angle)]])
    ref['cam2world'][:3,:3]=r@ref['cam2world'][:3,:3]
    yy,xx=np.mgrid[:512,:512];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(ref['intrinsics']).T
    direction=rays@ref['cam2world'][:3,:3].T
    ref['axial_depth']=-ref['cam2world'][2,3]/direction[...,2]
    out=reference_normal_field(ref);expected=np.array([0.,0.,1.])@ref['cam2world'][:3,:3]
    np.testing.assert_allclose(out['normal_xy'][:,10:-10,10:-10],np.broadcast_to(expected[:2,None,None],(2,108,108)),atol=5e-5,rtol=0)


def test_target_view_basis_uses_true_camera_and_negative_down_column():
    m,g,_=unit_fixture();world=np.array([r['world']for r in m['physical_points']]);ref=g['reference']['cam2world'];query=g['query']['cam2world']
    v=training_view_basis(world,ref,query)
    los=query[:3,3]-world;los/=np.linalg.norm(los,axis=1,keepdims=True)
    np.testing.assert_allclose(v[:,1:4],los@ref[:3,:3],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(v[:,4:],np.tile([0.,-1.,0.],(len(v),1)))
    assert not np.array_equal(v[:,4:],np.tile([0.,1.,0.],(len(v),1)))
    np.testing.assert_allclose(v[:,0],1.)


def test_builder_retains_both_native_views_all_even_lights_and_same_full128_padding():
    u,m,g,lights,reader,teacher=build_fixture();n=len(m['physical_points'])
    assert len(reader.calls)==24 and set(reader.calls)=={(r,i)for r in ('reference','query')for i in EVEN}
    assert len(teacher.calls)==24 and all(a.shape==(128,128)for a,p in teacher.calls)
    assert u['teacher'].shape==(n,2,24,32)and u['available'][:,:,::2].all()
    assert not u['available'][:,:,1::2].any()and np.isnan(u['teacher'][:,:,1::2]).all()
    assert not np.array_equal(u['teacher'][:,0,0],u['teacher'][:,1,0])
    np.testing.assert_array_equal(u['reference_images'][0],cv2.resize(EvenToyReader().get('reference',0),(128,128),interpolation=cv2.INTER_AREA))
    for j,(image,p)in enumerate(teacher.calls):np.testing.assert_array_equal(p,u['coarse_xy'][:,j%2])
    assert not u['details']['new_coverage_or_teacher_omissions']
    assert u['details']['feature_context_covisibility'].startswith('unknown')


def test_condition_suns_normals_reference_view_and_scale_are_in_reference_camera():
    u,m,g,lights,_,_=build_fixture();ref=g['reference']['cam2world'];n=len(m['physical_points'])
    expected=np.asarray([x['sun_enu']for x in lights])@ref[:3,:3]
    np.testing.assert_allclose(u['target_suns'],expected,atol=1e-7,rtol=0)
    np.testing.assert_allclose(u['observed_conditions'][0,:,:3],expected[list(EVEN)],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(u['observed_conditions'][:,:,3:6],np.broadcast_to([0.,0.,-1.],(n,12,3)))
    np.testing.assert_allclose(u['observed_conditions'][:,0,6:9],u['target_view_basis'][:,0,1:4],atol=0,rtol=0)
    assert not u['observed_conditions'][:,:,9].any()


def test_legacy163_style_omissions_preserved_new_support_never_drops_camera_or_rows():
    u,m,_,_,_,_=build_fixture([[49.5,49.5],[257.5,257.5]])
    assert len(u['world'])==2 and u['details']['retained_point_rows']==2
    assert u['coarse_support'][0,0]and not u['coarse_support'][0,1]
    assert not u['available'][0,1].any()and np.isnan(u['teacher'][0,1]).all()
    assert u['details']['legacy_trial_support_omissions']==1
    assert u['details']['legacy_trial_support_failure_records']==m['teacher_extraction']['canonical_trial_support_failures']
    assert u['details']['legacy_selected_indices']==m['teacher_extraction']['selected_indices']
    new=u['details']['new_coverage_or_teacher_omissions']
    assert new==[dict(label_index=22,role='query',reason='coarse_receptive_field_or_geometry_support')]
    assert u['correspondence_visible'].all()and u['details']['all_legacy_rows_and_camera_retained']


def test_zero_teacher_is_availability_failure_not_visibility_and_nonfinite_is_fatal():
    u,*_=build_fixture(teacher=lambda image,points:np.zeros((len(points),32),np.float32))
    assert not u['available'].any()and np.isnan(u['teacher']).all()
    assert u['correspondence_visible'].all()and len(u['world'])==3
    assert all(r['reason']=='zero_norm_teacher_not_visibility_label'for r in u['details']['new_coverage_or_teacher_omissions'])
    with pytest.raises(ValueError,match='Finite'):
        build_fixture(teacher=lambda image,points:np.full((len(points),32),np.nan,np.float32))


@pytest.mark.parametrize('change',['odd_schedule','foreign_split','bad_projection','changed_truth','bad_normal'])
def test_bad_training_labels_rejected_before_any_teacher_or_image_access(change):
    m,g,l=unit_fixture();r=EvenToyReader();teacher=Full128ToyTeacher()
    if change=='odd_schedule':m['source_lights']=[1]+list(EVEN[1:])
    elif change=='foreign_split':m['split']='model_selection'
    elif change=='bad_projection':m['physical_points'][0]['query_xy'][0]+=1.
    elif change=='changed_truth':m['query_training_truth_cam2world'][0][3]+=1.
    else:m['physical_points'][0]['normal']=[0.,0.,2.]
    with pytest.raises(ValueError):build_coarse_training_unit(m,g,r,l,teacher)
    assert not r.calls and not teacher.calls


def inventory_fixture(groups=16):
    bank=dict(split='training',lighting_indices=list(EVEN),heldout_image_reads=0,records=[]);metadata=[]
    for group in range(groups):
        for camera in range(4):
            m,_,_=unit_fixture();i=group*4+camera;name='train'+str(group)
            m.update(unit_index=i,component=name)
            bank['records'].append(dict(unit_index=i,component=name,split='training'))
            metadata.append(m)
    return bank,metadata,list(range(groups*4)),['train'+str(i)for i in range(groups)]


def test_full64_metadata_inventory_and_odd_or_foreign_or_dropped_unit_guards():
    b,m,ids,g=inventory_fixture();assert validate_metadata_inventory(b,m,ids,g)
    for mutate in ['odd','reserve','dropped','duplicate']:
        bb,mm=deepcopy(b),deepcopy(m)
        if mutate=='odd':mm[0]['teacher_extraction']['even_images_decoded'][0]['light']=1
        elif mutate=='reserve':mm[0]['component']='final_reserve'
        elif mutate=='dropped':bb['records'].pop();mm.pop()
        else:bb['records'][1]['unit_index']=0
        with pytest.raises(ValueError):validate_metadata_inventory(bb,mm,ids,g)


def test_full_geometry_gate_runs_before_bank_file_or_feature_access(monkeypatch,tmp_path):
    def blocked(*args):raise ValueError('independent full96 gate not passed')
    monkeypatch.setitem(sys.modules,'solar_training_bank_v68',SimpleNamespace(verify_geometry_gate=blocked))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='full96'):
        load_training_metadata_index(tmp_path,'nonexistent-bank.json','master.json','data.json','audit.json')


def test_metadata_loader_pins_all_records_without_opening_npz_or_images(monkeypatch,tmp_path):
    b,m,ids,groups=inventory_fixture(1);b.update(geometry_audit_sha256='a'*64,original_master_sha256='b'*64,dataset_summary_sha256='c'*64)
    for i,(record,meta)in enumerate(zip(b['records'],m)):
        path=tmp_path/(str(i)+'.json');path.write_text(json.dumps(meta));record.update(metadata_path=path.name,metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path='SHOULD_NOT_BE_OPENED.npz')
    (tmp_path/'bank.json').write_text(json.dumps(b))
    gate=SimpleNamespace(geometry_audit_sha256='a'*64,original_master_sha256='b'*64,dataset_summary_sha256='c'*64,eligible_training_indices=tuple(ids),training_groups=tuple(groups))
    monkeypatch.setitem(sys.modules,'solar_training_bank_v68',SimpleNamespace(verify_geometry_gate=lambda *args:gate))
    monkeypatch.chdir(tmp_path)
    result=load_training_metadata_index(tmp_path,'bank.json','master','data','audit')
    assert len(result[1])==4
    (tmp_path/'0.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):load_training_metadata_index(tmp_path,'bank.json','master','data','audit')


def test_view_prior_equal_group_camera_point_weighting_not_lighting_repeats():
    def unit(index,group,n,los):
        v=np.tile([1.,*los,0.,0.,1.],(n,2,1))
        return dict(details=dict(unit_index=index,component=group,split='training'),target_view_basis=v)
    units=[unit(0,'A',1,[1.,0.,0.]),unit(1,'A',3,[-1.,0.,0.]),unit(2,'B',20,[0.,1.,0.])]
    prior=fit_balanced_training_view_prior(units,allowed_training_groups=['A','B'],eligible_unit_indices=[0,1,2],source_sha256={'fixture':'0'*64})
    np.testing.assert_allclose(prior.mean,[1.,0.,.5,0.,0.,0.,1.],atol=1e-12,rtol=0)
    assert prior.provenance['sample_count']==24 and not prior.provenance['lighting_duplicates_counted']
    with pytest.raises(ValueError,match='camera'):
        fit_balanced_training_view_prior(units[:-1],allowed_training_groups=['A','B'],eligible_unit_indices=[0,1,2],source_sha256={'fixture':'0'*64})
