"""Procedural staged physics and control isolation; no lunar inference/BVH job."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from solar_assignment_v63 import CandidateGraph,initial_beams,assignments,solve,acquisition_predictions
from test_solar_assignment_v63 import fixture
from solar_lookup_worker_v70 import reference_anchor_intersections,lookup_states
from solar_map_v63 import reference_solar_table
from solar_physics_v63 import TerrainEnvelope,solar_direction
from test_solar_frontend_v63 import geometry
from solar_frontend_v63 import ReferenceGeometry


def test_off_beams_are_different_assignments_not_eight_unused_suns():
    graph,k,truth,false=fixture();n=len(graph.world)
    query=np.stack([np.column_stack([30+np.arange(n)%12*34+c,40+np.arange(n)//12*34+c])for c in range(8)],axis=1)
    graph=replace(graph,query_xy=query,appearance=np.tile(np.linspace(.9,.2,8),(n,1)),observed_shadow=np.full((n,8),.5))
    beams=initial_beams(graph,'off',8,False)
    assert len(beams)==8 and len(set(tuple(b['assignment']['candidate'])for b in beams))==8
    assert len(set(b['sun_index']for b in beams))==1


def test_staged_uncomputed_columns_cannot_masquerade_as_unknown_and_archive_not_mutated():
    graph,k,truth,false=fixture();n=len(graph.world)
    parameters=np.array([[40.,1.5],[41.25,1.5],[45.,1.5]])
    suns=np.array([solar_direction(80.,h,d)for h,d in parameters]);ideal=1+np.arange(n)%2
    states=np.column_stack([ideal,np.zeros(n,int),3-ideal]);observed=np.column_stack([np.where(ideal==2,.95,.05),np.full(n,.5)])
    g=CandidateGraph(graph.reference_xy,graph.world,graph.query_xy,graph.appearance,observed,states,suns,graph.world_variants,np.repeat(states[:,:,None],7,2),parameters,np.array([True,False,True]),np.array([True,False,True]))
    with pytest.raises(ValueError,match='uncomputed'):assignments(g,1)
    with pytest.raises(ValueError,match='source-pinned'):solve(g,k,lambda i,c,*_:dict(query_xy=g.query_xy[i,c]),budget=32)
    requested=[]
    def lookup(vectors):
        requested.extend(vectors);return dict(solar_states=np.repeat(ideal[:,None],len(vectors),1),solar_variants=np.repeat(np.repeat(ideal[:,None,None],len(vectors),1),7,2))
    result=solve(g,k,lambda i,c,*_:dict(query_xy=g.query_xy[i,c]),budget=32,solar_lookup=lookup)
    assert len(requested)==1;np.testing.assert_allclose(requested[0],suns[1]);assert result['solar_columns_materialized']==3
    assert not g.solar_materialized_mask[1]and(g.solar_states[:,1]==0).all()


def test_solar_off_active_predictions_do_not_read_hidden_unused_shadow_measurement():
    graph,k,truth,false=fixture();beams=initial_beams(graph,'off',8,False)
    for beam in beams:beam['pose']=truth
    actions=np.array([(0,0),(1,1)])
    pred,weights,valid=acquisition_predictions(graph,beams,actions,k,'off')
    assert np.isnan(pred[:,:,:,2]).all()and not valid[:,:,:,2].any()


def test_cached_reference_intersections_and_staged_rays_equal_original_dense_physics():
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=100.
    def raycast(origin,direction,limit):
        if direction[2]>=0:return None
        distance=-origin[2]/direction[2]
        if distance<=0 or distance>limit:return None
        return np.asarray(origin)+distance*np.asarray(direction),np.array([0.,0.,1.])
    world,normals=reference_anchor_intersections(k,pose,raycast)
    env=TerrainEnvelope(-100,100,-100,100,30.,True)
    suns=np.array([[0.,0.,1.],[1.,0.,.01]]);suns/=np.linalg.norm(suns,axis=1,keepdims=True)
    lazy=lookup_states(world,normals,suns,env,raycast)
    axis=49.5+16*np.rint(np.linspace(0,25,16));yy,xx=np.meshgrid(axis,axis,indexing='ij')
    dense,_=reference_solar_table(np.column_stack([xx.ravel(),yy.ravel()]),k,pose,suns,env,raycast)
    np.testing.assert_array_equal(lazy['solar_states'],dense['solar_states']);np.testing.assert_array_equal(lazy['solar_variants'],dense['solar_variants'])
    assert (np.asarray(lazy['solar_states'])[:,1]==0).all() # low ray loses map before ORIGINAL scene ceiling


def test_legacy_archive_materialized_default_and_new_mask_roundtrip(tmp_path):
    g,k=geometry();legacy={n:getattr(g,n)for n in g.__dataclass_fields__ if n not in('sun_parameters','coarse_sun_mask','solar_materialized_mask')}
    path=tmp_path/'legacy.npz';np.savez(path,**legacy)
    assert ReferenceGeometry.load(path).solar_materialized_mask.all()
    g.solar_materialized_mask[:]=False;np.savez(path,**{n:getattr(g,n)for n in g.__dataclass_fields__})
    assert not ReferenceGeometry.load(path).solar_materialized_mask.any()


def test_lookup_protocol_source_and_all_geometry_identity_are_pinned_without_worker(tmp_path,monkeypatch):
    import solar_lookup_v70 as lookup
    monkeypatch.setattr(lookup,'__file__',str(tmp_path/'solar_lookup_v70.py'))
    for name in lookup.SOURCES:(tmp_path/name).write_text(name+'\n')
    scene=tmp_path/'scene.npz';np.savez(scene,vertices=np.array([[0.,0.,2.],[1,1,3]]))
    mesh=tmp_path/'map.npz';np.savez(mesh,vertices=np.array([[0.,0.,0.]]))
    ref=tmp_path/'reference.npz';np.savez(ref,intrinsics=np.eye(3),cam2world=np.eye(4))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(passed=True,oracle_success=1.,approved_reference_inputs={'terrain/pair-0':dict(reference_geometry_sha256=lookup.sha(ref),scene_mesh_sha256=lookup.sha(scene),scene_maximum_height_m=3.)})))
    protocol=dict(schema_version=70,cpu_threads=2,query_truth_used=False,max_lookup_calendar_seconds=120,max_request_seconds=60,approved_reference_key='terrain/pair-0',source_sha256={n:lookup.sha(tmp_path/n)for n in lookup.SOURCES})
    for name,path in [('mesh',mesh),('scene_mesh',scene),('reference_geometry',ref),('geometry_audit',audit)]:protocol[name]=str(path);protocol[name+'_sha256']=lookup.sha(path)
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol));p,key=lookup.validate_protocol(path,'unused',False)
    provider=lookup.ReferenceSunLookup(path,'unused','never-launch',p,key)
    assert provider.process is None and provider.stats['computed_columns']==0
    protocol['true_query_sun']=[1,0,0];path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='extra/query'):lookup.validate_protocol(path,'unused',False)
    protocol.pop('true_query_sun');protocol['scene_mesh_sha256']='0'*64;path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='changed'):lookup.validate_protocol(path,'unused',False)


def test_arbitrary_secondary_sun_is_exact_and_not_added_to_primary_geometry(monkeypatch):
    import solar_frontend_v63 as frontend
    import torch
    from compact_lunar import CompactDescriptor
    g,k=geometry();p=frontend.reference_anchors();calls=[];seen=[]
    monkeypatch.setattr(frontend,'candidates',lambda *args:(p.copy(),np.repeat(p[:,None],8,1),np.ones((256,8)),np.zeros((256,8),int)))
    monkeypatch.setattr(frontend,'solve',lambda graph,*args,**kwargs:seen.append(graph)or dict(pose=None,reason='procedural_capture'))
    def lookup(suns):
        calls.append(suns.copy());return dict(solar_states=np.ones((256,len(suns)),np.uint8),solar_variants=np.ones((256,len(suns),7),np.uint8))
    model=CompactDescriptor(False);matcher=frontend.SolarMatcher(model,model)
    image=np.random.default_rng(2).integers(0,256,(512,512),dtype=np.uint8)
    exact=np.array([.123456789,.765432109,.631113579]);exact/=np.linalg.norm(exact)
    matcher.localize(image,image,g,k,solar_lookup=lookup)
    assert not calls
    matcher.localize(image,image,g,k,known_sun=exact,solar_lookup=lookup)
    assert len(calls)==1;np.testing.assert_array_equal(calls[0][0],exact);np.testing.assert_array_equal(seen[-1].sun_vectors[0],exact)
    assert len(g.sun_vectors)==1;np.testing.assert_array_equal(g.sun_vectors[0],[0.,0.,-1.])


def test_draft_or_integrated_frontend_source_origin_is_the_tested_folder():
    import solar_frontend_v63,solar_assignment_v63,solar_lookup_v70
    for module in(solar_frontend_v63,solar_assignment_v63,solar_lookup_v70):assert Path(module.__file__).resolve().parent==Path(__file__).resolve().parent
