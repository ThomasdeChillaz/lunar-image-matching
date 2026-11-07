"""Procedural schema/geometry/case controls only; no rays, renders or forwards."""
from pathlib import Path
import copy
import json
import numpy as np
import pytest
import solar_eval_data_v69 as m


def gate_fixture():
    p=dict(schema_version=69,phase='development_case_assets',split='model_selection',
      calibration_or_final_access=False,cpu_threads=2,max_calendar_seconds=1800,light_pairs=[list(x)for x in m.LIGHT_PAIRS],
      source_sha256={'solar_physics_v63.py':'physical-source'},geometry_audit_sha256='audit',
      original_master_sha256='master',dataset_summary_sha256='data',geometry_audit_source_sha256='auditor')
    units=[];master=[]
    for i in range(96):
        split='training'if i<64 else'model_selection';component=f'g{i//4:02d}'
        units.append(dict(unit_index=i,component=component,split=split,eligible=True))
        master.append(dict(site=dict(component=component,split=split),source_sha256={'solar_physics_v63.py':'physical-source'}))
    audit=dict(passed=True,complete_dataset=True,expected_units=96,completed_units=96,units=units,
      oracle_success=1.,oracle_pose_directions=192,correct_oracle_poses=192,terrain_counts={'training':16,'model_selection':8},
      original_master_sha256='master',completion_summary_sha256='data',audit_source_sha256='auditor',calibration_or_final_access=False)
    data=dict(complete=True,rendered_images=4608,units=[{'unit_index':i}for i in range(96)],original_master_sha256='master',matcher_forwards=0,optimizer_updates=0,calibration_or_final_access=False)
    assets=dict(schema_version=69,split='model_selection',geometry_audit_sha256='audit',calibration_or_final_access=False,units=[{'unit_index':i}for i in range(64,96)])
    return p,data,audit,{'units':master},assets


def unit_fixture():
    units={}
    for g in range(8):
        for camera in range(4):
            i=4*g+camera;component=f'g{g}';asset=dict(reference_geometry=f'archive-{i}.npz',reference_geometry_sha256=f'archive-sha-{i}',
              solar_lookup_protocol=f'lookup-{i}.json',scene_mesh_sha256=f'scene-{g}')
            units[i]=dict(unit_index=i,component=component,split='model_selection',eligible=True,
              family='motion'if camera<2 else'roll',pair_index=camera%2,body_bounds=[g*10.,0,g*10.+2,2.],
              enu_rows=np.eye(3),asset=asset,physical_inputs={},raw={'query':{'intrinsics':np.eye(3)}},
              suns={i:([1.,0,0]if i in(0,1)else[0,1.,0])for i in range(24)},
              sealed={'images':{(r,l):(f'{i}-{r}-{l}.png',f'hash-{i}-{r}-{l}')for r in('reference','query')for l in range(24)}})
    generated={i:dict(scoring_labels=f'labels-{i}.npz',scoring_geometry=f'scoring-{i}.npz',
      scoring_input_sha256={f'labels-{i}.npz':f'lh-{i}',f'scoring-{i}.npz':f'gh-{i}'})for i in units}
    return units,generated


def test_full_gate_requires_every_original_unit_and_split():
    args=gate_fixture();assert len(m.validate_gate(*args))==32
    bad=copy.deepcopy(args);bad[2]['completed_units']=95
    with pytest.raises(ValueError,match='Full'):m.validate_gate(*bad)
    bad=copy.deepcopy(args);bad[2]['units'][64]['split']='final'
    with pytest.raises(ValueError,match='split'):m.validate_gate(*bad)


def test_pose_failures_remain_eligible_and_in_oracle_denominator():
    p,d,a,master,assets=gate_fixture()
    a.update(correct_oracle_poses=190,oracle_success=190/192)
    assert len(m.validate_gate(p,d,a,master,assets))==32
    a['oracle_pose_directions']=190
    with pytest.raises(ValueError,match='denominator'):m.validate_gate(p,d,a,master,assets)


def test_nonfinite_gate_and_different_physical_lighting_source_rejected():
    args=gate_fixture();args[2]['oracle_success']=float('nan')
    with pytest.raises(ValueError,match='Full'):m.validate_gate(*args)
    args=gate_fixture();args[3]['units'][0]['source_sha256']['solar_physics_v63.py']='changed'
    with pytest.raises(ValueError,match='renderer physics'):m.validate_gate(*args)


def test_all_eligible_camera_lights_and_balanced_nonoverlap_retained():
    units,_=unit_fixture();rows=m.plan_cases(list(units.values()))
    positives=[r for r in rows if r['positive']];negatives=[r for r in rows if not r['positive']]
    assert len(positives)==len(negatives)==32*6
    assert {(r['unit_index'],r['light_a'],r['light_b'])for r in positives}=={(i,*p)for i in units for p in m.LIGHT_PAIRS}
    assert {r['matched_positive_case']for r in negatives}=={r['case_id']for r in positives}
    for r in negatives:
        a,b=units[r['unit_index']],units[r['query_unit_index']]
        assert a['component']!=b['component']and(a['family'],a['pair_index'])==(b['family'],b['pair_index'])
        assert m.bounds_disjoint(a['body_bounds'],b['body_bounds'])


def test_no_negative_certificate_never_silently_drops_cases():
    units,_=unit_fixture()
    for u in units.values():u['body_bounds']=[0,0,2,2]
    with pytest.raises(ValueError,match='No certified'):m.plan_cases(list(units.values()))


def test_full_scene_bounds_include_vertices_not_just_camera_hits():
    mesh=dict(vertices=np.array([[-100.,-20,2],[100,-20,10],[0,100,4]]),faces=np.array([[0,1,2]]),body_origin=np.array([1000.,2000,3000]),enu_rows=np.eye(3))
    bounds,ceiling,origin,rows=m.body_scene_bounds(mesh)
    assert bounds[0]<900 and bounds[2]>1100 and bounds[1]<1980 and bounds[3]>2100 and ceiling==10
    # A cropped reference map cannot lower the ORIGINAL scene ceiling.
    mesh['vertices'][1,2]=50
    assert m.body_scene_bounds(mesh)[1]==50


def test_negative_exact_sun_is_transformed_between_two_terrain_frames():
    rotation=np.array([[0,1.,0],[-1.,0,0],[0,0,1.]])
    assert np.allclose(m.convert_sun_frame([1.,0,0],rotation,np.eye(3)),[0,1,0])
    with pytest.raises(ValueError):m.convert_sun_frame([1.,0,0],np.zeros((3,3)),np.eye(3))


def test_pure_case_builder_keeps_truth_out_of_primary_inputs():
    units,generated=unit_fixture();plans=m.plan_cases(list(units.values()));positive=plans[0];negative=plans[1]
    a=m.materialize_case(positive,units,generated)
    assert set(a['inference_inputs'])=={'reference','query','reference_geometry','query_intrinsics','reference_sun_world','solar_lookup_protocol'}
    assert a['reference_sun_world']==units[0]['suns'][0]
    assert a['scoring_labels']in a['input_sha256']and a['scoring_geometry']in a['input_sha256']
    assert not({'scoring_labels','scoring_geometry','query_cam2world','query_depth','secondary_known_query_sun_world'}&set(a['inference_inputs']))
    query=units[negative['query_unit_index']];query['enu_rows']=np.array([[0,1.,0],[-1.,0,0],[0,0,1.]])
    b=m.materialize_case(negative,units,generated)
    assert b['scoring_inputs']=={}and'scoring_labels'not in b
    assert b['nonoverlap_certificate']['frame']==m.FRAME and b['nonoverlap_certificate']['full_image_footprints']
    assert np.allclose(b['secondary_known_query_sun_world'],[0,1,0])
    assert b['component']!=b['query_component']


def reference_archive():
    return dict(depth=np.ones((512,512)),intrinsics=np.eye(3),cam2world=np.eye(4),
      normals=np.tile([0.,0.,1.],(256,1)),sun_vectors=np.array([[0.,0.,1.]]),
      solar_states=np.zeros((256,1),np.uint8),solar_variants=np.zeros((256,1,7),np.uint8),
      reference_sun=np.array([0.,0.,1.]),world_variants=np.zeros((256,7,3)))


def test_reference_archive_rejects_query_truth_and_uncomputed_coarse(tmp_path):
    archive=reference_archive();archive['query_cam2world']=np.eye(4);p=tmp_path/'bad.npz';np.savez_compressed(p,**archive)
    with pytest.raises(ValueError,match='query truth'):m.validate_reference_archive(p)
    archive.pop('query_cam2world');archive.update(coarse_sun_mask=np.array([True]),solar_materialized_mask=np.array([False]));np.savez_compressed(p,**archive)
    with pytest.raises(ValueError,match='physically computed'):m.validate_reference_archive(p)


def test_primary_catalogue_is_generic_and_never_query_specific():
    catalogue=m.sun_search_catalogue(80.)
    geometry=dict(sun_vectors=np.array([x['sun_enu']for x in catalogue]),
      sun_parameters=np.array([[x['hour_angle_deg'],x['subsolar_latitude_deg']]for x in catalogue]),
      coarse_sun_mask=np.array([x['coarse']for x in catalogue],bool),solar_materialized_mask=np.array([x['coarse']for x in catalogue],bool))
    assert m.validate_generic_catalogue(geometry,80.)
    geometry['sun_vectors']=np.r_[geometry['sun_vectors'],[[1.,0,0]]]
    with pytest.raises(ValueError,match='a-priori'):m.validate_generic_catalogue(geometry,80.)


def physical_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def save(path,x):Path(path).write_text(json.dumps(x,indent=2)+'\n')
    for name in m.LOOKUP_SOURCES|m.PRECOMPUTE_SOURCES:Path(name).write_text('# PROCEDURAL SOURCE FIXTURE\n')
    for name in('map.npz','scene.npz','raw-reference.npz','audit.json','archive.npz'):Path(name).write_text('PROCEDURAL INPUT BYTES')
    key='g0/motion-0';auditsha=m.sha('audit.json');approved=dict(reference_geometry_sha256=m.sha('raw-reference.npz'),scene_mesh_sha256=m.sha('scene.npz'))
    common=dict(cpu_threads=2,query_truth_used=False,mesh='map.npz',mesh_sha256=m.sha('map.npz'),scene_mesh='scene.npz',scene_mesh_sha256=m.sha('scene.npz'),
      reference_geometry='raw-reference.npz',reference_geometry_sha256=m.sha('raw-reference.npz'),geometry_audit='audit.json',geometry_audit_sha256=auditsha,approved_reference_key=key)
    lookup=dict(common,schema_version=70,max_lookup_calendar_seconds=900,max_request_seconds=300,source_sha256={n:m.sha(n)for n in m.LOOKUP_SOURCES})
    pre=dict(common,output='archive.npz',max_calendar_seconds=900,source_sha256={n:m.sha(n)for n in m.PRECOMPUTE_SOURCES})
    save('lookup.json',lookup);save('precompute.json',pre)
    receipt=dict(complete=True,output_sha256=m.sha('archive.npz'),protocol_sha256=m.sha('precompute.json'),
      original_scene_mesh_sha256=m.sha('scene.npz'),original_scene_ceiling_m=100.,map_mesh_sha256=m.sha('map.npz'),
      query_truth_used=False,neural_forwards=0,source_sha256=pre['source_sha256'],calendar_seconds=12.)
    save('archive.npz.json',receipt)
    asset=dict(reference_geometry='archive.npz',reference_geometry_sha256=m.sha('archive.npz'),scene_mesh_sha256=m.sha('scene.npz'),scene_maximum_height_m=100.,
      solar_lookup_protocol='lookup.json',solar_lookup_protocol_sha256=m.sha('lookup.json'),reference_precompute_protocol='precompute.json',
      reference_precompute_protocol_sha256=m.sha('precompute.json'),reference_precompute_receipt='archive.npz.json',reference_precompute_receipt_sha256=m.sha('archive.npz.json'))
    monkeypatch.setattr(m.subprocess,'check_output',lambda argv:Path(argv[-1].split(':',1)[1]).read_bytes())
    return asset,approved,dict(geometry_audit_sha256=auditsha),key,receipt


def test_physical_precompute_receipt_and_lookup_lineage_are_required(tmp_path,monkeypatch):
    asset,approved,p,key,receipt=physical_fixture(tmp_path,monkeypatch)
    seconds,inputs=m.validate_physical_assets(asset,approved,p,'a'*40,key)
    assert seconds==12 and asset['reference_precompute_receipt']in inputs and asset['solar_lookup_protocol']in inputs
    receipt['original_scene_ceiling_m']=50
    Path(asset['reference_precompute_receipt']).write_text(json.dumps(receipt))
    asset['reference_precompute_receipt_sha256']=m.sha(asset['reference_precompute_receipt'])
    with pytest.raises(ValueError,match='ceiling'):m.validate_physical_assets(asset,approved,p,'a'*40,key)


def test_template_and_missing_publication_commit_cannot_build_assets():
    with pytest.raises(ValueError,match='publication commit'):m.verify_publication('does-not-exist',None,{})
    with pytest.raises(ValueError,match='Prepared template'):m.verify_publication('does-not-exist','a'*40,{'prepared_only':True})


def test_payload_paths_cannot_escape_unit():
    with pytest.raises(ValueError):m.safe_child('/dummy','../query_truth.npz')
    with pytest.raises(ValueError):m.safe_child('/dummy','/absolute/private')


def test_case_manifest_interoperates_with_independent_evaluation_adapter():
    import sys
    try:
        from solar_evaluation_v70 import validate_manifest
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solar_evaluation_v70'))
        from solar_evaluation_v70 import validate_manifest
    units,generated=unit_fixture();shifted={};sg={}
    for i,u in units.items():
        shifted[i+64]=dict(u,unit_index=i+64,component=f'g{(i+64)//4:02d}')
        sg[i+64]=generated[i]
    plans=m.plan_cases(list(shifted.values()))
    cases=[m.materialize_case(p,shifted,sg)for p in plans]
    manifest=dict(schema_version=70,split='model_selection',calibration_or_final_access=False,
      terrain_groups=sorted({u['component']for u in shifted.values()}),footprint_frame=m.FRAME,cases=cases)
    audit=gate_fixture()[2]
    assert len(validate_manifest(manifest,audit))==384


def test_reference_index_fails_before_geometry_or_output_on_incomplete_gate(tmp_path,monkeypatch):
    p,d,a,master,assets=gate_fixture();a['complete_dataset']=False
    paths={}
    for name,value in(('dataset_summary',d),('geometry_audit',a),('original_master',master)):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));paths[name]=str(path);paths[name+'_sha256']=m.sha(path)
    source=tmp_path/'auditor.py';source.write_text('# procedural source')
    p.update(paths,phase='reference_asset_index',geometry_audit_source=str(source),geometry_audit_source_sha256=m.sha(source),output=str(tmp_path/'MUST_NOT_EXIST.json'))
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(p))
    monkeypatch.setattr(m,'verify_publication',lambda *args:None)
    monkeypatch.setattr(m.subprocess,'check_output',lambda argv:source.read_bytes())
    monkeypatch.setattr(m,'load_npz',lambda *args:(_ for _ in()).throw(AssertionError('Geometry read before gate')))
    with pytest.raises(ValueError,match='Full'):m.build_reference_index(str(protocol),'a'*40)
    assert not Path(p['output']).exists()


def test_computed_coverage_unknown_is_distinct_from_uncomputed(tmp_path):
    archive=reference_archive();archive.update(coarse_sun_mask=np.array([True]),solar_materialized_mask=np.array([True]))
    path=tmp_path/'computed_unknown.npz';np.savez_compressed(path,**archive)
    z=m.validate_reference_archive(path)
    assert z['solar_materialized_mask'].all()and not z['solar_states'].any()
