"""Procedural/stub adapter tests. NO actual pretrained network is executed.

Temporary receipt-shaped fixtures below are synthetic unit-test inputs, never
scientific control evidence, never saved in runs/reports, and never published
as successful pretrained results. Existing real receipts are read separately.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import native_pretrained_adapters_v73 as adapters
# Pin the intended draft modules before repository tests modify sys.path.
import solar_evaluation_v70
import run_solar_evaluation_v70


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n');return path


def raw_control():
    yy,xx=np.mgrid[40:441:80,40:441:80];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(float);q=p+[8,-4]
    pose=np.eye(4);pose[:3,3]=[-.2,.1,0]
    return dict(passed=True,scope='SYNTHETIC TEST FIXTURE, NOT ACTUAL PRETRAINED EVIDENCE',reference_points_native=p.tolist(),query_points_native=q.tolist(),estimated_cam2world=pose.tolist(),matches=len(p),inliers=len(p),median_native_error_px=0.,translation_error_m=0.,rotation_error_deg=0.,warm_match_seconds=.01,blank_empty=True,invalid_mask_empty=True,known_translation_native=[8.,-4.],image_seed=20261101)


def fixture(tmp_path,method='matchanything',author_profile=None):
    spec=adapters.METHODS[method];author_profile=author_profile or spec['profiles'][0];repo=tmp_path/'repository';repo.mkdir();out=tmp_path/'synthetic-unit-test-evidence';out.mkdir()
    profile=dict(name=method,backend='native_pretrained',method=method,author_profile=author_profile,repository_root=str(repo),cpu_threads=2)
    needed=adapters.required_native_sources(profile)-{'native_pretrained_adapters_v73.py'}
    for name in needed:(repo/name).write_text('Synthetic source pin fixture for '+name+'\n')
    lock=dict(source_revision='a'*40,safe_weight_sha256='b'*64,source_repository='author/repository',source_license='TEST-ONLY',weight_repository='test/weights',weight_revision='c'*40,dependencies={'torch':'TEST-ONLY'})
    write(repo/spec['lock'],lock)
    internal=832 if author_profile=='training_size832' else 512
    provenance=dict(method=spec['display'],profile=author_profile,native_input_size=512,internal_input_size=internal,source_revision=lock['source_revision'],source_repository=lock['source_repository'],source_license=lock['source_license'],source_lock_sha256=adapters.sha(repo/spec['lock']),weight_sha256=lock['safe_weight_sha256'],safe_weight_sha256=lock['safe_weight_sha256'],npe=[832,832,internal,internal],reparameterization=False,cpu_threads=2,trainable_parameters=0,environment={'torch':'TEST-ONLY'},weight_repository=lock['weight_repository'],weight_revision=lock['weight_revision'],dependencies=lock['dependencies'],preprocessing='TEST-ONLY full512 image',matching='TEST-ONLY',query_geometry_in_interface=False,total_parameters=123)
    protocol=dict(cpu_threads=2,max_seconds=60,lunar_images_loaded=0,calibration_or_test_pixels_loaded=0,source_sha256={name:adapters.sha(repo/name)for name in needed})
    controls=raw_control();summary=dict(passed=True,actual_native_pretrained_forward_attempted=True,elapsed_seconds=.1,controls=deepcopy(controls),source_sha256=protocol['source_sha256'][spec['control']])
    for name,value in [('protocol',protocol),('provenance',provenance),('controls',controls),('summary',summary)]:write(out/(name+'.json'),value)
    profile=adapters.native_profile(method,method,author_profile,out,repo)
    return profile,repo,out,provenance


def update(profile,name,mutator):
    evidence=profile['pretrained_evidence'];path=Path(evidence[name]);value=adapters.read(path);mutator(value);write(path,value);evidence[name+'_sha256']=adapters.sha(path)


@pytest.mark.parametrize('method,author_profile',[('matchanything','native512'),('matchanything','training_size832'),('moonst3r','common_native512_full_field_of_view')])
def test_matching_actual_gate_schema_can_be_replayed_without_loading_network(tmp_path,method,author_profile):
    profile,repo,out,provenance=fixture(tmp_path,method,author_profile)
    gate=adapters.validate_native_profile(profile)
    assert gate['control_replay']['matches']==36 and not gate['isolated_latency_established']and not gate['primary_query_truth_inputs']
    assert gate['semantic_provenance']['internal_input_size']==(832 if author_profile=='training_size832' else 512)


@pytest.mark.parametrize('change,match',[
    (lambda c:c.update(blank_empty=False),'all pass'),
    (lambda c:c.update(invalid_mask_empty=False),'all pass'),
    (lambda c:c.update(passed=False),'all pass'),
    (lambda c:c.update(reference_points_native=[]),'correspondences'),
    (lambda c:c.update(median_native_error_px=.5),'arithmetic'),
    (lambda c:c.update(known_translation_native=[2.,-1.]),'transform differs'),
    (lambda c:c.update(warm_match_seconds=0),'timing'),
])
def test_control_boolean_does_not_replace_raw_valid_native_transform_replay(change,match):
    control=raw_control();change(control)
    with pytest.raises(ValueError,match=match):adapters.replay_native_control(control)


def test_coarse128_coordinates_cannot_pass_as_native512_control():
    control=raw_control();control['reference_points_native']=(np.asarray(control['reference_points_native'])/4).tolist();control['query_points_native']=(np.asarray(control['query_points_native'])/4).tolist()
    with pytest.raises(ValueError,match='arithmetic'):adapters.replay_native_control(control)


def test_missing_failed_or_mutated_gate_rejected_before_constructing_any_model(tmp_path,monkeypatch):
    profile,repo,out,provenance=fixture(tmp_path)
    monkeypatch.setattr(adapters,'_construct_model',lambda *args:pytest.fail('Invalid gate reached model construction'))
    original=profile.pop('pretrained_evidence')
    with pytest.raises(ValueError,match='Four exact'):adapters.make_native_pretrained(profile)
    profile['pretrained_evidence']=original;(out/'FAILED.json').write_text('{}')
    with pytest.raises(ValueError,match='Failed/incomplete'):adapters.make_native_pretrained(profile)
    (out/'FAILED.json').unlink();(out/'controls.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):adapters.make_native_pretrained(profile)


def test_source_profile_thread_and_weight_identity_mismatches_fail_closed(tmp_path):
    profile,repo,out,provenance=fixture(tmp_path)
    update(profile,'protocol',lambda p:p.update(cpu_threads=1))
    with pytest.raises(ValueError,match='TWO-thread'):adapters.validate_native_profile(profile)
    update(profile,'protocol',lambda p:p.update(cpu_threads=2))
    update(profile,'provenance',lambda p:p.update(weight_sha256='wrong'))
    with pytest.raises(ValueError,match='safe weights'):adapters.validate_native_profile(profile)
    update(profile,'provenance',lambda p:p.update(weight_sha256='b'*64,profile='training_size832'))
    with pytest.raises(ValueError,match='profile/native'):adapters.validate_native_profile(profile)
    update(profile,'provenance',lambda p:p.update(profile='native512'))
    (repo/'native_matchanything_v17.py').write_text('changed source')
    with pytest.raises(ValueError,match='source changed'):adapters.validate_native_profile(profile)


def test_wrapper_returns_native_subpixel_matches_without_affine_prefilter_and_never_exposes_geometry(tmp_path,monkeypatch):
    profile,repo,out,provenance=fixture(tmp_path);calls=[]
    p=np.array([[50.25,70.75],[130.125,350.875],[450.5,400.25]])
    q=p+np.array([[8.,-4.],[7.,3.],[-30.,4.]]) # deliberately not a common affine translation
    class Stub:
        def __init__(self):self.provenance=provenance
        def points(self,*args):
            assert len(args)==4 and all(np.asarray(x).shape==(512,512)for x in args)
            calls.append(args);return p.copy(),q.copy()
        def sync(self):pass
    monkeypatch.setattr(adapters,'_construct_model',lambda *args:Stub())
    matcher=adapters.make_native_pretrained(profile);image=np.indices((512,512)).sum(0).astype(np.uint8)
    actual=matcher.points(image,image)
    np.testing.assert_array_equal(actual[0],p);np.testing.assert_array_equal(actual[1],q);assert len(calls)==1
    assert matcher.provenance['pretrained_gate']['semantic_provenance']['weight_sha256']=='b'*64
    assert matcher.provenance['sun_metadata_used']is False


def test_blank_invalid_masks_and_wrong_shapes_never_reach_pretrained_network(tmp_path,monkeypatch):
    profile,repo,out,provenance=fixture(tmp_path)
    class Forbidden:
        def __init__(self):self.provenance=provenance
        def points(self,*args):pytest.fail('Invalid image reached network')
        def sync(self):pass
    monkeypatch.setattr(adapters,'_construct_model',lambda *args:Forbidden());matcher=adapters.make_native_pretrained(profile)
    image=np.indices((512,512)).sum(0).astype(np.uint8)
    for a,b,mask in[(np.zeros_like(image),image,None),(image,image,np.zeros_like(image,bool))]:assert matcher.points(a,b,mask)[0].shape==(0,2)
    with pytest.raises(ValueError,match='native512'):matcher.points(image[:384],image[:384])
    with pytest.raises(ValueError,match='native512'):matcher.points(image.astype(float),image)
    with pytest.raises(ValueError,match='mask shape'):matcher.points(image,image,np.zeros((128,128),bool))


def test_loaded_profile_must_equal_actual_gate_provenance(tmp_path,monkeypatch):
    profile,repo,out,provenance=fixture(tmp_path)
    changed=dict(provenance,npe=[832,832,832,832])
    monkeypatch.setattr(adapters,'_construct_model',lambda *args:SimpleNamespace(provenance=changed))
    with pytest.raises(ValueError,match='NPE/config'):adapters.make_native_pretrained(profile)


def test_backend_uses_common_reference_geometry_estimator_and_ref_to_query_pixels(monkeypatch):
    import solar_evaluation_v70 as evaluation
    profile=dict(name='native_stub',backend='native_pretrained',method='moonst3r',author_profile='common_native512_full_field_of_view')
    yy,xx=np.mgrid[50:451:80,50:451:80];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(float);q=p+[8,-4]
    class PointsOnly:
        provenance={'test_scope':'Stub, not actual pretrained'}
        def points(self,reference,query):return p.copy(),q.copy()
        def sync(self):pass
    monkeypatch.setattr(adapters,'make_native_pretrained',lambda profile:PointsOnly())
    backend=evaluation.Backend(profile);k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    inp=evaluation.InferenceInput(np.zeros((512,512),np.uint8),np.zeros((512,512),np.uint8),SimpleNamespace(depth=np.full((512,512),100.),intrinsics=k,cam2world=np.eye(4)),k,np.array([0.,0.,1.]))
    result=backend.infer(inp)
    np.testing.assert_allclose(result['pose'][:3,3],[-2.,1.,0.],atol=1e-6)
    np.testing.assert_array_equal(result['reference_xy'],p);np.testing.assert_array_equal(result['query_xy'],q)
    assert result['common_estimator']['iterations_requested']==2000 and result['common_estimator']['reprojection_threshold_native_px']==3.
    assert result['common_estimator']['solver']=='ap3p_reclassify'


def test_unimplemented_author_crop_profile_never_substitutes_principal512():
    with pytest.raises(ValueError,match='Unsupported author'):adapters.specification(dict(backend='native_pretrained',method='moonst3r',author_profile='author_512x384_crop'))


@pytest.mark.parametrize('bad_output',[
    (np.array([[512.,10.]]),np.array([[10.,10.]])),
    (np.array([[10.,10.]]),np.array([[np.nan,10.]])),
    (np.array([10.,10.]),np.array([10.,10.])),
])
def test_invalid_native_outputs_are_pipeline_defects_not_silently_refitted_or_resized(tmp_path,monkeypatch,bad_output):
    profile,repo,out,provenance=fixture(tmp_path)
    class Invalid:
        def __init__(self):self.provenance=provenance
        def points(self,*args):return bad_output
        def sync(self):pass
    monkeypatch.setattr(adapters,'_construct_model',lambda *args:Invalid());matcher=adapters.make_native_pretrained(profile)
    image=np.indices((512,512)).sum(0).astype(np.uint8)
    with pytest.raises(ValueError,match='invalid/unmapped'):matcher.points(image,image)


def test_evaluation_admission_binds_full_native_source_and_actual_evidence_pins(tmp_path,monkeypatch):
    import run_solar_evaluation_v70 as runner
    import solar_evaluation_protocol_v74 as freeze
    # Isolate native source/evidence admission; full schema74 budget/deadline
    # behavior is exercised by test_solar_evaluation_protocol_v74.py.
    monkeypatch.setattr(freeze,"validate_runtime",lambda *args:dict())
    profile,repo,out,provenance=fixture(tmp_path)
    protocol=dict(schema_version=70,builder_schema_version=74,split='model_selection',calibration_or_final_access=False,cpu_threads=2,tracks=['estimated_sun'],max_calendar_seconds=100,purpose='modern_comparison',profiles=[profile],source_sha256={name:'0'*64 for name in runner.REQUIRED_SOURCES},input_sha256={})
    path=tmp_path/'evaluation-protocol.json';write(path,protocol)
    with pytest.raises(ValueError,match='native pretrained adapter/control/source/weight-lock'):runner.admission(str(path),'not-used')
    for name in adapters.required_native_sources(profile,adapters.read(profile['pretrained_evidence']['protocol'])):protocol['source_sha256'][name]='0'*64
    write(path,protocol)
    with pytest.raises(ValueError,match='evidence must be bound'):runner.admission(str(path),'not-used')


def test_draft_backend_and_adapter_sources_are_the_intended_tested_folder():
    import solar_evaluation_v70,run_solar_evaluation_v70
    for module in(adapters,solar_evaluation_v70,run_solar_evaluation_v70):assert Path(module.__file__).resolve().parent==Path(__file__).resolve().parent


def test_aggregate_and_correct_looking_pose_cannot_hide_degenerate_saved_matches():
    control=raw_control();control['reference_points_native']=np.repeat([[100.,100.]],36,axis=0).tolist();control['query_points_native']=np.repeat([[108.,96.]],36,axis=0).tolist()
    with pytest.raises(ValueError,match='pose differs'):adapters.replay_native_control(control)
