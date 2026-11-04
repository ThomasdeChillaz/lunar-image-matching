"""Stub-only adapter controls: no lunar image read, neural forward or rendering."""
from types import SimpleNamespace,ModuleType
from pathlib import Path
import sys
import numpy as np
import pytest
import historical_compact_adapters_v73 as h
import direct_render_adapter_v73 as d


def inputs():
    geometry=SimpleNamespace(depth=np.ones((512,512)),intrinsics=np.eye(3),cam2world=np.eye(4))
    physical=SimpleNamespace(protocol={'reference_geometry_sha256':'reference','mesh_sha256':'scene','scene_mesh_sha256':'scene'})
    return SimpleNamespace(reference=np.ones((512,512),np.uint8),query=np.ones((512,512),np.uint8),
        geometry=geometry,query_intrinsics=np.eye(3),reference_sun=np.array([1.,0,0]),known_query_sun=None,solar_lookup=physical)


def known_outputs():
    p=dict(method='historical_original_native512',configuration={'native_profile':'native512_raw'},artifact_sha256={'checkpoint':'abc'},source_sha256={'adapter':'def'})
    a=np.array([[40.,40],[80,40],[120,40],[40,80],[80,80],[120,80]])
    o=dict(method=p['method'],configuration_sha256=h.value_digest(p['configuration']),artifact_sha256=p['artifact_sha256'],source_sha256=p['source_sha256'],
        native_shape=[512,512],cpu_threads=2,actual_pretrained_forward=True,query_truth_used=False,
        known_transform={'reference_xy':a.tolist(),'query_xy':(a+[8,-4]).tolist(),'reference_to_query_affine':[[1.,0,8],[0,1.,-4]]},
        blank={'reference_xy':[],'query_xy':[],'pose_returned':False},invalid_support={'reference_xy':[],'query_xy':[],'pose_returned':False})
    return p,o


def test_control_outputs_are_independently_replayed_not_just_booleans():
    p,o=known_outputs();assert h.validate_control_outputs(o,p)['known_matches']==6
    o['known_transform']['query_xy']=(np.asarray(o['known_transform']['query_xy'])+5).tolist()
    with pytest.raises(ValueError,match='known-transform'):h.validate_control_outputs(o,p)
    # Fitting the label to the wrong output cannot rescue a failed control.
    o['known_transform']['reference_to_query_affine']=[[1.,0,13],[0,1.,1]]
    with pytest.raises(ValueError,match='matcher-fitted'):h.validate_control_outputs(o,p)


def test_changed_checkpoint_or_leaked_query_truth_invalidates_control():
    p,o=known_outputs();p['artifact_sha256']={'different':'weights'}
    with pytest.raises(ValueError,match='another model'):h.validate_control_outputs(o,p)
    p,o=known_outputs();o['query_truth_used']=True
    with pytest.raises(ValueError):h.validate_control_outputs(o,p)


def test_unprepared_methods_fail_closed_without_loading_anything():
    for method in h.METHODS:
        with pytest.raises(h.BaselineUnavailable):h.make_historical_compact({'method':method,'prepared_only':True})
    with pytest.raises(h.BaselineUnavailable):d.make_direct_render({'prepared_only':True})
    with pytest.raises(ValueError):h.required_sources('solar_response_instead_of_history')


def test_historical_wrapper_returns_common_pose_not_internal_legacy_pose(monkeypatch):
    monkeypatch.setattr(h,'validate_historical_profile',lambda p:{'historical':True})
    calls=[]
    class Runtime:
        def __init__(self,p):pass
        def infer_points(self,inp):
            assert not hasattr(inp,'query_depth')and not hasattr(inp,'query_cam2world')
            return np.array([[20.,20],[40,40]]),np.array([[21.,20],[41,40]]),{'legacy_pose':'must_not_escape'}
    def estimator(p,q,inp):calls.append((p.copy(),q.copy(),inp));return np.eye(4),{'reason':None,'inliers':2}
    adapter=h.HistoricalCompactAdapter({'method':'historical_frozen_surface'},Runtime,estimator)
    result=adapter.infer(inputs())
    assert len(calls)==1 and np.array_equal(result['pose'],np.eye(4))and result['query_truth_used']is False


def test_surface_runtime_receives_reference_only_geometry_and_fresh_cache(monkeypatch):
    fake=ModuleType('geometry_warp_native_v11');caches=[]
    def infer(matcher,a,b,depth,k,pose,qk,features,feature_ids):
        caches.append(features);assert depth.shape==(512,512)and pose.shape==(4,4)
        return {'p':np.array([[30.,30]]),'q':np.array([[31.,30]]),'pose':None},[]
    fake.infer=infer;monkeypatch.setitem(sys.modules,'geometry_warp_native_v11',fake)
    runtime=object.__new__(h._HistoricalRuntime);runtime.profile={'method':'historical_frozen_surface'};runtime.matcher=object()
    runtime.infer_points(inputs());runtime.infer_points(inputs())
    assert len(caches)==2 and caches[0]is not caches[1]


def test_original_runtime_calls_actual_original_raw_extractor_contract(monkeypatch):
    fake=ModuleType('raw_correspondences_v2');seen=[]
    def raw(matcher,method,a,b,ma,mb):
        seen.append(method);assert method=='original'and a.shape==(512,512)and ma.all()and mb.all()
        return np.array([[16.,16]]),np.array([[17.,16]])
    fake.raw_points=raw;monkeypatch.setitem(sys.modules,'raw_correspondences_v2',fake)
    runtime=object.__new__(h._HistoricalRuntime);runtime.profile={'method':'historical_original_native512'};runtime.matcher=object()
    runtime.infer_points(inputs());assert seen==['original']


def test_native_input_and_output_boundaries():
    inp=inputs();inp.reference=np.ones((128,128),np.uint8)
    with pytest.raises(ValueError,match='native512'):h.validate_images(inp)
    inp=inputs();inp.geometry.depth[0,0]=np.nan
    with pytest.raises(ValueError,match='valid native'):h.validate_images(inp)
    with pytest.raises(ValueError,match='left native'):h.validate_points([[512,20]],[[20,20]])


def render_fixture(monkeypatch):
    inp=inputs();entries=[dict(light_index=0,sun_world=[1.,0,0],primary_generic=True,source_role='reference'),
        dict(light_index=1,sun_world=[0,1.,0],primary_generic=True,source_role='reference'),
        dict(light_index=2,sun_world=[0,0,1.],primary_generic=False,source_role='query')]
    library=dict(entries=entries,map_mesh_sha256='scene',scene_mesh_sha256='scene',
        reference_geometry_array_sha256=d.geometry_digest(inp.geometry.depth,inp.geometry.intrinsics,inp.geometry.cam2world),original_reference_preparation_seconds=123.)
    monkeypatch.setattr(d,'validate_render_profile',lambda p:({}, {'reference':library}, {'offline':123.}))
    calls=[]
    class Matcher:
        def __init__(self,c):pass
        def query_features(self,q):calls.append('query_features');return q
        def match(self,render,features):
            light=int(render[0,0]);a=np.array([[20.,20],[40,40],[60,60],[80,80],[100,100],[120,120]])
            return a,a+[light+1.,0],float(light)
    def loader(entry):calls.append(entry['light_index']);return np.full((512,512),entry['light_index'],np.uint8)
    def estimator(p,q,inp):
        t=np.eye(4);t[0,3]=float((q-p)[0,0]);return t,dict(reason=None,inliers=6,clipped_squared_reprojection_cost=0.)
    adapter=d.DirectRenderAdapter({'configuration':{'top_pose_candidates':2}},Matcher,loader,estimator)
    return adapter,inp,calls


def test_render_search_uses_generic_reference_library_and_common_final_pose(monkeypatch):
    adapter,inp,calls=render_fixture(monkeypatch);result=adapter.infer(inp)
    assert calls==['query_features',0,1]and 2 not in calls
    assert result['render_search']['library_entries_considered']==2
    assert result['render_search']['original_reference_preparation_seconds']==123.
    assert result['render_search']['optical_accuracy_verified']is False
    assert np.array_equal(result['pose'],adapter.estimator(result['reference_xy'],result['query_xy'],inp)[0])


def test_exact_secondary_sun_and_unavailable_are_never_rejections(monkeypatch):
    adapter,inp,calls=render_fixture(monkeypatch);inp.known_query_sun=np.array([0.,1,0]);result=adapter.infer(inp)
    assert calls==['query_features',1]and result['render_search']['exact_known_sun']
    calls.clear();inp.known_query_sun=np.array([2**-.5,2**-.5,0])
    with pytest.raises(h.BaselineUnavailable,match='no nearest'):adapter.infer(inp)
    assert calls==[]


def test_different_or_perturbed_primary_map_never_reuses_full_scene_library(monkeypatch):
    adapter,inp,_=render_fixture(monkeypatch);inp.solar_lookup.protocol['mesh_sha256']='cropped-or-perturbed'
    with pytest.raises(h.BaselineUnavailable,match='differs'):adapter.infer(inp)


def test_direct_builder_prepared_template_cannot_read_lunar_assets():
    import build_direct_render_library_v73 as b
    import tempfile,json
    with tempfile.TemporaryDirectory()as directory:
        p=Path(directory)/'protocol.json';p.write_text(json.dumps({'prepared_only':True}))
        with pytest.raises(ValueError,match='Prepared'):b.run(str(p),'a'*40)


def test_reference_prep_cost_keeps_startup_io_and_unknowns_explicit():
    summary={'calendar_seconds':10.,'views':{'reference':{'centre_ray_seconds':2.}},
      'images':[{'role':'reference','render_seconds':3.},{'role':'query','render_seconds':4.}]}
    assert d.reference_preparation_cost({'outer_calendar_seconds':12.},summary)==(12.,5.)
    with pytest.raises(h.BaselineUnavailable):d.reference_preparation_cost({'outer_calendar_seconds':None},summary)
    with pytest.raises(ValueError):d.reference_preparation_cost({'outer_calendar_seconds':6.},summary)
