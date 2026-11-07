"""Procedural image arrays and stub encoders only; no Torch import or NN job."""
from contextlib import nullcontext
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import solar_pose_free_features_v85 as adapter


class ReferenceOnlyGeometry:
    def __getattribute__(self, name):
        if name.startswith('query_') or name.startswith('true_'):
            raise AssertionError('Query truth is forbidden to feature extraction')
        return object.__getattribute__(self,name)


def inputs(sparse=True):
    y,x = np.mgrid[:512,:512]
    reference = ((x+3*y)%256).astype(np.uint8)
    query = ((2*x+y+7)%256).astype(np.uint8)
    g = ReferenceOnlyGeometry()
    g.intrinsics = np.array([[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]])
    r = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    g.cam2world = np.eye(4); g.cam2world[:3,:3] = r; g.cam2world[:3,3] = [10.,20.,30.]
    g.depth = np.full((512,512),np.nan if sparse else 1000.)
    if sparse: g.depth[34:67,34:67] = 1000.
    p = adapter.reference_anchor_grid()
    rays = np.column_stack([p,np.ones(len(p))])@np.linalg.inv(g.intrinsics).T
    world = (rays*1000.)@r.T+g.cam2world[:3,3]
    g.world_variants = np.repeat(world[:,None,:],7,axis=1)
    g.normals = np.tile([0.,0.,1.],(256,1))
    camera_suns = np.array([[1.,0.,0.],[0.,1.,0.]])
    g.sun_vectors = camera_suns@r.T
    g.sun_parameters = np.array([[0.,0.],[5.,0.]])
    g.coarse_sun_mask = np.ones(2,bool)
    g.solar_states = np.zeros((256,2),np.uint8)
    g.solar_materialized_mask = np.ones(2,bool)
    known_ref_sun = np.array([1.,0.,0.])@r.T
    g.reference_sun = known_ref_sun.copy()
    return reference,query,g,known_ref_sun,object()


class QuarterTurnStub:
    """Pure-array test substitute; actual adapter lazily uses RotationBank."""
    def __init__(self,image): self.images=[np.rot90(image,k).copy() for k in range(4)]
    def native_support(self,point,index):
        center = np.rint(point).astype(int)
        return bool((center-28>=0).all() and (center+29<=512).all())
    def transform(self,points,index,inverse=False):
        p = np.asarray(points,float); k = (-index if inverse else index)%4
        x,y = p.T
        if k == 0: return p.copy()
        if k == 1: return np.column_stack([y,511-x])
        if k == 2: return np.column_stack([511-x,511-y])
        return np.column_stack([511-y,x])


class StubRuntime:
    is_test_stub = True
    def __init__(self,model,caps,guard):
        self.caps = caps; self.guard = guard; self.patch_batches=[]; self.query_calls=[]; self.finished=0
        self.identity=dict(model_sha256='SOFTWARE_STUB_ONLY',total_parameters=95808,extra_encoder_parameters=0)
        self.counts=dict(reference_fine_forwards=0,reference_frozen_coarse_forwards=0,
                         query_frozen_coarse_forwards=0,response_decodes=0,decoder_builds=0)
    def reference_features(self,patches):
        self.patch_batches.append(patches.copy())
        fine=np.zeros((len(patches),32),np.float32); fine[:,9]=1.
        teacher=np.zeros_like(fine); teacher[:,0]=1.
        self.counts['reference_fine_forwards']+=1; self.counts['reference_frozen_coarse_forwards']+=1
        return fine,teacher
    def query_features(self,image,points):
        self.query_calls.append((image.copy(),points.copy()))
        self.counts['query_frozen_coarse_forwards']+=1
        out=np.zeros((len(points),32),np.float32);out[:,0]=1.
        return out
    def bind(self,base,condition,suns): self.bound=(base.copy(),condition.copy(),suns.copy())
    def decode(self,base,condition,target_sun,*,pooled):
        np.testing.assert_array_equal(base,self.bound[0]); np.testing.assert_array_equal(condition,self.bound[1])
        assert any(np.array_equal(target_sun,s) for s in self.bound[2])
        self.counts['response_decodes']+=1
        out=np.zeros((len(base),32),np.float32);out[:,0]=1.
        return out
    def finish(self): self.finished+=1;self.guard.check()


@pytest.fixture
def stubs(monkeypatch):
    runtimes=[];banks=[]
    def runtime(model,caps,guard):
        r=StubRuntime(model,caps,guard);runtimes.append(r);return r
    def bank(image):
        b=QuarterTurnStub(image);banks.append(b);return b
    monkeypatch.setattr(adapter,'_make_runtime',runtime)
    monkeypatch.setattr(adapter,'_make_rotation_bank',bank)
    # Execute the actual guard's exact source nodes, not a mirrored guard. Its
    # class/helper have only NumPy/hashlib dependencies; no Torch import/NN.
    path=Path(importlib.util.find_spec('solar_response_cache_v84').origin)
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('_array_digest','GeometryOwnerGuardV84')]
    namespace=dict(np=np,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    monkeypatch.setattr(adapter,'_make_geometry_guard',namespace['GeometryOwnerGuardV84'])
    return runtimes,banks


def test_native_half_pixel_grid_four_rotations_inverse_mapping_and_counts(stubs):
    args=inputs(); result=adapter.extract_native_features(*args)
    runtime=stubs[0][0]; bank=stubs[1][0]
    assert result.metadata['software_stub_only']
    assert result.inputs['query_descriptors'].shape==(10404,32)
    assert result.inputs['query_valid'].all()
    assert len(runtime.query_calls)==4 and runtime.counts['query_frozen_coarse_forwards']==4
    axis=49.5+8*np.arange(51);y,x=np.meshgrid(axis,axis,indexing='ij')
    grid=np.column_stack([x.ravel(),y.ravel()])
    for k,(image,points) in enumerate(runtime.query_calls):
        np.testing.assert_array_equal(points,grid)
        np.testing.assert_array_equal(image,np.rot90(args[1],k))
        mapped=result.inputs['query_native_xy'][k*2601:(k+1)*2601]
        np.testing.assert_array_equal(mapped,bank.transform(grid,k,True))
        np.testing.assert_array_equal(bank.transform(mapped,k,False),grid)
    assert result.metadata['preflight']['dense_feature_output_bytes']==32*512*512*4


def test_reference_support_subpixel_patch_teacher_and_condition_frames(stubs):
    reference,query,g,sun,model=inputs()
    result=adapter.extract_native_features(reference,query,g,sun,model)
    runtime=stubs[0][0]
    np.testing.assert_array_equal(result.anchor_indices,[0])
    assert result.metadata['reference_support_omissions']==255
    p=adapter.reference_anchor_grid()[0]
    y,x=np.mgrid[-16:17,-16:17]
    expected=cv2.remap(reference,(p[0]+x).astype(np.float32),(p[1]+y).astype(np.float32),cv2.INTER_LINEAR)
    np.testing.assert_array_equal(runtime.patch_batches[0][0],expected)
    assert result.inputs['reference_response_features'][0,9]==1.
    assert result.inputs['frozen_reference_descriptors'][0,0]==1.
    assert result.inputs['frozen_reference_descriptors'][0,9]==0.
    r=g.cam2world[:3,:3];world=g.world_variants[0,0]
    view=g.cam2world[:3,3]-world;view/=np.linalg.norm(view)
    expected_condition=np.concatenate([sun@r,g.normals[0]@r,view@r]).astype(np.float32)
    np.testing.assert_array_equal(runtime.bound[1][0],expected_condition)
    np.testing.assert_array_equal(result.inputs['sun_vectors'],g.sun_vectors@r)
    assert result.metadata['support_is_not_query_occlusion_or_shadow']


def test_reference_batches_never_exceed_32_and_query_batch_remains_one(stubs):
    result=adapter.extract_native_features(*inputs(sparse=False))
    runtime=stubs[0][0]
    assert len(runtime.patch_batches)==8 and all(len(p)<=32 for p in runtime.patch_batches)
    assert runtime.counts['reference_fine_forwards']==runtime.counts['reference_frozen_coarse_forwards']==8
    assert len(runtime.query_calls)==4
    assert result.metadata['query_image_batch']==1


def test_native_support_mask_excludes_features_without_visibility_labels(stubs,monkeypatch):
    class MaskedBank(QuarterTurnStub):
        def native_support(self,point,index):
            return super().native_support(point,index) and not(index==1 and np.array_equal(point,[49.5,49.5]))
    monkeypatch.setattr(adapter,'_make_rotation_bank',MaskedBank)
    result=adapter.extract_native_features(*inputs())
    assert result.inputs['query_valid'].sum()==10403 and not result.inputs['query_valid'][2601]
    assert np.isnan(result.inputs['query_descriptors'][2601]).all()
    assert np.isfinite(result.inputs['query_soft_shadow']).all()


def test_math_integration_uses_reference_provider_and_always_finishes(stubs):
    extracted=adapter.extract_native_features(*inputs())
    runtime=stubs[0][0];before=runtime.finished
    result=adapter.retrieve_from_extracted(extracted,arm='sh')
    assert result['prototype_only'] and result['software_stub_only']
    assert result['candidates']['query_native_xy'].shape==(1,8,2)
    assert runtime.counts['response_decodes']>0 and runtime.finished==before+1
    before=runtime.finished
    adapter.retrieve_from_extracted(extracted,arm='native')
    assert runtime.finished==before+1


@pytest.mark.parametrize('change',['query_image','depth','world'])
def test_input_byte_mutation_fails_end_guard(stubs,change):
    args=inputs();extracted=adapter.extract_native_features(*args)
    if change=='query_image':args[1][0,0]^=1
    elif change=='depth':args[2].depth[40,40]+=1.
    else:args[2].world_variants[0,0,0]+=.1
    with pytest.raises((ValueError,RuntimeError),match='inputs changed|geometry attribute replaced or mutated'):extracted.finish()


def test_blank_missing_surface_and_caps_fail_before_encoders(stubs):
    args=list(inputs());args[1]=np.zeros((512,512),np.uint8)
    with pytest.raises(adapter.ExtractionUnavailable,match='blank'):adapter.extract_native_features(*args)
    args=list(inputs());args[2].depth[:]=np.nan
    with pytest.raises(adapter.ExtractionUnavailable,match='surface'):adapter.extract_native_features(*args)
    for caps in [adapter.ExtractionCaps(reference_patch_batch=33),adapter.ExtractionCaps(query_image_batch=2),
                 adapter.ExtractionCaps(max_query_encoder_forwards=3),adapter.ExtractionCaps(max_explicit_array_bytes=1),
                 adapter.ExtractionCaps(max_calendar_seconds=float('nan'))]:
        with pytest.raises(ValueError):adapter.extract_native_features(*inputs(),caps=caps)
    assert stubs[0]==[]


def test_uncomputed_solar_is_not_silently_unknown_and_physical_can_be_omitted(stubs):
    args=inputs();args[2].solar_materialized_mask[0]=False
    with pytest.raises(ValueError,match='actually computed'):adapter.extract_native_features(*args)
    result=adapter.extract_native_features(*args,include_physical_evidence=False)
    assert 'solar_states'not in result.inputs and 'query_soft_shadow'not in result.inputs
    assert result.metadata['physical_evidence_observation']is None


def test_no_query_geometry_arguments_and_stub_results_are_honestly_marked(stubs):
    names=inspect.signature(adapter.extract_native_features).parameters
    for forbidden in ('query_pose','query_depth','query_sun','query_intrinsics','true_query_pose'):
        assert forbidden not in names
        with pytest.raises(TypeError):adapter.extract_native_features(*inputs(),**{forbidden:None})
    result=adapter.extract_native_features(*inputs())
    assert result.metadata['model_identity']['model_sha256']=='SOFTWARE_STUB_ONLY'
    assert result.metadata['software_stub_only']
    assert any('Half-pixel' in s for s in result.metadata['limitations'])
    assert any('same-reference-view' in s for s in result.metadata['limitations'])


def test_actual_runtime_reference_path_uses_fine_base_and_coarse_center16_without_torch():
    class Tensor:
        def __init__(self,a):self.a=np.asarray(a)
        @property
        def shape(self):return self.a.shape
        def float(self):return Tensor(self.a.astype(np.float32))
        def __getitem__(self,k):return Tensor(self.a[k])
        def __truediv__(self,v):return Tensor(self.a/v)
        def cpu(self):return self
        def numpy(self):return self.a
    class Model:
        def patch_features(self,x):return Tensor(np.full((len(x.a),32),9.,np.float32))
        def coarse(self,x):
            a=np.broadcast_to(np.arange(33*33,dtype=np.float32).reshape(1,1,33,33),(len(x.a),32,33,33)).copy()
            return Tensor(a),None
    runtime=object.__new__(adapter._TorchRuntime)
    runtime.torch=SimpleNamespace(from_numpy=Tensor,inference_mode=nullcontext)
    runtime.model=Model();runtime.caps=adapter.ExtractionCaps();runtime.check=lambda:None
    runtime.counts=dict(reference_fine_forwards=0,reference_frozen_coarse_forwards=0)
    learned,frozen=runtime.reference_features(np.zeros((2,33,33),np.uint8))
    np.testing.assert_array_equal(learned,np.full((2,32),9.,np.float32))
    np.testing.assert_array_equal(frozen,np.full((2,32),16*33+16,np.float32))
    assert runtime.counts['reference_fine_forwards']==runtime.counts['reference_frozen_coarse_forwards']==1


@pytest.mark.parametrize('attribute',['depth','intrinsics','cam2world','normals','sun_vectors','reference_sun','world_variants'])
@pytest.mark.parametrize('equal_bytes',[False,True])
def test_live_geometry_replacement_after_extraction_is_rejected(stubs,attribute,equal_bytes):
    args=inputs();result=adapter.extract_native_features(*args);g=args[2]
    replacement=getattr(g,attribute).copy()
    if not equal_bytes:replacement.flat[0]+=.125
    setattr(g,attribute,replacement)
    with pytest.raises(RuntimeError,match='geometry attribute replaced or mutated: '+attribute):result.finish()


@pytest.mark.parametrize('derive_reference_sun',[False,True])
def test_geometry_replacement_on_failed_extraction_is_checked_from_original_entry(stubs,monkeypatch,derive_reference_sun):
    args=inputs();original=args[2];geometry=copy.copy(original)if derive_reference_sun else original
    if derive_reference_sun:geometry.reference_sun=args[3].copy()
    def broken(image,points):
        geometry.depth=geometry.depth.copy()
        raise RuntimeError('deliberate encoder failure after owner replacement')
    def runtime(model,caps,guard):
        result=StubRuntime(model,caps,guard);result.query_features=broken;return result
    monkeypatch.setattr(adapter,'_make_runtime',runtime)
    with pytest.raises(RuntimeError,match='geometry attribute replaced or mutated: depth'):
        adapter.extract_native_features(args[0],args[1],geometry,args[3],args[4])


def test_native_retrieval_cannot_finish_after_geometry_replacement(stubs,monkeypatch):
    args=inputs();result=adapter.extract_native_features(*args)
    import solar_pose_free_retrieval_v85 as math
    def replace(**kwargs):args[2].intrinsics=args[2].intrinsics.copy();return {}
    monkeypatch.setattr(math,'retrieve_candidates',replace)
    with pytest.raises(RuntimeError,match='geometry attribute replaced or mutated: intrinsics'):
        adapter.retrieve_from_extracted(result,arm='native')
