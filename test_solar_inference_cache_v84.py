"""Flow identity and complete procedural solver replay; no lunar study."""
import ast
import copy
import hashlib
import inspect
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

import solar_assignment_v63 as legacy
import solar_inference_cache_v84 as cached
from solar_frontend_v63 import SolarMatcher, ReferenceGeometry, reference_anchors
from solar_response_v63 import LightingResponseModel
from compact_lunar import CompactDescriptor
from lunar_geometry_v5 import world_points_z


def function(path, name, cls=None):
    tree = ast.parse(Path(path).read_text())
    if cls: tree = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    return copy.deepcopy(next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name))


def tree(value): return ast.dump(value, include_attributes=False)


def test_solver_copy_has_only_declared_cache_injections_and_finally_guard():
    original = function('solar_assignment_v63.py', 'solve')
    candidate = function('solar_inference_cache_v84.py', 'solve_v84'); candidate.name = 'solve'
    index = next(i for i, x in enumerate(candidate.body) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'acquisition_cache' for t in x.targets))
    assert tree(candidate.body[index]) == tree(ast.parse('acquisition_cache=AcquisitionProjectionCacheV84(graph,actions,intrinsics,capacity=4)').body[0])
    guarded = candidate.body[index+1]
    assert isinstance(guarded, ast.Try) and not guarded.handlers and not guarded.orelse
    assert tree(guarded.finalbody[0]) == tree(ast.parse('acquisition_cache.finish()').body[0])
    candidate.body[index:] = guarded.body
    final = candidate.body[-1].value
    assert final.keywords[-1].arg == 'acquisition_projection_cache'
    assert tree(final.keywords.pop().value) == tree(ast.parse('acquisition_cache.stats', mode='eval').body)
    calls = [n for n in ast.walk(candidate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'acquisition_predictions_v84']
    assert len(calls) == 1 and calls[0].keywords[-1].arg == 'cache'
    assert tree(calls[0].keywords.pop().value) == tree(ast.Name(id='acquisition_cache', ctx=ast.Load()))
    calls[0].func.id = 'acquisition_predictions'
    assert tree(candidate) == tree(original)


def test_frontend_copy_has_only_cache_injection_and_pre_emission_boundary_checks():
    original = function('solar_frontend_v63.py', 'localize', 'SolarMatcher')
    candidate = function('solar_inference_cache_v84.py', 'localize', 'SolarMatcherV84')
    assert tree(candidate.body[1]) == tree(ast.parse('original_geometry_guard=GeometryOwnerGuardV84(geometry)').body[0])
    outer = candidate.body[2]
    assert isinstance(outer, ast.Try) and not outer.handlers and not outer.orelse
    assert tree(outer.finalbody[0]) == tree(ast.parse('original_geometry_guard.check()').body[0])
    candidate.body = candidate.body[:1]+outer.body
    guarded = candidate.body[-1]
    assert isinstance(guarded, ast.Try) and not guarded.handlers and not guarded.orelse
    assert tree(guarded.finalbody[0]) == tree(ast.parse('if responder is not None and not responder.closed:\n    responder.finish()').body[0])
    body = guarded.body
    checks = [i for i, n in enumerate(body) if tree(n) == tree(ast.parse('if responder is not None:\n    responder.finish()').body[0])]
    assert len(checks) == 1
    following = body[checks[0]+1]
    assert isinstance(following, ast.Expr) and any(k.arg == 'elapsed_seconds' for k in following.value.keywords)
    del body[checks[0]]; candidate.body[-1:] = body
    class Restore(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'PairResponseCacheV84': node.id = 'PairResponseCache'
            elif node.id == 'solve_v84': node.id = 'solve'
            return node
    assert tree(Restore().visit(candidate)) == tree(original)
    for name, expected in cached.FROZEN_COPY_SOURCE_SHA256.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected


def graph():
    p = reference_anchors().astype(float); n = len(p)
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    z = 100.+3*np.sin(p[:, 0]/60.)
    world = world_points_z(p, z, k, np.eye(4))
    offsets = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.], [2., 1.], [-2., 1.], [1., -2.]])
    q = p[:, None]+offsets[None]
    states = np.column_stack([1+np.arange(n)%2, 2-np.arange(n)%2]).astype(np.uint8)
    suns = np.array([[.5, 0, np.sqrt(.75)], [-.5, 0, np.sqrt(.75)]])
    observed = np.repeat(np.where(states[:, :1] == 2, .9, .1), 8, axis=1)
    appearance = np.tile(np.linspace(.95, .7, 8), (n, 1))
    variants = legacy.height_variants(world, np.eye(4))
    g = legacy.CandidateGraph(p, world, q, appearance, observed, states, suns, variants, np.repeat(states[:, :, None], 7, axis=2))
    return g, k


def equal(a, b):
    if isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, float) and np.isnan(a): assert np.isnan(b)
    else: assert a == b


@pytest.mark.parametrize('a,b,c', list(itertools.product((False, True), repeat=3)))
def test_complete_256_anchor_eight_beam_128_search_solver_decisions_are_identical(a, b, c):
    g, k = graph(); before = g.query_xy.copy(); traces = []
    def run(solve):
        calls = []; responses = []; finals = []
        def measure(i, candidate, pose, sun):
            calls.append((i, candidate, sun)); return dict(query_xy=g.query_xy[i, candidate].copy(), observed_shadow=float(g.observed_shadow[i, candidate]))
        def response(i, candidate, query_xy, pose, suns):
            responses.append((i, candidate, tuple(suns), pose.copy()))
            return np.asarray([.98-.02*candidate+.001*int(s) for s in suns])
        def final(p, q):
            finals.append((p.copy(), q.copy()))
            world, _ = __import__('lunar_geometry_v5').map_points_z(p, np.full((512, 512), 100.), k, np.eye(4))
            pose, info = legacy.estimate(world, q, k, dict(solver='ap3p', iterations=2000, reclassify=True))
            return pose, dict(info, iterations_requested=2000)
        result = solve(g, k, measure, solar_mode='shared' if a else 'off', response_enabled=b,
            active=c, policy='active', budget=128, response_refresh=response if b else None, final_estimator=final)
        traces.append((calls, responses, finals)); return result
    original = run(legacy.solve); result = run(cached.solve_v84)
    stats = result.pop('acquisition_projection_cache')
    equal(original, result); equal(traces[0], traces[1])
    assert len(result['hypotheses']) == 8 and len(result['measurements']) == 128
    assert result['proposal_samples_requested'] == 4096 and result['final_estimator_iterations'] == 2000
    assert len(traces[1][2]) == 1 and stats['capacity'] == 4 and stats['entries'] <= 4
    assert stats['projection_cache_hits'] > 0 and stats['projection_computations'] < stats['requests']
    if b: assert result['response_measurements'] > 0
    np.testing.assert_array_equal(g.query_xy, before)


@pytest.mark.parametrize('initialization', ['strict3', 'quantization_aware'])
def test_factory_requires_explicit_revision_and_retains_old_default(tmp_path, initialization):
    from localize_terrain import load_matcher
    from solar_evaluation_protocol_v74 import _configuration
    with torch.random.fork_rng():
        torch.manual_seed(84); model = LightingResponseModel(CompactDescriptor(False)).eval()
    path = tmp_path/'procedural-untrained.pt'
    torch.save(dict(format='solar-response-v63', training_complete=False, model=model.state_dict()), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    base = dict(model_version='solar-experimental-v63', checkpoint_sha256=digest, pose_initialization=initialization)
    old = load_matcher(path, base, allow_untrained=True)
    new = load_matcher(path, dict(base, inference_revision=cached.CACHE_REVISIONS[initialization]), allow_untrained=True)
    assert type(old) is SolarMatcher and type(new) is cached.SolarMatcherV84
    assert old.total_parameters == new.total_parameters == 95808
    settings = dict(inference_configuration=dict(proposal_angles=[0.,90.,180.,270.], pose_initialization=initialization, inference_revision=new.inference_revision))
    assert _configuration(settings)['inference_revision'] == new.inference_revision
    other = cached.CACHE_REVISIONS['quantization_aware' if initialization == 'strict3' else 'strict3']
    with pytest.raises(ValueError): load_matcher(path, dict(base, inference_revision=other), allow_untrained=True)


def frontend_fixture():
    with torch.random.fork_rng():
        torch.manual_seed(85); model = LightingResponseModel(CompactDescriptor(False)).eval()
    p = reference_anchors(); k = np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    world = world_points_z(p, np.full(256,100.), k, np.eye(4))
    geometry = ReferenceGeometry(np.full((512,512),100.), k, np.eye(4), np.tile([0.,0.,-1.],(256,1)), np.array([[0.,0.,-1.]]), np.ones((256,1),np.uint8), np.ones((256,1,7),np.uint8), np.array([0.,0.,-1.]), np.repeat(world[:,None],7,1))
    image = np.random.default_rng(85).integers(0,256,(512,512),dtype=np.uint8)
    q = np.repeat(p[:,None],8,axis=1); appearance = np.tile(np.linspace(.9,.6,8),(256,1))
    return model, p, q, appearance, geometry, image, k


def test_frontend_failure_still_finishes_case_cache(monkeypatch):
    model,p,q,appearance,g,image,k = frontend_fixture(); instances=[]; real=cached.PairResponseCacheV84
    def capture(*args,**kwargs): value=real(*args,**kwargs); instances.append(value); return value
    monkeypatch.setattr(cached,'PairResponseCacheV84',capture)
    monkeypatch.setattr(cached,'candidates',lambda *args:(p,q,appearance,np.zeros((256,8),int)))
    def fail(*args,**kwargs): raise RuntimeError('procedural solver failure')
    monkeypatch.setattr(cached,'solve_v84',fail)
    with pytest.raises(RuntimeError,match='procedural solver failure'):
        cached.SolarMatcherV84(model.coarse,model.coarse,model).localize(image,image,g,k,response=True)
    assert len(instances)==1 and instances[0].closed and instances[0].cost['immutable_boundary_checks']==1


def test_frontend_model_data_mutation_cannot_emit_case(monkeypatch):
    model,p,q,appearance,g,image,k = frontend_fixture()
    monkeypatch.setattr(cached,'candidates',lambda *args:(p,q,appearance,np.zeros((256,8),int)))
    def mutate(*args,**kwargs): model.coefficients.weight.data.add_(.01); return {}
    monkeypatch.setattr(cached,'solve_v84',mutate)
    with pytest.raises(RuntimeError,match='model bytes mutated'):
        cached.SolarMatcherV84(model.coarse,model.coarse,model).localize(image,image,g,k,response=True)


@pytest.mark.parametrize('equal_bytes', [False, True])
@pytest.mark.parametrize('override_reference_sun', [False, True])
def test_frontend_rejects_live_geometry_replacement_in_final_estimator(monkeypatch, equal_bytes, override_reference_sun):
    model,p,q,appearance,g,image,k = frontend_fixture(); calls=[]; original=g.depth
    monkeypatch.setattr(cached,'candidates',lambda *args:(p,q,appearance,np.zeros((256,8),int)))
    def final_estimator(reference_xy, query_xy):
        replacement=g.depth.copy()
        if not equal_bytes: replacement[0,0]+=1.
        g.depth=replacement
        calls.append((g.depth is original, float(g.depth[0,0])))
        return np.eye(4),dict(reason=None,iterations_requested=2000)
    def solve_stub(*args,**kwargs):
        pose,info=kwargs['final_estimator'](p[:8],p[:8])
        return dict(pose=pose,reason=info['reason'],reference_xy=p[:8],query_xy=p[:8])
    monkeypatch.setattr(cached,'solve_v84',solve_stub)
    with pytest.raises(RuntimeError,match='geometry attribute replaced or mutated: depth'):
        cached.SolarMatcherV84(model.coarse,model.coarse,model).localize(image,image,g,k,response=True,final_estimator=final_estimator,reference_sun=g.reference_sun.copy() if override_reference_sun else None)
    assert calls==[(False,100. if equal_bytes else 101.)]


def test_original_geometry_guard_also_protects_blank_early_return(monkeypatch):
    model,p,q,appearance,g,image,k = frontend_fixture(); derive=g.with_reference_sun
    def replace_then_derive(sun):
        derived=derive(sun);g.intrinsics=g.intrinsics.copy();return derived
    monkeypatch.setattr(g,'with_reference_sun',replace_then_derive)
    blank=np.zeros_like(image)
    with pytest.raises(RuntimeError,match='geometry attribute replaced or mutated: intrinsics'):
        cached.SolarMatcherV84(model.coarse,model.coarse,model).localize(blank,blank,g,k,reference_sun=g.reference_sun.copy())


def test_solver_error_still_checks_copied_graph_cache(monkeypatch):
    g,k = graph(); instances=[]; real=cached.AcquisitionProjectionCacheV84
    class Observer(real):
        def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs);self.finished=False;instances.append(self)
        def finish(self): self.finished=True;return super().finish()
    monkeypatch.setattr(cached,'AcquisitionProjectionCacheV84',Observer)
    def fail(*args): raise RuntimeError('procedural measurement failure')
    with pytest.raises(RuntimeError,match='measurement failure'): cached.solve_v84(g,k,fail,budget=32)
    assert len(instances)==1 and instances[0].finished and instances[0]._graph is not g
