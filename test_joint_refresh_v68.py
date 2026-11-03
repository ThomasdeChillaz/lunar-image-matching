"""Constructed solver/geometry controls; no lunar matching or accuracy claim."""
import numpy as np
import pytest
from test_solar_assignment_v63 import fixture
from solar_assignment_v63 import CandidateGraph,assignments,solve,project,acquisition_predictions,height_variants
from solar_physics_v63 import solar_direction


def test_bounded_fine_sun_reaches_unseen_quarter_degree_direction_from_coarse_hypothesis():
    g,k,truth,false=fixture();n=len(g.world);p=g.reference_xy
    params=np.array([[40.,1.5],[41.25,1.5],[45.,1.5]])
    suns=np.array([solar_direction(80.,h,d)for h,d in params]);ideal=1+np.arange(n)%2
    a=ideal.copy();a[::5]=3-a[::5];b=ideal.copy();b[::2]=3-b[::2]
    states=np.column_stack([a,ideal,b]);observed=np.column_stack([np.where(ideal==2,.95,.05),np.full(n,.5)])
    graph=CandidateGraph(p,g.world,g.query_xy,np.tile([.8,.78],(n,1)),observed,states,suns,g.world_variants,np.repeat(states[:,:,None],7,2),params,np.array([True,False,True]))
    result=solve(graph,k,lambda i,c,pose,sun:dict(query_xy=graph.query_xy[i,c].copy()),budget=64)
    assert result['pose']is not None and result['fine_sun_search_available']
    np.testing.assert_allclose(result['estimated_sun_parameters'],[41.25,1.5],atol=0,rtol=0)
    np.testing.assert_allclose(result['pose'],truth,atol=1e-5)
    assert result['proposal_samples_requested']<=4096 and result['final_estimator_iterations']==2000


def test_every_acquired_patch_updates_posterior_before_next_selection():
    graph,k,truth,false=fixture()
    def measure(i,c,pose,sun):return dict(query_xy=graph.query_xy[i,c].copy(),observed_shadow=float(1-graph.observed_shadow[i,c]))
    result=solve(graph,k,measure,active=True,budget=32)
    updates=result['incremental_updates']
    assert len(updates)==len(result['measurements'])==32
    assert [r['after_search']for r in updates]==list(range(1,33))
    assert all(np.isclose(sum(r['weights']),1)for r in updates)
    assert any(not np.allclose(a['weights'],b['weights'],atol=1e-12,rtol=0)for a,b in zip(updates,updates[1:]))


def test_response_refresh_and_final_scores_use_last_updated_pose_and_common_final_estimator(monkeypatch):
    import solar_assignment_v63 as solver
    graph,k,truth,false=fixture();estimates=[];refreshes=[];final_calls=[]
    def estimate(world,pixels,intrinsics,configuration):
        pose=np.eye(4);pose[0,3]=.01*(len(estimates)+1);estimates.append(pose.copy())
        return pose,dict(reason=None,inliers=len(world))
    monkeypatch.setattr(solver,'estimate',estimate);monkeypatch.setattr(solver,'_lm_update',lambda pose,*args:pose)
    def refresh(i,c,q,pose,suns):
        refreshes.append((i,c,pose.copy(),tuple(suns)))
        return np.full(len(suns),.8+.1*pose[0,3])
    common=np.eye(4);common[:3,3]=[7.,8.,9.]
    def final(p,q):final_calls.append((p.copy(),q.copy()));return common.copy(),dict(reason='external_common_control')
    result=solve(graph,k,lambda i,c,*args:dict(query_xy=graph.query_xy[i,c].copy()),budget=32,response_enabled=True,response_refresh=refresh,final_estimator=final)
    assert len(final_calls)==1;np.testing.assert_array_equal(result['pose'],common)
    assert result['final_geometry']=='caller_supplied_common_reference_geometry'
    assert result['proposal_samples_requested']==sum(1024//len(result['hypotheses'])for _ in estimates)
    assert result['proposal_samples_requested']<=4096
    assert any(np.array_equal(r[2],estimates[-1])for r in refreshes)
    for beam in result['hypotheses']:
        response=graph.appearance.copy()
        for record in result['measurements']:response[record['anchor'],record['candidate']]=.8+.1*beam['pose'][0,3]
        expected=assignments(graph,beam['sun_index'],'shared',beam['pose'],k,response)
        assert np.isclose(beam['score'],expected['total'],atol=1e-12,rtol=0)
    assert all(tuple(p)in set(map(tuple,graph.reference_xy))for p in final_calls[0][0])


def test_vectorized_active_geometry_and_independent_sun_predictions_match_physical_projection():
    graph,k,truth,false=fixture();beams=[]
    for j,pose in enumerate((truth,false)):
        assignment=assignments(graph,j,'independent');assignment['independent_sun_indices']=np.arange(len(graph.world))%2
        beams.append(dict(pose=pose,sun_index=j,assignment=assignment,score=float(j)))
    actions=[(i,c)for i in range(len(graph.world))for c in range(2)]
    prediction,weights,valid=acquisition_predictions(graph,beams,actions,k)
    for h,beam in enumerate(beams):
        for action,(i,c)in enumerate(actions):
            xy,front=project(graph.world_variants[i],beam['pose'],k)
            np.testing.assert_allclose(prediction[h,action,:,:2],(xy-graph.query_xy[i,c])/16,atol=1e-12)
            selected=beam['assignment']['independent_sun_indices'][i]
            states=graph.solar_variants[i,selected]
            expected=np.where(front&((xy>=0)&(xy<=511)).all(-1)&(states!=0),(states==2).astype(float),np.nan)
            np.testing.assert_allclose(prediction[h,action,:,2],expected,equal_nan=True)
    assert np.isfinite(prediction[valid]).all()and np.all(weights>0)
