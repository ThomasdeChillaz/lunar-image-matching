"""Geometry/stub QA only: no images, neural forwards, lunar cases or Sun rays."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import solar_assignment_v63 as solver
from solar_assignment_v63 import CandidateGraph,assignments,_matched,project


def fixture(side=4,candidates=2):
    axis=np.linspace(60,440,side);yy,xx=np.meshgrid(axis,axis,indexing='ij');p=np.column_stack([xx.ravel(),yy.ravel()])
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    world=np.column_stack([(p[:,0]-255.5)/4,(p[:,1]-255.5)/4,np.full(len(p),100.)])
    pose=np.eye(4);pose[:3,3]=[-2.,1.,0.];target=p+[8.,-4.]
    q=np.stack([target+[float(c),0.]for c in range(candidates)],axis=1)
    appearance=np.repeat(np.linspace(.99,.8,candidates)[None],len(p),axis=0)
    states=np.ones((len(p),1),np.uint8)
    graph=CandidateGraph(p,world,q,appearance,np.full(appearance.shape,.5),states,np.array([[0.,0.,1.]]),np.repeat(world[:,None],7,1),np.repeat(states[:,:,None],7,2))
    return graph,k,pose,target


def test_unobserved_full_assignment_cannot_erase_paid_native_measurements():
    graph,k,pose,target=fixture();graph.query_xy[:,0]=target+[12.,0.];graph.query_xy[:,1]=target+[.1,0.]
    state=np.tile([0,1],(len(target),1)).astype(np.uint8)
    a=assignments(graph,0,'off',pose,k,measurement_state=state)
    assert(a['candidate']==0).all()and(a['native_candidate']==1).all()
    assert len(a['native_keep'])==len(target)
    records={(i,1):dict(query_xy=graph.query_xy[i,1].copy())for i in range(len(target))}
    p,q,w,ids=_matched(graph,dict(assignment=a),records)
    np.testing.assert_array_equal(p,graph.reference_xy);np.testing.assert_array_equal(q,graph.query_xy[:,1])


def test_native_candidate_ranking_uses_current_geometry_not_only_appearance():
    graph,k,pose,target=fixture();graph.query_xy[:,0]=target+[2.8,0.];graph.query_xy[:,1]=target+[.1,0.]
    state=np.ones(graph.appearance.shape,np.uint8)
    a=assignments(graph,0,'off',pose,k,measurement_state=state,native_reprojection_gate=3.)
    assert(a['native_candidate']==1).all()and len(a['native_inlier_keep'])==len(target)


def test_native_duplicate_resolution_is_independent_of_unobserved_full_assignment():
    graph,k,pose,target=fixture();graph.query_xy[1,1]=graph.query_xy[0,1];graph.appearance[1,1]=.7
    state=np.zeros(graph.appearance.shape,np.uint8);state[:2,1]=1
    a=assignments(graph,0,'off',measurement_state=state)
    assert len(a['keep'])==len(target)and a['native_keep'].tolist()==[0]
    assert a['native_candidate'][0]==a['native_candidate'][1]==1


def test_failed_and_unobserved_candidates_never_become_native_matches():
    graph,k,pose,target=fixture();state=np.zeros(graph.appearance.shape,np.uint8);state[:,1]=2
    a=assignments(graph,0,'off',pose,k,measurement_state=state,native_reprojection_gate=3.)
    assert len(a['native_keep'])==len(a['native_inlier_keep'])==0
    assert(a['candidate']==0).all()
    # Even a contradictory external record cannot override the explicit state.
    records={(i,1):dict(query_xy=target[i].copy())for i in range(len(target))}
    assert len(_matched(graph,dict(assignment=a),records)[0])==0


@pytest.mark.parametrize('pose_present',[False,True])
@pytest.mark.parametrize('bad_state',[np.full((16,2),3),np.zeros((16,1)),np.full((16,2),.5)])
def test_invalid_measurement_state_rejected_before_any_pose_branch(pose_present,bad_state):
    graph,k,pose,target=fixture()
    with pytest.raises(ValueError,match='measurement states'):assignments(graph,0,'off',pose if pose_present else None,k,measurement_state=bad_state)


def test_current_inlier_diagnostic_does_not_censor_corrective_native_measurements():
    graph,k,pose,target=fixture();state=np.ones(graph.appearance.shape,np.uint8)
    graph.query_xy[:,0]=target+[4.,0.];graph.query_xy[:,1]=target+[8.,0.]
    # All observations currently lie outside3px. They must remain available to
    # the next RANSAC, rather than freezing an earlier biased pose forever.
    a=assignments(graph,0,'off',pose,k,measurement_state=state,native_reprojection_gate=3.)
    assert len(a['native_keep'])==len(target)and len(a['native_inlier_keep'])==0
    records={(i,c):dict(query_xy=graph.query_xy[i,c])for i in range(len(target))for c in range(2)}
    assert len(_matched(graph,dict(assignment=a),records)[0])==len(target)
    # The diagnostic must change when the TRIAL pose changes; no historical
    # native_pose boolean can substitute for current geometric consistency.
    changed=pose.copy();changed[0,3]-=1.
    b=assignments(graph,0,'off',changed,k,measurement_state=state,native_reprojection_gate=3.)
    assert len(b['native_inlier_keep'])==len(target)


def test_native_verification_excludes_points_behind_the_trial_camera():
    graph,k,pose,target=fixture();state=np.ones(graph.appearance.shape,np.uint8)
    behind=np.eye(4);behind[:3,:3]=np.diag([-1.,1.,-1.]);behind[:3,3]=[0.,0.,0.]
    a=assignments(graph,0,'off',behind,k,measurement_state=state,native_reprojection_gate=3.)
    assert len(a['native_keep'])==len(target)and len(a['native_inlier_keep'])==0


def test_fixed_native_verification_threshold_cannot_be_relaxed_by_configuration():
    graph,k,pose,target=fixture();state=np.ones(graph.appearance.shape,np.uint8)
    for bad in(4.,16.,float('nan')):
        with pytest.raises(ValueError,match='fixed3px'):assignments(graph,0,'off',pose,k,measurement_state=state,native_reprojection_gate=bad)
    with pytest.raises(ValueError,match='fixed3px'):assignments(graph,0,'off',None,k,measurement_state=state,native_reprojection_gate=3.)


def test_coarse_disc_is_explicitly_only_an_outer_bound_of_native_search_square():
    graph,k,pose,target=fixture(candidates=1);graph.query_xy[:,0]=target+[20.,0.]
    state=np.zeros(graph.appearance.shape,np.uint8)
    broad=assignments(graph,0,'off',pose,k,measurement_state=state)
    precise=assignments(graph,0,'off',pose,k,measurement_state=np.ones_like(state))
    assert np.isclose(broad['total'],graph.appearance.mean()) #20<sqrt(2)*16.5
    assert precise['total']<broad['total']
    # At angle0this centre is OUTSIDE the actual33x33local search, by3.5px.
    # This test documents the conservative proxy; it must not be described as
    # an exact oriented-window residual or evidence that a patch is reachable.
    exact_square_distance=np.linalg.norm(np.maximum(np.array([20.,0.])-16.5,0.))
    assert exact_square_distance==3.5 and 20.<np.sqrt(2.)*16.5


def test_depleted_native_evidence_returns_null_without_spending_final_estimator(monkeypatch):
    import solar_pose_proposals_v73 as proposals
    graph,k,pose,target=fixture()
    monkeypatch.setattr(proposals,'propose',lambda *args:(pose.copy(),dict(reason='procedural_stub',minimal_sets_executed=args[-1])))
    def forbidden(*args):pytest.fail('No native evidence reached final estimator')
    result=solver.solve(graph,k,lambda *args:dict(query_xy=None,reason='procedural_failed_measurement'),solar_mode='off',budget=32,pose_initialization='quantization_aware',final_estimator=forbidden)
    assert result['pose']is None and result['reason']=='insufficient_supported_native_matches'
    assert result['proposal_samples_requested']<=4096


def test_eight_beam_proposal_budget_and_single_common_final_solve(monkeypatch):
    import solar_pose_proposals_v73 as proposals
    graph,k,pose,target=fixture(side=8,candidates=8);initial=[];native=[];final=[]
    def propose(world,pixels,intrinsics,allocation):
        initial.append(allocation);return pose.copy(),dict(reason='procedural_stub',minimal_sets_executed=allocation)
    def estimate(world,pixels,intrinsics,config):
        native.append(config['iterations']);return(None,dict(reason='insufficient'))if len(world)<6 else(pose.copy(),dict(reason=None))
    def finish(p,q):
        # Final points must be paid subpixel measurements, not higher-scoring
        # free coarse candidates at the SAME reference anchors.
        np.testing.assert_allclose(q-p,np.repeat([[8.6,-4.]],len(p),axis=0),rtol=0,atol=1e-12)
        final.append((p.copy(),q.copy()));return pose.copy(),dict(reason=None,iterations_requested=2000)
    monkeypatch.setattr(proposals,'propose',propose);monkeypatch.setattr(solver,'estimate',estimate);monkeypatch.setattr(solver,'_lm_update',lambda pose,*args:pose)
    result=solver.solve(graph,k,lambda i,c,*args:dict(query_xy=target[i]+[.6,0.]),solar_mode='off',budget=32,pose_initialization='quantization_aware',final_estimator=finish)
    assert len(initial)==8 and len(native)==24 and set(initial+native)=={128}
    assert sum(initial+native)==result['proposal_samples_requested']==4096
    assert len(final)==1 and result['final_estimator_iterations']==2000
    assert len(result['measurements'])==32 and result['pose']is not None
    assert any(any(b['assignment']['candidate'][i]!=b['assignment']['native_candidate'][i]for i in b['assignment']['native_keep'])for b in result['hypotheses'])


def test_initial_proposal_budget_counts_four_point_sets_even_with_multiple_AP3P_roots(monkeypatch):
    import cv2
    import solar_pose_proposals_v73 as proposals
    graph,k,pose,target=fixture();calls=[]
    rv=np.zeros((3,1));tv=graph.world.mean(0).reshape(3,1)
    def roots(world,pixels,intrinsics,distortion,flags):
        assert world.shape==(4,3)and pixels.shape==(4,2)and flags==cv2.SOLVEPNP_AP3P
        calls.append(1);return(4,[rv.copy()for _ in range(4)],[tv.copy()for _ in range(4)])
    monkeypatch.setattr(cv2,'solvePnPGeneric',roots)
    monkeypatch.setattr(cv2,'solvePnPRefineLM',lambda world,pixels,k,d,rv,tv:(rv,tv))
    trial,info=proposals.propose(graph.world,target,k,128)
    assert len(calls)==info['minimal_sets_executed']==info['iterations_requested']==128
    assert info['roots_evaluated']==512 and trial is not None and info['accepted']is False
    assert info['strict3_support_diagnostic']==0 and info['final_reprojection_threshold_native_px']==3.
