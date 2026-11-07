import numpy as np
import pytest
from solar_assignment_v63 import CandidateGraph,assignments,initial_beams,height_variants,solve,project


def fixture():
    yy,xx=np.mgrid[50:451:80,50:451:60];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(float)
    n=len(p);z=40+np.sin(p[:,0]/40);k=np.array([[200.,0,255.5],[0,200.,255.5],[0,0,1.]])
    world=np.column_stack([(p[:,0]-255.5)*z/200,(p[:,1]-255.5)*z/200,z])
    true=np.eye(4);false=np.eye(4);false[:3,3]=[3.,1.,0.]
    qp,_=project(world,true,k);wrong,_=project(world,false,k);query=np.stack([qp,wrong],1)
    states=np.column_stack([1+np.arange(n)%2,2-np.arange(n)%2]);sun=np.array([[.5,0,np.sqrt(.75)],[-.5,0,np.sqrt(.75)]])
    obs=np.column_stack([np.where(states[:,0]==2,.95,.05),np.where(np.arange(n)%3==0,.95,.05)])
    appearance=np.tile([.8,.85],(n,1));variants=np.repeat(states[:,:,None],7,2)
    g=CandidateGraph(p,world,query,appearance,obs,states,sun,height_variants(world,np.eye(4)),variants)
    return g,k,true,false


def test_one_global_sun_changes_assignments_and_unknown_is_not_support():
    g,k,truth,false=fixture();off=assignments(g,0,'off');shared=assignments(g,0,'shared');independent=assignments(g,0,'independent')
    assert (off['candidate']==1).all()and(independent['candidate']==1).all()
    assert (shared['candidate']==0).any()and shared['solar_supported']
    g.solar_states[:]=0
    assert not assignments(g,0)['solar_supported']
    np.testing.assert_array_equal(assignments(g,0)['candidate'],off['candidate'])


def test_joint_geometry_and_shared_lighting_recover_pose_in_constructed_counterexample():
    g,k,truth,false=fixture();before=g.query_xy.copy();calls=[]
    def measure(i,c,pose,sun):calls.append((i,c));return dict(query_xy=g.query_xy[i,c].copy())
    result=solve(g,k,measure,budget=128)
    assert result['pose']is not None and len(calls)<=128 and len(calls)==len(set(calls))
    np.testing.assert_allclose(result['pose'],truth,atol=1e-5,rtol=0)
    assert result['proposal_samples_requested']<=4096 and result['final_estimator_iterations']==2000
    np.testing.assert_array_equal(g.query_xy,before)
    # This is a constructed candidate/geometry software counterexample. It
    # contains identity patch callbacks and establishes NO lunar accuracy.


def test_reference_height_variants_preserve_reference_camera_rays():
    g,k,_,_=fixture();v=height_variants(g.world,np.eye(4));p,front=project(v,np.eye(4),k)
    assert front.all();np.testing.assert_allclose(p,np.repeat(g.reference_xy[:,None],7,1),atol=1e-10)


def test_corrupt_graph_and_undeclared_budget_are_rejected():
    g,k,_,_=fixture()
    with pytest.raises(ValueError):solve(g,k,None,budget=100)
    values={n:getattr(g,n).copy()for n in g.__dataclass_fields__};values['solar_states'][0,0]=4
    with pytest.raises(ValueError):CandidateGraph(**values)
