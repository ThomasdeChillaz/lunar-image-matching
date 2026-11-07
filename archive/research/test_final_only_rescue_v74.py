"""Even failed short trial searches must receive the common final pose solve."""
import numpy as np
import pytest
import solar_assignment_v63 as solver
from native_baseline_pose_v7 import propose_pose
from test_solar_native_assignment_v73 import fixture


@pytest.mark.parametrize('initialization',('strict3','quantization_aware'))
def test_valid_paid_native_matches_reach_common_final_despite_no_trial_pose(monkeypatch,initialization):
    import solar_pose_proposals_v73 as initial
    graph,k,truth,target=fixture(side=8,candidates=1)
    short=[];final=[]
    def failed_trial(*args):
        short.append(args[-1]);return None,dict(reason='constructed_short_budget_miss')
    monkeypatch.setattr(solver,'estimate',failed_trial)
    monkeypatch.setattr(initial,'propose',failed_trial)
    def finish(p,q):
        final.append((p.copy(),q.copy()))
        pose,info=propose_pose(p,q,np.full((512,512),100.),k,np.eye(4),k)
        return pose,dict(info,iterations_requested=2000)
    result=solver.solve(graph,k,lambda i,*unused:dict(query_xy=target[i]),solar_mode='off',budget=32,
        pose_initialization=initialization,final_estimator=finish)
    assert result['pose']is not None and result['final_only_rescue'] is True
    assert len(final)==1 and result['final_estimator_iterations']==2000
    assert result['proposal_samples_requested']==4096
    assert len(result['measurements'])==32
    np.testing.assert_allclose(result['pose'],truth,atol=1e-5,rtol=0)
    # Independent common estimator arithmetic reproduces the saved proposal.
    replay,_=propose_pose(result['reference_xy'],result['query_xy'],np.full((512,512),100.),k,np.eye(4),k)
    np.testing.assert_allclose(replay,result['pose'],atol=1e-10,rtol=0)


def test_completely_depleted_measurements_remain_null(monkeypatch):
    graph,k,truth,target=fixture()
    monkeypatch.setattr(solver,'estimate',lambda *args:(None,dict(reason='constructed_trial_miss')))
    def forbidden(*args):pytest.fail('No paid native measurements available')
    result=solver.solve(graph,k,lambda *args:dict(query_xy=None),solar_mode='off',budget=32,final_estimator=forbidden)
    assert result['pose']is None and result['final_estimator_iterations']==0
    assert result['response_measurements']==0
    assert result['proposal_samples_requested']==4096
