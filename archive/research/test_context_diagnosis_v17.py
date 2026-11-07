import numpy as np
import pytest

torch=pytest.importorskip('torch')
from context_supervision_v15 import proposal_grid
from diagnose_context_proposals_v17 import score_proposals,target_likelihood
from localization_diagnostics_v3 import native_points
from rotation_bank_pose_v4 import quarter_turn


def test_native_local_window_uses_rounded_center_and_keeps_visibility_separate():
    p=q=np.array([[12.,12.]]*3,np.float32)
    target=np.array([[66.,50.],[66.001,50.],[50.,50.]])
    metrics,coordinates=score_proposals(p,q,target,np.array([True,True,False]),np.arange(3),0)
    # Native q=49.5 rounds to50; the33px search ends at66. Euclidean
    # distance to49.5 is larger than16, so this differs from a radial test.
    assert coordinates['reachable_33px']==[True,False,False]
    assert metrics['geometrically_visible']==2
    assert metrics['geometrically_reachable_33px']==1
    assert metrics['correct_at_native_px']['12']==0


@pytest.mark.parametrize('rotation',[0,1,2,3])
def test_native_pixel_center_and_rotation_diagnosis_agree(rotation):
    grid=proposal_grid()[:11];native=native_points(grid,(512,512))
    rotated=quarter_turn(grid,rotation,128)
    metrics,_=score_proposals(grid,rotated,native,np.ones(len(grid),bool),np.arange(len(grid)),rotation)
    assert metrics['correct_at_native_px']['1']==len(grid)
    assert metrics['geometrically_reachable_33px']==len(grid)


def test_likelihood_distinguishes_exact_targets_from_high_confidence_distractors():
    torch.set_num_threads(1);native=native_points(proposal_grid(),(512,512));visible=np.ones(676,bool)
    uniform=target_likelihood(torch.zeros(676,676),native,visible,0)
    exact=target_likelihood(torch.eye(676)*.9,native,visible,0)
    wrong=target_likelihood(torch.roll(torch.eye(676),1,1)*.9,native,visible,0)
    assert uniform['points']==676 and np.isclose(uniform['mean_nll'],np.log(676),atol=1e-5)
    assert exact['mean_nll']<uniform['mean_nll']<wrong['mean_nll']
    assert np.isclose(exact['mean_true_score'],.9,atol=1e-6)
    assert exact['mean_best_other_score']==0
