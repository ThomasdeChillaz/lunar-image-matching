import pytest
pytest.importorskip('torch')
import numpy as np
from diagnose_proposal_coverage_v4 import rank_coverage


def test_coverage_uses_visual_ranking_then_scores_truth_separately():
    scores=np.array([[.9,.8,.1],[.9,.8,.1],[.9,.8,.1]])
    candidates=np.array([[30.,30.],[100.,100.],[200.,200.]])
    target=np.array([[100.,100.],[31.,31.],[31.,31.]])
    visible=np.array([True,True,False])
    assert rank_coverage(scores,candidates,target,visible,16,1).tolist()==[False,True,False]
    assert rank_coverage(scores,candidates,target,visible,16,2).tolist()==[True,True,False]
