import numpy as np
import pytest
from verify_north_optimization_v54 import independent_objective


def test_independent_numpy_loss_agrees_with_original_torch_on_mixed_visibility():
    import torch
    from search_window_learning_v18 import objective
    rng=np.random.default_rng(5411)
    scores=rng.normal(.4,.15,(676,676)).astype(np.float32)
    baseline=rng.uniform(-.3,.9,(676,676)).astype(np.float32)
    target={}
    for direction in ('forward','reverse'):
        target[direction]=rng.uniform(-8,136,(676,2)).astype(np.float32)
        target[direction+'_visible']=rng.random(676)>.25
    expected,labels=objective(torch.from_numpy(scores[None]),{k:torch.from_numpy(v)for k,v in target.items()},torch.from_numpy(baseline[None]),anchor_weight=.5)
    actual,independent_labels=independent_objective(scores.astype(float),baseline.astype(float),target)
    assert independent_labels==labels
    assert actual==pytest.approx(float(expected),abs=2e-6)


def test_uniform_all_negative_scores_have_closed_form_no_match_loss():
    scores=np.full((676,676),.5)
    target={key:value for direction in ('forward','reverse')for key,value in [(direction,np.zeros((676,2))),(direction+'_visible',np.zeros(676,bool))]}
    loss,labels=independent_objective(scores,scores,target)
    assert loss==pytest.approx(.025*np.log(2),abs=1e-12)
    assert all(v==dict(positive=0,negative=676,anchors=0)for v in labels.values())
