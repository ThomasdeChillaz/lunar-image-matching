"""No CNN: check that contrastive negatives share the physical target light."""
import math
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from solar_response_v63 import counterfactual_loss


class ConstantResponse(nn.Module):
    def __init__(self):
        super().__init__()
        coefficients=torch.zeros(4,36,32);coefficients[:,0,0]=1.
        self.coefficients_stub=nn.Parameter(coefficients)
    def encode(self,patches,condition):return self.coefficients_stub
    def decode(self,coefficients,suns,pooled=False):return F.normalize(coefficients[:,0],dim=-1)


def fixture(shared=True):
    model=ConstantResponse();patches=torch.zeros(4,1,33,33);condition=torch.zeros(4,9)
    suns=torch.tensor([[[0.,0.,1.],[1.,0.,0.]]]).expand(4,-1,-1).clone()
    if shared:available=torch.ones(4,2,dtype=torch.bool)
    else:available=torch.tensor([[True,False],[True,False],[False,True],[False,True]])
    teacher=torch.full((4,2,32),float('nan'));teacher[available]=torch.eye(32)[0]
    return model,patches,condition,suns,teacher,available,[[0,1],[2,3]]


def test_unavailable_counterfactual_negatives_do_not_create_a_spurious_loss():
    model,p,c,s,t,a,pairs=fixture(False)
    loss,parts=counterfactual_loss(model,p,c,s,t,a,pairs)
    assert parts['discrimination']==0.
    assert parts['discrimination_pair_light_count']==0
    assert parts['discrimination_light_count']==0
    loss.backward()
    assert torch.isfinite(model.coefficients_stub.grad).all()


@pytest.mark.parametrize('pooled',(False,True))
def test_collapsed_features_are_penalized_when_same_sun_negatives_exist(pooled):
    model,p,c,s,t,a,pairs=fixture()
    loss,parts=counterfactual_loss(model,p,c,s,t,a,pairs,pooled=pooled)
    assert parts['discrimination']==pytest.approx(math.log(2.),abs=1e-7)
    assert parts['discrimination_pair_light_count']==4
    assert parts['discrimination_light_count']==2
    loss.backward()
    assert torch.isfinite(model.coefficients_stub.grad).all()


def test_invalid_later_target_sun_is_detected_even_when_first_target_agrees():
    model,p,c,s,t,a,pairs=fixture()
    s[1,1]=torch.tensor([0.,1.,0.])
    with pytest.raises(ValueError,match='same target Sun frame'):
        counterfactual_loss(model,p,c,s,t,a,pairs)


def test_cross_point_negative_sun_frames_cannot_be_mixed():
    model,p,c,s,t,a,pairs=fixture()
    s[2:,1]=torch.tensor([0.,1.,0.])
    with pytest.raises(ValueError,match='same target Sun frame'):
        counterfactual_loss(model,p,c,s,t,a,pairs)
