"""Inference algebra, checkpoint invalidation and frozen foundation controls."""
import math
import numpy as np
import torch
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel,FusedLightingDecoder
from solar_frontend_v63 import SolarMatcher


def test_fused_and_unfused_response_agree_after_nontrivial_head_and_film_changes():
    torch.manual_seed(20261101);model=LightingResponseModel(CompactDescriptor(False)).eval()
    with torch.no_grad():
        model.coefficients.weight.normal_(0,.1);model.coefficients.bias.normal_(0,.1)
        model.condition[-1].weight.normal_(0,.1);model.condition[-1].bias.normal_(0,.1)
    base=torch.randn(12,32);condition=torch.randn(12,9);suns=torch.randn(7,3);ids=torch.tensor([0,2,3,4,5,6,1,2,3,4,5,6])
    for pooled in(False,True):
        fused=FusedLightingDecoder(model,suns,pooled)
        actual=fused(base,condition,ids)
        expected=model.decode(model.condition_features(base,condition),suns[ids],pooled)
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=2e-5)
        with torch.no_grad():model.coefficients.weight.add_(.03)
        changed=fused(base,condition,ids)
        expected=model.decode(model.condition_features(base,condition),suns[ids],pooled)
        assert fused.rebuilds==2;torch.testing.assert_close(changed,expected,atol=3e-6,rtol=2e-5)


def test_response_fine_updates_do_not_change_frozen_native_refinement_or_parameter_cap():
    torch.manual_seed(20261101);model=LightingResponseModel(CompactDescriptor(False)).eval();matcher=SolarMatcher(model.coarse,model.fine,model)
    assert matcher.fine is model.coarse and matcher.total_parameters==95808
    assert any(p.requires_grad for p in model.fine.parameters())
    image=torch.rand(1,1,33,33)
    with torch.inference_mode():before=matcher.fine(image)[0].clone()
    with torch.no_grad():next(model.fine.parameters()).add_(.5)
    with torch.inference_mode():after=matcher.fine(image)[0]
    torch.testing.assert_close(before,after,atol=0,rtol=0)
