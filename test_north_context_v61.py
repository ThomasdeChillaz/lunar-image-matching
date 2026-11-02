"""Procedural/software controls; no released lunar images or pretrained forward."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from compact_lunar import CompactDescriptor
from search_window_learning_v18 import SearchWindowModel
from train_search_window_v18 import make_optimizer,verify_frozen


def toy_model():
    torch.manual_seed(20261101)
    coarse=SimpleNamespace(model=CompactDescriptor(False),metadata={'variant':'expanded'})
    return SearchWindowModel(coarse,variant='cross_context',train_coarse=True)


def test_nearly_matched_capacity_and_frozen_fine_encoder():
    model=toy_model()
    assert sum(p.numel() for p in model.parameters())==82208
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==53936
    assert all(not p.requires_grad for p in model.fine.parameters())
    assert all(p.requires_grad for p in model.coarse.parameters())
    assert len(model.context.blocks)==2
    assert all(b.cross_attention.dropout==0 for b in model.context.blocks)
    model.train()
    assert not model.fine.training and model.coarse.training


def test_starting_procedural_scores_exactly_equal_the_original_backbone():
    model=toy_model().eval()
    with torch.inference_mode():
        scores,baseline=model(torch.rand(2,1,128,128))
    assert scores.shape==(1,676,676)
    assert torch.equal(scores,baseline)


def test_original_optimizer_groups_and_protected_encoder_detection():
    model=toy_model();optimizer=make_optimizer(model)
    assert [g['lr'] for g in optimizer.param_groups]==[1e-4,1e-5]
    assert all(g['weight_decay']==1e-4 for g in optimizer.param_groups)
    assert {id(p) for p in optimizer.param_groups[0]['params']}=={id(p) for p in model.context.parameters()}
    initial={k:v.clone() for k,v in model.state_dict().items()}
    verify_frozen(model,initial)
    with torch.no_grad():next(model.fine.parameters()).add_(1)
    with pytest.raises(ValueError,match='Protected encoder'):verify_frozen(model,initial)


def test_sampler_loss_and_gradient_update_are_unchanged():
    def fitting(path):
        tree=ast.parse(Path(path).read_text())
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fitting_step')
    assert ast.dump(fitting('train_north_pilot_v38.py'),include_attributes=False)==ast.dump(fitting('train_north_context_v61.py'),include_attributes=False)


def test_invalid_public_source_cannot_create_experiment(tmp_path):
    from train_north_context_v61 import run
    with pytest.raises(ValueError,match='Full public source'):
        run('missing.json',tmp_path/'never-created','invalid')
    assert not (tmp_path/'never-created').exists()
