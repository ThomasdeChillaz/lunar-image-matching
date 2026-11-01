"""Controls for coarse/fine optimizer separation and exact interrupted continuation."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from compact_lunar import CompactDescriptor
from context_supervision_v15 import proposal_grid
import train_search_window_v18 as runner


def fake_sample(tiles,prepared,rng,negative_pair=False):
    images=torch.from_numpy(rng.random((2,1,128,128),dtype=np.float32))
    grid=torch.from_numpy(proposal_grid());visible=torch.full((676,),not negative_pair)
    return images,dict(forward=grid,reverse=grid,forward_visible=visible,reverse_visible=visible),dict(verified_nonoverlap=negative_pair)


def test_coarse_learning_rate_and_protected_fine_encoder(monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(123)
    frozen=SimpleNamespace(model=CompactDescriptor(False).eval(),metadata=dict(variant='expanded'),device='cpu')
    model=runner.make_model(frozen,'coarse_adaptation');optimizer=runner.make_optimizer(model)
    assert [g['lr'] for g in optimizer.param_groups]==[.0001,.00001]
    ids={id(p) for g in optimizer.param_groups for p in g['params']}
    assert not ids.intersection(id(p) for p in model.fine.parameters())
    initial=copy.deepcopy(model.state_dict());monkeypatch.setattr(runner,'sample_training',fake_sample)
    runner.fitting_step(model,optimizer,{},None,np.random.default_rng(22),1,'coarse_adaptation')
    runner.verify_frozen(model,initial)
    assert any(not torch.equal(v,initial['coarse.'+k]) for k,v in model.coarse.state_dict().items())


def test_resume_restores_sampler_optimizer_and_future_weights_exactly(monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(123)
    frozen=SimpleNamespace(model=CompactDescriptor(False).eval(),metadata=dict(variant='expanded'),device='cpu')
    model=runner.make_model(frozen,'window_only');optimizer=runner.make_optimizer(model);rng=np.random.default_rng(22)
    monkeypatch.setattr(runner,'sample_training',fake_sample)
    first=runner.fitting_step(model,optimizer,{},None,rng,1,'window_only')
    saved=copy.deepcopy(runner.checkpoint(model,optimizer,rng,1,[first],0))
    expected=runner.fitting_step(model,optimizer,{},None,rng,2,'window_only');weights=copy.deepcopy(model.state_dict())
    replay=runner.make_model(frozen,'window_only');opt=runner.make_optimizer(replay);random=np.random.default_rng(999)
    runner.restore(saved,replay,opt,random)
    assert runner.fitting_step(replay,opt,{},None,random,2,'window_only')==expected
    assert all(torch.equal(v,weights[k]) for k,v in replay.state_dict().items())


def test_checkpoint_selection_counts_failed_cases_and_keeps_initialization_on_tie():
    rows=[dict(pose_correct=True,strict_pose_correct=False,estimated_cam2world=[[1]],correct_at_native_px={'2':2},returned=3,scorable=3),
          dict(pose_correct=False,strict_pose_correct=False,estimated_cam2world=None,correct_at_native_px={'2':0},returned=0,scorable=0)]
    counts=runner.counts(rows)
    assert counts['cases']==2 and counts['correct_poses']==1 and counts['estimated_poses']==1
    assert runner.selection_key(counts,0)>runner.selection_key(counts,500)


def test_small_complete_runner_preserves_initialization_and_replays_completion(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(123)
    coarse=SimpleNamespace(model=CompactDescriptor(False).eval(),metadata=dict(variant='expanded'),device='cpu')
    row=dict(pose_correct=True,strict_pose_correct=False,estimated_cam2world=[[1]],correct_at_native_px={'2':2},returned=3,scorable=3)
    def evaluate(*args,**kwargs):
        rows=[dict(row,pair_id=str(i)) for i in range(16)]
        return runner.counts(rows),rows
    monkeypatch.setattr(runner,'sample_training',fake_sample);monkeypatch.setattr(runner,'evaluate',evaluate)
    monkeypatch.setattr(runner,'validate_initial',lambda rows:None);monkeypatch.setattr(runner,'check_sources',lambda p:None)
    monkeypatch.setattr(runner,'SOURCES',[])
    protocol=dict(steps=2,interval=1,pilot=True,per_invocation_wall_cap_seconds=600)
    result=runner.train(tmp_path,'window_only',20261101,({}, {}, {}),coarse,None,protocol,{})
    assert result['completed_steps']==2 and result['selected_step']==0 and len(result['history'])==2
    assert len(result['validation'])==3 and result['total_parameters']<100000
    assert runner.train(tmp_path,'window_only',20261101,({}, {}, {}),coarse,None,protocol,{})==result
