"""Controls for paired sampling, coordinate labels and checkpoint-bound maps."""
import numpy as np
import pytest
import torch
from north_fine_data_v58 import prepare,sample_batch,physical_samples,CheckpointFeatureCache,make_fine,FineMatcher
from north_training_bundle_v38 import proposal_grid_native
from verify_north_fine_v58 import identical


def fixture():
    grid=proposal_grid_native();a=np.random.default_rng(7).integers(0,256,(12,512,512),dtype=np.uint8)
    tiles={'a':dict(split='train',component='g')}
    data={'a':dict(images=a,forward_target=grid+.23,forward_visible=np.ones(676,bool))}
    return tiles,data


def test_paired_physical_draws_and_exact_phase_labels():
    tiles,data=fixture();prepared=prepare(tiles,data);ra=np.random.default_rng(42);rb=np.random.default_rng(42)
    a,ma=sample_batch(tiles,data,prepared,ra,'aligned');b,mb=sample_batch(tiles,data,prepared,rb,'fractional')
    assert physical_samples(ma)==physical_samples(mb)and ra.bit_generator.state==rb.bit_generator.state
    assert np.array_equal(a[0],b[0])and np.array_equal(a[2],b[2])and not np.array_equal(a[1],b[1])
    assert np.all(a[2]==13.5)
    assert a[0].max()>.5 and 0<=a[0].min()and a[0].max()<=1
    for i,(x,y)in enumerate(zip(ma,mb)):
        np.testing.assert_allclose(a[3][i],np.array(x['target_native'])-x['integer_query_center']+16,atol=1e-6)
        np.testing.assert_allclose(b[3][i],a[3][i]-np.array(y['drawn_fractional_phase']),atol=2e-6)


def test_training_only_and_no_silent_geometry_drop():
    tiles,data=fixture();tiles['a']['split']='selection'
    with pytest.raises(ValueError,match='Only training'):prepare(tiles,data)
    tiles['a']['split']='train';data['a']['forward_visible'][:]=False
    with pytest.raises(ValueError,match='Every training geometry'):prepare(tiles,data)


def test_feature_cache_tracks_actual_weight_identity():
    c=CheckpointFeatureCache();a=c.for_weights('a');assert c.for_weights('a')is a
    b=c.for_weights('b');assert b is not a
    c.invalidate();assert c.for_weights('b')is not b
    with pytest.raises(ValueError):c.for_weights('')


def test_fine_initialized_but_not_shared_with_frozen_proposal(tmp_path,monkeypatch):
    import json
    import compact_lunar
    from compact_lunar import CompactDescriptor,CompactMatcher
    from dataset import sha256
    monkeypatch.chdir(tmp_path)
    checkpoint=tmp_path/'procedural-descriptor'
    checkpoint.mkdir()
    # Exercise the real sealed loader without requiring a trained artifact.
    # fork_rng preserves the global RNG for the other procedural controls.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(58)
        initial=CompactDescriptor(reliability=False)
        torch.save(initial.state_dict(),checkpoint/'selected.pt')
        metadata=dict(variant='expanded',seed=58,updates=0,procedural_fixture=True,
            scope='Procedural initialization only; no training or real imagery.',
            source_sha256={'compact_lunar.py':sha256(compact_lunar.__file__)})
        (checkpoint/'training.json').write_text(json.dumps(metadata,sort_keys=True))
        seal=dict(checkpoint_sha256=sha256(checkpoint/'selected.pt'),
            training_sha256=sha256(checkpoint/'training.json'),
            procedural_fixture=True,updates=0)
        (checkpoint/'COMPLETE.json').write_text(json.dumps(seal,sort_keys=True))
        coarse=CompactMatcher(checkpoint);coarse.model.eval().requires_grad_(False)
        fine=make_fine(coarse)
    assert all(not p.requires_grad for p in coarse.model.parameters())
    assert all(p.requires_grad for p in fine.parameters())
    assert sum(p.numel()for p in fine.parameters())==28272
    assert {p.data_ptr()for p in fine.parameters()}.isdisjoint({p.data_ptr()for p in coarse.model.parameters()})
    a=torch.linspace(0,1,64*64).reshape(1,1,64,64)
    with torch.inference_mode():
        assert torch.equal(coarse.model(a)[0],FineMatcher(coarse,fine).model(a)[0])
    assert identical({'a':[torch.ones(2)]},{'a':[torch.ones(2)]})
    assert not identical({'a':torch.ones(2)},{'a':torch.zeros(2)})
