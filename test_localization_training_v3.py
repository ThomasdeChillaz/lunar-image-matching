import numpy as np
import pytest

from prepare_localization_training_v3 import allowed_records


def test_development_components_cannot_cross_training_and_selection():
    with pytest.raises(ValueError, match='Disjoint'):
        allowed_records({'tiles':[dict(split='train',component='same'),
                                 dict(split='model_selection',component='same')]})
    rows = allowed_records({'tiles':[dict(split='train',component='a'),
                                    dict(split='model_selection',component='b'),
                                    dict(split='test',component='unopened',cache='forbidden')]})
    assert len(rows) == 2 and all(r['split'] != 'test' for r in rows)


def test_refiner_parameter_cap_includes_frozen_coarse_network():
    pytest.importorskip('torch')
    from localization_refiner_v3 import LocalRefiner
    counts = {v:sum(p.numel() for p in LocalRefiner(v).parameters())+28272
              for v in ('descriptor_3d','refinement','reliability')}
    assert counts == dict(descriptor_3d=56544, refinement=63505, reliability=64098)


def test_local_supervision_and_no_match_gradient_are_finite():
    torch = pytest.importorskip('torch')
    from localization_refiner_v3 import LocalRefiner,local_loss
    torch.set_num_threads(2); model = LocalRefiner('reliability')
    a,b = torch.rand(2,1,27,27),torch.rand(2,1,57,57)
    logits,match = model(a,b,torch.tensor([[13.5,12.5],[13,13.]]))
    assert logits.shape == (2,33,33) and match.shape == (2,)
    loss = local_loss(logits,match,torch.tensor([[17.3,14.1],[0,0.]]),torch.tensor([1.,0.]),True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_peak_decoding_stays_on_strongest_mode():
    torch = pytest.importorskip('torch')
    from localization_refiner_v3 import decode_peak
    logits = torch.full((1,33,33),-100.)
    logits[0,9,7]=10; logits[0,28,28]=9
    np.testing.assert_allclose(decode_peak(logits).numpy(),[[-9.,-7.]],atol=.01)


def test_patch_training_matches_native_dense_coordinates():
    torch = pytest.importorskip('torch')
    import cv2
    from localization_refiner_v3 import LocalRefiner,refine_dense
    from contrastive_terrain import sample_descriptors
    torch.set_num_threads(2);torch.manual_seed(8)
    model=LocalRefiner('refinement').eval()
    rng=np.random.default_rng(8);image=rng.integers(0,256,(96,96),dtype=np.uint8)
    a=cv2.getRectSubPix(image,(27,27),(48.,48.))
    b=cv2.getRectSubPix(image,(57,57),(50.,50.))
    with torch.inference_mode():
        dense=model.encoder(torch.from_numpy(image.copy()).float()[None,None]/255)
        logits,_=model(torch.from_numpy(a.copy()).float()[None,None]/255,
                       torch.from_numpy(b.copy()).float()[None,None]/255,
                       torch.tensor([[13.5,12.5]]))
        vector=sample_descriptors(dense,torch.tensor([[[48.5,47.5]]]))[0]
        expected,_=model.score(vector,dense[:,:,34:67,34:67])
        torch.testing.assert_close(logits,expected,atol=5e-5,rtol=1e-5)
        p,q,score,indices=refine_dense(model,[[48,48]],[[50,50]],dense,dense)
        assert p.shape==q.shape==(1,2) and indices.tolist()==[0]
        np.testing.assert_allclose(q,[[48,48]],atol=.5)


def test_nonoverlap_negatives_require_separated_world_bounds():
    pytest.importorskip('torch')
    from train_localization_v3 import nonoverlap
    assert not nonoverlap(dict(world_bounds=[[0,0,0],[10,10,10]]),
                          dict(world_bounds=[[5,5,5],[15,15,15]]))
    assert nonoverlap(dict(world_bounds=[[0,0,0],[10,10,10]]),
                      dict(world_bounds=[[1000,0,0],[1010,10,10]]))


def test_prepared_loader_rejects_test_pixels_before_opening_cache(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    import json
    from train_localization_v3 import load_bundle
    path=tmp_path/'bad.json'
    path.write_text(json.dumps(dict(calibration_or_test_pixels_loaded=0,tiles=[
        dict(split='train',component='a'),dict(split='model_selection',component='b'),
        dict(split='test',component='c',native_cache='forbidden') ])))
    def forbidden(*args,**kwargs):raise AssertionError('Forbidden data was opened')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='Forbidden pixels'):
        load_bundle(path)


def test_candidate_loader_rejects_changed_checkpoint(tmp_path):
    torch=pytest.importorskip('torch')
    import json
    from dataset import sha256
    from localization_refiner_v3 import LocalRefiner
    from refine_lunar_development import load_candidate
    model=LocalRefiner('reliability');torch.save(model.state_dict(),tmp_path/'selected.pt')
    meta=dict(variant='reliability',final_test_enabled=False,calibration_or_test_pixels_loaded=0,
              total_parameters=64098,source_sha256={'localization_refiner_v3.py':sha256('localization_refiner_v3.py')})
    (tmp_path/'training.json').write_text(json.dumps(meta))
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(training_sha256=sha256(tmp_path/'training.json'),
                                                       checkpoint_sha256=sha256(tmp_path/'selected.pt'))))
    loaded,_=load_candidate(tmp_path);assert loaded.variant=='reliability'
    with (tmp_path/'selected.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='bytes differ'):load_candidate(tmp_path)
