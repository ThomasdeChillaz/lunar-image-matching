import cv2
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from context_supervision_v15 import make_context, prepare_supervision, proposal_grid, sample_training, label_masks, objective
from rotation_bank_pose_v4 import quarter_turn


def data_fixture(dark=False):
    image = np.zeros((512, 512), np.uint8) if dark else np.random.default_rng(7).integers(0, 256, (512, 512), np.uint8)
    query = cv2.warpAffine(image, np.array([[1., 0, 8.], [0, 1., -4.]]), (512, 512))
    k = [[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]]
    ca = np.eye(4); cb = np.eye(4); cb[:2, 3] = [-.2, .1]
    cameras = [dict(intrinsics=k, cam2world=ca.tolist()), dict(intrinsics=k, cam2world=cb.tolist())]
    tiles = dict(a=dict(tile_id='a', split='train', component='one', cameras=cameras, world_bounds=[[-3, -3, 9], [3, 3, 11]]),
                 selection=dict(tile_id='selection', split='model_selection'))
    data = dict(a=dict(images=np.stack([image, query]*3), depth_a=np.full(image.shape, 10.), depth_b=np.full(image.shape, 10.)))
    return tiles, data


def test_forward_reverse_rotation_and_native_pixel_centers_are_geometrically_consistent():
    tiles, data = data_fixture(); prepared, stats = prepare_supervision(tiles, data)
    grid = proposal_grid(); labels = prepared['a']['rotations']
    np.testing.assert_allclose(labels[0]['forward'], grid+[2, -1], atol=1e-5)
    for k in range(4):
        np.testing.assert_allclose(labels[k]['forward'], quarter_turn(grid+[2, -1], k, 128), atol=1e-5)
        expected = quarter_turn(grid, k, 128, inverse=True)-[2, -1]
        np.testing.assert_allclose(labels[k]['reverse'], expected, atol=1e-5)
    assert stats['tiles'] == stats['groups'] == 1 and stats['points_per_image'] == 676


def test_darkness_does_not_create_occlusion_and_candidate_edge_band_is_not_a_negative():
    tiles, data = data_fixture(True); prepared, _ = prepare_supervision(tiles, data)
    assert prepared['a']['rotations'][0]['forward_visible'].all()
    target = torch.tensor([[10., 20.], [120., 20.], [20., 20.], [20., 20.]])
    positive, negative = label_masks(target, torch.tensor([True, True, True, False]))
    assert positive.tolist() == [False, False, True, False]
    assert negative.tolist() == [False, True, False, True]


def test_capacity_matched_factories_preserve_initial_scores_and_parameter_cap():
    torch.set_num_threads(2); numbers = []
    for variant in ('pointwise', 'self_context', 'cross_context'):
        model = make_context(variant); numbers.append(sum(p.numel() for p in model.parameters()))
        a = torch.nn.functional.normalize(torch.randn(1, 9, 32), dim=-1); p = torch.rand(1, 9, 2)*100
        assert torch.equal(model(a, a, p, p), torch.stack([x@x.T for x in a]))
        assert 28272+35233+numbers[-1] < 100000
    assert max(numbers)/min(numbers) < 1.03


def test_correspondence_loss_and_nonoverlap_labels_have_correct_preference():
    torch.set_num_threads(2); grid = torch.from_numpy(proposal_grid())
    targets = dict(forward=grid, reverse=grid, forward_visible=torch.ones(676, dtype=torch.bool), reverse_visible=torch.ones(676, dtype=torch.bool))
    good = torch.eye(676)[None]*.8+.1; bad = torch.roll(good, 9, -1)
    assert objective(good, targets)[0] < objective(bad, targets)[0]
    targets['forward_visible'][:] = False; targets['reverse_visible'][:] = False
    assert objective(torch.full_like(good, -.7), targets)[0] < objective(torch.full_like(good, .8), targets)[0]
    tiles, data = data_fixture(); prepared, _ = prepare_supervision(tiles, data)
    tiles['b'] = dict(tiles['a'], tile_id='b', component='two', world_bounds=[[997, -3, 9], [1003, 3, 11]])
    prepared['b'] = prepared['a']
    _, labels, meta = sample_training(tiles, prepared, np.random.default_rng(17), negative_pair=True)
    assert meta['tile_a'] != meta['tile_b'] and meta['verified_nonoverlap']
    assert not labels['forward_visible'].any() and not labels['reverse_visible'].any()
