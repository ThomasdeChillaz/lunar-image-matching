import inspect
import cv2
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from surface_learning_v13 import prepare_training, refine_surface, training_batch


def fixture(dark=False):
    y, x = np.mgrid[:512, :512]
    image = ((3*x+5*y) % 251).astype(np.uint8)
    query = cv2.warpAffine(image, np.array([[1., 0, 8], [0, 1., -4]]), (512, 512))
    if dark:
        query[:] = 0
    k = [[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]]
    ref = np.eye(4); other = np.eye(4); other[:2, 3] = [-.2, .1]
    tile = dict(tile_id='train', split='train', component='a', cameras=[
        dict(intrinsics=k, cam2world=ref.tolist()), dict(intrinsics=k, cam2world=other.tolist())])
    z = dict(images=np.stack([image, query]*3), depth_a=np.full(image.shape, 10.), depth_b=np.full(image.shape, 10.))
    # Selection has intentionally unusable query truth: training must skip it.
    return {'train': tile, 'selection': dict(tile_id='selection', split='model_selection')}, {'train': z}


def test_normalized_training_has_exact_common_geometry_and_no_selection_truth():
    tiles, data = fixture()
    report = prepare_training(tiles, data)
    assert report['tiles'] == report['groups'] == 1
    z = data['train']; p = z['surface_reference']
    np.testing.assert_array_equal(p, np.rint(p))
    np.testing.assert_allclose(z['surface_target'], p+[8, -4], atol=1e-5)
    points = p.astype(int)
    np.testing.assert_array_equal(z['surface_normalized_images'][0, points[:, 1], points[:, 0]],
                                  z['images'][0, points[:, 1], points[:, 0]])


def test_raw_normalized_ablations_keep_sampling_and_subpixel_targets_paired():
    tiles, data = fixture(); prepare_training(tiles, data)
    raw, rm = training_batch(tiles, data, np.random.default_rng(71), 8, False)
    normalized, nm = training_batch(tiles, data, np.random.default_rng(71), 8, True)
    torch.testing.assert_close(raw[3], normalized[3], rtol=0, atol=1e-5)
    torch.testing.assert_close(raw[0], normalized[0], rtol=0, atol=0)
    for a, b in zip(rm, nm):
        assert {k: v for k, v in a.items() if k != 'crop_center'} == {k: v for k, v in b.items() if k != 'crop_center'}
        np.testing.assert_allclose(np.array(a['crop_center'])-b['crop_center'], [8, -4], atol=1e-5)
    assert not torch.equal(raw[3], raw[3].round())


def test_shadows_are_not_occlusion_and_no_truth_enters_inference_interface():
    tiles, data = fixture(dark=True); prepare_training(tiles, data)
    batch, _ = training_batch(tiles, data, np.random.default_rng(4), 8, True)
    assert batch[4].bool().all()
    names = set(inspect.signature(refine_surface).parameters)
    assert not {'query_depth', 'query_pose', 'query_cam2world'} & names
    assert {'reference_depth', 'initial', 'query_intrinsics'} <= names


def test_invalid_reference_geometry_excluded_without_losing_valid_dark_pixels():
    tiles, data = fixture(); data['train']['depth_a'][200:260, 200:260] = np.nan
    prepare_training(tiles, data)
    p = data['train']['surface_reference']
    assert not (((p >= 159) & (p <= 300)).all(1)).any()
    assert len(p) > 100


@pytest.mark.parametrize('variant', ['encoder_3d', 'pair_refinement'])
def test_actual_fine_encoder_preserves_batch_dimension_and_recovers_known_pose(variant):
    from native_refinement_v7 import NativeRefiner
    from synthetic import texture
    torch.set_num_threads(2); torch.manual_seed(83)
    image = texture(733, 160); query = np.roll(image, 4, axis=1)
    k = np.array([[100., 0, 79.5], [0, 100., 79.5], [0, 0, 1.]])
    pose = np.eye(4); pose[0, 3] = -.4
    y, x = np.mgrid[45:116:10, 45:116:10]; p = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
    initial = dict(p=p, q=p+[4, 0], pose=pose, info={}, quarter_turn=0)
    result = refine_surface(NativeRefiner(variant), image, query, np.full(image.shape, 10.),
                            k, np.eye(4), k, initial)
    assert len(result['geometry_history']) == 2
    # Same analytic-control bounds used by the historical surface runner;
    # an untrained encoder's quadratic peak need not be exactly unbiased.
    assert np.linalg.norm(result['pose'][:3, 3]-pose[:3, 3]) < .05
    angle = np.degrees(np.arccos(np.clip((np.trace(result['pose'][:3, :3])-1)/2, -1, 1)))
    assert angle < .5
