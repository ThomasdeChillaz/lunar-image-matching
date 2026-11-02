import numpy as np
import pytest

from north_optimization_probe_v54 import (fixed_specs, fixture_arrays, grid128,
                                         score_metrics, decoded_indices, verify_fixture)


def manifest():
    return dict(calibration_or_test_pixels_loaded=0, tiles=[
        dict(split='model_selection', tile_id='a', component='selection', lighting_count=6),
        dict(split='train', tile_id='b', component='train-b', lighting_count=6),
        dict(split='train', tile_id='c', component='train-c', lighting_count=6)])


def labels():
    return dict(forward=grid128(), reverse=grid128(), forward_visible=np.ones(676, bool),
                reverse_visible=np.ones(676, bool))


def test_fixture_choice_is_fixed_to_training_metadata():
    specs = fixed_specs(manifest())
    assert {s['tile_id'] for s in specs} == {'b'}
    assert [s['query_image'] for s in specs] == [0, 8, 1, 9]
    assert [s['identity_geometry'] for s in specs] == [True, True, False, False]
    assert all(not s['sensor_augmentation'] for s in specs)


def test_selection_only_or_reserve_loaded_manifest_is_refused():
    m = manifest(); m['tiles'] = m['tiles'][:1]
    with pytest.raises(ValueError): fixed_specs(m)
    m = manifest(); m['calibration_or_test_pixels_loaded'] = 1
    with pytest.raises(ValueError): fixed_specs(m)


def test_identity_and_stereo_fixture_coordinates_have_correct_pixel_center_conversion():
    native = (grid128() + .5) * 4 - .5
    z = dict(images=np.stack([np.full((512, 512), i, np.uint8) for i in range(12)]),
        forward_target=native + [8, -4], reverse_target=np.repeat((native + [-8, 4])[None], 4, axis=0),
        forward_visible=np.ones(676, bool), reverse_visible=np.ones((4, 676), bool))
    specs = fixed_specs(manifest())
    a, b, identity = fixture_arrays(z, specs[1])
    assert np.all(a == 0) and np.all(b == 8)
    np.testing.assert_array_equal(identity['forward'], grid128())
    a, b, stereo = fixture_arrays(z, specs[3])
    assert np.all(a == 0) and np.all(b == 9)
    np.testing.assert_allclose(stereo['forward'] - grid128(), np.tile([2, -1], (676, 1)))
    np.testing.assert_allclose(stereo['reverse'] - grid128(), np.tile([-2, 1], (676, 1)))


def test_correct_image_only_score_matrix_preserves_all_positive_rows_and_return_budget():
    scores = np.eye(676) * .8 + .1; indices = np.arange(512)
    result = score_metrics(scores, labels(), indices, indices)
    assert result['positive_rows'] == result['positive_row_top1_reachable'] == 676
    assert result['returned_mutual_matches'] == result['returned_native_correct']['2'] == 512
    assert result['returned_native_precision']['2'] == 1
    assert not result['pose_accuracy_evaluated']


def test_uniform_scores_do_not_turn_one_returned_match_into_full_coverage():
    result = score_metrics(np.full((676, 676), .5), labels(), [0], [0])
    assert result['returned_reachable_precision'] == 1
    assert result['positive_row_top1_reachable_fraction'] < .01
    assert result['returned_mutual_matches'] == 1


def test_empty_returned_matches_keep_undefined_precision():
    result = score_metrics(np.eye(676), labels(), [], [])
    assert result['returned_mutual_matches'] == 0
    assert result['returned_reachable_precision'] is None
    assert result['returned_native_precision']['2'] is None


@pytest.mark.parametrize('ia,ib', [([0, 0], [0, 1]), ([0, 1], [0, 0]), ([676], [0]), ([.5], [0])])
def test_invalid_decoder_output_is_refused(ia, ib):
    with pytest.raises(ValueError): score_metrics(np.eye(676), labels(), ia, ib)


def test_actual_returned_match_must_agree_with_score_peak_beyond_ties():
    with pytest.raises(ValueError, match='decoding disagree'):
        score_metrics(np.eye(676), labels(), [0], [10])


def test_nonfinite_scores_and_off_grid_inference_are_refused():
    scores = np.eye(676); scores[0, 0] = float('nan')
    with pytest.raises(ValueError): score_metrics(scores, labels(), [0], [0])
    with pytest.raises(ValueError): decoded_indices([[12.1, 12]], [[12, 12]])


def test_returned_token_indices_use_xy_and_exact_canonical_order():
    ia, ib = decoded_indices([[12, 12], [16, 12], [12, 16]], [[16, 12], [12, 16], [12, 12]])
    np.testing.assert_array_equal(ia, [0, 1, 26])
    np.testing.assert_array_equal(ib, [1, 26, 0])


def test_resumed_fixture_requires_same_training_arrays_and_dtypes(tmp_path):
    a = np.zeros((3, 3), np.uint8); b = a.copy(); target = labels()
    path = tmp_path / 'fixture.npz'; np.savez_compressed(path, image_a=a, image_b=b, **target)
    verify_fixture(path, a, b, target)
    with pytest.raises(ValueError): verify_fixture(path, a.astype(np.float32), b, target)


def test_altered_cached_fixture_is_preserved_and_refused(tmp_path):
    a = np.zeros((3, 3), np.uint8); target = labels(); path = tmp_path / 'fixture.npz'
    np.savez_compressed(path, image_a=a, image_b=np.ones_like(a), **target)
    before = path.read_bytes()
    with pytest.raises(ValueError): verify_fixture(path, a, a, target)
    assert path.read_bytes() == before
