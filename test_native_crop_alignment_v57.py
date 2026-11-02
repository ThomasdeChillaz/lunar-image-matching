import numpy as np
import pytest
from native_crop_alignment_v57 import native_example,integer_crop


def image():return (np.arange(128*128).reshape(128,128)%251).astype(np.uint8)


def test_half_pixel_reference_is_not_resampled_before_encoding():
    a=image();e=native_example(a,a,[49.5,65.5],[70.5,80.5],[72.25,79.75],True)
    np.testing.assert_array_equal(e['reference_origin'],[36,52])
    np.testing.assert_array_equal(e['reference_xy'],[13.5,13.5])
    np.testing.assert_array_equal(e['reference_patch'],a[52:79,36:63])
    np.testing.assert_array_equal(e['query_center'],[70,80])
    np.testing.assert_array_equal(e['query_patch'],a[52:109,42:99])
    np.testing.assert_allclose(e['target_xy'],[18.25,15.75])


def test_darkness_never_changes_geometry_visibility_or_search_labels():
    a=np.zeros((128,128),np.uint8)
    e=native_example(a,a,[49.5,49.5],[64,64],[64,64],True)
    assert e['geometrically_visible'] and e['matchable']


def test_wrong_local_proposal_and_geometric_occlusion_remain_distinct():
    a=image();wrong=native_example(a,a,[49.5,49.5],[64,64],[90,90],True)
    assert wrong['geometrically_visible'] and not wrong['target_in_search'] and not wrong['matchable']
    occluded=native_example(a,a,[49.5,49.5],[64,64],[64,64],False)
    assert not occluded['geometrically_visible'] and occluded['target_in_search'] and not occluded['matchable']


def test_unsupported_crops_and_fractional_origins_cannot_be_silently_padded():
    a=image()
    with pytest.raises(ValueError,match='integer x,y'):
        integer_crop(a,[.5,4],27)
    with pytest.raises(ValueError,match='Complete native crop'):
        native_example(a,a,[8.5,49.5],[64,64],[64,64],True)


def test_query_local_grid_maps_to_exact_integer_inference_window():
    a=image();e=native_example(a,a,[49.5,49.5],[70.5,80.5],[72,80],True)
    offsets=np.arange(-16,17)
    np.testing.assert_array_equal(e['query_origin'][0]+np.arange(12,45),e['query_center'][0]+offsets)
    np.testing.assert_array_equal(e['query_origin'][1]+np.arange(12,45),e['query_center'][1]+offsets)
