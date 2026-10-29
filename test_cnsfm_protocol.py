import numpy as np
import pytest
from rasterio.transform import from_origin
from cnsfm_outputs import read_craters, read_matches, native_to_common
from prepare_cnsfm_common import crater_text
from run_cnsfm_sharded import partition


def test_shards_cover_every_original_pair_once_without_overlap():
    products = [f'observation{i}' for i in range(22)]
    shards = [partition(products, index, 16) for index in range(16)]
    flattened = [p for shard in shards for p in shard]
    assert len(flattened) == 231
    assert sorted(index for index, _ in flattened) == list(range(231))
    assert len({pair for _, pair in flattened}) == 231


def test_original_output_ids_and_window_offsets_are_audited():
    first = read_craters('0:20,30,8,8,.9,1\n1:40,50,9,9,.8,2\n')
    second = read_craters('0:22,33,8,8,.9,5\n1:42,53,9,9,.8,3\n')
    a, b = read_matches('20 30 22 33 0 8 0 8\n40 50 42 53 10000 9 10000 9', first, second)
    assert np.allclose(b-a, [2, 3])
    with pytest.raises(ValueError, match='coordinates/IDs'):
        read_matches('20 30 222 33 0 8 0 8', first, second)


def test_pixel_center_conversion_has_no_half_pixel_bias():
    native = from_origin(100, 200, 2, 2)
    common = from_origin(100, 200, 1, 1)
    assert np.allclose(native_to_common([[0, 0], [10, 20]], native, common), [[.5, .5], [20.5, 40.5]])
    assert np.allclose(native_to_common([[3, 4]], native, native), [[3, 4]])


def test_generated_crater_file_preserves_prediction_coordinates():
    detections = [dict(x=20.25, y=30.5, width=8., height=9., score=.9, detection_id=1)]
    table = read_craters(crater_text(detections))
    assert np.allclose(table[0], [20.25, 30.5])
