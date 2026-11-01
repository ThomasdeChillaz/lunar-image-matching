import numpy as np
import pytest
pytest.importorskip('torch')

from native_sampling_v8 import prepare_uniform,uniform_example


def test_uniform_positive_labels_are_projected_not_matcher_filtered_and_exclude_selection():
    k=np.array([[618.,0,256],[0,618.,256],[0,0,1.]])
    a=np.eye(4);b=np.eye(4);b[0,3]=5
    tiles={'train':dict(tile_id='train',component='train-group',split='train',cameras=[dict(intrinsics=k,cam2world=a),dict(intrinsics=k,cam2world=b)]),
           'forbidden':dict(split='model_selection')}
    z=dict(depth_a=np.full((512,512),100.),depth_b=np.full((512,512),100.),images=np.random.default_rng(3).integers(0,256,(6,512,512),dtype=np.uint8))
    data={'train':z};stats=prepare_uniform(tiles,data)
    assert stats['tiles']==1 and stats['points']>500 and stats['groups']==1
    np.testing.assert_allclose(z['uniform_target_native'],z['uniform_reference_native']-[30.9,0],atol=1e-8)
    first,meta=uniform_example(tiles,data,np.random.default_rng(5));second,again=uniform_example(tiles,data,np.random.default_rng(5))
    for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
    assert meta==again and meta['component']=='train-group'
    np.testing.assert_allclose(first[3],np.array(meta['target_native'])-meta['query_center_native']+16,atol=1e-6)
