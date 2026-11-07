import numpy as np
import pytest
pytest.importorskip('torch')

from coarse_training_v6 import canonical_quarter_turn,sample_training
from rotation_bank_pose_v4 import quarter_turn


def test_training_orientation_recovers_every_quarter_turn_without_changing_mask():
    p=np.array([[22.,35.],[40,90],[100,45],[80,110],[np.nan,0]])
    valid=np.array([True,True,True,True,False]);original=valid.copy()
    for k in range(4):
        q=quarter_turn(p,k,128)*1.1+np.array([3,-2])
        assert canonical_quarter_turn(p,q,valid)==(-k)%4
        np.testing.assert_array_equal(valid,original)
    with pytest.raises(ValueError,match='Degenerate'):canonical_quarter_turn(np.ones((3,2)),np.ones((3,2)),np.ones(3,bool))


def test_wide_training_control_is_identical_to_v5_for_fixed_rng():
    from coarse_training_v5 import sample_training as original
    yy,xx=np.mgrid[20:105:8,20:105:8];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    z=dict(p=p,q=p+1,visible=np.ones(len(p),bool),masks=[np.ones((128,128),bool)]*2,
           images=np.random.default_rng(17).integers(0,256,(6,128,128),dtype=np.uint8))
    tiles={'train':dict(split='train',tile_id='train',component='c0')}
    a=original('true_3d',{},tiles,{'train':z},np.random.default_rng(20261101))
    b=sample_training('true_3d',{},tiles,{'train':z},np.random.default_rng(20261101))
    for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x.numpy(),y.numpy())
    assert a[3]==b[3]


def test_canonical_sampler_never_reads_selection_images_and_is_reproducible():
    yy,xx=np.mgrid[20:105:8,20:105:8];p=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
    z=dict(p=p,q=quarter_turn(p,1,128),visible=np.ones(len(p),bool),masks=[np.ones((128,128),bool)]*2,
           images=np.random.default_rng(17).integers(0,256,(6,128,128),dtype=np.uint8))
    tiles={'train':dict(split='train',tile_id='train',component='c0'),'forbidden':dict(split='model_selection')}
    a=sample_training('canonical_3d',{},tiles,{'train':z},np.random.default_rng(1))
    b=sample_training('canonical_3d',{},tiles,{'train':z},np.random.default_rng(1))
    assert a[3]['training_quarter_turn']==3 and a[3]['points']>=8
    for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x.numpy(),y.numpy())
