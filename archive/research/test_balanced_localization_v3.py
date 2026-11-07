import numpy as np
import pytest


def test_balanced_sampling_preserves_exact_labels_and_separate_terrain_negatives():
    pytest.importorskip('torch')
    from train_balanced_localization_v3 import training_batch
    rng=np.random.default_rng(4)
    data={};tiles={};cases=[]
    for i,name in enumerate(('a','b')):
        tiles[name]=dict(tile_id=name,split='train',component=name,world_bounds=[[1000*i,0,0],[1000*i+1,1,1]])
        data[name]=dict(images=rng.integers(0,256,(6,128,128),dtype=np.uint8),
            p0=np.array([[48.,48.],[52.,52.]]),q0=np.array([[48.,48.],[52.,52.]]),
            target0=np.array([[49.,49.],[0.,0.]]),visible0=np.array([True,False]))
        cases.append(dict(split='train',component=name,tile_id=name,key=0,proposed_points=2,light_a=0,light_b=0))
    for seed in range(10):
        batch,kinds,n=training_batch(dict(cases=cases),tiles,data,np.random.default_rng(seed),4)
        assert n==2 and batch[4].tolist()==[1.,0.,1.,0.]
        assert batch[0].shape==(4,1,27,27) and batch[1].shape==(4,1,57,57)
        assert sum(kinds.values())==4


def test_balanced_sampler_rejects_unbalanced_batch_size():
    pytest.importorskip('torch')
    from train_balanced_localization_v3 import training_batch
    with pytest.raises(ValueError,match='even size'):
        training_batch({}, {}, {}, np.random.default_rng(4), 3)
