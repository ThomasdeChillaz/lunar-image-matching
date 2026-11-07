"""Six-light, split, global-negative and fractional-coordinate regressions."""
import numpy as np

from north_development_plan_v35 import development_plan,proposal_grid_native
from north_training_bundle_v38 import prepare_training,sample_training


def fixture():
    _,sites=development_plan();train=[s for s in sites if s['split']=='training'][:2]
    held=next(s for s in sites if s['split']=='model_selection');tiles={};prepared={}
    for i,site in enumerate(train+[held]):
        key='fixture-'+str(i);split='train'if i<2 else'model_selection'
        tiles[key]=dict(tile_id=key,component=site['component'],split=split,
            support=dict(x=site['x'],y=site['y']),world_bounds=[[0,0,0],[1,1,1]])
        rotations={k:dict(forward=np.full((676,2),12+k,np.float32),reverse=np.full((676,2),20+k,np.float32),
            forward_visible=np.ones(676,bool),reverse_visible=np.ones(676,bool))for k in range(4)}
        prepared[key]=dict(images=np.stack([np.full((128,128),20+10*j,np.uint8)for j in range(12)]),rotations=rotations)
    # Accessing held-out preparation must fail, even though its metadata exists.
    prepared['fixture-2']=None
    return tiles,prepared


def test_both_images_sample_all_six_lights_without_selection_inputs():
    tiles,prepared=fixture();rng=np.random.default_rng(20261101);a=set();b=set()
    for _ in range(100):
        images,target,meta=sample_training(tiles,prepared,rng)
        a.add(meta['light_a']);b.add(meta['light_b'])
        assert meta['tile_a']!='fixture-2'and meta['tile_b']!='fixture-2'
        assert images.shape==(2,1,128,128)and images.dtype==np.float32
        expected=prepared[meta['tile_a']]['rotations'][meta['query_quarter_turn']]
        np.testing.assert_array_equal(target['forward'],expected['forward'])
    assert a==set(range(6))and b==set(range(6))


def test_negative_uses_global_support_despite_identical_local_boxes():
    tiles,prepared=fixture();rng=np.random.default_rng(20261101)
    _,target,meta=sample_training(tiles,prepared,rng,negative_pair=True)
    assert meta['component_a']!=meta['component_b']and meta['verified_nonoverlap']
    assert not target['forward_visible'].any()and not target['reverse_visible'].any()
    assert meta['tile_a']!='fixture-2'and meta['tile_b']!='fixture-2'


def test_saved_rng_replays_images_targets_and_metadata_exactly():
    tiles,prepared=fixture();rng=np.random.default_rng(20261101);state=rng.bit_generator.state
    a=sample_training(tiles,prepared,rng,negative_pair=True)
    rng.bit_generator.state=state;b=sample_training(tiles,prepared,rng,negative_pair=True)
    np.testing.assert_array_equal(a[0],b[0]);assert a[2]==b[2]
    for k in a[1]:np.testing.assert_array_equal(a[1][k],b[1][k])


def test_native_fractional_labels_preserve_pixel_centers_when_rotated():
    grid=proposal_grid_native();images=np.zeros((12,512,512),np.uint8)
    tiles={'a':dict(split='train')};data={'a':dict(images=images,forward_target=grid,
        forward_visible=np.ones(676,bool),reverse_target=np.stack([grid]*4),reverse_visible=np.ones((4,676),bool))}
    prepared=prepare_training(tiles,data)['a']
    np.testing.assert_array_equal(prepared['rotations'][0]['forward'][0],[12,12])
    np.testing.assert_array_equal(prepared['rotations'][1]['forward'][0],[12,115])
    np.testing.assert_array_equal(prepared['rotations'][2]['forward'][0],[115,115])
    assert prepared['images'].shape==(12,128,128)
