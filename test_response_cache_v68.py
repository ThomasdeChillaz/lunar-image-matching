import numpy as np
import torch
from test_sun_search_v68 import geometry
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel
from solar_frontend_v63 import PairResponseCache,reference_anchors


def test_pair_response_cache_refreshes_pose_condition_and_checkpoint_without_research_truth():
    torch.manual_seed(12);g,k=geometry();model=LightingResponseModel(CompactDescriptor(False)).eval()
    a=np.random.default_rng(12).integers(0,256,(512,512),dtype=np.uint8);anchors=np.arange(256);ids=np.arange(len(g.sun_vectors))
    cache=PairResponseCache(model,a,a,g,k,anchors,ids)
    p=reference_anchors()[100];sun=np.flatnonzero(g.coarse_sun_mask)[:2]
    first=cache(100,0,p,g.cam2world,sun);assert first is not None
    before=cache.cost.copy();same=cache(100,0,p,g.cam2world,sun)
    np.testing.assert_array_equal(first,same);assert cache.cost['cache_hits']==before['cache_hits']+1 and cache.cost['canonicalizations']==before['canonicalizations']
    trial=g.cam2world.copy();trial[0,3]+=.01;cache(100,0,p,trial,sun)
    assert cache.cost['canonicalizations']==before['canonicalizations']+1
    with torch.no_grad():next(model.fine.parameters()).add_(.01)
    cache(100,0,p,trial,sun)
    assert cache.cost['canonicalizations']==before['canonicalizations']+2 and cache.cost['patch_encoder_calls']>before['patch_encoder_calls']
