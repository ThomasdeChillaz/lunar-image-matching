import inspect
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')

from coarse_pipeline_v5 import infer


def test_inference_interface_does_not_accept_query_depth_or_true_pose():
    names=set(inspect.signature(infer).parameters)
    assert 'reference_depth' in names and 'reference_cam2world' in names
    assert 'query_depth' not in names and 'query_pose' not in names and 'query_cam2world' not in names


def test_actual_frozen_model_recovers_controlled_quarter_turn_pose():
    model=Path('runs/compact-v2-expanded-20261030')
    if not (model/'selected.pt').exists():pytest.skip('Own released model artifact is not installed')
    import cv2
    from compact_lunar import CompactMatcher
    from synthetic import texture
    torch.set_num_threads(2);cv2.setNumThreads(1)
    matcher=CompactMatcher(model);image=texture(733,512);query=np.rot90(image,1).copy()
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    selected,_=infer(matcher,matcher.model,image,query,np.full(image.shape,10.),k,np.eye(4),k)
    assert selected is not None and selected['quarter_turn']==3
    expected=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    assert np.linalg.norm(selected['pose'][:3,:3]-expected)<.01
    assert np.linalg.norm(selected['pose'][:3,3])<.05
