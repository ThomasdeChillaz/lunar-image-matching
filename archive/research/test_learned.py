import json,os
from pathlib import Path
import numpy as np
import pytest
from fetch_models import verify_weights
from matchers import fit_geometry


def test_missing_weight_error_is_explicit(tmp_path):
    lock=json.loads(Path('model_lock.json').read_text())
    with pytest.raises(RuntimeError,match='fetch_models.py'): verify_weights(tmp_path,lock)


def test_corrupt_weights_rejected_before_loading(tmp_path):
    path=tmp_path/'hub/checkpoints';path.mkdir(parents=True);(path/'fake.pth').write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='corrupt'):verify_weights(tmp_path,{'weights':[{'filename':'fake.pth','sha256':'0'*64}]})


def test_degenerate_geometry_is_failure():
    a=np.column_stack((np.arange(10),np.zeros(10)))
    assert fit_geometry(a,a).matrix is None


@pytest.mark.skipif(os.environ.get('RUN_LEARNED')!='1',reason='Explicit pretrained integration test; requires pinned dependencies/weights')
def test_actual_pretrained_forward_pass():
    from learned import LearnedMatcher
    from synthetic import texture,make_pair
    from geometry import endpoint_error,evaluation_points
    m=LearnedMatcher();a=texture(42)
    b,h=make_pair(a,dict(angle=12,dx=8,dy=-6,gain=.7,gamma=1.2,noise=.005),43)
    m.warmup(a);r=m(a,b)
    error=endpoint_error(r.matrix,h,evaluation_points(a.shape,h))
    assert error is not None and error<1
    blank=np.zeros_like(a)
    assert m(blank,blank).reason=='low_texture'
    no_valid=np.zeros_like(a,bool)
    assert m(a,a,no_valid,no_valid).matrix is None
