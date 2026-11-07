import numpy as np
import pytest
from probe_native_crop_v57 import error_stats
from verify_native_crop_v57 import independent_stats


def test_feature_error_has_closed_form_for_a_known_perturbation():
    a=np.zeros((1089,32),np.float32);b=np.full_like(a,.25)
    stats=error_stats(a,b)
    assert stats['max_abs']==.25
    assert stats['mean_l2']==pytest.approx(np.sqrt(2),abs=1e-12)
    assert stats==independent_stats(a,b)


def test_feature_error_rejects_empty_malformed_or_nonfinite_data():
    for a,b in [(np.empty((0,32)),np.empty((0,32))),
        (np.zeros((1,32)),np.zeros((2,32))),
        (np.full((1,32),np.nan),np.zeros((1,32)))]:
        with pytest.raises(ValueError,match='paired feature'):
            error_stats(a,b)


def test_saved_feature_replay_detects_a_changed_vector():
    rng=np.random.default_rng(57);a=rng.normal(size=(1089,32));b=a.copy()
    assert independent_stats(a,b)==dict(max_abs=0.,mean_l2=0.,max_l2=0.)
    b[-1,-1]+=1
    assert independent_stats(a,b)['max_l2']==1
