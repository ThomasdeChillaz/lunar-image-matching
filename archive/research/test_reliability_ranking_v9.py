import numpy as np
import pytest
pytest.importorskip('torch')
from diagnose_no_match_generalization_v9 import rank_auc


def test_ranking_ties_inversion_and_missing_class():
    assert rank_auc([.9, .7, .2, .1], [True, True, False, False]) == 1
    assert rank_auc([.1, .2, .7, .9], [True, True, False, False]) == 0
    assert rank_auc([.5]*4, [True, True, False, False]) == .5
    assert rank_auc([.5], [True]) is None
    with pytest.raises(ValueError):
        rank_auc([np.nan], [True])
