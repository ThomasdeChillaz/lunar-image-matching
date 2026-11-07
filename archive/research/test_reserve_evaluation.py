import copy

import numpy as np
import pytest

from matchers import Match
from research_dataset import BANDS
from reserve_evaluation import infer, metrics, reserve_plan
from synthetic import texture


def manifest():
    tiles=[]
    for split,band in BANDS.items():
        for i in range(4):
            tiles.append(dict(tile_id=f'{split}_{i}',observation_group=f'{split}_acq{i}',split=split,
                bounds_lonlat_deg=[band[0]+.06,-44+i*.3,band[0]+.08,-43.98+i*.3],
                images=[dict(sha256=f'{split}_{i}_{j}',constant_zero_render=False) for j in range(9)]))
    return dict(bands_lon_deg=BANDS,limits={s:4 for s in BANDS},tiles=tiles,previously_evaluated_terrain=[])


def test_reserve_plan_exhaustive_and_negative_components_are_separate():
    data=manifest()
    cases,lookup,groups=reserve_plan(data)
    assert len(groups)==8
    assert len(cases)==8*(144+27)
    assert {c['split'] for c in cases}=={'reserve_a','reserve_b'}
    assert len({c['pair_id'] for c in cases})==len(cases)
    for case in cases:
        if case['geometry_family']=='nonoverlap_negative':
            assert lookup[case['tile_id']]!=lookup[case['query_tile_id']]
    # Validation rejects a fresh tile from a previously used acquisition.
    contaminated=copy.deepcopy(data)
    contaminated['previously_evaluated_terrain']=[dict(observation_group='reserve_a_acq0',bounds_lonlat_deg=[1,1,2,2])]
    with pytest.raises(ValueError,match='previously evaluated'):
        reserve_plan(contaminated)


def test_public_inference_has_no_transform_when_frozen_gate_rejects():
    image=texture(8,128)
    mask=np.ones_like(image,bool)
    identity=np.array([[1.,0,0],[0,1.,0]])
    model=lambda *args:Match(identity)
    policy=dict(min_global=1.1,min_patch=-1,min_lit=0,min_support=0,min_agreement=0,max_cycle=8)
    rejected=infer(image,image,mask,mask,model,policy)
    assert rejected['raw'].matrix is not None
    assert rejected['deployed'].matrix is None
    assert not rejected['accepted']
    policy['min_global']=-1
    accepted=infer(image,image,mask,mask,model,policy)
    assert accepted['accepted']
    assert np.array_equal(accepted['deployed'].matrix,identity)


def test_reserve_metrics_keep_rejections_and_negatives_in_denominators():
    rows=[dict(label_source='exact_geometry',correct=True,estimated=True),
          dict(label_source='exact_geometry',correct=False,estimated=False),
          dict(label_source='known_nonoverlap',correct=False,estimated=True)]
    result=metrics(rows)
    assert result['positive_cases']==2 and result['correct_rate']==.5
    assert result['rejected_or_failed_positive']==1
    assert result['negative_accepted']==1 and result['accepted_error_fraction']==.5
