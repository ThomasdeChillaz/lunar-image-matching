from dataclasses import replace
import inspect

import numpy as np
import pytest

from catalogue_v138 import Catalogue, ProjectionCache, batch_losses, coarse, rank, select_seeds
from blind_matcher_v138 import run
from procedural_fixture_v138 import fixture


def test_joint_camera_and_shared_light_recover_known_generated_control():
    cat,q,d,k,poses=fixture()
    result=rank(cat,q,d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=poses)
    first=result['seeds'][0]
    assert first['light_id']==8 and first['pose_index']==1
    assert result['diagnostics']['global_light_per_hypothesis']
    assert result['diagnostics']['cameras_scored']==3
    assert not result['diagnostics']['query_truth_used']


def test_radiance_and_query_changes_do_not_reuse_camera_or_light_scores():
    cat,q,d,k,poses=fixture(); cache=ProjectionCache()
    a=rank(cat,q,d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=poses,cache=cache)
    swapped=replace(cat,images=cat.images[[1,0,2]])
    b=rank(swapped,q,d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=poses,cache=cache)
    assert b['diagnostics']['projection_cache']['hit']
    assert a['seeds'][0]['light_id']==8 and b['seeds'][0]['light_id']==0
    dark=rank(cat,np.zeros_like(q),d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=poses,cache=cache)
    assert not dark['seeds'] and dark['diagnostics']['reason']=='uniform_query_on_fixed_cells'
    changed=poses.copy(); changed[0,1,3]=10.
    c=rank(cat,q,d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=changed,cache=cache)
    assert not c['diagnostics']['projection_cache']['hit']


def test_missing_catalogue_coverage_cannot_win_by_discarding_residuals():
    cat,q,d,k,poses=fixture(); images=cat.images.copy()
    images[2,100:412,100:412]=np.nan
    result=rank(replace(cat,images=images),q,d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),poses=poses)
    assert not result['seeds'] and result['diagnostics']['cameras_scored']==0
    empty=rank(cat,q,d,k,np.eye(4),k,reference_valid=np.zeros_like(d,bool),poses=poses)
    assert not empty['seeds'] and empty['diagnostics']['reason']=='no_reference_geometry'


def test_metric_matches_independent_weighted_huber_arithmetic():
    observed=np.array([1.,3.,2.,7.,5.]); predicted=np.array([[3.,7.,5.,15.,11.],[7.,5.,2.,3.,1.]])
    weights=np.array([.1,.1,.3,.3,.2])
    values=batch_losses(observed,predicted,weights)
    y=observed-weights@observed; y/=np.sqrt(weights@(y*y))
    for index,p in enumerate(predicted):
        t=p-weights@p; t/=np.sqrt(weights@(t*t)); gain=weights@(y*t)
        if gain<=0: assert values[index] is None
        else:
            r=np.abs(y-gain*t); expected=weights@np.where(r<=1.,.5*r*r,r-.5)
            assert values[index]==pytest.approx(expected,abs=1e-14)
    assert values[0]==pytest.approx(0.,abs=1e-14)


def test_seed_budget_is_global_and_reproducible_under_catalogue_permutation():
    scores=[[.3,.2,.1,.4,.8],[.6,.7,.2,.5,.9],[.5,.6,.7,.8,.9]]
    a=select_seeds(scores,(0,8,16)); b=select_seeds([scores[2],scores[1],scores[0]],(16,8,0))
    assert [(v['light_id'],v['pose_index']) for v in a]==[(v['light_id'],v['pose_index']) for v in b]
    assert len(a)<=8 and len({s['light_id'] for s in a})==2


@pytest.mark.parametrize('change',[{'light_ids':(0,0,1)},{'sun_world':np.zeros((3,3))},{'images':np.zeros((25,256,256))}])
def test_invalid_catalogues_and_truth_arguments_are_refused(change):
    cat,*_=fixture()
    with pytest.raises(ValueError): replace(cat,**change).validate()
    assert not {'query_pose','query_depth','query_sun','known_query_sun'} & set(inspect.signature(run).parameters)
    with pytest.raises(TypeError): run(None,None,None,None,None,None,query_sun=[1,0,0])


def test_blank_public_result_withholds_raw_pose_and_has_no_private_camera_output():
    cat,q,d,k,poses=fixture()
    result=run(cat,np.zeros_like(q),d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),
        height_direction_reference=[0,0,-1],poses=poses,budget=0)
    assert result==dict(pose=None,acceptance_status='uncalibrated',rejection_reason='candidate_not_calibrated')
    with pytest.raises(ValueError):
        run(cat,np.zeros_like(q),d,k,np.eye(4),k,reference_valid=np.ones_like(d,bool),
            height_direction_reference=[0,0,-1],poses=poses,budget=33)
