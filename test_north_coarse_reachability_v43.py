import numpy as np
import pytest

from north_coarse_reachability_v43 import source_indices,measure,rotate,union_reachability,oracle_inputs


def labels(point):
    target=np.zeros((676,2));target[0]=point;visible=np.zeros(676,bool);visible[0]=True
    return target,visible


def test_fractional_native_grid_maps_exactly_and_rejects_interpolated_labels():
    np.testing.assert_array_equal(source_indices([[49.5,49.5],[449.5,449.5]]),[0,675])
    with pytest.raises(ValueError,match='exact fractional'):
        source_indices([[49.51,49.5]])


def test_square_search_reachability_uses_rounded_center_not_euclidean_radius():
    target,visible=labels([66.,66.])
    m=measure([[49.5,49.5]],[[49.5,49.5]],target,visible,0)
    assert m['reachable']==[True]and m['correct_at_native_px']['16']==0
    target[0,0]=66.001
    assert measure([[49.5,49.5]],[[49.5,49.5]],target,visible,0)['reachable']==[False]


def test_nonvisible_points_never_become_reachable_due_to_darkness_or_location():
    target,visible=labels([49.5,49.5]);visible[:]=False
    m=measure([[49.5,49.5]],[[49.5,49.5]],target,visible,0)
    assert m['reachable_proposals']==0 and m['nonvisible_returned']==1


def test_rotated_query_frame_is_used_for_local_search_scoring():
    target,visible=labels([80.,90.]);q=rotate([[80.,90.]],1)
    m=measure([[49.5,49.5]],q,target,visible,1)
    assert m['reachable']==[True]and m['correct_at_native_px']['1']==1


def test_oracle_union_deduplicates_source_points_and_is_not_pose_success():
    target,visible=labels([80.,90.]);branches=[]
    for k in range(4):
        m=measure([[49.5,49.5]],rotate([[80.,90.]],k),target,visible,k)
        branches.append(dict(quarter_turn=k,measurement=m))
    u=union_reachability(branches)
    assert u['oracle_source_points_reachable_in_any_branch']==1
    assert u['oracle_reachable_fraction']==1 and u['actual_pose_success_claim']is False


def test_oracle_control_uses_unrotated_truth_and_drops_unreachable_matches():
    target,visible=labels([80.,90.]);target[1]=[400.,400.];visible[1]=True
    p=np.array([[49.5,49.5],[65.5,49.5]])
    q=rotate([[80.,90.],[49.5,49.5]],1)
    m=measure(p,q,target,visible,1)
    op,oq=oracle_inputs(p,m,target)
    np.testing.assert_array_equal(op,[[49.5,49.5]])
    np.testing.assert_array_equal(oq,[[80.,90.]])
