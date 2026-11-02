import numpy as np
import pytest

from report_north_lighting_failure_v42 import point_counts,summarize,lighting_decomposition,training_loss_profile


def row():
    return dict(pair_id='fixture',reference_points_native=[[0,0],[1,0],[2,0],[3,0]],
        query_points_native=[[0,0],[1,0],[2,0],[100,0]],visible=[True,False,True,True],
        target_points_native=[[0,0],[4,0],[3,0]],returned=4,scorable=3,
        correct_at_native_px={'1':1,'2':2,'4':2,'8':2,'12':2},
        estimated_cam2world=None,pose_correct=False,strict_pose_correct=False)


def test_known_nonvisible_returns_do_not_disappear_from_precision():
    r=row();p=point_counts(r);s=summarize([r])
    assert p['nonvisible_returned']==1
    assert s['precision_all_returned']['2']==.5
    assert s['precision_scorable_only']['2']==pytest.approx(2/3)
    assert s['no_pose']==1 and s['pose_success_all_eligible']==0


def test_existing_native_count_tampering_is_rejected():
    r=row();r['correct_at_native_px']['2']=3
    with pytest.raises(ValueError,match='point count differs'):point_counts(r)


def test_zero_matches_have_undefined_precision_but_failed_pose_still_counts():
    r=row();r.update(reference_points_native=[],query_points_native=[],visible=[],
        target_points_native=[],returned=0,scorable=0,
        correct_at_native_px={str(t):0 for t in (1,2,4,8,12)})
    s=summarize([r])
    assert s['eligible_positive_pairs']==1 and s['no_pose']==1
    assert s['precision_all_returned']['5']is None


def test_incomplete_or_reordered_cases_cannot_be_reported_as_complete():
    with pytest.raises(ValueError,match='All96'):
        lighting_decomposition([row()],[dict(pair_id='other')])


def test_threshold_five_is_analysis_only_and_not_a_new_pose_criterion():
    r=row();r['query_points_native'][-1]=[8,0]
    r['correct_at_native_px'].update({'8':3,'12':3})
    p=point_counts(r)
    assert p['correct_at_native_px']['5']==3
    assert r['pose_correct']is False


def test_loss_profile_exposes_negative_only_console_sampling():
    history=[]
    for step in range(1,2001):
        negative=step%4==0
        history.append(dict(step=step,loss=.1 if negative else 5.,gradient_norm=1.,
            sample=dict(verified_nonoverlap=negative,light_a=0,light_b=1),
            labels=dict(forward=dict(positive=0 if negative else 600))))
    profile=training_loss_profile(history);first=profile['windows']['1-500']
    assert first['nonoverlap']['updates']==125
    assert first['changed_light_positive']['updates']==375
    assert first['nonoverlap']['mean_loss']==pytest.approx(.1)
    assert first['changed_light_positive']['mean_loss']==5.
    assert profile['pose_improvement_claim']is False
