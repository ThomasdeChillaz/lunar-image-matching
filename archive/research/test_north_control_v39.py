"""Failure-denominator, partial-evaluation resume and freeze regressions."""
from datetime import datetime, timedelta, timezone
import json
from time import perf_counter

import numpy as np
import pytest

import train_north_control_v39 as study
import verify_north_results_v39 as replay


def setup(tmp_path, monkeypatch):
    (tmp_path/'protocol.json').write_text('{}\n')
    protocol=dict(root=str(tmp_path))
    cases=[dict(pair_id='case-'+str(i),tile_id='tile',positive=True) for i in range(2)]
    inputs=(np.zeros((2,2),np.uint8),np.ones((2,2),np.uint8),
            np.full((2,2),5.),np.eye(3),np.eye(4),np.eye(3))
    calls=[]
    def infer(matcher,*args):
        # Six permitted inference inputs followed by cache and image IDs.
        assert len(args)==8
        for actual,expected in zip(args[:6],inputs):
            np.testing.assert_array_equal(actual,expected)
        calls.append(True)
        return None,[dict(p=np.empty((0,2)),q=np.empty((0,2)),info={})]
    monkeypatch.setattr(study,'clock_check',lambda *a:None)
    monkeypatch.setattr(study,'inference_inputs',lambda *a:inputs)
    monkeypatch.setattr(study,'infer',infer)
    monkeypatch.setattr(study,'score_case',lambda bundle,case,p,q,pose:
        dict(pose_correct=False,strict_pose_correct=False,estimated_cam2world=None))
    for case in cases:
        case.update(light_a=0,light_b=1)
    return protocol,cases,calls


def test_interrupted_case_evaluation_resumes_without_replacing_saved_evidence(tmp_path,monkeypatch):
    protocol,cases,calls=setup(tmp_path,monkeypatch)
    checks=[]
    def stop_after_one(*a):
        checks.append(True)
        if len(checks)==2:raise TimeoutError('test interruption')
    monkeypatch.setattr(study,'clock_check',stop_after_one)
    folder=tmp_path/'partial'
    with pytest.raises(TimeoutError):study.evaluate(None,None,cases,folder,{},protocol,0)
    saved=list(folder.glob('*.json'))
    before={p.name:p.read_bytes() for p in saved}
    assert len(calls)==1
    monkeypatch.setattr(study,'clock_check',lambda *a:None)
    rows=study.evaluate(None,None,cases,folder,{},protocol,0)
    assert len(rows)==2 and len(calls)==2
    assert all((folder/name).read_bytes()==value for name,value in before.items())


def test_inference_failure_remains_in_eligible_denominator_with_null_pose(tmp_path,monkeypatch):
    protocol,cases,_=setup(tmp_path,monkeypatch)
    def fail(*a):raise ValueError('no usable matching support')
    monkeypatch.setattr(study,'infer',fail)
    rows=study.evaluate(None,None,cases,tmp_path/'partial',{},protocol,0)
    assert len(rows)==2
    assert all(r['estimated_cam2world'] is None and not r['pose_correct'] for r in rows)
    assert all(r['inference_error']['type']=='ValueError' for r in rows)


def test_resumed_evaluation_rejects_changed_case_metadata(tmp_path,monkeypatch):
    protocol,cases,_=setup(tmp_path,monkeypatch)
    study.evaluate(None,None,cases,tmp_path/'partial',{},protocol,0)
    changed=[dict(c) for c in cases];changed[0]['light_b']=5
    with pytest.raises(ValueError,match='case changed'):
        study.evaluate(None,None,changed,tmp_path/'partial',{},protocol,0)


def test_exact_initial_equivalence_checks_native_points_not_just_pose_count():
    old=dict(pair_id='one',method='frozen_surface',pose_correct=True,strict_pose_correct=True,
        reference_points_native=[[1.,2.]],query_points_native=[[3.,4.]],estimated_cam2world=np.eye(4).tolist())
    study.initial_equivalence([old],[old])
    changed=dict(old,query_points_native=[[3.01,4.]])
    with pytest.raises(ValueError,match='initialization differs'):
        study.initial_equivalence([changed],[old])


def test_checkpoint_selection_keeps_initialization_when_training_worsens():
    records=[dict(step=0,selection=dict(correct_poses=70,correct_at_two_px=100)),
        dict(step=500,selection=dict(correct_poses=69,correct_at_two_px=900)),
        dict(step=1000,selection=dict(correct_poses=70,correct_at_two_px=100))]
    winner=max(records,key=lambda r:study.selection_key(r['selection'],r['step']))
    assert winner['step']==0


def test_absolute_deadline_still_expires_on_a_new_invocation():
    now=datetime.now(timezone.utc)
    protocol=dict(created_utc=(now-timedelta(hours=7)).isoformat(),
        absolute_deadline_utc=(now-timedelta(hours=1)).isoformat())
    with pytest.raises(TimeoutError,match='Original cohort deadline'):
        study.clock_check(protocol,perf_counter())


def test_metric_replay_detects_tampered_result(monkeypatch):
    row=dict(pair_id='one',reference_points_native=[],query_points_native=[],
        estimated_cam2world=None,pose_correct=True)
    monkeypatch.setattr(replay,'score_case',lambda *a:dict(pose_correct=False))
    with pytest.raises(ValueError,match='metric replay differs'):
        replay.replay_row(None,row)


def test_native_feature_cache_is_bounded_and_preserves_recent_reference():
    cache=study.FineFeatureCache(capacity=2)
    cache['reference']='a';cache['query-0']='b'
    assert cache['reference']=='a'
    cache['query-1']='c'
    assert len(cache)==2 and 'query-0'not in cache
    assert cache['reference']=='a'and cache['query-1']=='c'
