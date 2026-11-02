import ast
from datetime import datetime,timedelta,timezone
from pathlib import Path
from time import perf_counter

import pytest

import train_north_rate_v56 as study


def test_cost_gate_covers_all_five_positive_checkpoints_and_selected_negatives():
    assert study.cost_admission(17.2,1.87)==pytest.approx(2*(17.2*20+1.87*576+30))


@pytest.mark.parametrize('fit,case', [(0,1),(1,0),(-1,1),(float('nan'),1),(1,float('inf')),(100,3)])
def test_cost_gate_rejects_missing_nonfinite_or_over_budget_evidence(fit,case):
    with pytest.raises(ValueError,match='fixed one-hour'):
        study.cost_admission(fit,case)


def protocol(start):
    return dict(created_utc=start.isoformat(),absolute_deadline_utc=(start+timedelta(seconds=3600)).isoformat())


def test_original_absolute_deadline_is_not_reset_by_a_fresh_invocation():
    with pytest.raises(TimeoutError,match='do not reset'):
        study.clock_check(protocol(datetime.now(timezone.utc)-timedelta(seconds=3601)),perf_counter())


def test_monotonic_cap_and_backwards_calendar_are_both_enforced():
    with pytest.raises(TimeoutError):
        study.clock_check(protocol(datetime.now(timezone.utc)),perf_counter()-3601)
    with pytest.raises(TimeoutError):
        study.clock_check(protocol(datetime.now(timezone.utc)+timedelta(seconds=10)),perf_counter())


def test_inference_errors_remain_in_the_complete_case_denominator(tmp_path,monkeypatch):
    import json
    root=tmp_path/'root';root.mkdir();(root/'protocol.json').write_text('{}')
    cases=[dict(pair_id=str(i),tile_id='tile',light_a=0,light_b=1)for i in range(2)]
    monkeypatch.setattr(study,'clock_check',lambda *a:None)
    monkeypatch.setattr(study,'inference_inputs',lambda *a:[])
    def fail(*a):raise RuntimeError('declared fixture failure')
    monkeypatch.setattr(study,'infer',fail)
    monkeypatch.setattr(study,'score_case',lambda bundle,case,p,q,pose:dict(pose_correct=False,estimated_cam2world=None))
    rows=study.evaluate(None,None,cases,root/'cases',{},dict(root=str(root)),perf_counter())
    assert [r['pair_id']for r in rows]==['0','1']
    assert all(r['inference_error']['type']=='RuntimeError'and not r['pose_correct']for r in rows)
    assert len(list((root/'cases').glob('*.json')))==3


def test_numerical_inference_and_training_loop_match_the_original_control():
    def tree(path):return ast.parse(Path(path).read_text().replace('expanded_data_architecture_control','coarse_learning_rate_control'))
    old,new=map(tree,['train_north_control_v39.py','train_north_rate_v56.py'])
    def node(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
    assert ast.dump(node(old,'evaluate'),include_attributes=False)==ast.dump(node(new,'evaluate'),include_attributes=False)
    def loop(t):return next(n for n in ast.walk(node(t,'one_seed'))if isinstance(n,ast.While))
    assert ast.dump(loop(old),include_attributes=False)==ast.dump(loop(new),include_attributes=False)


def test_invalid_source_revision_rejects_before_creating_experiment(tmp_path):
    with pytest.raises(ValueError,match='Exact public'):
        study.run(tmp_path/'experiment','not-a-source-revision')
    assert not (tmp_path/'experiment').exists()
