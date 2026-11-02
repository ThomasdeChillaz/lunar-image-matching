import ast
from datetime import datetime,timedelta,timezone
from pathlib import Path
from time import perf_counter
import pytest
import train_north_context_study_v61 as study


def test_cost_gate_covers_full_study_and_all_576_eligible_case_runs():
    assert study.cost_admission(100,2)==pytest.approx(2*(2000+2*576+60))


@pytest.mark.parametrize('fit,case',[(0,1),(-1,1),(1,float('nan')),(float('inf'),1),(1,0),(500,3)])
def test_missing_invalid_or_unaffordable_costs_do_not_start_study(fit,case):
    with pytest.raises(ValueError,match='fixed three-hour'):study.cost_admission(fit,case)


def test_deadline_stays_fixed_across_invocations():
    start=datetime.now(timezone.utc)-timedelta(seconds=10801)
    protocol=dict(created_utc=start.isoformat(),absolute_deadline_utc=(start+timedelta(seconds=10800)).isoformat())
    with pytest.raises(TimeoutError,match='do not reset'):study.clock_check(protocol,perf_counter())


def test_original_numerical_inference_and_training_loop_are_preserved():
    def node(path,name):
        tree=ast.parse(Path(path).read_text().replace('expanded_data_architecture_control','cross_image_context_control'))
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    old='train_north_control_v39.py';new='train_north_context_study_v61.py'
    assert ast.dump(node(old,'evaluate'),include_attributes=False)==ast.dump(node(new,'evaluate'),include_attributes=False)
    def loop(path):return next(n for n in ast.walk(node(path,'one_seed'))if isinstance(n,ast.While))
    assert ast.dump(loop(old),include_attributes=False)==ast.dump(loop(new),include_attributes=False)
    assert study.SEEDS==(20261101,)and study.STEPS==2000 and study.INTERVAL==500
