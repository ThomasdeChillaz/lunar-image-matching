import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from north_optimization_rate_v55 import validate_rate, rate_optimizer, check_optimizer
from verify_north_optimization_rate_v55 import only_rate_changed


def model():
    import torch
    return SimpleNamespace(context=torch.nn.Linear(2,2), coarse=torch.nn.Linear(2,2), train_coarse=True)


@pytest.mark.parametrize('rate', [1e-4, 1e-3])
def test_original_optimizer_membership_and_all_other_settings_remain_fixed(rate):
    from train_search_window_v18 import make_optimizer
    m=model(); control=make_optimizer(m); candidate=rate_optimizer(m,rate)
    only_rate_changed(control.state_dict()['param_groups'],candidate.state_dict()['param_groups'],rate)
    check_optimizer(m,candidate,rate)
    assert [id(p) for p in candidate.param_groups[0]['params']]==[id(p) for p in m.context.parameters()]


@pytest.mark.parametrize('rate', [True, 0, -1e-4, 1e-5, 1e-2, float('nan')])
def test_undeclared_rate_is_rejected_before_training(rate):
    with pytest.raises(ValueError,match='two declared'):
        validate_rate(rate)


def test_changed_context_rate_weight_decay_or_group_assignment_is_rejected():
    from train_search_window_v18 import make_optimizer
    m=model(); old=make_optimizer(m).state_dict()['param_groups']; candidate=rate_optimizer(m,1e-3)
    for group,field,value in [(0,'lr',1e-3),(1,'weight_decay',0.),(1,'eps',1e-6)]:
        tampered=deepcopy(candidate.state_dict()['param_groups']);tampered[group][field]=value
        with pytest.raises(ValueError,match='other than coarse'):
            only_rate_changed(old,tampered,1e-3)
    candidate.param_groups[0]['params']=list(m.coarse.parameters())
    with pytest.raises(ValueError,match='membership'):
        check_optimizer(m,candidate,1e-3)


def test_training_update_and_snapshot_evaluation_code_is_identical_to_v54():
    def nodes(name):
        tree=ast.parse(Path(name).read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
        update=next(n for n in ast.walk(run)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='update')
        evaluate=next(n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)and n.name=='evaluate')
        return [ast.dump(n,include_attributes=False)for n in (update,evaluate)]
    assert nodes('north_optimization_rate_v55.py')==nodes('north_optimization_probe_v54.py')
