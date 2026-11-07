import hashlib,json
import pytest
from solar_compute_budget_v67 import reserve


def files(tmp_path):
    p=tmp_path/'ledger.json';p.write_text(json.dumps(dict(accounting_complete=True,remaining_training_wall_seconds=100,remaining_baseline_wall_seconds=120)))
    return p,hashlib.sha256(p.read_bytes()).hexdigest(),tmp_path/'journal.json'


def test_failed_and_duplicate_invocations_cannot_reset_cumulative_allowance(tmp_path):
    p,h,j=files(tmp_path);reserve(p,h,j,'failed-job','training',60)
    assert json.loads(j.read_text())['reservations'][0]['refund_on_failure']is False
    with pytest.raises(ValueError,match='already charged'):reserve(p,h,j,'failed-job','training',60)
    with pytest.raises(ValueError,match='allowance exceeded'):reserve(p,h,j,'second-job','training',41)
    assert reserve(p,h,j,'second-job','training',40)['remaining_after_reservation']==0
    assert reserve(p,h,j,'baseline','baseline',120)['remaining_after_reservation']==0


def test_missing_accounting_or_changed_ledger_never_admits_science(tmp_path):
    p,h,j=files(tmp_path);p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='ledger changed'):reserve(p,h,j,'one','training',1)
    z=json.loads(p.read_text());z['accounting_complete']=False;p.write_text(json.dumps(z));h=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='Reconstruct'):reserve(p,h,j,'one','training',1)
    assert not j.exists()
