import numpy as np
import pytest

from north_coarse_reachability_v43 import measure,write_once
from verify_north_coarse_reachability_v43 import independent_measure,check_measurement


def test_independent_matrix_rotation_and_window_replay_matches_all_four_branches():
    rng=np.random.default_rng(3309)
    target=rng.uniform(0,511,(676,2));visible=rng.random(676)>.2
    ids=rng.choice(676,80,replace=False)
    p=np.column_stack([49.5+16*(ids%26),49.5+16*(ids//26)])
    for k in range(4):
        q=rng.uniform(20,490,(80,2))
        assert independent_measure(p,q,target,visible,k)==measure(p,q,target,visible,k)


def test_replay_rejects_changed_reachability_even_when_other_counts_look_valid():
    target=np.zeros((676,2));target[0]=[66,66];visible=np.zeros(676,bool);visible[0]=True
    p=q=[[49.5,49.5]];m=measure(p,q,target,visible,0);m['reachable']=[False]
    row=dict(reference_points_native=p,query_points_rotated_native=q,quarter_turn=0,measurement=m)
    with pytest.raises(ValueError,match='measurement differs'):check_measurement(row,target,visible)


def test_atomic_record_resume_preserves_partial_evidence_and_never_replaces_complete_data(tmp_path):
    path=tmp_path/'case.json';pending=tmp_path/'case.json.pending'
    pending.write_text('interrupted original bytes')
    write_once(path,dict(value=1))
    saved=path.read_bytes();prior=list(tmp_path.glob('case.json.pending.interrupted-*'))
    assert len(prior)==1 and prior[0].read_text()=='interrupted original bytes'
    write_once(path,dict(value=1));assert path.read_bytes()==saved
    with pytest.raises(ValueError,match='evidence differs'):write_once(path,dict(value=2))
    assert path.read_bytes()==saved
