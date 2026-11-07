import pytest

from combine_native_baselines_v8 import select_assigned


def rows(ids, correct=False):
    return [dict(pair_id=p, method=m, pose_correct=correct) for p in ids for m in ('loma_r', 'compact_orientation')]


def test_fixed_assignment_never_selects_the_more_successful_duplicate():
    ids = [str(i) for i in range(72)]
    original = rows(ids, True)
    tail = rows(ids[36:], False)
    selected = select_assigned(original, tail, ids, 'loma_r')
    assert len(selected) == 144
    assert all(r['pose_correct'] for r in selected[:72])
    assert not any(r['pose_correct'] for r in selected[72:])


def test_missing_assigned_case_cannot_be_replaced_by_an_original_duplicate():
    ids = [str(i) for i in range(72)]
    with pytest.raises(ValueError, match='Missing assigned'):
        select_assigned(rows(ids), rows(ids[37:]), ids, 'loma_r')


def test_duplicate_and_out_of_range_rows_fail_closed():
    ids = [str(i) for i in range(72)]
    for tail in (rows(ids[36:])+rows(ids[36:37]), rows(ids[35:])):
        with pytest.raises(ValueError, match='Unexpected or duplicated'):
            select_assigned(rows(ids), tail, ids, 'loma_r')
