import pytest


def test_paired_interval_refuses_mixed_historical_and_corrected_group_names():
    from report_coarse_training_v5 import paired_interval
    reference={'a':dict(component='corrected-group',pose_correct=False)}
    with pytest.raises(ValueError,match='terrain groups differ'):
        paired_interval({'a':dict(component='old-group',pose_correct=True)},reference)
    with pytest.raises(ValueError,match='identical cases'):paired_interval({},reference)
    assert paired_interval({'a':dict(component='corrected-group',pose_correct=True)},reference)==[100.,100.]
