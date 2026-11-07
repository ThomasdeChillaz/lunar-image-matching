import numpy as np
import pytest
from physical_lighting_analysis import aggregate, bin_index, shared_support, solar_separation


@pytest.mark.parametrize('first,second,expected', [
    ((0,90),(180,90),0),  # Opposite azimuth at zenith is the same Sun direction.
    ((350,0),(10,0),20),
    ((0,0),(180,0),180),
    ((45,10),(45,70),60)])
def test_solar_separation_uses_full_direction_not_azimuth(first, second, expected):
    angle = lambda p:dict(azimuth_deg=p[0], elevation_deg=p[1])
    assert solar_separation(angle(first), angle(second)) == pytest.approx(expected, abs=1e-6)


def test_support_uses_jointly_valid_raw_terrain_and_retains_dark_pixels():
    first = np.array([[0., .006], [.015, .03]])
    second = np.array([[0., .007], [.025, np.nan]])
    valid = np.ones((2,2), bool)
    assert shared_support(first, second, valid, valid, .005) == pytest.approx(2/3)
    assert shared_support(first, second, valid, valid, .01) == pytest.approx(1/3)
    assert shared_support(first, second, valid, valid, .02) == 0
    assert shared_support(first, second, ~valid, ~valid, .01) is None
    assert bin_index(1., [0,.5,1]) == 1
    assert bin_index(.5, [0,.5,1]) == 1
    assert bin_index(None, [0,.5,1]) is None


def fixtures():
    metadata = {name:dict(pair_id=name, split='test', observation_group='one-acquisition',
        solar_separation_deg=10, shared_support={str(t):.1 for t in [0., .005, .01, .02]}, query_zero_render=False)
        for name in ['correct', 'no-fit']}
    rows = [dict(pair_id=name, method='fixture', split='test', dataset_kind='physical_render',
        label_source='exact_geometry', estimated=name == 'correct', correct=name == 'correct') for name in metadata]
    return rows, metadata


def test_strata_keep_no_fit_in_denominator_and_do_not_invent_independent_groups():
    rows, metadata = fixtures()
    result = aggregate(rows, metadata, ['fixture'])
    sun = next(r for r in result if r['split'] == 'test' and r['metric'] == 'solar_separation_deg' and r['bin_index'] == 0)
    assert sun['pairs'] == 2 and sun['no_fit'] == 1 and sun['success_rate'] == .5
    assert sun['source_acquisitions'] == 1
    assert all(r['success_rate'] is None for r in result if r['pairs'] == 0)
    with pytest.raises(ValueError, match='Missing or repeated'):
        aggregate(rows[:1], metadata, ['fixture'])
    with pytest.raises(ValueError, match='Missing or repeated'):
        aggregate(rows+[rows[0]], metadata, ['fixture'])


def test_unknown_reference_labels_cannot_be_turned_into_accuracy():
    rows, metadata = fixtures()
    rows[0]['label_source'] = None
    with pytest.raises(ValueError, match='Known geometry'):
        aggregate(rows, metadata, ['fixture'])
