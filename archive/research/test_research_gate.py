import pytest
from research_gate import accepts, calibrate
from test_research_protocol import tiny_manifest


def row(name, correct, cycle, label='exact_geometry'):
    return dict(pair_id=name, tile_id='tile1', split='dev', method='example', label_source=label,
        correct=correct, estimated=True, reverse_estimated=True, support=20,
        coverage_reference=.2, coverage_query=.2, cycle_p95_px=cycle,
        translation_fraction=.7, correlation_score=None, peak_margin=None)


def test_gate_does_not_reject_large_motion_or_read_correctness_at_inference():
    policy = dict(min_coverage=.05, max_cycle_p95_px=2., min_support=6, min_correlation_score=0., min_peak_margin=0.)
    example = row('a', True, .3)
    assert accepts(example, policy)
    example['correct'] = False
    example['error_px'] = 1000.
    assert accepts(example, policy)


def test_calibration_uses_only_development_labels_and_keeps_negative_errors_explicit():
    rows = [row('a', True, .2), row('b', False, 20), row('c', False, 40, 'known_nonoverlap')]
    result = calibrate(rows, tiny_manifest())
    count = result['policies']['example']['development_counts']
    assert count['accepted_correct'] == 1 and count['accepted_incorrect'] == count['accepted_nonoverlap'] == 0
    rows[0]['split'] = 'reserve_a'
    with pytest.raises(ValueError, match='development labels'):
        calibrate(rows, tiny_manifest())
    rows[0]['split'] = 'dev'
    rows[0]['label_source'] = 'map_grid'
    with pytest.raises(ValueError, match='development labels'):
        calibrate(rows, tiny_manifest())
