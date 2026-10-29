import pytest
from research_quality_gate import accepts


def example():
    return dict(estimated=True, reverse_estimated=True, correct=True, error_px=0., cycle_p95_px=.2,
        translation_fraction=.8, inlier_residual_p95_px=0.,
        selection_quality=dict(global_correlation=.8, patch_median_correlation=.7, joint_nondark_fraction=.5,
            geometric_support_fraction=.7, agreement_fraction=.5))


def test_quality_gate_never_uses_truth_or_template_residual_or_displacement():
    policy = dict(min_global=.4, min_patch=.2, min_lit=.1, min_support=.4, min_agreement=.25, max_cycle=1.)
    row = example()
    assert accepts(row, policy)
    row.update(correct=False, error_px=999., inlier_residual_p95_px=999., translation_fraction=50.)
    assert accepts(row, policy)
    row['selection_quality']['global_correlation'] = .1
    assert not accepts(row, policy)


def test_missing_reverse_or_unknown_image_evidence_is_rejected():
    policy = dict(min_global=.4, min_patch=.2, min_lit=.1, min_support=.4, min_agreement=.25, max_cycle=1.)
    row = example()
    row['cycle_p95_px'] = None
    assert not accepts(row, policy)
    row = example()
    row['estimated'] = False
    assert not accepts(row, policy)
