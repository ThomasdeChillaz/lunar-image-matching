"""Small analytic algebra controls; no model forward, data or fitting required."""
import pytest

torch = pytest.importorskip('torch')
from mutual_assignment_v23 import log_assignment, window_correction


def mutual_pairs(scores):
    index = scores.argmax(1); reverse = scores.argmax(0)
    return {(i, int(j)) for i, j in enumerate(index) if int(reverse[j]) == i and scores[i, j] >= .5}


def row_set_nll(scores, good):
    logits = scores/.1; rows = good.any(1)
    return (logits[rows].logsumexp(1)-logits[rows].masked_fill(~good[rows], -torch.inf).logsumexp(1)).mean()


def test_row_set_objective_can_improve_while_mutual_proposal_count_falls():
    # Two neighboring references can both refine from either of two query
    # candidates. A third row/column is a genuine incompatible distractor.
    initial = torch.tensor([[.8, .7, .2], [.7, .8, .2], [.2, .2, .1]], dtype=torch.float64)
    collapsed = initial.clone(); collapsed[0, 1] = .95
    good = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    assert len(mutual_pairs(initial)) == 2 and len(mutual_pairs(collapsed)) == 1
    assert row_set_nll(collapsed, good) < row_set_nll(initial, good)
    # Coupled mass notices the competition that row-set likelihood misses.
    assert row_set_nll(collapsed, good)+window_correction(collapsed, good) > (
        row_set_nll(initial, good)+window_correction(initial, good))


def test_dual_mass_prefers_distinct_matches_and_has_bounded_expected_count():
    uniform = torch.zeros(4, 4, dtype=torch.float64)
    distinct = torch.eye(4, dtype=torch.float64)*2
    assert log_assignment(uniform).exp().sum().item() == pytest.approx(1.)
    assert log_assignment(distinct).exp().sum().item() > 3.99
    for scores in (uniform, distinct, torch.tensor([[.1, .9, .4], [.6, .7, .2]], dtype=torch.float64)):
        mass = log_assignment(scores).exp()
        assert (mass.sum(0) <= 1+1e-12).all() and (mass.sum(1) <= 1+1e-12).all()
        torch.testing.assert_close(log_assignment(scores.T), log_assignment(scores).T)


def test_correction_is_invariant_to_joint_reindexing_and_gradients_remain_finite():
    scores = torch.tensor([[.6, .9, -.1], [.7, .5, .3]], dtype=torch.float64, requires_grad=True)
    good = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    loss = window_correction(scores, good)
    rows = torch.tensor([1, 0]); cols = torch.tensor([2, 0, 1])
    torch.testing.assert_close(loss, window_correction(scores[rows][:, cols], good[rows][:, cols]))
    assert loss >= 0; loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    zero = window_correction(scores, torch.zeros_like(good)); assert zero == 0
    zero.backward(); assert torch.isfinite(scores.grad).all()


def test_coupled_mass_gradient_matches_independent_finite_differences():
    scores = torch.tensor([[.3, .8, -.2], [.6, .2, .4]], dtype=torch.float64, requires_grad=True)
    good = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda s: window_correction(s, good), (scores,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_invalid_scores_and_masks_fail_explicitly_and_large_logits_stay_finite():
    for scores in (torch.empty(0, 2), torch.tensor([[float('nan')]])):
        with pytest.raises(ValueError):log_assignment(scores)
    with pytest.raises(ValueError):log_assignment(torch.ones(2, 2), 0.)
    with pytest.raises(ValueError):window_correction(torch.ones(2, 2), torch.ones(2, 2))
    scores = torch.tensor([[1e4, -1e4], [-1e4, 1e4]], requires_grad=True)
    loss = window_correction(scores, torch.eye(2, dtype=torch.bool)); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()


def test_independent_numpy_scalar_and_native_window_boundary_agree():
    import numpy as np
    from verify_mutual_pilot_v23 import independent_correction
    scores = np.array([[.3, .8, .1], [.7, .4, -.2]], np.float32)
    grid = np.array([[12, 12], [16, 12], [20, 12]], np.float32)
    target = np.array([[16.125, 12], [16.1251, 12]], np.float32)
    value, mask = independent_correction(scores, target, [True, True], grid)
    # Native candidates are50,66,82; target66 includes both endpoints at16px.
    assert mask.tolist() == [[True, True, True], [False, True, True]]
    actual = window_correction(torch.from_numpy(scores), torch.from_numpy(mask))
    assert float(actual) == pytest.approx(value, abs=2e-6)
    zero, mask = independent_correction(scores, target, [False, False], grid)
    assert zero == 0 and not mask.any()
