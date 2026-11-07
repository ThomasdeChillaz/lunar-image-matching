"""Selection and exact interruption recovery controls for the context cohort."""
import io

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from context_supervision_v15 import make_context
from train_global_context_v16 import checkpoint, restore, selection_key, validate_initial


def update(model, optimizer, rng):
    a = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(1, 11, 32)).astype(np.float32)), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(1, 13, 32), dim=-1)
    pa, pb = torch.rand(1, 11, 2)*128, torch.rand(1, 13, 2)*128
    optimizer.zero_grad(set_to_none=True)
    loss = model(a, b, pa, pb).square().mean()
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
    return float(loss)


@pytest.mark.parametrize('variant', ['pointwise','self_context','cross_context'])
def test_interrupted_optimizer_and_both_random_streams_resume_bit_exactly(variant):
    torch.set_num_threads(2); torch.manual_seed(20261101); rng = np.random.default_rng(20261101)
    model = make_context(variant); optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    update(model, optimizer, rng)
    stream = io.BytesIO(); torch.save(checkpoint(model, optimizer, rng, 1, [], .1), stream)
    expected_loss = update(model, optimizer, rng)
    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    expected_numpy, expected_torch = rng.random(19), torch.rand(19)
    # Reconstructing a module consumes random values. restore must restore both
    # streams after module/optimizer creation, including Adam moments and step.
    alternate = make_context(variant); other_optimizer = torch.optim.AdamW(alternate.parameters(), lr=.9)
    other_rng = np.random.default_rng(7); stream.seek(0)
    state = torch.load(stream, weights_only=False); restore(state, alternate, other_optimizer, other_rng)
    assert update(alternate, other_optimizer, other_rng) == expected_loss
    assert all(torch.equal(v, alternate.state_dict()[k]) for k, v in expected_weights.items())
    np.testing.assert_array_equal(other_rng.random(19), expected_numpy)
    assert torch.equal(torch.rand(19), expected_torch)
    assert all(int(s['step']) == 2 for s in other_optimizer.state.values())


def test_selection_prioritizes_unfiltered_pose_count_and_keeps_initialization_on_a_tie():
    assert selection_key(dict(correct_poses=55, correct_at_two_px=1), 2000) > selection_key(dict(correct_poses=54, correct_at_two_px=999999), 0)
    assert selection_key(dict(correct_poses=54, correct_at_two_px=1), 0) > selection_key(dict(correct_poses=54, correct_at_two_px=1), 500)


def test_actual_initial_control_detects_missing_and_displaced_pose_rows(monkeypatch):
    row = dict(pair_id='control', pose_correct=True, strict_pose_correct=False,
        reference_points_native=[[10.,20.]],query_points_native=[[14.,25.]],estimated_cam2world=[[1.]])
    monkeypatch.setattr('train_global_context_v16.read_rows', lambda path: [row])
    validate_initial([row.copy()])
    with pytest.raises(ValueError, match='equivalence failed'):
        validate_initial([dict(row, estimated_cam2world=None)])
    with pytest.raises(ValueError, match='equivalence failed'):
        validate_initial([dict(row, query_points_native=[[14.,26.]])])
