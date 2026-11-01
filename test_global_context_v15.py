import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F
from global_context_v15 import PairContext, ContextMatcher, VARIANTS


@pytest.mark.parametrize('variant', VARIANTS)
def test_context_has_exact_initial_scores_swapping_symmetry_and_trainable_outputs(variant):
    torch.set_num_threads(2); torch.manual_seed(73)
    # Deliberately retain the channel-major layout of sampled descriptors.
    a = F.normalize(torch.randn(1, 32, 37).transpose(1, 2), dim=-1)
    b = F.normalize(torch.randn(1, 32, 43).transpose(1, 2), dim=-1)
    pa = torch.rand(1, 37, 2)*127; pb = torch.rand(1, 43, 2)*127
    model = PairContext(variant)
    original = torch.stack([x@y.T for x, y in zip(a, b)])
    output = model(a, b, pa, pb)
    assert torch.equal(output, original)
    # Swapping GEMM operands changes floating-point accumulation even for the
    # original frozen scores. Exact initialization above remains mandatory.
    torch.testing.assert_close(output, model(b, a, pb, pa).transpose(1, 2), rtol=1e-6, atol=2e-7)
    loss = F.cross_entropy(output[0]/.1, torch.arange(37))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(model(a, b, pa, pb), original)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert 28272+35233+sum(p.numel() for p in model.parameters()) < 100000


def test_actual_frozen_proposal_forward_matches_initial_context_and_rejects_invalid_inputs():
    from compact_lunar import CompactMatcher, CompactDescriptor
    from synthetic import texture
    torch.set_num_threads(2); torch.manual_seed(81)
    coarse = CompactMatcher.__new__(CompactMatcher)
    coarse.model = CompactDescriptor(False).eval(); coarse.metadata = dict(variant='expanded'); coarse.device = 'cpu'
    coarse.last_evidence = {}
    matcher = ContextMatcher(coarse, PairContext('cross_context'))
    image = texture(733, 128); query = np.roll(image, 4, axis=1)
    expected = coarse.points(image, query); actual = matcher.points(image, query)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    blank = np.zeros_like(image)
    assert len(matcher.points(blank, blank)[0]) == 0
    assert len(matcher.points(image, query, np.zeros_like(image, bool))[0]) == 0
    with pytest.raises(ValueError):
        matcher.points(image[:64], query[:64])
