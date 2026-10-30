import numpy as np
import pytest

# The core CI intentionally installs requirements.txt without neural runtimes.
# These numerical checks execute in the full research environment (not mocks).
torch = pytest.importorskip('torch', reason='Optional dense-descriptor tests require the research neural runtime')

from contrastive_terrain import (TerrainDescriptor, contrastive_loss,
    sample_descriptors, valid_centers)
from train_contrastive_terrain import crop_pair, training_tiles


def test_fractional_native_descriptor_coordinates():
    y, x = torch.meshgrid(torch.arange(19.), torch.arange(27.), indexing='ij')
    dense = torch.stack([x, y, torch.ones_like(x)])[None]
    points = torch.tensor([[[3.25, 5.75], [21.1, 12.3]]])
    actual = sample_descriptors(dense, points)
    expected = torch.nn.functional.normalize(torch.cat([points, torch.ones(1, 2, 1)], -1), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mask_complete_cnn_receptive_field():
    torch.manual_seed(4)
    model = TerrainDescriptor().eval()
    a = torch.rand(1, 1, 80, 80)
    b = a.clone()
    b[:, :, :20] = torch.rand_like(b[:, :, :20])
    mask = np.ones((80, 80), bool)
    mask[:20] = False
    valid = valid_centers(mask)
    assert not valid[31, 40] and valid[32, 40]
    with torch.no_grad():
        fa, fb = model(a), model(b)
    # Invalid pixels farther than the declared support cannot affect retained
    # descriptors. An image-wide normalizer would fail this contract.
    torch.testing.assert_close(fa[0, :, valid], fb[0, :, valid], atol=1e-6, rtol=1e-6)


def test_contrastive_loss_correspondence_and_gradients():
    a = torch.eye(8, requires_grad=True)
    aligned = contrastive_loss(a, torch.eye(8))
    shuffled = contrastive_loss(a, torch.eye(8).roll(1, 0))
    assert aligned < shuffled
    shuffled.backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().max() > 0


def test_training_crop_known_identity_and_nodata_exclusion():
    rng = np.random.default_rng(4)
    image = rng.integers(20, 255, (128, 128), dtype=np.uint8)
    images = np.stack([image, image])[None]
    masks = np.ones_like(images, bool)
    sample = crop_pair(images, masks, (0, 0, 1), np.random.default_rng(5), False)
    assert sample is not None
    np.testing.assert_array_equal(sample[0][0], sample[0][1])
    np.testing.assert_array_equal(sample[1], sample[2])
    masks[:] = False
    assert crop_pair(images, masks, (0, 0, 1), np.random.default_rng(5), True) is None


def test_reserve_never_reaches_training_loader():
    manifest = {'tiles': [dict(tile_id=s, split=s) for s in ['train', 'dev', 'reserve_a', 'reserve_b']]}
    assert [t['split'] for t in training_tiles(manifest)] == ['train', 'dev']
