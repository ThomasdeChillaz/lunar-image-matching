import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from tycho_dataset import intersection, load_pair, validate
from rebase_review import conversion, rebase
from score_independent import score


def test_real_common_grid_preserves_centers_and_invalid_interpolation(tmp_path):
    values = np.tile(np.arange(128, dtype=np.float32)/255, (128, 1))
    values[64, 64] = np.nan
    path = tmp_path/'gradient.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=128, height=128, count=1, dtype='float32') as image:
        image.write(values, 1)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assets = {'a':dict(path='gradient.tif', sha256=digest, bounds_lonlat_deg=[0, 0, 128, 128])}
    pair = dict(reference='a', query='a', bounds_lonlat_deg=[0, 0, 128, 128])
    a, b, ma, mb = load_pair(pair, assets, tmp_path)
    assert np.array_equal(a[10], np.arange(128))
    assert not ma[64, 64] and ma.sum() == 128*128-1
    assert np.array_equal(ma, mb) and np.array_equal(a, b)
    pair['bounds_lonlat_deg'] = [.5, 0, 128.5, 128]
    _, _, mask, _ = load_pair(pair, assets, tmp_path)
    assert not mask[64, 63] and not mask[64, 64]


def test_new_real_manifest_excludes_physical_sources_and_training_terrain():
    path = Path('tycho_real_manifest.json')
    if not path.exists():
        pytest.skip('Optional full real-appearance archive not acquired')
    manifest, physical = json.loads(path.read_text()), json.loads(Path('moon_manifest.json').read_text())
    validate(manifest, physical)
    assert len({p['region_id'] for p in manifest['pairs']}) == 5
    corrupt = copy.deepcopy(manifest)
    corrupt['assets'][0]['observation_group'] = physical['tiles'][0]['observation_group']
    with pytest.raises(ValueError, match='leakage'):
        validate(corrupt, physical)
    assert intersection([0,0,1,1], [1,0,2,1]) is None


def test_manual_landmark_conversion_uses_known_pixel_center_geometry():
    points, bound = conversion([[10, 20]], [0, 0, 512, 512], [0, 256, 256, 512], 1.)
    assert np.allclose(points, [[20.5, 40.5]])
    assert bound == 2.5
    original = json.loads(Path('lunar_manifest.json').read_text())
    expanded = json.loads(Path('expanded_solar_manifest.json').read_text())
    # Merely having a source dataset can never create independent landmarks.
    with pytest.raises(ValueError, match='No completed independent human'):
        rebase(original, expanded, root=None)


def test_unreviewed_real_data_cannot_produce_independent_scores(tmp_path):
    original = json.loads(Path('expanded_solar_manifest.json').read_text())
    # Avoid an optional asset download: the reference gate itself is exercised
    # after the same structural checks, with no method or human claims created.
    from unittest.mock import patch
    destination = tmp_path/'must-not-be-written'
    with patch('score_independent.validate_manifest', return_value={}):
        with pytest.raises(ValueError, match='No independent human references'):
            score(original, original, [], destination)
    assert not destination.exists()
