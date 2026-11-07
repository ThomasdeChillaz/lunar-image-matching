import copy
import json
from pathlib import Path
import numpy as np
import pytest
from dataset import validate_manifest, safe_path, common_points


def manifest():
    return json.loads(Path('lunar_manifest.json').read_text())


def test_frozen_manifest_has_region_separation():
    m=manifest(); validate_manifest(m)
    assert len(m['pairs']) == 12
    assert len({p['region_id'] for p in m['pairs']}) == 3
    assert all(p['reference_uncertainty_px'] is None for p in m['pairs'])


def test_region_leakage_rejected():
    m=manifest(); m['pairs'][1]['split']='test'
    with pytest.raises(ValueError,match='leakage'): validate_manifest(m)


def test_source_content_leakage_rejected():
    m=manifest(); m['assets'][5]['sha256']=m['assets'][0]['sha256']
    with pytest.raises(ValueError,match='leakage'): validate_manifest(m)


def test_missing_corrupt_assets_rejected(tmp_path):
    with pytest.raises(ValueError,match='Missing/corrupt'): validate_manifest(manifest(),tmp_path)


def test_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError,match='escapes'): safe_path(tmp_path,'../outside')


def test_grid_requires_valid_pixels_in_both_images():
    a=np.ones((128,128),bool); b=a.copy(); b[:,64:]=False
    points=common_points(a,b,spacing=16,margin=16)
    assert len(points)>0 and (points[:,0]<64).all()
    assert len(common_points(a,np.zeros_like(a))) == 0


def test_independent_reference_requires_ten_points():
    m=manifest(); m['pairs'][0]['reference_type']='independent_checkpoints'
    with pytest.raises(ValueError,match='10 independent'): validate_manifest(m)
