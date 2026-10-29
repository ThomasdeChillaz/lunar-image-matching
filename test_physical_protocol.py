import copy
import json
from pathlib import Path
import pytest
from moon_dataset import validate, geographic_split
from evaluate_moon import disjoint


def manifest():
    return json.loads(Path('moon_manifest.json').read_text())


def test_physical_split_has_wide_geographic_gaps_and_separate_acquisitions():
    data = manifest()
    validate(data)
    assert {s:sum(t['split'] == s for t in data['tiles']) for s in ['train', 'dev', 'test']} == dict(train=48, dev=16, test=16)
    for first in data['tiles']:
        for second in data['tiles']:
            if first['split'] != second['split']:
                assert first['observation_group'] != second['observation_group']
                assert disjoint(first, second)


def test_physical_manifest_rejects_acquisition_and_nonblank_content_leakage():
    data = manifest()
    train = next(t for t in data['tiles'] if t['split'] == 'train')
    test = next(t for t in data['tiles'] if t['split'] == 'test')
    test['observation_group'] = train['observation_group']
    with pytest.raises(ValueError, match='acquisition'):
        validate(data)
    data = manifest()
    train = next(t for t in data['tiles'] if t['split'] == 'train')
    test = next(t for t in data['tiles'] if t['split'] == 'test')
    first = next(i for i in train['images'] if not i['constant_zero_render'])
    second = next(i for i in test['images'] if not i['constant_zero_render'])
    second['sha256'] = first['sha256']
    with pytest.raises(ValueError, match='content'):
        validate(data)


def test_gap_tiles_cannot_be_assigned_to_a_nearby_split():
    assert geographic_split([348.1, -44, 348.2, -43.9]) is None
    data = manifest()
    data['tiles'][0]['bounds_lonlat_deg'] = [348.1, -44, 348.2, -43.9]
    with pytest.raises(ValueError, match='geographic'):
        validate(data)
