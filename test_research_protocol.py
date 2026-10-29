import copy
import numpy as np
import pytest

from geometry import transform
from research_dataset import BANDS, overlap, split_for, validate
from research_geometry import FAMILIES, motion, plan, prepare


def tiny_manifest():
    tiles = []
    for i, (split, band) in enumerate(BANDS.items()):
        tiles.append(dict(tile_id=f'tile{i}', observation_group=f'acq{i}', split=split,
            bounds_lonlat_deg=[band[0]+.05, -43.5, band[0]+.08, -43.48],
            images=[dict(sha256=f'{i}-{j}', constant_zero_render=False) for j in range(9)]))
    return dict(bands_lon_deg=BANDS, limits={s: 1 for s in BANDS}, tiles=tiles, previously_evaluated_terrain=[])


def test_fresh_reserves_reject_earlier_acquisitions_and_buffered_terrain():
    manifest = tiny_manifest()
    validate(manifest)
    target = manifest['tiles'][2]
    changed = copy.deepcopy(manifest)
    changed['previously_evaluated_terrain'] = [dict(observation_group=target['observation_group'], bounds_lonlat_deg=[1, 1, 2, 2])]
    with pytest.raises(ValueError, match='previously evaluated'):
        validate(changed)
    changed['previously_evaluated_terrain'][0]['observation_group'] = 'other'
    changed['previously_evaluated_terrain'][0]['bounds_lonlat_deg'] = target['bounds_lonlat_deg']
    with pytest.raises(ValueError, match='previously evaluated'):
        validate(changed)


def test_research_splits_reject_content_and_acquisition_leakage():
    manifest = tiny_manifest()
    manifest['tiles'][1]['observation_group'] = manifest['tiles'][0]['observation_group']
    with pytest.raises(ValueError, match='Acquisition'):
        validate(manifest)
    manifest = tiny_manifest()
    manifest['tiles'][1]['images'][0]['sha256'] = manifest['tiles'][0]['images'][0]['sha256']
    with pytest.raises(ValueError, match='content leakage'):
        validate(manifest)


def test_geometry_families_are_separate_seeded_and_score_only_valid_support():
    image = np.arange(128*128, dtype=np.uint8).reshape(128, 128)
    mask = np.ones_like(image, bool)
    mask[30:40, 50:60] = False
    for family in FAMILIES:
        matrix, params = motion('held-out-example', family)
        again, _ = motion('held-out-example', family)
        assert np.array_equal(matrix, again)
        if family == 'translation':
            assert np.allclose(matrix[:, :2], np.eye(2))
        if family in ('rotation', 'scale'):
            assert params['dx'] == params['dy'] == 0
        if family in ('rotation', 'translation'):
            assert params['scale'] == 1
        if family in ('scale', 'translation'):
            assert params['angle_deg'] == 0
        query, valid, truth, points, targets, _ = prepare(image, image, mask, mask, 'held-out-example', family)
        assert np.allclose(transform(points, truth), targets)
        assert valid[np.rint(targets[:, 1]).astype(int), np.rint(targets[:, 0]).astype(int)].all()
        assert mask[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)].all()
        assert (query[~valid] == 0).all()


def test_geometry_plan_covers_each_light_pair_and_motion_family_once():
    cases = plan(tiny_manifest(), 'dev')
    assert len(cases) == 36*4
    assert len({c['pair_id'] for c in cases}) == len(cases)
    assert {c['geometry_family'] for c in cases} == set(FAMILIES)
    assert all(c['split'] == 'dev' for c in cases)
