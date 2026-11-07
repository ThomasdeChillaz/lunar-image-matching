import pytest
from uncertainty_groups import geographic_groups,paired_registration_interval


def tile(identity,source,bounds):
    return dict(tile_id=identity,observation_group=source,bounds_lonlat_deg=bounds)


def test_spatial_and_acquisition_dependence_are_joined_transitively():
    tiles = [tile('a','source1',[0,0,1,1]),tile('b','source2',[.5,0,1.5,1]),
        tile('c','source2',[10,0,11,1]),tile('d','source3',[11,0,12,1])]
    lookup,groups,overlaps = geographic_groups(tiles)
    assert lookup['a'] == lookup['b'] == lookup['c'] != lookup['d']
    assert groups == [['a','b','c'],['d']] and overlaps == [['a','b']]


def test_paired_interval_keeps_failed_cases_and_refuses_unpaired_methods():
    manifest = dict(tiles=[tile('a','source1',[0,0,1,1]),tile('b','source2',[.5,0,1.5,1])])
    rows = [dict(pair_id=identity,tile_id=identity,split='test',method=method,dataset_kind='physical_render',
        label_source='exact_geometry',correct=method == 'disk_lightglue_lunar') for identity in ['a','b'] for method in ['disk_lightglue','disk_lightglue_lunar']]
    result = paired_registration_interval(rows,manifest,100)
    assert result['groups'] == 1 and result['source_acquisitions'] == 2
    assert result['delta'] == 1 and result['percentile_interval_95'] == [1,1]
    with pytest.raises(ValueError,match='same known-geometry'):
        paired_registration_interval(rows[:-1],manifest,100)
