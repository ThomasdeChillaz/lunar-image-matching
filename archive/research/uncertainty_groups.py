"""Conservative uncertainty groups join shared acquisitions and overlapping terrain."""
from collections import defaultdict
import numpy as np


def geographic_groups(tiles):
    tiles = sorted(tiles,key=lambda t:t['tile_id'])
    if len({t['tile_id'] for t in tiles}) != len(tiles):
        raise ValueError('Duplicate uncertainty-group tile')
    parent = list(range(len(tiles)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    overlaps = []
    for index,tile in enumerate(tiles):
        bounds = tile['bounds_lonlat_deg']
        if not np.isfinite(bounds).all() or not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
            raise ValueError('Invalid declared tile bounds')
        for other in range(index):
            previous = tiles[other]
            box = previous['bounds_lonlat_deg']
            overlap = min(bounds[2],box[2]) > max(bounds[0],box[0]) and min(bounds[3],box[3]) > max(bounds[1],box[1])
            if overlap:
                overlaps.append([previous['tile_id'],tile['tile_id']])
            if overlap or tile['observation_group'] == previous['observation_group']:
                parent[find(index)] = find(other)
    members = defaultdict(list)
    for index,tile in enumerate(tiles):
        members[find(index)].append(tile['tile_id'])
    groups = sorted(members.values(),key=lambda g:g[0])
    return {tile:index for index,group in enumerate(groups) for tile in group},groups,overlaps


def paired_registration_interval(rows,manifest,repetitions=5000):
    selected = [r for r in rows if r['split'] == 'test' and r['method'] in ['disk_lightglue','disk_lightglue_lunar']
        and r['dataset_kind'] in ['physical_render','physical_render_fixed_translation']]
    identities = [(r['pair_id'],r['method']) for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated paired outcome')
    by_method = {method:{r['pair_id'] for r in selected if r['method'] == method} for method in ['disk_lightglue','disk_lightglue_lunar']}
    if not selected or by_method['disk_lightglue'] != by_method['disk_lightglue_lunar']:
        raise ValueError('Both methods must cover the same known-geometry cases')
    tile_ids = {r['tile_id'] for r in selected}
    tiles = [t for t in manifest['tiles'] if t['tile_id'] in tile_ids]
    if len(tiles) != len(tile_ids):
        raise ValueError('Unknown uncertainty-group tile')
    lookup,members,overlaps = geographic_groups(tiles)
    values = np.zeros((len(members),3),int)
    for row in selected:
        if row.get('label_source') != 'exact_geometry' or type(row['correct']) is not bool:
            raise ValueError('Known-geometry correctness required for physical uncertainty')
        group = values[lookup[row['tile_id']]]
        group[0 if row['method'] == 'disk_lightglue' else 1] += int(row['correct'])
        if row['method'] == 'disk_lightglue':
            group[2] += 1
    samples = values[np.random.default_rng(20261029).integers(0,len(values),size=(repetitions,len(values)))].sum(axis=1)
    deltas = (samples[:,1]-samples[:,0])/samples[:,2]
    return dict(groups=len(members),source_acquisitions=len({t['observation_group'] for t in tiles}),tiles=len(tiles),
        grouping='Connected components of overlapping declared footprints OR shared source acquisition; repeated lighting cases remain in one component.',
        group_members=members,overlapping_footprint_pairs=overlaps,
        delta=float((values[:,1].sum()-values[:,0].sum())/values[:,2].sum()),
        percentile_interval_95=np.percentile(deltas,[2.5,97.5]).tolist(),bootstrap_repetitions=repetitions,seed=20261029,
        warning='Exploratory small-sample interval within previously inspected Tycho terrain; disconnected footprints need not be globally independent. Not evidence of global or real-image superiority.')
