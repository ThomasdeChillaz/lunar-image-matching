"""Freeze new real Tycho repeat pairs using metadata, never matcher outcomes.

LunarPhoto's render/ folder contains orthorectified real LRO appearance crops
(MoonAnything section 3.3). They are distinct from the Hapke renders. Geographic
map alignment remains provisional and cannot supply independent checkpoints.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import warnings
import zipfile
import cv2
import numpy as np
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning
from dataset import sha256, safe_path
from landmark_review import fingerprint
from solar_metadata import local_sun

REGIONS = [
    ('Tycho_dev_north', 'dev', [349.1, -42.4, 349.4, -42.1]),
    ('Tycho_dev_south', 'dev', [349.1, -44.5, 349.4, -44.2]),
    ('Tycho_test_north', 'test', [350.1, -42.4, 350.4, -42.1]),
    ('Tycho_test_mid', 'test', [350.1, -43.4, 350.4, -43.1]),
    ('Tycho_test_south', 'test', [350.1, -44.5, 350.4, -44.2])]


def intersection(a, b):
    result = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return result if result[2] > result[0] and result[3] > result[1] else None


def area(bounds):
    return (bounds[2]-bounds[0])*(bounds[3]-bounds[1])


def sun_separation(a, b):
    az0, el0, az1, el1 = np.deg2rad([a['solar_azimuth_deg'], a['solar_elevation_deg'], b['solar_azimuth_deg'], b['solar_elevation_deg']])
    return float(np.rad2deg(np.arccos(np.clip(np.sin(el0)*np.sin(el1)+np.cos(el0)*np.cos(el1)*np.cos(az0-az1), -1, 1))))


def validate(manifest, physical, root=None):
    if manifest['physical_manifest_sha256'] != fingerprint(physical):
        raise ValueError('Physical training provenance changed')
    forbidden = {t['observation_group'] for t in physical['tiles']}
    assets = {a['tile_id']:a for a in manifest['assets']}
    groups, hashes = {}, {}
    regions = {r[0]:(r[1], r[2]) for r in REGIONS}
    seen = set()
    for pair in manifest['pairs']:
        if pair['pair_id'] in seen:
            raise ValueError('Duplicate real Tycho pair')
        seen.add(pair['pair_id'])
        split, region = regions[pair['region_id']]
        if pair['split'] != split or pair['reference_type'] != 'provisional_map':
            raise ValueError('Real Tycho split/reference mismatch')
        aa, bb = [assets[pair[key]] for key in ['reference', 'query']]
        if aa['observation_group'] == bb['observation_group']:
            raise ValueError('Repeat observations require different source acquisitions')
        expected = intersection(aa['bounds_lonlat_deg'], bb['bounds_lonlat_deg'])
        if expected is None or not np.allclose(pair['bounds_lonlat_deg'], expected, atol=1e-12, rtol=0):
            raise ValueError('Incorrect shared real Tycho footprint')
        for asset in [aa, bb]:
            bounds = asset['bounds_lonlat_deg']
            group = asset['observation_group']
            if (group in forbidden or groups.setdefault(group, split) != split
                    or hashes.setdefault(asset['sha256'], split) != split):
                raise ValueError('Real/physical acquisition or content leakage')
            if not region[0] < bounds[0] < bounds[2] < region[2] or not region[1] < bounds[1] < bounds[3] < region[3]:
                raise ValueError('Real Tycho footprint crosses the frozen geographic box')
            for tile in physical['tiles']:
                if tile['split'] in ('train', 'dev') and intersection(bounds, tile['bounds_lonlat_deg']):
                    raise ValueError('Real evaluation terrain overlaps physical train/dev terrain')
            if root is not None and sha256(safe_path(root, asset['path'])) != asset['sha256']:
                raise ValueError('Real appearance hash mismatch')
    return assets


def prepare(physical, data='data/moonanything'):
    data = Path(data)
    archives = {name:data/(name+'.zip') for name in ['metadata', 'render']}
    for path in archives.values():
        evidence = json.loads(path.with_suffix('.zip.provenance.json').read_text())
        if sha256(path) != evidence['sha256']:
            raise ValueError('Real source archive failed its saved hash')
    forbidden = {t['observation_group'] for t in physical['tiles']}
    by_region, group_splits = defaultdict(list), defaultdict(set)
    with zipfile.ZipFile(archives['metadata']) as archive:
        for member in sorted(n for n in archive.namelist() if n.endswith('.json')):
            raw = archive.read(member)
            metadata = json.loads(raw)
            tile = Path(member).stem.removeprefix('metadata_')
            group = re.match(r'M\d+', tile)[0]
            if group in forbidden or metadata['rot90'] != 0:
                continue
            bounds = [metadata['top_left'][0], metadata['bottom_left'][1], metadata['top_right'][0], metadata['top_left'][1]]
            # Keep the new real reference set clear of all terrain used for
            # adapter fitting or checkpoint selection, not just source IDs.
            if any(t['split'] in ('train', 'dev') and intersection(bounds, t['bounds_lonlat_deg']) for t in physical['tiles']):
                continue
            for name, split, region in REGIONS:
                if region[0] < bounds[0] < bounds[2] < region[2] and region[1] < bounds[1] < bounds[3] < region[3]:
                    pds = metadata['pds_metadata']
                    az, el = local_sun((bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2, pds['sub_solar_lat'], pds['sub_solar_lon'])
                    by_region[name].append(dict(tile_id=tile, observation_group=group, bounds_lonlat_deg=bounds,
                        metadata_member=member, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                        product_id=pds['name'], acquired_utc=pds['start_time'], solar_azimuth_deg=az, solar_elevation_deg=el,
                        source_subsolar_latitude_deg=pds['sub_solar_lat'], source_subsolar_longitude_deg=pds['sub_solar_lon'], rot90=0))
                    group_splits[group].add(split)
    assets, pairs = {}, []
    for name, split, _ in REGIONS:
        candidates = [r for r in by_region[name] if len(group_splits[r['observation_group']]) == 1]
        bins = defaultdict(list)
        for a, b in itertools.combinations(candidates, 2):
            shared = intersection(a['bounds_lonlat_deg'], b['bounds_lonlat_deg'])
            if a['observation_group'] == b['observation_group'] or not shared:
                continue
            overlap = area(shared)/min(area(a['bounds_lonlat_deg']), area(b['bounds_lonlat_deg']))
            if overlap < .6:
                continue
            solar = sun_separation(a, b)
            stratum = min(2, int(solar//30))
            identity = a['tile_id']+'_'+b['tile_id']
            order = hashlib.sha256(('20261029:'+identity).encode()).hexdigest()
            bins[stratum].append((order, a, b, shared, solar, overlap))
        used, footprints, selected = set(), [], []
        for stratum in range(3):
            count = 0
            for _, a, b, bounds, solar, overlap in sorted(bins[stratum], key=lambda p:p[0]):
                if count == 2:
                    break
                if any(t['tile_id'] in used for t in [a, b]) or any(intersection(bounds, old) for old in footprints):
                    continue
                selected.append(dict(pair_id=f'{name}_{len(selected)+1:02}', region_id=name, split=split,
                    reference=a['tile_id'], query=b['tile_id'], bounds_lonlat_deg=bounds,
                    reference_type='provisional_map', reference_uncertainty_px=None,
                    solar_separation_deg=solar, overlap_fraction_smaller_source=overlap, solar_stratum=stratum))
                used.update([a['tile_id'], b['tile_id']]); footprints.append(bounds)
                assets.update({t['tile_id']:t for t in [a, b]}); count += 1
        if not selected:
            raise ValueError('No metadata-selected repeat pairs in '+name)
        pairs += selected
    with zipfile.ZipFile(archives['render']) as archive:
        for tile, asset in assets.items():
            member = 'render/render_'+tile+'.tif'
            raw = archive.read(member)
            destination = data/'real_selected'/(tile+'.tif')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError('Refuse to overwrite different selected real image')
            destination.write_bytes(raw)
            asset.update(path=destination.as_posix(), source_member=member, sha256=hashlib.sha256(raw).hexdigest())
    manifest = dict(schema_version=1, dataset='Real LRO orthorectified LunarPhoto repeat-observation subset',
        source_doi='10.5281/zenodo.18415119', license='CC BY 4.0', physical_manifest_sha256=fingerprint(physical),
        archive_sha256={name:sha256(path) for name, path in archives.items()}, seed=20261029, image_size=128,
        regions=REGIONS, selection='Fixed five geographic boxes; metadata overlap >=60%; two deterministic hash-ordered pairs per available Sun-separation stratum (0-30,30-60,60+ degrees). No image/matcher outcomes used. Nonoverlapping selected shared footprints within each box.',
        independence='No physical train/dev/test source acquisitions; no physical train/dev terrain; source groups and content disjoint between real dev/test. All new real areas are within Tycho. Existing physical test terrain may overlap; this real transfer evaluation is exploratory.',
        geometry='North-up orthorectified 128px appearance crops (rot90=0), remapped from metadata corner bounds to a 128px shared intersection. Metadata-to-pixel alignment is provisional; no independent real correctness labels.',
        photometry='Fixed round(255*I/F) clipped [0,255]; real observed appearance, distinct from Hapke render archive.',
        assets=list(assets.values()), pairs=pairs)
    validate(manifest, physical, '.')
    return manifest


def load_pair(pair, assets, root='.'):
    left, bottom, right, top = pair['bounds_lonlat_deg']
    lon, lat = np.meshgrid(left+(np.arange(128)+.5)*(right-left)/128,
        top-(np.arange(128)+.5)*(top-bottom)/128)
    arrays, masks = [], []
    for key in ['reference', 'query']:
        asset = assets[pair[key]]
        path = safe_path(root, asset['path'])
        if sha256(path) != asset['sha256']:
            raise ValueError('Selected real appearance hash mismatch')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NotGeoreferencedWarning)
            with MemoryFile(path.read_bytes()) as memory:
                with memory.open() as source:
                    values = source.read(1)
        if values.shape != (128, 128):
            raise ValueError('Unexpected real source raster dimensions')
        west, south, east, north = asset['bounds_lonlat_deg']
        xx = ((lon-west)/(east-west)*128-.5).astype(np.float32)
        yy = ((north-lat)/(north-south)*128-.5).astype(np.float32)
        # Bilinear interpolation needs all four source neighbors to be valid.
        valid = np.isfinite(values)
        byte = np.rint(np.clip(np.where(valid, values, 0), 0, 1)*255).astype(np.uint8)
        image = cv2.remap(byte, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        support = cv2.remap(valid.astype(np.float32), xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= 1-1e-7
        image[~support] = 0
        arrays.append(image); masks.append(support)
    return *arrays, *masks


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-manifest', default='moon_manifest.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = prepare(json.loads(Path(args.physical_manifest).read_text()))
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print('Frozen', len(manifest['assets']), 'real images and', len(manifest['pairs']), 'repeat pairs')
