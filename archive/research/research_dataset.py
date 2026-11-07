"""Freeze expanded training terrain and two untouched geographic reserves.

Selection uses source metadata only. No image matcher, pixel appearance, success
label or downstream outcome participates. All areas are within Tycho: this is
geographic separation within one crater region, not a south-pole transfer claim.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import warnings
import zipfile

import numpy as np
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning

from dataset import safe_path, sha256


SEED = 'lunar-research-20261029-v1'
BANDS = dict(train=[346.6, 347.85], dev=[349.1, 349.45],
             reserve_a=[348.15, 348.45], reserve_b=[348.6, 348.9])
LIMITS = dict(train=384, dev=64, reserve_a=32, reserve_b=32)
MAX_PER_ACQUISITION = dict(train=12, dev=8, reserve_a=4, reserve_b=4)


def overlap(a, b, buffer=0.):
    return min(a[2], b[2])+buffer > max(a[0], b[0]) and min(a[3], b[3])+buffer > max(a[1], b[1])


def split_for(bounds):
    return next((name for name, (west, east) in BANDS.items() if bounds[0] > west and bounds[2] < east), None)


def previous_evidence(root):
    root = Path(root)
    physical = json.loads((root/'moon_manifest.json').read_text())['tiles']
    real = json.loads((root/'tycho_real_manifest.json').read_text())['assets']
    return physical+real


def inventory(root='.'):
    root = Path(root)
    earlier = previous_evidence(root)
    old_groups = {t['observation_group'] for t in earlier}
    candidates, groups = [], defaultdict(set)
    with zipfile.ZipFile(root/'data/moonanything/multilighting_hapke_rendering.zip') as renders, zipfile.ZipFile(root/'data/moonanything/metadata.zip') as metadata:
        for member in sorted(n for n in renders.namelist() if n.endswith('/lighting_metadata.json')):
            light = json.loads(renders.read(member))
            tile_id = light['dem']
            name = f'metadata/metadata_{tile_id}.json'
            original = json.loads(metadata.read(name))
            corners = [original[k] for k in ['top_left', 'top_right', 'bottom_left', 'bottom_right']]
            bounds = [min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners)]
            split = split_for(bounds)
            if split is None:
                continue
            group = re.match(r'M\d+', tile_id)[0]
            if split.startswith('reserve') and (group in old_groups or any(overlap(bounds, t['bounds_lonlat_deg'], .03) for t in earlier)):
                continue
            groups[group].add(split)
            candidates.append(dict(tile_id=tile_id, observation_group=group, split=split,
                bounds_lonlat_deg=bounds, lighting_member=member, metadata_member=name))
    # Remove acquisitions spanning new splits, before selecting individual patches.
    return [c for c in candidates if len(groups[c['observation_group']]) == 1]


def choose(candidates, limits=None):
    limits = LIMITS if limits is None else limits
    result = []
    digest = lambda name: hashlib.sha256((SEED+':'+name).encode()).hexdigest()
    for split, limit in limits.items():
        groups = defaultdict(list)
        for candidate in candidates:
            if candidate['split'] == split:
                groups[candidate['observation_group']].append(candidate)
        for values in groups.values():
            values.sort(key=lambda c: digest(c['tile_id']))
        selected = []
        for depth in range(MAX_PER_ACQUISITION[split]):
            for group in sorted(groups, key=digest):
                if depth < len(groups[group]) and len(selected) < limit:
                    selected.append(groups[group][depth])
        if len(selected) != limit:
            raise ValueError(f'Insufficient metadata-only samples for {split}: {len(selected)}/{limit}')
        result.extend(selected)
    return result


def validate(manifest, root=None, check_files=False):
    if manifest['bands_lon_deg'] != BANDS or set(manifest['limits']) != set(BANDS):
        raise ValueError('Research split policy differs from frozen geographic bands')
    groups, hashes, seen = {}, {}, set()
    tiles = manifest['tiles']
    if Counter(t['split'] for t in tiles) != Counter(manifest['limits']):
        raise ValueError('Incomplete geographic split counts')
    for tile in tiles:
        split = tile['split']
        if tile['tile_id'] in seen or split_for(tile['bounds_lonlat_deg']) != split:
            raise ValueError('Duplicate tile or geographic split mismatch')
        seen.add(tile['tile_id'])
        if groups.setdefault(tile['observation_group'], split) != split:
            raise ValueError('Acquisition leakage')
        if len(tile['images']) != 9:
            raise ValueError('Missing physical Sun renders')
        for image in tile['images']:
            if hashes.setdefault(image['sha256'], split) != split and not image['constant_zero_render']:
                raise ValueError('Nonblank render content leakage')
            if check_files and sha256(safe_path(root, image['path'])) != image['sha256']:
                raise ValueError('Physical render changed')
    for i, tile in enumerate(tiles):
        for other in tiles[i+1:]:
            if tile['split'] != other['split'] and overlap(tile['bounds_lonlat_deg'], other['bounds_lonlat_deg'], .03):
                raise ValueError('Cross-split geographic footprint/buffer leakage')
    old = manifest['previously_evaluated_terrain']
    old_groups = {t['observation_group'] for t in old}
    for tile in tiles:
        if tile['split'].startswith('reserve'):
            if tile['observation_group'] in old_groups or any(overlap(tile['bounds_lonlat_deg'], t['bounds_lonlat_deg'], .03) for t in old):
                raise ValueError('Fresh reserve overlaps previously evaluated source or buffered footprint')
    return manifest


def prepare(output, root='.'):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    candidates = inventory(root)
    selected = choose(candidates)
    archive_path = root/'data/moonanything/multilighting_hapke_rendering.zip'
    metadata_path = root/'data/moonanything/metadata.zip'
    tiles = []
    with zipfile.ZipFile(archive_path) as renders, zipfile.ZipFile(metadata_path) as metadata:
        for number, candidate in enumerate(selected, 1):
            lighting_bytes = renders.read(candidate['lighting_member'])
            original_bytes = metadata.read(candidate['metadata_member'])
            lighting = json.loads(lighting_bytes)
            folder = Path('data/moonanything/research_v1')/candidate['split']/candidate['tile_id']
            (root/folder).mkdir(parents=True, exist_ok=True)
            images = []
            for position in lighting['positions']:
                member = str(Path(candidate['lighting_member']).parent/position['hapke_filename'])
                content = renders.read(member)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', NotGeoreferencedWarning)
                    with MemoryFile(content) as memory, memory.open() as raster:
                        values = raster.read(1)
                if values.shape != (128, 128):
                    raise ValueError('Unexpected physical render size')
                path = folder/Path(member).name
                (root/path).write_bytes(content)
                images.append(dict(path=str(path), source_member=member, sha256=hashlib.sha256(content).hexdigest(),
                    constant_zero_render=bool(np.isfinite(values).all() and np.all(values == 0)),
                    azimuth_deg=position['local_azimuth'], elevation_deg=90-position['local_incidence'], acquired_utc=position['start_time']))
            images.sort(key=lambda i: -i['elevation_deg'])
            tiles.append(candidate | dict(images=images, lighting_metadata_sha256=hashlib.sha256(lighting_bytes).hexdigest(),
                original_metadata_sha256=hashlib.sha256(original_bytes).hexdigest()))
            print(f'{number}/{len(selected)} {candidate["split"]} {candidate["tile_id"]}', flush=True)
    manifest = dict(schema_version=1, dataset='Expanded MoonAnything LunarPhoto physical research subset',
        source_doi='10.5281/zenodo.18415119', license='CC BY 4.0; Grethen et al. MoonAnything (2026)',
        archive_sha256=sha256(archive_path), metadata_archive_sha256=sha256(metadata_path),
        seed=SEED, native_size=128, bands_lon_deg=BANDS, limits=LIMITS, max_per_acquisition=MAX_PER_ACQUISITION,
        created_utc=datetime.now(timezone.utc).isoformat(), selection_source_sha256=sha256(__file__),
        candidate_counts=dict(Counter(c['split'] for c in candidates)),
        selection='Metadata-only fixed bands; remove acquisitions crossing splits; deterministic acquisition-round-robin hash order. Never rank by pixels or matcher outcomes.',
        geography='Four separated Tycho longitude bands; two uninspected geographic reserves. Within-Tycho transfer only, not a different crater or polar environment.',
        reserve_protocol='Do not use reserve_a/reserve_b for fitting, calibration, candidate choice or model selection. Bind each final evaluation to a preregistered frozen hypothesis/model/policy artifact. An inspected reserve is never fresh again.',
        previous_evidence_manifest_sha256={f: sha256(root/f) for f in ['moon_manifest.json', 'tycho_real_manifest.json']},
        previously_evaluated_terrain=[{k: t[k] for k in ['tile_id', 'observation_group', 'bounds_lonlat_deg']} for t in previous_evidence(root)],
        cross_split_buffer_lonlat_deg=.03, photometry='round(255*I/F), clipped [0,255]; identical to original benchmark.',
        geometry='Co-rendered terrain gives pixel identity before a known artificial similarity transform; never independent real-image ground truth.',
        tiles=tiles)
    validate(manifest, root, check_files=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args()
    if args.inventory_only:
        candidates = inventory()
        print(dict(Counter(c['split'] for c in candidates)))
        print(dict(Counter(c['split'] for c in choose(candidates))))
    else:
        manifest = prepare(args.output)
        print(dict(Counter(t['split'] for t in manifest['tiles'])))
