"""Freeze geographically and acquisition-separated physical lighting samples."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile
import warnings
import numpy as np
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning
from dataset import sha256, safe_path


def geographic_split(bounds):
    west, south, east, north = bounds
    if east < 348.:
        return 'train'
    if west > 349. and east < 349.5:
        return 'dev'
    if west > 350.:
        return 'test'
    return None


def validate(manifest, root=None):
    sources, hashes, tiles, by_split = {}, {}, set(), defaultdict(list)
    for tile in manifest['tiles']:
        if tile['tile_id'] in tiles or geographic_split(tile['bounds_lonlat_deg']) != tile['split']:
            raise ValueError('Duplicate tile or geographic split mismatch')
        tiles.add(tile['tile_id'])
        source = tile['observation_group']
        if sources.setdefault(source, tile['split']) != tile['split']:
            raise ValueError('Source acquisition leakage')
        by_split[tile['split']].append(tile)
        for image in tile['images']:
            if hashes.setdefault(image['sha256'], tile['split']) != tile['split'] and not image.get('constant_zero_render'):
                raise ValueError('Rendered content leakage')
            if root is not None and sha256(safe_path(root, image['path'])) != image['sha256']:
                raise ValueError('Corrupt physical render')
    if set(by_split) != {'train', 'dev', 'test'}:
        raise ValueError('All geographic splits must be populated')
    # Wide longitude gaps exceed all 128px patch footprints; check actual bounds.
    for a in manifest['tiles']:
        for b in manifest['tiles']:
            if a['split'] == b['split']:
                continue
            x, y = a['bounds_lonlat_deg'], b['bounds_lonlat_deg']
            if min(x[2], y[2]) > max(x[0], y[0]) and min(x[3], y[3]) > max(x[1], y[1]):
                raise ValueError('Geographic footprint leakage')
    return manifest


def prepare(root='.', data='data/moonanything', limits=None):
    root, data = Path(root), Path(data)
    limits = limits or dict(train=48, dev=16, test=16)
    archive_path = root/data/'multilighting_hapke_rendering.zip'
    metadata_path = root/data/'metadata.zip'
    renders, original_metadata = zipfile.ZipFile(archive_path), zipfile.ZipFile(metadata_path)
    candidates, observations = [], defaultdict(set)
    for name in sorted(n for n in renders.namelist() if n.endswith('/lighting_metadata.json')):
        lighting = json.loads(renders.read(name))
        tile_id = lighting['dem']
        metadata_name = 'metadata/metadata_'+tile_id+'.json'
        original = json.loads(original_metadata.read(metadata_name))
        corners = [original[p] for p in ['top_left', 'top_right', 'bottom_left', 'bottom_right']]
        bounds = [min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners)]
        split = geographic_split(bounds)
        if not split:
            continue
        group = re.match(r'M\d+', tile_id)[0]
        observations[group].add(split)
        candidates.append(dict(tile_id=tile_id, observation_group=group, split=split,
            bounds_lonlat_deg=bounds, lighting_member=name, metadata_member=metadata_name,
            lighting=lighting, original_metadata=original))
    candidates = [c for c in candidates if len(observations[c['observation_group']]) == 1]
    selected = []
    for split in ['train', 'dev', 'test']:
        groups = defaultdict(list)
        for candidate in candidates:
            if candidate['split'] == split:
                groups[candidate['observation_group']].append(candidate)
        for group in groups:
            groups[group].sort(key=lambda c:hashlib.sha256(('20261026:'+c['tile_id']).encode()).hexdigest())
        count = 0
        for depth in range(2):
            for group in sorted(groups, key=lambda g:hashlib.sha256(g.encode()).hexdigest()):
                if depth < len(groups[group]) and count < limits[split]:
                    selected.append(groups[group][depth])
                    count += 1
        if count < limits[split]:
            raise ValueError(f'Insufficient acquisition-separated samples for {split}: {count}')
    tiles = []
    for candidate in selected:
        folder = data/'selected'/candidate['split']/candidate['tile_id']
        (root/folder).mkdir(parents=True, exist_ok=True)
        images = []
        for position in candidate['lighting']['positions']:
            name = str(Path(candidate['lighting_member']).parent/position['hapke_filename'])
            content = renders.read(name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NotGeoreferencedWarning)
                with MemoryFile(content) as memory:
                    with memory.open() as source:
                        values = source.read(1)
            constant_zero = bool(np.isfinite(values).all() and np.all(values == 0))
            destination = folder/Path(name).name
            (root/destination).write_bytes(content)
            images.append(dict(path=destination.as_posix(), source_member=name,
                sha256=hashlib.sha256(content).hexdigest(), constant_zero_render=constant_zero, azimuth_deg=position['local_azimuth'],
                elevation_deg=90-position['local_incidence'], acquired_utc=position['start_time']))
        images.sort(key=lambda i: -i['elevation_deg'])
        lighting_bytes = renders.read(candidate['lighting_member'])
        original_bytes = original_metadata.read(candidate['metadata_member'])
        (root/folder/'lighting_metadata.json').write_bytes(lighting_bytes)
        (root/folder/'metadata.json').write_bytes(original_bytes)
        tiles.append({k:candidate[k] for k in ['tile_id', 'observation_group', 'split', 'bounds_lonlat_deg']} | dict(
            images=images, lighting_metadata_sha256=hashlib.sha256(lighting_bytes).hexdigest(),
            original_metadata_sha256=hashlib.sha256(original_bytes).hexdigest()))
    manifest = dict(schema_version=1, dataset='MoonAnything LunarPhoto Hapke physical lighting subset',
        source_doi='10.5281/zenodo.18415119', archive_sha256=sha256(archive_path), metadata_archive_sha256=sha256(metadata_path),
        seed=20261026, native_size=128, limits=limits,
        selection='Fixed longitude bands with wide gaps; retain source observation groups confined to one band; deterministic hash order, at most two patches per acquisition. Frozen before matcher outcomes.',
        split_policy='Train east boundary <348; dev 349<longitude<349.5; test west boundary >350 degrees east. All actual footprints and source-acquisition groups checked. All regions are within Tycho; this is within-region geographic transfer, not south-pole transfer.',
        geometry='Different Sun positions rendered for the same source tile, projection and camera; pixel identity before an additional known 2D warp. Synthetic supervision, never independent real checkpoints.',
        blank_render_policy='Exactly all-zero renders can recur across splits because they carry no terrain information. Retain as explicit evaluation failures, exclude from descriptor training. All nonblank content and source acquisitions remain split-separated.',
        photometry='Fixed uint8 mapping round(255*I/F), clipping [0,255], no per-image contrast normalization unless explicitly labelled CLAHE.',
        license='CC BY 4.0; cite Grethen et al., MoonAnything (2026) and Zenodo record.', tiles=tiles)
    return validate(manifest, root)


def load_image(image, root='.'):
    path = safe_path(root, image['path'])
    if sha256(path) != image['sha256']:
        raise ValueError('Physical render hash mismatch')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with MemoryFile(path.read_bytes()) as memory:
            with memory.open() as source:
                values = source.read(1)
    if values.shape != (128, 128):
        raise ValueError('Unexpected native render size')
    valid = np.isfinite(values)
    byte = np.rint(np.clip(np.where(valid, values, 0), 0, 1)*255).astype(np.uint8)
    return byte, valid, values


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--output', default='moon_manifest.json')
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = prepare(args.root)
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print({s:sum(t['split'] == s for t in manifest['tiles']) for s in ['train', 'dev', 'test']})
