"""Checksummed lunar dataset, masked common-grid loading, and leakage validation."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest, root=None):
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest schema')
    assets = {a['asset_id']: a for a in manifest['assets']}
    if len(assets) != len(manifest['assets']):
        raise ValueError('Duplicate asset IDs')
    regions, observations, fingerprints, pair_ids = {}, {}, {}, set()
    for pair in manifest['pairs']:
        if pair['pair_id'] in pair_ids:
            raise ValueError('Duplicate pair ID')
        pair_ids.add(pair['pair_id'])
        if pair['split'] not in ('train', 'dev', 'test'):
            raise ValueError('Unknown split')
        if pair['dataset_kind'] != 'real_repeat_observation':
            raise ValueError('This manifest requires repeat observations')
        region, split = pair['region_id'], pair['split']
        if regions.setdefault(region, split) != split:
            raise ValueError('Region leakage across splits')
        for key in ('reference', 'query'):
            if pair[key] not in assets:
                raise ValueError('Pair references unknown asset')
            asset = assets[pair[key]]
            if asset['region_id'] != region:
                raise ValueError('Region mismatch')
            for field, table in [('product_id', observations), ('sha256', fingerprints)]:
                if table.setdefault(asset[field], split) != split:
                    raise ValueError('Source observation or content leakage across splits')
            if len(asset['sha256']) != 64 or not asset['source_url'].startswith('https://'):
                raise ValueError('Missing asset provenance')
        if pair['reference'] == pair['query']:
            raise ValueError('Repeat observations must be distinct assets')
        if pair['reference_type'] not in ('map_grid', 'independent_checkpoints'):
            raise ValueError('Unknown reference type')
        if pair['reference_type'] == 'independent_checkpoints' and len(pair.get('checkpoints', [])) < 10:
            raise ValueError('At least 10 independent checkpoints required')
    if root is not None:
        for asset in assets.values():
            path = safe_path(root, asset['path'])
            if not path.is_file() or sha256(path) != asset['sha256']:
                raise ValueError(f'Missing/corrupt asset: {asset["asset_id"]}; run dataset.py fetch')
    return assets


def safe_path(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset path escapes data root')
    return path


def fetch(manifest, root):
    validate_manifest(manifest)
    for asset in manifest['assets']:
        path = safe_path(root, asset['path']); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha256(path) == asset['sha256']:
            continue
        temporary = path.with_suffix(path.suffix+'.part')
        try:
            with urllib.request.urlopen(asset['source_url'], timeout=120) as response, temporary.open('wb') as f:
                while chunk := response.read(1024*1024): f.write(chunk)
            if sha256(temporary) != asset['sha256']:
                raise ValueError(f'Checksum mismatch: {asset["asset_id"]}')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    validate_manifest(manifest, root)


def load_pair(pair, assets, root, size=512):
    """Reproject each uint8 orthoimage and its native nodata mask to one fixed grid.

    Pixel centers follow GDAL's half-pixel convention. Georeferencing is a reference,
    not independent subpixel truth; image content is not used to align the inputs.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    box = pair['common_bounds_m']; destination = from_bounds(*box, size, size)
    images, masks = [], []
    with rasterio.open(safe_path(root, assets[pair['reference']]['path'])) as ds:
        crs = ds.crs
    for key in ('reference', 'query'):
        with rasterio.open(safe_path(root, assets[pair[key]]['path'])) as ds:
            if ds.crs != crs or ds.count != 1 or ds.dtypes[0] != 'uint8':
                raise ValueError('Expected matching lunar CRS and single-band uint8 rasters')
            data = np.zeros((size, size), np.float32); valid = np.zeros((size, size), np.uint8)
            reproject(ds.read(1), data, src_transform=ds.transform, src_crs=crs,
                      src_nodata=ds.nodata, dst_transform=destination, dst_crs=crs,
                      dst_nodata=0, resampling=Resampling.bilinear)
            reproject(ds.read_masks(1), valid, src_transform=ds.transform, src_crs=crs,
                      dst_transform=destination, dst_crs=crs, resampling=Resampling.nearest)
        mask = cv2.erode((valid>0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        data[~mask] = 0
        images.append(np.clip(data, 0, 255).astype(np.uint8)); masks.append(mask)
    return images[0], images[1], masks[0], masks[1], destination


def common_points(refmask, querymask, spacing=32, margin=32):
    h, w = refmask.shape
    xx, yy = np.meshgrid(np.arange(margin, w-margin, spacing), np.arange(margin, h-margin, spacing))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return points[(refmask & querymask)[points[:, 1], points[:, 0]]].astype(float)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['fetch', 'validate'])
    p.add_argument('--manifest', type=Path, default=Path('lunar_manifest.json'))
    p.add_argument('--root', type=Path, default=Path('.'))
    args = p.parse_args(); data = json.loads(args.manifest.read_text())
    if args.action == 'fetch': fetch(data, args.root)
    validate_manifest(data, args.root)
    print(f'Validated {len(data["assets"])} checksummed assets and {len(data["pairs"])} pairs.')
