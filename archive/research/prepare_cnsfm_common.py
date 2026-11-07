"""Prepare original CNSFM inputs on the exact same 512px baseline grids.

The detector output is a prediction file, never reference measurements. Refuse
to run if a platform's raster resampling produces different image/mask bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from dataset import fetch, load_pair, validate_manifest, sha256
from landmark_review import fingerprint


def crater_text(detections):
    ids = [p['detection_id'] for p in detections]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate crater prediction IDs')
    entries = []
    for p in detections:
        values = [p['x'], p['y'], p['width'], p['height'], p['score'], p['detection_id']]
        if not np.isfinite(values).all() or min(p['width'], p['height']) <= 0 or not 0 <= p['score'] <= 1:
            raise ValueError('Invalid crater prediction')
        entries.append(','.join(format(v, '.9g') for v in values))
    # Prefix identifies the detector window, not the semantic class.
    return '0:'+';'.join(entries)+'\n'


def prepare(manifest, predictions, output, root='.'):
    if predictions['source_manifest_sha256'] != fingerprint(manifest):
        raise ValueError('Detector predictions belong to a different image manifest')
    assets = validate_manifest(manifest, root)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    products, log = {}, []
    for item in predictions['assets']:
        asset = assets[item['product_id']]
        pair = next(p for p in manifest['pairs'] if p['region_id'] == asset['region_id'])
        pair = dict(pair, reference=asset['asset_id'], query=asset['asset_id'])
        image, _, mask, _, grid = load_pair(pair, assets, root)
        image[~mask] = 0
        if (hashlib.sha256(image.tobytes()).hexdigest() != item['image_bytes_sha256']
                or hashlib.sha256(mask.tobytes()).hexdigest() != item['mask_bytes_sha256']):
            raise ValueError('Common-grid bytes differ from detector input; stop before comparing methods')
        folder = output/'Dataset'/asset['region_id']
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/(asset['product_id']+'.tif')
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path, 'w', driver='GTiff', width=512, height=512, count=1, dtype='uint8',
                crs=asset['crs_wkt'], transform=grid, nodata=255) as target:
                target.write(image, 1)
                target.write_mask(mask.astype(np.uint8)*255)
        path.with_suffix('.craters').write_text(crater_text(item['detections']))
        products.setdefault(asset['region_id'], []).append(asset['product_id'])
        log.append(dict(product_id=asset['product_id'], image_bytes_sha256=item['image_bytes_sha256'],
            mask_bytes_sha256=item['mask_bytes_sha256'], tif_sha256=sha256(path),
            craters_sha256=sha256(path.with_suffix('.craters')), detections=len(item['detections'])))
    if set(p['product_id'] for p in log) != set(assets):
        raise ValueError('Detector output must cover all frozen image assets')
    for region, names in products.items():
        (output/'Dataset'/region/'list_images.txt').write_text('\n'.join(name+'.tif' for name in names)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(source_manifest_sha256=fingerprint(manifest),
        predictions_sha256=fingerprint(predictions), variant='common_512_redetected', assets=log), indent=2))
    return log


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='expanded_solar_manifest.json')
    p.add_argument('--predictions', default='cnsfm_detector_outputs.json')
    p.add_argument('--output', required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--fetch', action='store_true')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if args.fetch:
        fetch(manifest, args.root)
    rows = prepare(manifest, json.loads(Path(args.predictions).read_text()), args.output, args.root)
    print('Verified and prepared', len(rows), 'common-grid images')
