"""Audit original CNSFM output coordinates against detector files, then score.

Crater matches are algorithm predictions, never independently verified targets.
Do not interpret the executable's console 'right matches' text as ground truth.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from confidence import diagnostics
from dataset import common_points, load_pair, validate_manifest, sha256
from evaluate import endpoint_statistics
from matchers import Match, fit_geometry


def read_craters(text):
    points = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        prefix, entries = line.split(':', 1)
        window = int(prefix)
        for index, entry in enumerate(e for e in entries.split(';') if e.strip()):
            values = np.array([float(x) for x in entry.split(',')])
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError('Invalid crater file')
            # Verified against all three author-binary probe outputs: CNSFM
            # assigns window-local array indices, ignoring the detector's ID.
            identity = window*10000+index
            if identity in points:
                raise ValueError('Repeated crater identity')
            points[identity] = values[:2]
    return points


def read_matches(text, reference_craters, query_craters):
    if not text.strip():
        return np.empty((0, 2)), np.empty((0, 2))
    rows = np.loadtxt(io.StringIO(text), ndmin=2)
    if rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError('Expected eight finite columns in original CNSFM .match output')
    for row in rows:
        for xy, identifier, table in [(row[:2], row[4], reference_craters), (row[2:4], row[6], query_craters)]:
            if identifier != int(identifier) or int(identifier) not in table or not np.allclose(xy, table[int(identifier)], atol=.001, rtol=0):
                raise ValueError('Output coordinates/IDs do not correspond to the supplied crater predictions')
    return rows[:, :2], rows[:, 2:4]


def native_to_common(points, native_transform, common_transform):
    # GDAL affine origins refer to pixel corners; keypoints use pixel centers.
    points = np.asarray(points, float)
    if not len(points):
        return points.reshape(0, 2)
    world = np.array([native_transform*(float(x)+.5, float(y)+.5) for x, y in points])
    return np.array([(~common_transform)*(float(x), float(y)) for x, y in world])-.5


def load_prediction(folder, source_folder, common_transform, masks, variant):
    folder, source_folder = Path(folder), Path(source_folder)
    provenance = json.loads((folder/'provenance.json').read_text())
    if provenance['status'] != 'executed' or provenance['returncode'] != 0:
        raise ValueError('Original executable did not complete successfully')
    if (provenance.get('source_revision') != '94cebaaa7fd3e070f8040620975554c3067651ef'
            or provenance.get('binaries_sha256', {}).get('CNSFM.exe') != 'bdfbaa00ee549c81349aae0c7b06d37e73ae91a0dd6d4eaa7472a27512cfc0a0'
            or provenance.get('input_variant', 'authors_native') != variant):
        raise ValueError('Original engine provenance or input variant mismatch')
    products = provenance.get('products')
    if products is None:
        products = [Path(line.strip().replace('\\', '/')).stem for line in (folder/'list_images.txt').read_text().splitlines() if line.strip()]
    if len(products) != 2:
        raise ValueError('This importer expects per-pair original outputs')
    recorded = {Path(key.replace('\\', '/')).name:value for key, value in provenance['input_sha256'].items()}
    common = provenance.get('common_grid_provenance', {})
    grid_assets = {a['product_id']:a for a in common.get('assets', [])}
    for product in products:
        for suffix in ['.tif', '.craters']:
            file = source_folder/(product+suffix)
            data = file.read_bytes()
            if variant == 'common_512_redetected' and suffix == '.tif':
                import hashlib
                item = grid_assets.get(product)
                if not item or recorded.get(file.name) != item['tif_sha256']:
                    raise ValueError('Common-grid input audit trail missing')
                # TIFF container bytes can differ by platform; the exact pixel
                # and valid-mask arrays must agree with the preparation audit.
                with rasterio.open(file) as image:
                    pixel_hash = hashlib.sha256(image.read(1).tobytes()).hexdigest()
                    mask_hash = hashlib.sha256((image.dataset_mask() > 0).tobytes()).hexdigest()
                if pixel_hash != item['image_bytes_sha256'] or mask_hash != item['mask_bytes_sha256']:
                    raise ValueError('Common-grid pixel/mask mismatch')
                continue
            acceptable = {sha256(file)}
            if suffix == '.craters':
                # Git's Windows checkout may have converted LF to CRLF.
                import hashlib
                lf = data.replace(b'\r\n', b'\n')
                acceptable |= {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()}
            if recorded.get(file.name) not in acceptable:
                raise ValueError('Original input hash mismatch')
    match_path = folder/'original_outputs/0_1.match'
    if not match_path.exists():
        return Match(reason='original_no_match_output'), products
    record = provenance['outputs'].get('original_outputs/0_1.match')
    if not record or sha256(match_path) != record['sha256']:
        raise ValueError('Original output hash mismatch')
    tables = [read_craters((source_folder/(p+'.craters')).read_text()) for p in products]
    first, second = read_matches(match_path.read_text(), *tables)
    if variant == 'authors_native':
        converted = []
        for points, product in zip([first, second], products):
            with rasterio.open(source_folder/(product+'.tif')) as image:
                converted.append(native_to_common(points, image.transform, common_transform))
        first, second = converted
    elif variant != 'common_512_redetected':
        raise ValueError('Unknown input variant')
    keep = np.ones(len(first), bool)
    for points, mask in zip([first, second], masks):
        coordinates = np.rint(points).astype(int)
        valid = (coordinates >= 0).all(axis=1) & (coordinates < 512).all(axis=1)
        valid &= mask[coordinates[:, 1].clip(0, 511), coordinates[:, 0].clip(0, 511)]
        keep &= valid
    first, second = first[keep], second[keep]
    # Apply the one-to-one destination convention used by the SIFT baseline.
    seen_a, seen_b, selected = set(), set(), []
    for index, (a, b) in enumerate(zip(first, second)):
        if tuple(a) not in seen_a and tuple(b) not in seen_b:
            selected.append(index)
            seen_a.add(tuple(a)); seen_b.add(tuple(b))
    return fit_geometry(first[selected], second[selected]), products


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', required=True)
    p.add_argument('--dataset-folder', required=True)
    p.add_argument('--manifest', default='expanded_solar_manifest.json')
    p.add_argument('--variant', choices=['authors_native', 'common_512_redetected'], required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    provenance = json.loads((Path(args.execution)/'provenance.json').read_text())
    products = provenance.get('products') or [Path(line.strip().replace('\\', '/')).stem for line in (Path(args.execution)/'list_images.txt').read_text().splitlines() if line.strip()]
    pair = next(p for p in manifest['pairs'] if p['reference'] == products[0] and p['query'] == products[1])
    a, b, ma, mb, grid = load_pair(pair, validate_manifest(manifest, '.'), '.')
    result, products = load_prediction(args.execution, args.dataset_folder, grid, (ma, mb), args.variant)
    points = common_points(ma, mb)
    error, rmse = endpoint_statistics(result.matrix, points, points)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(dict(pair_id=pair['pair_id'], method='cnsfm', variant=args.variant,
        estimated=result.matrix is not None, support=result.support, map_disagreement_px=error, rmse_px=rmse,
        warning='Algorithm predictions scored against provisional map coordinates. Not independently verified registration accuracy.'), indent=2))
