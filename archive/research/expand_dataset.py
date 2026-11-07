"""Freeze all 321 pairs from the pinned CNSFM release, preserving the pilot."""
import argparse
import itertools
import json
import shutil
from pathlib import Path
import rasterio
from openpyxl import load_workbook
from dataset import sha256, validate_manifest

REVISION = '94cebaaa7fd3e070f8040620975554c3067651ef'


def build(source, root):
    source, root = Path(source), Path(root)
    assets, pairs, upstream = [], [], []
    for region, split in [('S1', 'train'), ('S2', 'dev'), ('S3', 'test')]:
        folder = source/'Dataset'/region
        sheet = folder/'Image_illumination_angles.xlsx'
        rows = [r for r in list(load_workbook(sheet, data_only=True).active.values)[1:] if r[0]]
        bounds = []
        for row in rows:
            product = row[0]
            origin = folder/(product+'.tif')
            path = Path('data/raw')/region/origin.name
            destination = root/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(origin, destination)
            if sha256(destination) != sha256(origin):
                raise ValueError('Cached raster differs from pinned source')
            with rasterio.open(origin) as ds:
                bounds.append(ds.bounds)
                assets.append(dict(asset_id=product, product_id=product, region_id=region,
                    path=path.as_posix(), sha256=sha256(origin),
                    source_url=f'https://raw.githubusercontent.com/Bin501/CNSFM/{REVISION}/Dataset/{region}/{origin.name}',
                    acquired_utc=row[1], pixel_scale_m=abs(ds.transform.a), crs_wkt=ds.crs.to_wkt(),
                    bounds_m=list(ds.bounds), shape=list(ds.shape), nodata=ds.nodata,
                    incidence_deg=row[4], emission_deg=row[5], phase_deg=row[6],
                    north_azimuth_deg=row[3], solar_azimuth_deg=None,
                    source_processing='Published map-projected crop; co-registration uncertainty unknown',
                    use_conditions='Upstream research release; no dataset-specific license. Download originals.'))
            crater = folder/(product+'.craters')
            upstream.append(dict(path=crater.relative_to(source).as_posix(), sha256=sha256(crater)))
        # One content-independent map grid per scene keeps all acquisition pairs comparable.
        box = [max(b.left for b in bounds), max(b.bottom for b in bounds),
               min(b.right for b in bounds), min(b.top for b in bounds)]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('No scene-wide geographic intersection')
        for first, second in itertools.combinations([r[0] for r in rows], 2):
            pairs.append(dict(pair_id=f'{region}_{first}_{second}', region_id=region, split=split,
                dataset_kind='real_repeat_observation', reference=first, query=second,
                common_bounds_m=box, reference_type='map_grid', reference_uncertainty_px=None,
                checkpoints=[], audit_status='not_independently_annotated'))
        upstream.append(dict(path=sheet.relative_to(source).as_posix(), sha256=sha256(sheet)))
    manifest = dict(schema_version=1, dataset='CNSFM complete published 321-pair evaluation',
        source_revision=REVISION, selection='All 42 source images and all unordered pairs within each scene; no outcome-based selection.',
        split_policy='S1 train/reserved; S2 development; S3 test. Sites/source observations never cross splits. S3 was inspected in the pilot and is no longer a pristine confirmatory test for new choices.',
        reference_policy='Map reference only until independent landmarks are imported. Not a reproduction of published success percentages.',
        upstream_inputs=upstream, assets=assets, pairs=pairs)
    validate_manifest(manifest, root)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Checkout of pinned Bin501/CNSFM revision')
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='expanded_manifest.json')
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    manifest = build(args.source, args.root)
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f'{len(manifest["assets"])} images; {len(manifest["pairs"])} pairs: {target}')
