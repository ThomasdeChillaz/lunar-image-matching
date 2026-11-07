"""Move human-picked landmarks between verified coordinate grids, never fit them.

The image bytes and coordinate reference system must be identical. The known
map-grid affine supplies the conversion, not a matcher-estimated registration.
Original reviewer provenance and an extra interpolation uncertainty are retained.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from rasterio.transform import from_bounds
from dataset import load_pair, validate_manifest
from landmark_review import fingerprint
from cnsfm_outputs import native_to_common


def conversion(points, source_bounds, target_bounds, uncertainty):
    old = from_bounds(*source_bounds, 512, 512)
    new = from_bounds(*target_bounds, 512, 512)
    result = native_to_common(points, old, new)
    scale = max(abs(old.a/new.a), abs(old.e/new.e))
    # Conservative half-pixel extra picking allowance for a differently sampled
    # grid. This is an explicit bound convention, not an estimated noise model.
    return result, float(uncertainty*scale+.5)


def rebase(reviewed, target, root='.'):
    old_assets, new_assets = validate_manifest(reviewed, root), validate_manifest(target, root)
    result = copy.deepcopy(target)
    new_pairs = {frozenset([p['reference'], p['query']]):p for p in result['pairs']}
    imported = 0
    for source in reviewed['pairs']:
        if source.get('reference_validation_status') != 'reviewed' or source['reference_type'] != 'independent_checkpoints':
            continue
        provenance = source.get('review_provenance', {})
        if not provenance.get('primary_reviewer') or not provenance.get('primary_review_sha256'):
            raise ValueError('Source-bound manual review provenance required')
        pair = new_pairs.get(frozenset([source['reference'], source['query']]))
        if pair is None or pair['split'] != source['split'] or pair['region_id'] != source['region_id']:
            raise ValueError('Reviewed pair absent or split changed in expanded dataset')
        for side in ['reference', 'query']:
            key = source[side]
            if old_assets[key]['sha256'] != new_assets[key]['sha256'] or old_assets[key]['crs_wkt'] != new_assets[key]['crs_wkt']:
                raise ValueError('Landmark transfer requires identical source image bytes and CRS')
        reverse = source['reference'] != pair['reference']
        points = []
        for point in source['checkpoints']:
            converted = dict(landmark_id=point['landmark_id'])
            for side in ['reference', 'query']:
                destination = ('query' if side == 'reference' else 'reference') if reverse else side
                xy, uncertainty = conversion([point[side]], source['common_bounds_m'], pair['common_bounds_m'], point[side+'_uncertainty_px'])
                if (xy < 0).any() or (xy > 511).any():
                    raise ValueError('Reviewed point falls outside the expanded grid; explicit re-review required')
                converted[destination] = xy[0].tolist()
                converted[destination+'_uncertainty_px'] = uncertainty
            points.append(converted)
        old_grid = from_bounds(*source['common_bounds_m'], 512, 512)
        new_grid = from_bounds(*pair['common_bounds_m'], 512, 512)
        scale = max(abs(old_grid.a/new_grid.a), abs(old_grid.e/new_grid.e))
        # Keep the original aggregate bound, which may include disagreement
        # from a second reviewer beyond the primary per-point picking bounds.
        bound = max(max(np.hypot(p['reference_uncertainty_px'], p['query_uncertainty_px']) for p in points),
            source['reference_uncertainty_px']*scale+np.sqrt(.5))
        pair.update(checkpoints=points, reference_type='independent_checkpoints', reference_validation_status='reviewed',
            audit_status='independent_manual_checkpoints_imported',
            reference_uncertainty_px=float(bound),
            review_provenance=provenance | dict(source_reviewed_manifest_sha256=fingerprint(reviewed), source_pair_id=source['pair_id'],
                target_source_manifest_sha256=fingerprint(target), grid_conversion='Known GDAL pixel-center affine; no matcher predictions used',
                added_per_image_uncertainty_px=.5, coordinate_order_reversed=reverse))
        _, _, ma, mb, _ = load_pair(pair, new_assets, root)
        for point in points:
            for side, mask in [('reference', ma), ('query', mb)]:
                x, y = np.rint(point[side]).astype(int)
                if not mask[y, x]:
                    raise ValueError('Converted landmark falls in invalid source data')
        imported += 1
    if not imported:
        raise ValueError('No completed independent human reviews to transfer')
    validate_manifest(result, root)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reviewed_manifest')
    p.add_argument('--target', default='expanded_solar_manifest.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = rebase(json.loads(Path(args.reviewed_manifest).read_text()), json.loads(Path(args.target).read_text()))
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
