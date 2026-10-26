"""Import source-bound manual reviews with per-point uncertainty and audit trail.

Reviewer attestations document provenance; software cannot verify a reviewer's
identity or expertise. An optional second review quantifies picking disagreement.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from confidence import coverage
from dataset import validate_manifest, load_pair
from landmark_review import fingerprint


def validate_review(manifest, review):
    if review.get('schema_version') != 2 or review.get('source_manifest_sha256') != fingerprint(manifest):
        raise ValueError('Review must be bound to this exact source manifest')
    if (review.get('coordinate_system') != '512px common-grid pixel centers'
            or review.get('reviewer_type') != 'human' or not review.get('manual_blind_review_attested')
            or not str(review.get('reviewer', '')).strip() or not review.get('reviewed_utc')):
        raise ValueError('Named human manual review and timestamp required; generated coordinates are not independent landmarks')
    known = {p['pair_id'] for p in manifest['pairs']}
    for name, pair in review['pairs'].items():
        if name not in known:
            raise ValueError('Unknown pair in review')
        if pair.get('unscorable'):
            if not pair.get('reason', '').strip():
                raise ValueError('Unscorable pair needs an explanation')
            continue
        points = pair['points']
        if not 10 <= len(points) <= 20:
            raise ValueError('Review needs 10–20 landmarks per scorable pair')
        ids = [p['landmark_id'].strip() for p in points]
        if any(not name for name in ids) or len(set(ids)) != len(ids):
            raise ValueError('Unique named stable landmarks required')
        for side in ['reference', 'query']:
            coordinates = np.asarray([p[side] for p in points], float)
            uncertainty = np.asarray([p[side+'_uncertainty_px'] for p in points], float)
            if (coordinates.shape != (len(points), 2) or not np.isfinite(coordinates).all()
                    or (coordinates < 0).any() or (coordinates > 511).any()
                    or not np.isfinite(uncertainty).all() or (uncertainty <= 0).any()):
                raise ValueError('Finite in-grid landmarks and positive uncertainties required')
            if len(np.unique(coordinates, axis=0)) != len(points) or coverage(coordinates, (512, 512)) < .10:
                raise ValueError('Landmarks must be distinct and distributed across both images')


def apply_reviews(manifest, first, second=None, root=None):
    validate_review(manifest, first)
    if second:
        validate_review(manifest, second)
        if first['reviewer'].strip().casefold() == second['reviewer'].strip().casefold():
            raise ValueError('Second reviewer must be distinct')
    result = copy.deepcopy(manifest)
    assets = validate_manifest(result, root)
    by_id = {p['pair_id']:p for p in result['pairs']}
    for name, reviewed in first['pairs'].items():
        pair = by_id[name]
        pair['review_provenance'] = dict(primary_reviewer=first['reviewer'], primary_review_sha256=fingerprint(first),
            reviewed_utc=first['reviewed_utc'], source_manifest_sha256=fingerprint(manifest),
            second_review_sha256=fingerprint(second) if second else None,
            second_reviewer=second['reviewer'] if second else None,
            identity_verification='Self-attested human manual review; identity/expertise not verified by software')
        if reviewed.get('unscorable'):
            pair.update(reference_validation_status='unscorable', reference_validation_reason=reviewed['reason'])
            continue
        points = copy.deepcopy(reviewed['points'])
        uncertainty = np.array([np.hypot(p['reference_uncertainty_px'], p['query_uncertainty_px']) for p in points])
        if second:
            other = second['pairs'].get(name)
            if not other or other.get('unscorable'):
                raise ValueError('Second reviewer must score the same pairs; disagreements need explicit adjudication')
            other_points = {p['landmark_id']:p for p in other['points']}
            if set(other_points) != {p['landmark_id'] for p in points}:
                raise ValueError('Second review must identify the same named landmarks independently')
            disagreement = []
            for i, point in enumerate(points):
                other_point = other_points[point['landmark_id']]
                dr = np.linalg.norm(np.array(point['reference'])-other_point['reference'])
                dq = np.linalg.norm(np.array(point['query'])-other_point['query'])
                disagreement.append(float(np.hypot(dr, dq)))
                uncertainty[i] = max(uncertainty[i], disagreement[-1], np.hypot(other_point['reference_uncertainty_px'], other_point['query_uncertainty_px']))
            pair['interreviewer_disagreement_px'] = disagreement
        if root is not None:
            _, _, ma, mb, _ = load_pair(pair, assets, root)
            for point in points:
                for side, mask in [('reference', ma), ('query', mb)]:
                    x, y = np.rint(point[side]).astype(int)
                    if not mask[y, x]:
                        raise ValueError('Landmark falls in invalid source data')
        pair.update(reference_type='independent_checkpoints', checkpoints=points,
            reference_uncertainty_px=float(max(uncertainty)), reference_validation_status='reviewed',
            checkpoint_uncertainty_px=uncertainty.tolist(),
            uncertainty_convention='Conservative maximum combined picking estimate; not a calibrated confidence interval. Coordinates from primary reviewer; second-review differences retained.',
            audit_status='manual_review_imported')
    validate_manifest(result, root)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review')
    p.add_argument('--second-review')
    p.add_argument('--manifest', default='lunar_manifest.json')
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads(Path(args.manifest).read_text())
    first = json.loads(Path(args.review).read_text())
    second = json.loads(Path(args.second_review).read_text()) if args.second_review else None
    destination.write_text(json.dumps(apply_reviews(manifest, first, second, args.root), indent=2, allow_nan=False))
