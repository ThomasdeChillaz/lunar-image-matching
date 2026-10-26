"""Alignment diagnostics and development-only selection of a rejection policy.

Diagnostics never use reference errors. Calibration must use separately supplied
independent labels; map-reference disagreement cannot become a correctness label.
"""
import argparse
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from geometry import transform


def independent_correctness(error, uncertainty, threshold):
    """Conservative label with an uncertainty band; None means unresolved."""
    if error is None or uncertainty is None:
        return None
    if not np.isfinite([error, uncertainty, threshold]).all() or uncertainty < 0 or threshold <= 0:
        raise ValueError('Invalid reference error/uncertainty')
    if error+uncertainty <= threshold:
        return True
    if error-uncertainty > threshold:
        return False
    return None


def coverage(points, shape):
    if points is None or len(points) < 3:
        return None
    hull = cv2.convexHull(np.asarray(points, np.float32))
    return float(cv2.contourArea(hull)/(shape[0]*shape[1]))


def diagnostics(forward, reverse, shape):
    result = dict(coverage_reference=coverage(forward.reference_points, shape),
        coverage_query=coverage(forward.query_points, shape), cycle_median_px=None,
        cycle_p95_px=None, scale_min=None, scale_max=None, rotation_deg=None,
        translation_fraction=None, inlier_residual_p95_px=None, reverse_estimated=reverse.matrix is not None)
    if forward.matrix is None:
        return result
    matrix = np.asarray(forward.matrix, float)
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    result.update(scale_min=float(min(singular)), scale_max=float(max(singular)),
        rotation_deg=float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        translation_fraction=float(np.linalg.norm(matrix[:, 2])/min(shape)))
    if forward.reference_points is not None:
        residuals = np.linalg.norm(transform(forward.reference_points, matrix)-forward.query_points, axis=1)
        result['inlier_residual_p95_px'] = float(np.percentile(residuals, 95))
    if reverse.matrix is not None:
        h, w = shape
        points = np.array([[x, y] for y in np.linspace(0, h-1, 5) for x in np.linspace(0, w-1, 5)])
        recovered = transform(transform(points, matrix), reverse.matrix)
        cycle = np.linalg.norm(recovered-points, axis=1)
        result.update(cycle_median_px=float(np.median(cycle)), cycle_p95_px=float(np.percentile(cycle, 95)))
    return result


def accepts(row, policy):
    if not row['estimated'] or not row.get('reverse_estimated'):
        return False
    for field, minimum in [('scale_min', policy['min_scale'])]:
        if row.get(field) is None or row[field] < minimum:
            return False
    for field, maximum in [('scale_max', policy['max_scale']), ('cycle_p95_px', policy['max_cycle_px']),
                           ('translation_fraction', policy['max_translation_fraction'])]:
        if row.get(field) is None or row[field] > maximum:
            return False
    if row.get('rotation_deg') is None or abs(row['rotation_deg']) > policy['max_rotation_deg']:
        return False
    if row['method'] != 'ncc':
        for field in ['coverage_reference', 'coverage_query']:
            if row.get(field) is None or row[field] < policy['min_coverage']:
                return False
    return True


def quality_counts(rows, policy):
    # Unknown labels remain unknown; they are not silently counted as incorrect.
    known = [r for r in rows if isinstance(r.get('correct'), bool)]
    positive = [r for r in known if r['correct']]
    negative = [r for r in known if not r['correct']]
    accepted_positive = sum(accepts(r, policy) for r in positive)
    accepted_negative = sum(accepts(r, policy) for r in negative)
    return dict(labelled=len(known), unknown_labels=len(rows)-len(known), correct=len(positive),
        incorrect=len(negative), accepted_correct=accepted_positive, accepted_incorrect=accepted_negative,
        correct_retention=accepted_positive/len(positive) if positive else None,
        incorrect_acceptance_rate=accepted_negative/len(negative) if negative else None)


def calibrate(rows, max_incorrect_acceptance=0):
    if not 0 <= max_incorrect_acceptance <= 1:
        raise ValueError('Invalid incorrect-acceptance budget')
    if not rows or any(r['split'] != 'dev' for r in rows):
        raise ValueError('Calibration accepts development rows only')
    if any(r.get('label_source') not in ('independent_checkpoints', 'exact_geometry', 'known_nonoverlap') for r in rows):
        raise ValueError('Independent labels required; map references cannot calibrate correctness')
    if {r.get('correct') for r in rows} != {True, False}:
        raise ValueError('Both correct and incorrect labelled development outcomes required')
    policies = []
    for cover, cycle, translation in itertools.product([0, .01, .03, .08, .15], [.5, 1., 3., 8.], [.05, .15, .3, .6]):
        policy = dict(min_coverage=cover, max_cycle_px=cycle, max_translation_fraction=translation,
            min_scale=.75, max_scale=1.25, max_rotation_deg=20.)
        counts = quality_counts(rows, policy)
        policies.append(dict(policy=policy, **counts))
    feasible = [r for r in policies if r['incorrect_acceptance_rate'] <= max_incorrect_acceptance]
    if not feasible:
        raise ValueError('No candidate policy meets the development acceptance budget')
    best = max(feasible, key=lambda r: (r['accepted_correct'], -r['accepted_incorrect'],
        r['policy']['min_coverage'], -r['policy']['max_cycle_px'], -r['policy']['max_translation_fraction']))
    return dict(schema_version=1, selected_on_split='dev', max_development_incorrect_acceptance=max_incorrect_acceptance,
        policy=best['policy'], development_counts={k:v for k,v in best.items() if k != 'policy'}, tradeoff=policies,
        warning='Development selection is not a guarantee of held-out performance. Report each reference/dataset type separately.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rows', help='JSON array with independent correctness labels and diagnostics')
    p.add_argument('--output', required=True)
    p.add_argument('--max-incorrect-acceptance', type=float, default=0.)
    args = p.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    result = calibrate(json.loads(Path(args.rows).read_text()), args.max_incorrect_acceptance)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
