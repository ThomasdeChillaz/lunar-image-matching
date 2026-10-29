"""Score saved predictions on completed human landmarks; freeze dev-only gates.

No matcher is fitted to the review. Unreviewed and uncertain pairs retain unknown
labels, and imported pilot references cover only the corresponding pair subset.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from confidence import calibrate, independent_correctness, quality_counts
from dataset import load_pair, sha256, validate_manifest
from evaluate import endpoint_statistics
from landmark_review import fingerprint


def read_run(folder, expected_manifest):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if sha256(folder/'results.json') != complete['results_sha256']:
        raise ValueError('Saved result digest mismatch')
    if (folder/'frozen_inputs.json').exists():
        source = json.loads((folder/'frozen_inputs.json').read_text())
        actual = fingerprint(source['manifest'])
    else:
        source = json.loads((folder/'audit.json').read_text())
        actual = source['manifest_sha256']
    if actual != fingerprint(expected_manifest):
        raise ValueError('Prediction and reviewed-reference source grids differ')
    return json.loads((folder/'results.json').read_text())


def score(reviewed, original, runs, output):
    assets = validate_manifest(reviewed, '.')
    if reviewed['assets'] != original['assets']:
        raise ValueError('Review changed source images or grids')
    base = {p['pair_id']:p for p in original['pairs']}
    if set(base) != {p['pair_id'] for p in reviewed['pairs']}:
        raise ValueError('Review changed pair selection')
    reference = {}
    for pair in reviewed['pairs']:
        source = base[pair['pair_id']]
        for key in ['reference', 'query', 'common_bounds_m', 'split', 'region_id']:
            if pair[key] != source[key]:
                raise ValueError('Review changed pair source geometry')
        if pair.get('reference_validation_status') != 'reviewed' or pair['reference_type'] != 'independent_checkpoints':
            continue
        provenance = pair.get('review_provenance', {})
        if provenance.get('target_source_manifest_sha256') != fingerprint(original):
            raise ValueError('Known-grid conversion provenance required')
        _, _, ma, mb, _ = load_pair(pair, assets, '.')
        points, targets = [np.asarray([p[key] for p in pair['checkpoints']], float) for key in ['reference', 'query']]
        if len(points) < 10 or pair.get('reference_uncertainty_px') is None:
            raise ValueError('Incomplete independent reference')
        for xy, mask in [(points, ma), (targets, mb)]:
            if (xy < 0).any() or (xy > 511).any() or not mask[np.rint(xy[:, 1]).astype(int), np.rint(xy[:, 0]).astype(int)].all():
                raise ValueError('Independent reference outside valid source data')
        reference[pair['pair_id']] = points, targets, pair['reference_uncertainty_px']
    if not reference:
        raise ValueError('No independent human references; cannot score true real registration accuracy')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    def label_row(saved):
        row = dict(saved)
        if row['pair_id'] in reference:
            points, targets, uncertainty = reference[row['pair_id']]
            error, rmse = endpoint_statistics(row['transform'], points, targets)
            row.update(original_map_disagreement_px=row['error_px'], error_px=error, rmse_px=rmse,
                error_kind='independent_checkpoint_error', label_source='independent_checkpoints',
                reference_uncertainty_px=uncertainty, scored_points=len(points),
                correct=False if row['transform'] is None else independent_correctness(error, uncertainty, 3),
                failure_with_visible_independent_landmarks=row['transform'] is None)
        else:
            row.update(label_source=None, correct=None)
        return row
    rows, seen = [], set()
    for run in runs:
        for saved in read_run(run, original):
            row = dict(saved)
            key = (row['pair_id'], row['method'], row['preprocessing'])
            if key in seen:
                raise ValueError('Duplicate saved method/pair')
            seen.add(key)
            if row['split'] != 'test':
                row = label_row(row)
            rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method'], row['preprocessing']].append(row)
    policies, summaries = {}, []
    for (method, preparation), subset in grouped.items():
        key = method+'__'+preparation
        dev = [r for r in subset if r['split'] == 'dev' and r['label_source'] == 'independent_checkpoints' and isinstance(r['correct'], bool)]
        try:
            policies[key] = calibrate(dev, 0.)
        except ValueError as exc:
            policies[key] = dict(unavailable_reason=str(exc), labelled_development_rows=len(dev))
    # Persist thresholds before computing any independent held-out errors.
    (output/'frozen_policies.json').write_text(json.dumps(policies, indent=2, allow_nan=False))
    rows = [label_row(r) if r['split'] == 'test' else r for r in rows]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method'], row['preprocessing']].append(row)
    for (method, preparation), subset in grouped.items():
        key = method+'__'+preparation
        for split in ['dev', 'test']:
            split_rows = [r for r in subset if r['split'] == split]
            counts = dict(split=split, method=method, preprocessing=preparation, pairs=len(split_rows),
                independently_reviewed=sum(r['label_source'] == 'independent_checkpoints' for r in split_rows),
                known_correct=sum(r['correct'] is True for r in split_rows), known_incorrect=sum(r['correct'] is False for r in split_rows),
                unresolved=sum(r['correct'] is None for r in split_rows))
            if 'policy' in policies[key]:
                counts['frozen_development_policy'] = quality_counts(split_rows, policies[key]['policy'])
            summaries.append(counts)
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(dict(measurements=summaries,
        correctness='Primary 3px threshold with conservative reviewer/grid-conversion uncertainty band. Visible independently annotated pairs returning no fit are explicit failures.',
        limitation='Independent reference coverage is limited to reviewed pairs. Real policies use known development labels only; ambiguous/unreviewed outcomes stay unknown. S3 is exploratory after pilot inspection.'), indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(reviewed_manifest_sha256=fingerprint(reviewed),
        original_manifest_sha256=fingerprint(original), runs={str(run):sha256(Path(run)/'results.json') for run in runs},
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ['score_independent.py', 'confidence.py', 'evaluate.py']},
        reviewed_pairs=list(reference), rows=len(rows)), indent=2))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'),
        policies_sha256=sha256(output/'frozen_policies.json'), summary_sha256=sha256(output/'summary.json'))))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed-manifest', required=True)
    p.add_argument('--original-manifest', default='expanded_solar_manifest.json')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    score(json.loads(Path(args.reviewed_manifest).read_text()), json.loads(Path(args.original_manifest).read_text()), args.runs, args.output)
