"""Resumable bidirectional diagnostics on frozen real lunar observations.

Map-reference errors are never converted into correctness labels. Learned
features are cached on exact image/mask bytes; extraction times and matching
times are recorded separately. Existing pilot outputs are never overwritten.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from confidence import diagnostics, independent_correctness
from dataset import common_points, load_pair, sha256, validate_manifest
from evaluate import endpoint_statistics
from learned import LearnedMatcher
from matchers import Match, fit_geometry, ncc, preprocess, sift


class CachedLearnedMatcher:
    def __init__(self, cache='data/models', device='cpu', max_keypoints=512):
        self.model = LearnedMatcher(cache, max_keypoints, device)
        self.cache = {}

    def features(self, image, mask):
        key = hashlib.sha256(image.tobytes()+mask.tobytes()).hexdigest()
        if key not in self.cache:
            self.model.sync()
            start = perf_counter()
            with self.model.torch.inference_mode():
                result = self.model.features(image, mask)
            self.model.sync()
            self.cache[key] = (result, (perf_counter()-start)*1000)
        return self.cache[key]

    def __call__(self, a, b, ma, mb):
        if min(a.std(), b.std()) < 1:
            return Match(reason='low_texture'), dict(feature_reference_ms=0., feature_query_ms=0., matching_ms=0.)
        aa, ta = self.features(a, ma)
        bb, tb = self.features(b, mb)
        self.model.sync()
        start = perf_counter()
        with self.model.torch.inference_mode():
            if min(aa['keypoints'].shape[1], bb['keypoints'].shape[1]) < 6:
                result = Match(reason='insufficient_features')
            else:
                matches = self.model.matcher({'image0': aa, 'image1': bb})['matches'][0]
                if len(matches) < 6:
                    result = Match(reason='insufficient_matches', support=len(matches))
                else:
                    ap = aa['keypoints'][0][matches[:, 0]].cpu().numpy()
                    bp = bb['keypoints'][0][matches[:, 1]].cpu().numpy()
                    result = fit_geometry(ap, bp)
        self.model.sync()
        return result, dict(feature_reference_ms=ta, feature_query_ms=tb, matching_ms=(perf_counter()-start)*1000)


def circular_difference(a, b):
    return float(abs((a-b+180) % 360-180))


def illumination_metrics(a, b, ma, mb):
    valid = ma & mb
    result = {}
    for threshold in (5, 20, 50):
        first, second = (a > threshold) & ma, (b > threshold) & mb
        both = first & second
        union = (first | second) & valid
        result[f'bright_overlap_at_{threshold}_fraction'] = float(both.sum()/valid.sum()) if valid.any() else None
        result[f'bright_jaccard_at_{threshold}'] = float(both.sum()/union.sum()) if union.any() else None
    return result


def run(manifest, config, output, root='.', resume=False):
    if (not config['methods'] or len(set(config['methods'])) != len(config['methods'])
            or any(m not in ('ncc', 'sift', 'disk_lightglue') for m in config['methods'])
            or any(p not in ('none', 'clahe') for p in config['preprocessing'])
            or config['correctness_threshold_px'] <= 0):
        raise ValueError('Invalid extended evaluation configuration')
    assets = validate_manifest(manifest, root)
    output = Path(output)
    frozen = dict(manifest=manifest, config=config,
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ['evaluate_extended.py', 'confidence.py', 'matchers.py', 'learned.py', 'dataset.py', 'geometry.py', 'fetch_models.py']})
    frozen_text = json.dumps(frozen, sort_keys=True, allow_nan=False)
    if output.exists():
        if not resume or (output/'frozen_inputs.json').read_text() != frozen_text:
            raise ValueError('Existing run requires --resume and identical inputs, configuration and source')
    else:
        output.mkdir(parents=True)
        (output/'frozen_inputs.json').write_text(frozen_text)
        environment = dict(platform=platform.platform(), numpy=np.__version__, opencv=cv2.__version__,
            timing='Bidirectional inference. Learned extraction cached by exact image/mask bytes; feature times measured once per unique input and reported separately. Do not compare these timings to the pilot.',
            bright_overlap='Intensity-threshold proxy in provisional map coordinates; not a physical shadow mask or independently registered common illumination.',
            reference='Map disagreement is provisional. No correctness labels are derived from it.')
        (output/'environment.json').write_text(json.dumps(environment, indent=2))
    rows = []
    log = output/'results.jsonl'
    if log.exists():
        # Interrupted writes are not silently repaired; preserve evidence for inspection.
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    completed = {(r['pair_id'], r['method'], r['preprocessing']) for r in rows}
    cv2.setNumThreads(1)
    model = CachedLearnedMatcher(device=config.get('device', 'cpu')) if 'disk_lightglue' in config['methods'] else None
    for number, pair in enumerate(manifest['pairs'], 1):
        a, b, ma, mb, grid = load_pair(pair, assets, root)
        metadata = illumination_metrics(a, b, ma, mb)
        first, second = assets[pair['reference']], assets[pair['query']]
        if all(x.get('solar_azimuth_deg') is not None for x in (first, second)):
            metadata.update(solar_azimuth_difference_deg=circular_difference(first['solar_azimuth_deg'], second['solar_azimuth_deg']),
                reference_solar_elevation_deg=first['solar_elevation_deg'], query_solar_elevation_deg=second['solar_elevation_deg'])
        if pair['reference_type'] == 'independent_checkpoints':
            points = np.asarray([p['reference'] for p in pair['checkpoints']], float)
            targets = np.asarray([p['query'] for p in pair['checkpoints']], float)
            keep = ma[np.rint(points[:, 1]).astype(int).clip(0, 511), np.rint(points[:, 0]).astype(int).clip(0, 511)]
            keep &= mb[np.rint(targets[:, 1]).astype(int).clip(0, 511), np.rint(targets[:, 0]).astype(int).clip(0, 511)]
            points, targets = points[keep], targets[keep]
            error_kind = 'independent_checkpoint_error'
            label_source = 'independent_checkpoints' if pair.get('reference_validation_status') == 'reviewed' else None
        else:
            points = common_points(ma, mb)
            targets = points.copy()
            error_kind, label_source = 'map_reference_disagreement', None
        for preparation in config['preprocessing']:
            aa, bb = preprocess(a, preparation == 'clahe'), preprocess(b, preparation == 'clahe')
            aa[~ma], bb[~mb] = 0, 0
            for method in config['methods']:
                if (pair['pair_id'], method, preparation) in completed:
                    continue
                cv2.setRNGSeed(config['seed'])
                start = perf_counter()
                if method == 'disk_lightglue':
                    forward, forward_timing = model(aa, bb, ma, mb)
                    cv2.setRNGSeed(config['seed'])
                    reverse, reverse_timing = model(bb, aa, mb, ma)
                else:
                    matcher = {'ncc': ncc, 'sift': sift}[method]
                    forward = matcher(aa, bb, ma, mb)
                    cv2.setRNGSeed(config['seed'])
                    reverse = matcher(bb, aa, mb, ma)
                    forward_timing = reverse_timing = None
                duration = (perf_counter()-start)*1000
                error, rmse = endpoint_statistics(forward.matrix, points, targets)
                row = dict(pair_id=pair['pair_id'], region_id=pair['region_id'], split=pair['split'],
                    reference=pair['reference'], query=pair['query'], method=method, preprocessing=preparation,
                    dataset_kind='real_repeat_observation', error_kind=error_kind, label_source=label_source,
                    correct=independent_correctness(error, pair.get('reference_uncertainty_px'), config['correctness_threshold_px']) if label_source else None,
                    estimated=forward.matrix is not None, error_px=error, rmse_px=rmse,
                    scored_points=len(points), reference_uncertainty_px=pair.get('reference_uncertainty_px'),
                    reason=forward.reason, support=forward.support, support_type='template_pixels' if method == 'ncc' else 'geometric_inliers',
                    transform=None if forward.matrix is None else forward.matrix.tolist(),
                    reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
                    reference_points=None if forward.reference_points is None else forward.reference_points.tolist(),
                    query_points=None if forward.query_points is None else forward.query_points.tolist(),
                    bidirectional_elapsed_ms=duration, forward_timing=forward_timing, reverse_timing=reverse_timing,
                    **diagnostics(forward, reverse, a.shape), **metadata)
                with log.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                rows.append(row)
                completed.add((pair['pair_id'], method, preparation))
        print(f'{number}/{len(manifest["pairs"])} {pair["pair_id"]}: {len(completed)} completed evaluations', flush=True)
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), results_sha256=sha256(output/'results.json'))))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='expanded_solar_manifest.json')
    p.add_argument('--config', default='configs/extended.json')
    p.add_argument('--output', required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(json.loads(Path(args.manifest).read_text()), json.loads(Path(args.config).read_text()), args.output, args.root, args.resume)
