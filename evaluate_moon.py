"""Frozen versus lunar-adapted matching on physically changing illumination.

Evaluate development before test; freeze each rejection policy on development
only. Native 128px geometric errors stay separate from 512px real-image map
diagnostics. Negative pairs use geographically disjoint terrain, never invented
identity targets. Dark renders remain in the denominator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from confidence import calibrate, diagnostics, quality_counts, accepts
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_extended import CachedLearnedMatcher, circular_difference
from geometry import evaluation_points, transform
from landmark_review import fingerprint
from matchers import preprocess, sift
from moon_dataset import load_image, validate
from synthetic import make_pair


class AdaptedMatcher:
    def __init__(self, cached, adapter):
        self.cached = cached
        self.matrix = cached.model.torch.tensor(adapter['matrix'], dtype=cached.model.torch.float32)
        if self.matrix.shape != (128, 128) or not cached.model.torch.isfinite(self.matrix).all():
            raise ValueError('Invalid descriptor adapter')

    def __call__(self, a, b, ma, mb):
        from matchers import Match, fit_geometry
        if min(a.std(), b.std()) < 1:
            return Match(reason='low_texture')
        model, torch = self.cached.model, self.cached.model.torch
        aa, _ = self.cached.features(a, ma)
        bb, _ = self.cached.features(b, mb)
        with torch.inference_mode():
            if min(aa['keypoints'].shape[1], bb['keypoints'].shape[1]) < 6:
                return Match(reason='insufficient_features')
            first, second = dict(aa), dict(bb)
            first['descriptors'] = torch.nn.functional.normalize(aa['descriptors']@self.matrix.T, dim=-1)
            second['descriptors'] = torch.nn.functional.normalize(bb['descriptors']@self.matrix.T, dim=-1)
            matches = model.matcher({'image0':first, 'image1':second})['matches'][0]
            if len(matches) < 6:
                return Match(reason='insufficient_matches', support=len(matches))
            return fit_geometry(aa['keypoints'][0][matches[:, 0]].numpy(), bb['keypoints'][0][matches[:, 1]].numpy())


def disjoint(a, b):
    x, y = a['bounds_lonlat_deg'], b['bounds_lonlat_deg']
    return x[2]+.1 < y[0] or y[2]+.1 < x[0] or x[3]+.1 < y[1] or y[3]+.1 < x[1]


def cases(manifest, split):
    selected = [t for t in manifest['tiles'] if t['split'] == split]
    for tile in selected:
        a, ma, raw_a = load_image(tile['images'][0])
        for index, image in enumerate(tile['images'][1:], 1):
            b, mb, raw_b = load_image(image)
            case = dict(angle=[-8, 0, 8][index % 3], dx=6, dy=-4, gain=1, gamma=1, noise=0)
            query, truth = make_pair(b, case, 20261026)
            qm = cv2.warpAffine(mb.astype(np.uint8), truth, (128, 128), flags=cv2.INTER_NEAREST).astype(bool)
            query[~qm] = 0
            points = evaluation_points(a.shape, truth)
            targets = transform(points, truth)
            keep = ma[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
            keep &= qm[np.rint(targets[:, 1]).astype(int), np.rint(targets[:, 0]).astype(int)]
            valid = ma & mb
            yield dict(pair_id=tile['tile_id']+f'_light_{index}', tile_id=tile['tile_id'], split=split,
                case='physical_light_known_warp', dataset_kind='physical_render', label_source='exact_geometry',
                reference=a, query=query, reference_mask=ma, query_mask=qm,
                points=points[keep], targets=targets[keep], truth=truth,
                solar_azimuth_difference_deg=circular_difference(tile['images'][0]['azimuth_deg'], image['azimuth_deg']),
                shared_nondark_fraction=float(((raw_a > .01) & (raw_b > .01) & valid).sum()/valid.sum()) if valid.any() else 0.,
                shared_nondark_convention='Same rendered terrain pixels before known warp; fixed I/F threshold .01. Not a labelled physical shadow mask.',
                reference_zero_render=tile['images'][0]['constant_zero_render'], query_zero_render=image['constant_zero_render'])
        other = next((t for t in selected if disjoint(tile, t)), None)
        if other:
            b, mb, _ = load_image(other['images'][0])
            yield dict(pair_id=tile['tile_id']+'_nonoverlap_'+other['tile_id'], tile_id=tile['tile_id'], split=split,
                case='known_nonoverlap', dataset_kind='physical_nonoverlap', label_source='known_nonoverlap',
                reference=a, query=b, reference_mask=ma, query_mask=mb, points=None, targets=None, truth=None,
                negative_query_tile=other['tile_id'], solar_azimuth_difference_deg=None, shared_nondark_fraction=None)


def summarize(rows):
    summaries = []
    for split in ['dev', 'test']:
        for method in sorted({r['method'] for r in rows}):
            subset = [r for r in rows if r['split'] == split and r['method'] == method and r['dataset_kind'] == 'physical_render']
            if not subset:
                continue
            error = [r['error_px'] for r in subset if r['error_px'] is not None]
            negative = [r for r in rows if r['split'] == split and r['method'] == method and r['dataset_kind'] == 'physical_nonoverlap']
            summaries.append(dict(split=split, method=method, positive_pairs=len(subset),
                estimated=sum(r['estimated'] for r in subset), correct=sum(r['correct'] for r in subset),
                success_rate=sum(r['correct'] for r in subset)/len(subset),
                median_error_estimated_px=float(np.median(error)) if error else None,
                p95_error_estimated_px=float(np.percentile(error, 95)) if error else None,
                dark_queries=sum(r.get('query_zero_render', False) for r in subset), negative_pairs=len(negative),
                negative_pairs_returning_alignment=sum(r['estimated'] for r in negative)))
    return summaries


def run(manifest, adapter, output):
    validate(manifest, '.')
    if adapter['manifest_sha256'] != fingerprint(manifest) or adapter['model_lock_sha256'] != sha256('model_lock.json'):
        raise ValueError('Adapter training provenance does not match inputs')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output/'frozen_adapter.json').write_text(json.dumps(adapter, indent=2))
    config = dict(seed=20261026, methods=['sift', 'disk_lightglue', 'disk_lightglue_clahe', 'disk_lightglue_lunar'],
        threshold_px=3, image_size=128, max_incorrect_development_acceptance=0,
        protocol='Physical illumination plus known 2D warps; all positives including dark failures and explicit geographically disjoint negatives. Calibrate dev, save policy, then evaluate test.',
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ['evaluate_moon.py', 'moon_dataset.py', 'confidence.py', 'evaluate_extended.py', 'learned.py', 'matchers.py', 'synthetic.py', 'geometry.py']})
    (output/'frozen_config.json').write_text(json.dumps(config, indent=2))
    cv2.setNumThreads(1)
    cached = CachedLearnedMatcher()
    adapted = AdaptedMatcher(cached, adapter)
    rows, policies = [], {}
    for split in ['dev', 'test']:
        for index, case in enumerate(cases(manifest, split), 1):
            for method in config['methods']:
                a, b = case['reference'], case['query']
                ma, mb = case['reference_mask'], case['query_mask']
                if method == 'disk_lightglue_clahe':
                    a, b = preprocess(a, True), preprocess(b, True)
                a, b = a.copy(), b.copy()
                a[~ma], b[~mb] = 0, 0
                def match(x, y, mx, my):
                    cv2.setRNGSeed(config['seed'])
                    if method == 'sift':
                        return sift(x, y, mx, my)
                    if method == 'disk_lightglue_lunar':
                        return adapted(x, y, mx, my)
                    return cached(x, y, mx, my)[0]
                forward, reverse = match(a, b, ma, mb), match(b, a, mb, ma)
                error, rmse = endpoint_statistics(forward.matrix, case['points'], case['targets']) if case['truth'] is not None else (None, None)
                metadata = {k:v for k,v in case.items() if k not in ['reference', 'query', 'reference_mask', 'query_mask', 'points', 'targets', 'truth']}
                row = dict(**metadata, method=method, estimated=forward.matrix is not None, reason=forward.reason,
                    error_px=error, rmse_px=rmse, correct=bool(error is not None and error <= config['threshold_px']) if case['truth'] is not None else False,
                    transform=None if forward.matrix is None else forward.matrix.tolist(), support=forward.support,
                    **diagnostics(forward, reverse, a.shape))
                if split == 'test':
                    row['accepted'] = accepts(row, policies[method]['policy']) if method in policies else False
                with (output/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
            print(split, index, case['pair_id'], flush=True)
        if split == 'dev':
            for method in config['methods']:
                subset = [r for r in rows if r['method'] == method]
                try:
                    policies[method] = calibrate(subset, 0.)
                except ValueError as exc:
                    # Missing positive/negative outcomes cannot yield a fitted policy.
                    policies[method] = dict(calibration_error=str(exc))
            (output/'frozen_policies.json').write_text(json.dumps(policies, indent=2, allow_nan=False))
            # Save before any held-out sample is evaluated.
            (output/'development_results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
            policies = {m:p for m,p in policies.items() if 'policy' in p}
    summary = summarize(rows)
    rejection = {method:{kind:quality_counts([r for r in rows if r['method'] == method and r['split'] == 'test' and r['dataset_kind'] == kind], policy['policy'])
        for kind in ['physical_render', 'physical_nonoverlap']} for method, policy in policies.items()}
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (output/'summary.json').write_text(json.dumps(dict(registration=summary, heldout_rejection=rejection), indent=2, allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows), result_sha256=sha256(output/'results.json'),
        policies_sha256=sha256(output/'frozen_policies.json'))))
    print(json.dumps(summary, indent=2))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='moon_manifest.json')
    p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(json.loads(Path(args.manifest).read_text()), json.loads(Path(args.adapter).read_text()), args.output)
