"""Audit every saved all-Sun-pair fit against exact images and known shifts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from confidence import accepts, diagnostics, quality_counts
from dataset import sha256
from evaluate import endpoint_statistics
from evaluate_crosslighting import METHODS, WARP, plan
from geometry import evaluation_points, transform
from landmark_review import fingerprint
from matchers import Match
from moon_dataset import load_image, validate
from physical_lighting_analysis import solar_separation, shared_support
from verify_extension import source_check, same


def translated_input(query, mask):
    # Independently reconstruct the integer image shift using array slices,
    # rather than invoking the synthetic image generator used by inference.
    shifted = np.zeros_like(query)
    shifted_mask = np.zeros_like(mask)
    shifted[:124,6:] = query[4:,:122]
    shifted_mask[:124,6:] = mask[4:,:122]
    shifted[~shifted_mask] = 0
    return shifted, shifted_mask


def verify(folder, original='runs/physical-adaptation-v1'):
    folder, original = Path(folder), Path(original)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    frozen = json.loads((folder/'frozen_inputs.json').read_text())
    if (sha256(folder/'results.json') != complete['results_sha256']
            or sha256(folder/'frozen_inputs.json') != complete['frozen_inputs_sha256']):
        raise ValueError('Cross-lighting completion hashes differ')
    source_check(frozen['source_sha256'])
    original_complete = json.loads((original/'COMPLETE.json').read_text())
    if (frozen['physical_source_result_sha256'] != original_complete['result_sha256']
            or sha256(original/'results.json') != original_complete['result_sha256']
            or sha256(original/'frozen_policies.json') != original_complete['policies_sha256']):
        raise ValueError('Original model-selection experiment changed')
    policies = frozen['original_physical_policies']
    if (policies != json.loads((original/'frozen_policies.json').read_text())
            or frozen['adapter'] != json.loads((original/'frozen_adapter.json').read_text())
            or frozen['manifest'] != json.loads((original/'frozen_manifest.json').read_text())):
        raise ValueError('Model, policy or dataset changed after original evaluation')
    for value in policies.values():
        if 'policy' in value and value['selected_on_split'] != 'dev':
            raise ValueError('Test-selected policy is not permitted')
    manifest = validate(frozen['manifest'], '.')
    if frozen['warp'] != WARP or frozen['methods'] != METHODS or frozen['threshold_px'] != 3 or frozen['image_size'] != 128:
        raise ValueError('Fixed cross-lighting protocol drift')
    planned = plan(manifest,frozen['split'])
    if planned != frozen['planned']:
        raise ValueError('Planned Sun combinations changed')
    rows = json.loads((folder/'results.json').read_text())
    expected = {(p['pair_id'],m) for p in planned for m in METHODS}
    identities = [(r['pair_id'],r['method']) for r in rows]
    if len(rows) != complete['rows'] or len(planned) != complete['pairs'] or set(identities) != expected or len(identities) != len(expected):
        raise ValueError('Missing, repeated or unplanned outcomes')
    tiles = {t['tile_id']:t for t in manifest['tiles']}
    cases = {p['pair_id']:p for p in planned}
    loaded, calculated = {}, {}
    truth = np.array([[1.,0,6],[0,1.,-4]])
    for row in rows:
        case = cases[row['pair_id']]
        if (any(row[k] != case[k] for k in case) or row['label_source'] != 'exact_geometry'
                or row['dataset_kind'] != 'physical_render_fixed_translation'):
            raise ValueError('Physical source labels or pair provenance changed')
        tile = tiles[case['tile_id']]
        if row['observation_group'] != tile['observation_group']:
            raise ValueError('Source acquisition identity drift')
        if row['pair_id'] not in calculated:
            def image(index):
                key = (tile['tile_id'],index)
                if key not in loaded:
                    loaded[key] = load_image(tile['images'][index])
                return loaded[key]
            a,ma,raw_a = image(case['reference_light_index'])
            base,mb,raw_b = image(case['query_light_index'])
            b,qm = translated_input(base,mb)
            points = evaluation_points((128,128),truth)
            targets = transform(points,truth)
            keep = ma[np.rint(points[:,1]).astype(int),np.rint(points[:,0]).astype(int)]
            keep &= qm[np.rint(targets[:,1]).astype(int),np.rint(targets[:,0]).astype(int)]
            hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=qm).items()}
            first,second = tile['images'][case['reference_light_index']],tile['images'][case['query_light_index']]
            metadata = dict(solar_separation_deg=solar_separation(first,second),
                shared_nondark_fraction=shared_support(raw_a,raw_b,ma,mb,.01),
                reference_solar_elevation_deg=first['elevation_deg'],query_solar_elevation_deg=second['elevation_deg'],
                reference_zero_render=first['constant_zero_render'],query_zero_render=second['constant_zero_render'])
            calculated[row['pair_id']] = (hashes,points[keep],targets[keep],metadata)
        hashes,points,targets,metadata = calculated[row['pair_id']]
        if row['input_array_sha256'] != hashes or row['scored_points'] != len(points):
            raise ValueError('Known-shift pixels, masks or evaluation support changed')
        for key,value in metadata.items():
            same(row[key],value,key)
        error,rmse = endpoint_statistics(row['transform'],points,targets)
        same(row['error_px'],error,'cross-lighting endpoint error')
        same(row['rmse_px'],rmse,'cross-lighting RMSE')
        if row['correct'] != bool(error is not None and error <= 3) or row['estimated'] != (row['transform'] is not None):
            raise ValueError('Known-geometry correctness drift')
        forward = Match(matrix=row['transform'],reference_points=None if row['reference_points'] is None else np.asarray(row['reference_points']),
            query_points=None if row['query_points'] is None else np.asarray(row['query_points']))
        reverse = Match(matrix=row['reverse_transform'])
        for key,value in diagnostics(forward,reverse,(128,128)).items():
            same(row[key],value,key)
        expected_gate = accepts(row,policies[row['method']]['policy']) if 'policy' in policies.get(row['method'],{}) else None
        same(row['accepted_by_original_development_policy'],expected_gate,'transported frozen policy')
    stored_summary = json.loads((folder/'summary.json').read_text())
    if len(stored_summary) != len(METHODS) or {s['method'] for s in stored_summary} != set(METHODS):
        raise ValueError('Incomplete method summary')
    for summary in stored_summary:
        selected = [r for r in rows if r['method'] == summary['method']]
        if (summary['pairs'] != len(selected) or summary['correct'] != sum(r['correct'] for r in selected)
                or summary['returned'] != sum(r['estimated'] for r in selected)):
            raise ValueError('Cross-lighting summary mismatch')
        errors = [r['error_px'] for r in selected if r['error_px'] is not None]
        same(summary['median_error_returned_px'],float(np.median(errors)) if errors else None,'returned-fit median')
        same(summary['dark_input_pairs'],sum(r['reference_zero_render'] or r['query_zero_render'] for r in selected),'dark inputs')
        if 'policy' in policies.get(summary['method'],{}):
            if summary['transported_rejection'] != quality_counts(selected,policies[summary['method']]['policy']):
                raise ValueError('Frozen rejection summary mismatch')
    return dict(pairs=len(planned),evaluations=len(rows),source_acquisitions=len({r['observation_group'] for r in rows}),
        source_files_verified=len(frozen['source_sha256']),results_sha256=complete['results_sha256'],
        frozen_inputs_sha256=complete['frozen_inputs_sha256'],manifest_sha256=fingerprint(manifest),
        summary_sha256=sha256(folder/'summary.json'),
        original_adapter_sha256=sha256(original/'frozen_adapter.json'),
        original_policy_sha256=original_complete['policies_sha256'],fixed_translation=[6,-4],
        scope='Exhaustive Sun-pair coverage, exact raw image and mask bytes reconstructed by slicing, known-geometry scoring and unchanged development policy. Physical follow-up on previously inspected terrain; no independent real-image validation.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--original',default='runs/physical-adaptation-v1')
    p.add_argument('--output',required=True)
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    result = verify(args.run,args.original)
    destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
