"""Preregister, execute and audit untouched geography with a deployed rejection gate.

This evaluator does not fit models or thresholds. The default public result is
None when the frozen gate rejects; a raw proposal is retained only for research.
Real-photo warps are a separate control, never repeat-observation ground truth.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from confidence import diagnostics
from dataset import sha256
from evaluate import endpoint_statistics
from landmark_review import fingerprint
from matchers import Match, sift
from moon_dataset import load_image
from research_dataset import overlap, validate
from research_geometry import plan, prepare
from research_quality_gate import accepts, counts
from structural_refinement import AlignmentObjective
from uncertainty_groups import geographic_groups


SEED = 20261030
SOURCES = ['reserve_evaluation.py', 'contrastive_terrain.py', 'research_dataset.py',
    'research_geometry.py', 'research_quality_gate.py', 'structural_refinement.py',
    'structural_matchers.py', 'confidence.py', 'matchers.py', 'geometry.py',
    'evaluate.py', 'moon_dataset.py', 'dataset.py', 'uncertainty_groups.py',
    'landmark_review.py', 'candidate_selection.py', 'evaluate_contrastive_dev.py',
    'synthetic.py', 'train_contrastive_terrain.py']


def reserve_plan(manifest):
    """All frozen reserve patches/lights/motions, plus three disjoint partners.

    Partner selection is deterministic metadata-only and spans components. No
    texture screening, outcomes, model calls or pixels participate in planning.
    """
    validate(manifest)
    tiles = [t for t in manifest['tiles'] if t['split'].startswith('reserve_')]
    lookup, components, _ = geographic_groups(tiles)
    cases = []
    for split in ('reserve_a', 'reserve_b'):
        cases.extend(plan(manifest, split))
        for tile in [t for t in tiles if t['split'] == split]:
            candidates = [t for t in tiles if lookup[t['tile_id']] != lookup[tile['tile_id']]
                and not overlap(t['bounds_lonlat_deg'], tile['bounds_lonlat_deg'], .1)]
            candidates.sort(key=lambda t: hashlib.sha256((str(SEED)+tile['tile_id']+t['tile_id']).encode()).hexdigest())
            partners, seen = [], set()
            for t in candidates:
                group = lookup[t['tile_id']]
                if group not in seen:
                    partners.append(t)
                    seen.add(group)
                if len(partners) == 3:
                    break
            if len(partners) != 3:
                raise ValueError('Three disjoint negative components required')
            for partner in partners:
                for index in range(9):
                    cases.append(dict(pair_id=f'{tile["tile_id"]}_negative_{partner["tile_id"]}_{index}',
                        tile_id=tile['tile_id'], query_tile_id=partner['tile_id'], split=split,
                        reference_light_index=index, query_light_index=(index+3) % 9,
                        geometry_family='nonoverlap_negative'))
    return cases, lookup, components


def freeze(manifest_path, training, development, gate_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(manifest_path).read_text())
    cases, lookup, components = reserve_plan(manifest)
    training, development = Path(training), Path(development)
    terminal = json.loads((development/'COMPLETE.json').read_text())
    gate = json.loads(Path(gate_path).read_text())
    if (sha256(development/'results.json') != terminal['results_sha256'] or
        sha256(development/'frozen_inputs.json') != terminal['frozen_inputs_sha256'] or
        gate['development_results_sha256'] != terminal['results_sha256'] or
        gate['selected_on_split'] != 'dev'):
        raise ValueError('Gate must bind a completed development-only experiment')
    dev = json.loads((development/'frozen_inputs.json').read_text())
    if dev['manifest'] != manifest:
        raise ValueError('Development and reserve manifests differ')
    chosen = gate['policies']['contrastive_selected']['empirical_accepted_error_budgets']
    policy = chosen['0.0']['policy']
    training_files = {str(training/name): sha256(training/name)
        for name in ('training.json', 'COMPLETE.json', 'selected.pt')}
    doc = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
        manifest=manifest, manifest_sha256=fingerprint(manifest), cases=cases,
        components=components, tile_component=lookup, training=str(training),
        training_files_sha256=training_files, source_sha256={p: sha256(p) for p in SOURCES},
        development_run=str(development), development_completion=terminal,
        development_gate_sha256=sha256(gate_path), policy=policy,
        development_policy_counts=chosen['0.0']['development_counts'],
        methods=['terrain_raw', 'terrain_deployed', 'sift'], threshold_px=3, seed=SEED,
        selection='Selected CNN checkpoint from completed training. Strict standalone development gate, fixed before reserve access. No reserve calibration or best-of-reserve selection.',
        primary_endpoint='Frozen deployed correct acceptance and incorrect acceptance, reported separately; failures and rejections stay in the denominator.',
        success_criterion=dict(accepted_error_fraction_max=.01, nonoverlap_false_acceptance_max=.01,
            positive_correct_acceptance_min=.40, minimum_components=12),
        criterion_note='Predeclared empirical evaluation targets, not a statistical risk guarantee. Failure is reported and never repaired on these reserves.',
        uncertainty='Report components and per-component outcomes. Cluster bootstrap positive rates by footprint/acquisition component; negative pairs reuse both components, so no independent-pair confidence bound.',
        geography='Both reserves are within Tycho, four plus eight footprint/acquisition components. No cross-crater or polar generalization claim.',
        real_accuracy='No independently verified repeat-observation navigation labels are available. Real-photo known-warp controls must remain a separate experiment.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(freeze_sha256=sha256(output), cases=len(cases), components=len(components),
        by_split=dict(Counter(c['split'] for c in cases)), policy=policy)), flush=True)
    return doc


def validate_freeze(path):
    doc = json.loads(Path(path).read_text())
    for name, digest in {**doc['source_sha256'], **doc['training_files_sha256']}.items():
        if sha256(name) != digest:
            raise ValueError(f'Frozen source or model changed: {name}')
    cases, lookup, components = reserve_plan(doc['manifest'])
    if cases != doc['cases'] or lookup != doc['tile_component'] or components != doc['components']:
        raise ValueError('Frozen reserve plan changed')
    return doc


def infer(reference, query, reference_mask, query_mask, model, policy):
    """Only pixels, masks, a fixed model and a fixed policy enter inference."""
    cv2.setRNGSeed(SEED)
    start = perf_counter()
    forward = model(reference, query, reference_mask, query_mask)
    cv2.setRNGSeed(SEED)
    reverse = model(query, reference, query_mask, reference_mask)
    diag = diagnostics(forward, reverse, reference.shape)
    q = AlignmentObjective(reference, query, reference_mask, query_mask).quality(forward.matrix)
    quality = {} if q is None else dict(q, agreement_fraction=0., agreeing_other_families=[])
    evidence = dict(estimated=forward.matrix is not None, selection_quality=quality, **diag)
    accepted = accepts(evidence, policy)
    return dict(raw=forward, reverse=reverse, deployed=forward if accepted else Match(reason='frozen_gate_rejected'),
        accepted=accepted, evidence=evidence, bidirectional_with_gate_ms=(perf_counter()-start)*1000)


def inputs(case, tiles, images):
    def get(tile, index):
        key = tile, index
        if key not in images:
            images[key] = load_image(tiles[tile]['images'][index])[:2]
        return images[key]
    a, ma = get(case['tile_id'], case['reference_light_index'])
    base, mb = get(case.get('query_tile_id', case['tile_id']), case['query_light_index'])
    if case['geometry_family'] == 'nonoverlap_negative':
        return a, base, ma, mb, None, None
    b, qm, _, points, targets, _ = prepare(a, base, ma, mb, case['pair_id'], case['geometry_family'])
    return a, b, ma, qm, points, targets


def array_hashes(a, b, ma, mb):
    return {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in
        dict(reference=a, query=b, reference_mask=ma, query_mask=mb).items()}


def run(freeze_path, output, resume=False):
    import torch
    from contrastive_terrain import ContrastiveTerrainMatcher
    from evaluate_contrastive_dev import control
    doc = validate_freeze(freeze_path)
    output = Path(output)
    binding = dict(freeze_sha256=sha256(freeze_path), freeze_path=str(freeze_path))
    if output.exists():
        if not resume or json.loads((output/'binding.json').read_text()) != binding:
            raise ValueError('Resume requires the identical frozen experiment')
    else:
        output.mkdir(parents=True)
        (output/'binding.json').write_text(json.dumps(binding, indent=2))
        (output/'frozen_protocol.json').write_bytes(Path(freeze_path).read_bytes())
        (output/'RESERVES_OPENED.json').write_text(json.dumps(dict(**binding,
            opened_utc=datetime.now(timezone.utc).isoformat(),
            notice='Both reserves are consumed by this frozen evaluation. Never describe them as untouched for a subsequent iteration.')))
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = ContrastiveTerrainMatcher(doc['training'])
    proof = control(model)
    if not proof['passed']:
        raise RuntimeError('Actual selected model software control failed')
    (output/'software_control.json').write_text(json.dumps(proof, indent=2))
    path = output/'results.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected = {(c['pair_id'], m) for c in doc['cases'] for m in doc['methods']}
    done = {(r['pair_id'], r['method']) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Duplicate or unplanned outcomes')
    tiles = {t['tile_id']:t for t in doc['manifest']['tiles'] if t['split'].startswith('reserve_')}
    images = {}
    for number, case in enumerate(doc['cases'], 1):
        if all((case['pair_id'], m) in done for m in doc['methods']):
            continue
        a, b, ma, mb, points, targets = inputs(case, tiles, images)
        prediction = infer(a, b, ma, mb, model, doc['policy'])
        cv2.setRNGSeed(SEED)
        started = perf_counter()
        baseline = sift(a, b, ma, mb)
        baseline_ms = (perf_counter()-started)*1000
        negative = case['geometry_family'] == 'nonoverlap_negative'
        for method, fit in [('terrain_raw', prediction['raw']), ('terrain_deployed', prediction['deployed']), ('sift', baseline)]:
            if (case['pair_id'], method) in done:
                continue
            error, rmse = (None, None) if negative else endpoint_statistics(fit.matrix, points, targets)
            row = dict(**case, method=method, component=doc['tile_component'][case['tile_id']],
                query_component=doc['tile_component'][case.get('query_tile_id', case['tile_id'])],
                dataset_kind='heldout_physical_render',
                label_source='known_nonoverlap' if negative else 'exact_geometry',
                estimated=fit.matrix is not None, correct=bool(not negative and error is not None and error <= doc['threshold_px']),
                error_px=error, rmse_px=rmse, reason=fit.reason, support=fit.support,
                transform=None if fit.matrix is None else np.asarray(fit.matrix).tolist(),
                input_array_sha256=array_hashes(a, b, ma, mb), freeze_sha256=binding['freeze_sha256'])
            if method != 'sift':
                row.update(raw_transform=None if prediction['raw'].matrix is None else np.asarray(prediction['raw'].matrix).tolist(),
                    reverse_transform=None if prediction['reverse'].matrix is None else np.asarray(prediction['reverse'].matrix).tolist(),
                    gate_accepted=prediction['accepted'], gate_evidence=prediction['evidence'],
                    bidirectional_with_gate_ms=prediction['bidirectional_with_gate_ms'])
            else:
                row['forward_ms'] = baseline_ms
            with path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)
            done.add((case['pair_id'], method))
        if number % 72 == 0:
            print(f'{number}/{len(doc["cases"])} cases completed', flush=True)
    if done != expected:
        raise ValueError('Missing planned outcomes')
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    summary = summarize(rows, doc)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),
        results_sha256=sha256(output/'results.json'), protocol_sha256=binding['freeze_sha256'],
        summary_sha256=sha256(output/'summary.json')), indent=2))
    print(json.dumps(summary['overall']), flush=True)


def metrics(rows):
    positive = [r for r in rows if r['label_source']=='exact_geometry']
    negative = [r for r in rows if r['label_source']=='known_nonoverlap']
    accepted_good = sum(r['correct'] for r in positive)
    accepted_wrong = sum(r['estimated'] and not r['correct'] for r in rows)
    accepted = sum(r['estimated'] for r in rows)
    return dict(positive_cases=len(positive), correct=accepted_good,
        correct_rate=accepted_good/len(positive) if positive else None,
        positive_wrong=sum(r['estimated'] and not r['correct'] for r in positive),
        rejected_or_failed_positive=sum(not r['estimated'] for r in positive),
        negative_cases=len(negative), negative_accepted=sum(r['estimated'] for r in negative),
        incorrect_accepted=accepted_wrong,
        accepted_error_fraction=accepted_wrong/accepted if accepted else None)


def summarize(rows, doc):
    overall, by_group, by_family = {}, {}, {}
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(doc['components']), (5000, len(doc['components'])))
    for method in doc['methods']:
        selected = [r for r in rows if r['method']==method]
        overall[method] = metrics(selected)
        component_counts = []
        for i in range(len(doc['components'])):
            group = [r for r in selected if r['component']==i]
            by_group[f'{method}:{i}'] = metrics(group)
            positive = [r for r in group if r['label_source']=='exact_geometry']
            component_counts.append([sum(r['correct'] for r in positive), len(positive)])
        values = np.asarray(component_counts)[samples].sum(1)
        overall[method]['positive_rate_cluster_percentile_95'] = np.percentile(values[:, 0]/values[:, 1], [2.5,97.5]).tolist()
        for split in ('reserve_a', 'reserve_b'):
            by_group[f'{method}:{split}'] = metrics([r for r in selected if r['split']==split])
        for family in ('translation', 'rotation', 'scale', 'combined'):
            by_family[f'{method}:{family}'] = metrics([r for r in selected if r['geometry_family']==family])
    deployed = overall['terrain_deployed']
    target = doc['success_criterion']
    met = bool(deployed['accepted_error_fraction'] is not None
        and deployed['accepted_error_fraction'] <= target['accepted_error_fraction_max']
        and deployed['negative_accepted']/deployed['negative_cases'] <= target['nonoverlap_false_acceptance_max']
        and deployed['correct_rate'] >= target['positive_correct_acceptance_min']
        and len(doc['components']) >= target['minimum_components'])
    return dict(overall=overall, by_group=by_group, by_geometry=by_family,
        components=len(doc['components']), tiles=len(doc['tile_component']),
        predeclared_targets_met=met, empirical_target=target,
        calibration_on_reserves=False, model_selection_on_reserves=False,
        uncertainty_warning=doc['uncertainty'], geography=doc['geography'], real_accuracy=doc['real_accuracy'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--manifest', default='research_manifest_v1.json')
    f.add_argument('--training', default='runs/contrastive-terrain-train-v1')
    f.add_argument('--development', default='runs/contrastive-terrain-evidence-v1')
    f.add_argument('--gate', default='reports/contrastive_terrain_quality_gates.json')
    f.add_argument('--output', required=True)
    r = sub.add_parser('run')
    r.add_argument('--freeze', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.action=='freeze':
        freeze(args.manifest, args.training, args.development, args.gate, args.output)
    else:
        run(args.freeze, args.output, args.resume)
