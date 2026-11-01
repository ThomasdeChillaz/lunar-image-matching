"""Hash-bound saved-only native pose comparison; never executes a matcher."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_cases(root, summary):
    files = sorted((root / 'cases').glob('*.json'))
    if {p.name: digest(p) for p in files} != summary['case_sha256']:
        raise ValueError('Saved case hashes differ: ' + str(root))
    return [read(p) for p in files]


def ordered(rows, method, cases):
    selected = [r for r in rows if r['method'] == method]
    by_id = {r['pair_id']: r for r in selected}
    if len(selected) != len(by_id) or set(by_id) != {c['pair_id'] for c in cases}:
        raise ValueError('Incomplete or duplicate method: ' + method)
    for c in cases:
        r = by_id[c['pair_id']]
        for k in ('tile_id', 'component', 'light_a', 'light_b'):
            if r[k] != c[k]:
                raise ValueError('Case definition differs: ' + c['pair_id'])
        if r['acceptance_policy'] is not None:
            raise ValueError('Raw comparison must not be relabelled accepted')
        if r['strict_pose_correct'] and not r['pose_correct']:
            raise ValueError('Strict correctness is not nested')
        if r['pose_correct'] and r['estimated_cam2world'] is None:
            raise ValueError('Correct pose is missing')
    return [by_id[c['pair_id']] for c in cases]


def totals(rows):
    correct = sum(bool(r['pose_correct']) for r in rows)
    estimated = sum(r['estimated_cam2world'] is not None for r in rows)
    return dict(cases=len(rows), correct=correct,
                strict=sum(bool(r['strict_pose_correct']) for r in rows),
                wrong=estimated-correct, no_pose=len(rows)-estimated)


def paired_interval(a, b):
    """Exact four-component bootstrap; descriptive exposed-development only."""
    groups = sorted({r['component'] for r in a})
    aa, bb = {r['pair_id']: r for r in a}, {r['pair_id']: r for r in b}
    if set(aa) != set(bb) or len(groups) != 4:
        raise ValueError('Expected matched four-component development cases')
    data = []
    for g in groups:
        ids = [r['pair_id'] for r in a if r['component'] == g]
        data.append((sum(int(aa[i]['pose_correct'])-int(bb[i]['pose_correct']) for i in ids), len(ids)))
    values = sorted(100*sum(data[g][0] for g in draw)/sum(data[g][1] for g in draw)
                    for draw in itertools.product(range(4), repeat=4))
    def percentile(q):
        x = (len(values)-1)*q
        lo, hi = math.floor(x), math.ceil(x)
        return values[lo] + (values[hi]-values[lo])*(x-lo)
    return dict(difference_pp=100*(totals(a)['correct']-totals(b)['correct'])/len(a),
                descriptive_component_bootstrap_95_pp=[percentile(.025), percentile(.975)],
                resamples=len(values), groups=4,
                scope='Exposed-development descriptive component bootstrap; not fresh-test confidence or proof of generalization.')


def report(artifacts, output):
    output = Path(output)
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Preserve prior reports')
    manifest_path = Path('reports/localization_training_data_v5.json')
    cases = [c for c in read(manifest_path)['cases'] if c['split'] == 'model_selection']
    if len(cases) != 72 or len({c['component'] for c in cases}) != 4 or len({c['tile_id'] for c in cases}) != 8:
        raise ValueError('Unexpected development cohort')
    seals = {str(manifest_path): digest(manifest_path)}
    methods = {}
    mr = Path('runs/localization-v28-moonst3r-amended')
    ma_path = Path('reports/native_moonst3r_amended_audit_v28.json')
    ma, ms = read(ma_path), read(mr/'summary.json')
    if not ma['passed'] or ma['records'] != 72 or ma['summary_sha256'] != digest(mr/'summary.json') or ms['original_calendar_protocol_completed']:
        raise ValueError('Amended MOONSt3R audit required; never reset original protocol')
    methods['MOONSt3R common-native, amended'] = ordered(checked_cases(mr, ms), 'moonst3r_native', cases)
    mt = totals(methods['MOONSt3R common-native, amended'])
    if any(mt[k] != ms[k] for k in ('cases', 'correct', 'strict', 'wrong')) or ms['original_cases_preserved'] != 71 or ms['supplemental_cases'] != 1:
        raise ValueError('MOONSt3R saved aggregate differs')
    seals.update({str(ma_path): digest(ma_path), str(mr/'summary.json'): digest(mr/'summary.json')})
    er = Path('runs/localization-v17-matchanything-native-development')
    ea_path = Path('reports/native_matchanything_audit_v17.json')
    ea, es = read(ea_path), read(er/'summary.json')
    if not ea['passed'] or ea['records'] != 144 or ea['summary_sha256'] != digest(er/'summary.json'):
        raise ValueError('MatchAnything replay required')
    erows = checked_cases(er, es)
    for name, method in [('MatchAnything-EfficientLoFTR native512', 'matchanything_native'), ('SIFT-first + surface pipeline', 'sift_first_surface')]:
        methods[name] = ordered(erows, method, cases)
        if any(totals(methods[name])[k] != es['results'][method][k] for k in ('cases','correct','strict','wrong')):
            raise ValueError('MatchAnything saved aggregate differs')
    seals.update({str(ea_path): digest(ea_path), str(er/'summary.json'): digest(er/'summary.json')})
    sr = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    sa_path = Path('reports/native_surface_audit_v11.json')
    sa = read(sa_path)
    if not sa['passed'] or sa['results_sha256'] != digest(sr):
        raise ValueError('Surface replay required')
    with gzip.open(sr, 'rt') as f:
        methods['Frozen CNN + surface'] = ordered(json.load(f), 'frozen_surface_warp', cases)
    seals.update({str(sa_path): digest(sa_path), str(sr): digest(sr)})
    # Fixed positional assignment, identical to the public native range plan.
    ids = [c['pair_id'] for c in cases]
    for method, label in [('loma_r','LoMa-R common'),('romav2','RoMa v2 fast/common')]:
        ap = Path('reports') / ('native_loma_comparison_v10.json' if method == 'loma_r' else 'native_romav2_comparison_v10.json')
        audit = read(ap)
        if not audit['passed'] or audit['eligible_pairs'] != 72 or audit['replayed_records'] != 144:
            raise ValueError('Complete modern replay required')
        selected = []
        compact = []
        for folder, evidence, assigned in zip(
                [f'native-v7-{method}-common',f'native-v8-{method}-common-36-36'],
                audit['source_runs'], [ids[:36],ids[36:]]):
            root = Path(artifacts)/folder/'native-comparison'
            files = sorted((root/'cases').glob('*.json'))
            if {p.name:digest(p) for p in files} != evidence['case_sha256']:
                raise ValueError('Preserved modern bytes differ')
            rows = [read(p) for p in files]
            selected += [r for r in rows if r['method'] == method and r['pair_id'] in assigned]
            compact += [r for r in rows if r['method'] == 'compact_orientation' and r['pair_id'] in assigned]
        methods[label] = ordered(selected, method, cases)
        comp = ordered(compact,'compact_orientation',cases)
        for key in ('correct','strict'):
            if totals(methods[label])[key] != audit['results'][method][key]:
                raise ValueError('Modern aggregate differs')
        if totals(methods[label])['wrong'] != audit['results'][method]['estimated']-audit['results'][method]['correct']:
            raise ValueError('Wrong raw count differs')
        if 'Frozen orientation CNN' in methods:
            if any([r['pose_correct'],r['strict_pose_correct']] != [s['pose_correct'],s['strict_pose_correct']] for r,s in zip(comp,methods['Frozen orientation CNN'])):
                raise ValueError('Paired frozen compact correctness differs')
        methods['Frozen orientation CNN'] = comp
        seals[str(ap)] = digest(ap)
    order = ['SIFT-first + surface pipeline','Frozen CNN + surface','RoMa v2 fast/common','Frozen orientation CNN','LoMa-R common','MatchAnything-EfficientLoFTR native512','MOONSt3R common-native, amended']
    table = [dict(method=name, **totals(methods[name])) for name in order]
    group_rows = {name:{g:totals([r for r in rows if r['component']==g]) for g in sorted({c['component'] for c in cases})} for name,rows in methods.items()}
    light_rows = {name:{label:totals([r for r in rows if (r['light_a']==r['light_b']) == same]) for label,same in [('same_rendered_light_index',True),('different_rendered_light_index',False)]} for name,rows in methods.items()}
    intervals = {name:paired_interval(methods['SIFT-first + surface pipeline'],methods[name]) for name in order if name != 'SIFT-first + surface pipeline'}
    result = dict(scope='Saved-only synthesis of independently replayed common-native exposed-development results, not a new metric implementation or fresh evaluation.',
        cases=72, terrain_components=4, stereo_geometries=8, table=table,
        per_component=group_rows, per_light_index=light_rows,
        paired_sift_surface_minus_other=intervals,
        input_seals=seals, reporter_sha256=digest(__file__),
        original_moon_calendar_protocol_completed=False,
        amended_moon_original_rows=71, amended_moon_supplemental_rows=1,
        no_new_matcher_forward=True, calibration_or_test_pixels_loaded=0,
        claims={'learned_upgrade':False,'fresh_generalization':False,'calibrated_acceptance':False,'global_sota':False,'verified_orbital_accuracy':False},
        limitations=['Only four exposed components/eight geometries;72lighting pairs are not independent terrain tests.',
            'Compact surface pipeline uses a four-rotation bank and reference-surface rectification; dense baselines use their declared single-pair profiles.',
            'Generic frozen pretraining versus lunar-trained compact features does not isolate architectural superiority; MOONSt3R exact training overlap is unknown.',
            'RoMa v2 is fast/common, not the uncompleted author precise profile. Remaining required author profiles remain incomplete.',
            'No cross-hardware latency ranking; MOON median is not paired timing.',
            'Raw wrong proposals are not calibrated accepted errors; confidence/nonoverlap/fresh test and transfer remain incomplete.',
            'MOONSt3R has21strict poses versus frozen-surface16; coarse-success ranking does not imply precision dominance.',
            'MOON original calendar budget was exceeded after observed clock/suspension divergence; original71rows remain unchanged; separate public600s amendment completed only one missing case.'])
    output.write_text(json.dumps(result,indent=2)+'\n')
    lines='\n'.join(f"| {r['method']} | {r['correct']}/72 ({100*r['correct']/72:.1f}%) | {r['strict']} | {r['wrong']} | {r['no_pose']} |" for r in table)
    output.with_suffix('.md').write_text(f'''# Native lunar pose results including MOONSt3R

These are all72 **exposed development pairs**, generated from eight geometries in four terrain components. They do not establish fresh-test accuracy. Correct means rotation ≤2° and translation ≤1% of median reference range; strict means the already frozen stricter rule recorded in the native protocol. All methods receive the same reference geometry and final3-native-pixel AP3P pose estimator; query depth/true query pose are scoring only.

| Exact evaluated configuration | Correct raw | Strict /72 | Wrong raw | No pose |
|---|---:|---:|---:|---:|
{lines}

The strongest completed simple pipeline remains **SIFT-first + reference-surface refinement:57/72**. Its advantage over the tested RoMa v2 fast profile is four pairs, which is not evidence of a general SOTA advantage. These are whole configured pipelines: the compact method uses four image rotations and reference-surface rectification, whereas the dense models use their declared single-pair settings. Lunar adaptation and unknown MOONSt3R training overlap further limit attribution. Remaining author-recommended profiles are still outstanding.

MOONSt3R recovers **28/72 raw correct poses, including21 strict**, versus frozen CNN+surface54/72 with16 strict. It is weaker at the main threshold on this dataset, but the strict result prevents a blanket claim that our correspondences are more precise. Neither result is calibrated acceptance. The compact surface pipeline still emits three wrong proposals; SIFT-first emits two. A tested model must improve the complete pose/coverage/error tradeoff before adoption.

## Resource amendment retained

The original four-calendar-hour MOONSt3R run was stopped **incomplete at71/72** after observed clock/suspension divergence:calendar age41,891seconds exceeded14,400seconds, while completed-case inference totalled3,239seconds. The exact cause is not established. A separately published600-calendar-second contract froze the71original row hashes and the single missing case before inference. The supplemental phase completed in42.48seconds and the amended72-case study independently replays. All71original rows remain byte-identical; the original expired protocol is never described as completed or reset. The amendment stays inside the user's unchanged72CPUwall-hour comparison allowance.

## What this means for the new training

The new81,696-parameter coupled-assignment study is a separate prospective experiment. Its toy loss counterexample, exact100-update optimizer replay and software checks are prerequisites, not localization gains. All three seeds and all checkpoint results must be retained; selection includes initialization. If trained models fail to beat the frozen baseline, they will not be adopted. No breakthrough is claimed from matching a limited subset of pretrained configurations.

The JSON companion binds all source summaries/audits/case hashes and reports counts by terrain and light-index equality. It includes an exact256-draw four-component descriptive bootstrap for paired raw success. Those intervals describe exposed development cases and cannot repair the missing fresh-data gate. Light-index equality is not a quantified Sun-angle measurement. No model forward, new raster input, final-test inference or metric/pose-estimator rerun occurs in this reporter.

No fair paired MOON timing or isolated-memory ranking is available. Full fresh terrain validation, calibrated acceptance/nonoverlap, transfer, remaining baseline profiles and independently characterized orbital accuracy remain unresolved.
''')
    print(json.dumps(dict(passed=True,table=table,outputs=[str(output),str(output.with_suffix('.md'))]),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',required=True)
    p.add_argument('--output',default='reports/native_pose_comparison_v29.json')
    a=p.parse_args();report(a.artifacts,a.output)
