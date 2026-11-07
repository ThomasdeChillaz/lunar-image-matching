"""Present audited all-Sun-pair outcomes without changing models or policies."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from dataset import sha256
from physical_lighting_analysis import SUN_EDGES, SUPPORT_EDGES, bin_index
from uncertainty_groups import paired_registration_interval

LABELS = {'ncc':'NCC', 'sift':'SIFT', 'disk_lightglue':'DISK + LightGlue',
    'disk_lightglue_clahe':'DISK + LightGlue + CLAHE', 'disk_lightglue_lunar':'Lunar adapter'}
COLORS = {'ncc':'#8056a3', 'sift':'#bc9475', 'disk_lightglue':'#2d83b8',
    'disk_lightglue_clahe':'#91b8c7', 'disk_lightglue_lunar':'#34a698'}


def strata(rows,methods):
    result = []
    for metric,edges in [('solar_separation_deg',SUN_EDGES),('shared_nondark_fraction',SUPPORT_EDGES)]:
        for index,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
            for method in methods:
                selected = [r for r in rows if r['method'] == method and bin_index(r[metric],edges) == index]
                result.append(dict(metric=metric,lower=low,upper=high,bin_index=index,method=method,pairs=len(selected),
                    source_acquisitions=len({r['observation_group'] for r in selected}),
                    correct=sum(r['correct'] for r in selected),returned=sum(r['estimated'] for r in selected),
                    no_fit=sum(not r['estimated'] for r in selected),
                    success_rate=sum(r['correct'] for r in selected)/len(selected) if selected else None))
    return result


def paired_interval(rows,manifest,repetitions=5000):
    return paired_registration_interval(rows,manifest,repetitions)


def render(folder,audit,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder,audit,output = Path(folder),Path(audit),Path(output)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    verified = json.loads(audit.read_text())
    if (sha256(folder/'results.json') != complete['results_sha256']
            or verified['results_sha256'] != complete['results_sha256']
            or sha256(folder/'summary.json') != verified['summary_sha256']
            or sha256(folder/'frozen_inputs.json') != verified['frozen_inputs_sha256']):
        raise ValueError('A matching completed geometry/pixel audit is required')
    frozen = json.loads((folder/'frozen_inputs.json').read_text())
    rows = json.loads((folder/'results.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    if len(rows) != verified['evaluations'] or complete['pairs'] != verified['pairs']:
        raise ValueError('Audited counts differ')
    methods = frozen['methods']
    grouped = strata(rows,methods)
    interval = paired_interval(rows,frozen['manifest'])
    output.mkdir(parents=True,exist_ok=True)
    if audit.resolve() != (output/'crosslighting_audit.json').resolve():
        shutil.copyfile(audit,output/'crosslighting_audit.json')
    for name,values,exclude in [('crosslighting_results.csv',rows,{'reference_points','query_points'}),('crosslighting_strata.csv',grouped,set())]:
        fields = sorted(set().union(*(v.keys() for v in values))-exclude)
        with (output/name).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for value in values:
                writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in value.items() if k in fields})
    (output/'crosslighting_summary.json').write_text(json.dumps(dict(methods=summary,paired_interval=interval,strata=grouped),indent=2,allow_nan=False))
    fig,axes = plt.subplots(1,2,figsize=(12,5.2),sharey=True)
    for ax,metric,label in zip(axes,['solar_separation_deg','shared_nondark_fraction'],
            ['3D Sun-direction separation (degrees)','Shared non-dark terrain (%) at I/F > .01; intensity proxy']):
        labels = []
        for method in methods:
            selected = [g for g in grouped if g['metric'] == metric and g['method'] == method]
            ax.plot(range(len(selected)),[g['success_rate']*100 if g['success_rate'] is not None else np.nan for g in selected],
                marker='o',color=COLORS[method],label=LABELS[method])
            if not labels:
                scale = 1 if metric == 'solar_separation_deg' else 100
                labels = [f"{g['lower']*scale:g}–{g['upper']*scale:g}\nn={g['pairs']}\nacq={g['source_acquisitions']}" for g in selected]
        ax.set_xticks(range(len(labels)),labels,fontsize=8);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2);ax.set_xlabel(label,fontsize=9)
    axes[0].set_ylabel('Correct alignments within 3 native pixels (%)')
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=5,fontsize=8)
    fig.suptitle(f"All {complete['pairs']} physical Sun pairs; identical translation (6, −4) throughout",fontsize=12)
    fig.text(.5,.02,f"Exploratory follow-up on {verified['source_acquisitions']} previously inspected Tycho acquisitions; all failed fits retained.",ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.87]);fig.savefig(output/'crosslighting_strata.png',dpi=170);plt.close(fig)
    baseline = {r['pair_id']:r for r in rows if r['method'] == 'disk_lightglue'}
    above_90 = [r for r in baseline.values() if r['solar_separation_deg'] >= 90]
    lines = ['# Physical lighting with fixed geometry', '',
        f"All {complete['pairs']} Sun pairs from {verified['source_acquisitions']} previously inspected Tycho acquisitions were evaluated with {len(methods)} unchanged methods ({len(rows):,} method evaluations). Every combination of the nine source lighting renders is included. The same known translation (6,-4) and zero rotation is applied to every case; gamma, gain and noise remain unchanged. NCC is therefore within its translation model. The learned feature methods retain their original shared partial-affine estimator.", '',
        'The complete design, source images, model and original development policies were frozen before these new outcomes. Neither retraining nor test-based threshold selection occurs. This is exploratory follow-up on previously inspected terrain, not a fresh geographic holdout. The earlier anchor-light experiment and its negative real-image transfer results remain unchanged.', '',
        f"Sun separation spans {min(r['solar_separation_deg'] for r in baseline.values()):.2f}–{max(r['solar_separation_deg'] for r in baseline.values()):.2f}°. There are {len(above_90)} cases at 90° or more. The original anchor-light experiment stopped at 77.94° on test cases. Raw non-dark support is measured on corresponding terrain before the known shift; I/F>.01 is an intensity proxy, not a shadow annotation.", '',
        '| Method | Correct / pairs | Returned / pairs | Median error among returned (px) | Dark-input pairs |', '|---|---:|---:|---:|---:|']
    for value in summary:
        median = f"{value['median_error_returned_px']:.2f}" if value['median_error_returned_px'] is not None else '—'
        lines.append(f"| {LABELS[value['method']]} | {value['correct']}/{value['pairs']} | {value['returned']}/{value['pairs']} | {median} | {value['dark_input_pairs']} |")
    low,high = interval['percentile_interval_95']
    lines += ['',f"The adapter-minus-frozen success difference is {interval['delta']*100:.2f} percentage points; an exploratory bootstrap interval is [{low*100:.2f}, {high*100:.2f}] percentage points over {interval['groups']} components joining overlapping footprints and shared acquisitions. Cases share terrain and images. This interval does not establish independent real-image transfer or global lunar superiority.", '',
        '![Fixed-geometry Sun and support strata](crosslighting_strata.png)', '',
        '| Sun separation | Pairs | Acquisitions | NCC correct | SIFT correct | Frozen correct | CLAHE correct | Adapter correct |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in [g for g in grouped if g['metric'] == 'solar_separation_deg' and g['method'] == 'disk_lightglue']:
        values = {g['method']:g for g in grouped if g['metric'] == group['metric'] and g['bin_index'] == group['bin_index']}
        lines.append(f"| {group['lower']:g}–{group['upper']:g}° | {group['pairs']} | {group['source_acquisitions']} | "+' | '.join(str(values[m]['correct']) for m in methods)+' |')
    lines += ['', '## Transported rejection policies', '',
        'The policies below were selected on the original physical development cases and are applied unchanged. They are not recalibrated on these follow-up outcomes. Counts concern same-terrain positives; original known non-overlap negatives remain a separate experiment.', '',
        '| Method | Correct fits retained | Incorrect fits accepted |', '|---|---:|---:|']
    for value in summary:
        counts = value.get('transported_rejection')
        lines.append(f"| {LABELS[value['method']]} | {counts['accepted_correct']}/{counts['correct']} | {counts['accepted_incorrect']} |" if counts else f"| {LABELS[value['method']]} | No original development policy | — |")
    true_fraction = float(np.hypot(frozen['warp']['dx'],frozen['warp']['dy'])/frozen['image_size'])
    policy_limit = frozen['original_physical_policies']['disk_lightglue']['policy']['max_translation_fraction']
    lines += ['', f"A concrete failure is visible in the original frozen learned policy: its translation limit is {policy_limit:g} of the image size, while the exact imposed translation has magnitude {true_fraction:.5f}. The correct transform itself exceeds that selected bound. This helps explain rejection of valid fits and shows that the old policy does not transport reliably to this design. The threshold is left unchanged; a replacement needs development-only design and a fresh evaluation, not selection from these inspected outcomes.", '']
    lines += ['', 'Failures to return a fit stay in correctness denominators. A low error median among returned fits does not compensate for missing fits. Dark inputs remain explicit failures when unmatchable; no stable terrain coordinates are invented from darkness.', '',
        'Evidence: [per-method outcomes](crosslighting_results.csv), [all strata](crosslighting_strata.csv), [summary and paired uncertainty](crosslighting_summary.json), [pixel/geometry audit](crosslighting_audit.json). The audit reconstructs the known image translation by array slicing, checks exact image/mask hashes and recomputes every saved error and gate decision.', '',
        'Source data: Grethen et al., [MoonAnything](https://arxiv.org/html/2604.00682v1), [Zenodo 18415119](https://zenodo.org/records/18415119), CC BY 4.0. Independent human landmarks and real-data rejection calibration remain open. No NASA/JPL affiliation or navigation certification is implied.']
    (output/'CROSS_LIGHTING_REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'crosslighting_report_provenance.json').write_text(json.dumps(dict(renderer_sha256=sha256(__file__),
        audit_sha256=sha256(audit),results_sha256=complete['results_sha256'],frozen_inputs_sha256=verified['frozen_inputs_sha256'],
        grouping_source_sha256=sha256(Path(__file__).with_name('uncertainty_groups.py')),
        bins_source_sha256=sha256(Path(__file__).with_name('physical_lighting_analysis.py')),
        report_sha256=sha256(output/'CROSS_LIGHTING_REPORT.md')),indent=2))
    return dict(pairs=complete['pairs'],evaluations=len(rows),above_90_pairs=len(above_90),methods=summary,paired_interval=interval,
        true_translation_fraction=true_fraction,frozen_learned_translation_limit=policy_limit)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/physical-crosslighting-v1')
    p.add_argument('--audit',default='reports/crosslighting_audit.json')
    p.add_argument('--output',default='reports')
    args = p.parse_args()
    print(json.dumps(render(args.run,args.audit,args.output),indent=2))
