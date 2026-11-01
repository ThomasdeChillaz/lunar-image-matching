"""Report completed native MatchAnything comparison without fresh/SOTA claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval


def report(run,audit_path,prefix):
    root=Path(run);prefix=Path(prefix);audit=json.loads(Path(audit_path).read_text());summary=json.loads((root/'summary.json').read_text())
    if not audit['passed'] or audit['summary_sha256']!=sha256(root/'summary.json'):raise ValueError('Complete independent audit required')
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['bank_only'] or audit['eligible_pairs']!=72:raise ValueError('All72 common development pairs required')
    rows=[json.loads(p.read_text()) for p in sorted((root/'cases').glob('*.json'))]
    methods=protocol['methods'];by={m:{r['pair_id']:r for r in rows if r['method']==m} for m in methods}
    a,b=[by[m] for m in methods];pairs=protocol['case_ids']
    intervals=paired_interval(a,b)
    groups=sorted({r['component'] for r in a.values()})
    per_component={g:{m:dict(cases=len(part),correct=sum(r['pose_correct'] for r in part)) for m in methods
        if (part:=[r for r in by[m].values() if r['component']==g])} for g in groups}
    ratios=np.array([a[i]['elapsed_seconds']/b[i]['elapsed_seconds'] for i in pairs])
    value=dict(scope='Full native512 exposed-development comparison; no fresh final, calibrated acceptance, transfer or SOTA claim.',
        results=summary['results'],matchanything_minus_sift_surface_gain_pp=100*(summary['results'][methods[0]]['correct']-summary['results'][methods[1]]['correct'])/72,
        descriptive_paired_component_interval_pp=intervals,median_paired_matchanything_to_sift_surface_latency_ratio=float(np.median(ratios)),
        per_component=per_component,provenance=json.loads((root/'provenance.json').read_text()),
        gained=[i for i in pairs if a[i]['pose_correct'] and not b[i]['pose_correct']],
        lost=[i for i in pairs if b[i]['pose_correct'] and not a[i]['pose_correct']],
        source_sha256=sha256(__file__),audit_sha256=sha256(audit_path),summary_sha256=sha256(root/'summary.json'),
        limits=['72 correlated lighting pairs, eight geometries/four previously exposed terrain components.',
            'Local lunar training versus generic pretrained weights is not a matched adaptation comparison.',
            'Native512 retains the same original field of view;832 positional-encoding profile is a separate pending comparison.',
            'Raw correct poses and wrong estimates are not calibrated accepted poses; no1%risk claim.',
            'Joint process memory includes both models/controls and cannot rank isolated model memory.',
            'Uncertainty is descriptive after repeated exposed-development hypothesis/model selection.'])
    for ext in ('.json','.md','.png'):
        if prefix.with_suffix(ext).exists():raise FileExistsError('Preserve report snapshots')
    write_new(prefix.with_suffix('.json'),value)
    lines=['# Native512 MatchAnything versus the compact surface baseline','',
        'Both methods receive the same full512px image content, reference axial geometry and3-original-pixel AP3P pose estimator. Query depth and query pose enter scoring only. Every eligible case is included. These72 pairs are eight exposed geometries/four terrain components, not a fresh final test.','',
        '| Method | Correct raw /72 | Strict /72 | Wrong raw | Median full inference (s) |',
        '|---|---:|---:|---:|---:|']
    for method in methods:
        r=summary['results'][method];lines.append(f'| {method} | {r["correct"]} | {r["strict"]} | {r["wrong"]} | {r["median_full_inference_seconds"]:.3f} |')
    lines+=['',f'MatchAnything minus the SIFT-first surface baseline: {value["matchanything_minus_sift_surface_gain_pp"]:+.2f} percentage points; descriptive paired terrain-component interval [{intervals[0]:.2f}, {intervals[1]:.2f}]. This interval uses only four exposed groups after extensive development and is not confirmatory evidence.', '',
        f'Median paired MatchAnything/SIFT-first latency ratio: {np.median(ratios):.2f}. Both models are loaded/warmed, order alternates and compact features are not reused across pairs. This ratio describes this CPU/profile and pipeline. Historical speed ratios from other workers or compact variants do not apply.', '',
        'MatchAnything uses pinned released weights and unchanged author matching confidence/fine matching, with no planar affine/homography prefilter. Its native512 NPE settings are explicit; the832 profile is a separate pending control, not an assumed author-recommended lunar configuration. Raw wrong poses remain visible. No rejection rule is fitted here.', '',
        'The comparison is reproducible and independently replays native geometry metrics and all144 common-estimator pose fits. It does not establish fresh geographic generalization, low-error accepted localization or orbital accuracy. MOONSt3R, remaining author profiles, transfer and calibrated acceptance remain incomplete.','']
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(14,4.6));x=np.arange(2)
    axes[0].bar(x,[summary['results'][m]['correct'] for m in methods],label='correct raw')
    axes[0].bar(x,[summary['results'][m]['wrong'] for m in methods],bottom=[summary['results'][m]['correct'] for m in methods],label='wrong raw',color='#e45756')
    axes[0].set(xticks=x,xticklabels=['MatchAnything','SIFT + surface'],ylim=(0,72),ylabel='Pairs /72',title='Complete common-pose denominator');axes[0].legend(fontsize=8)
    for i,m in enumerate(methods):
        axes[1].plot(np.arange(len(groups)),[per_component[g][m]['correct']/per_component[g][m]['cases'] for g in groups],marker='o',label=['MatchAnything','SIFT + surface'][i])
    axes[1].set(xticks=np.arange(len(groups)),xticklabels=[str(i+1) for i in range(len(groups))],ylim=(0,1.05),xlabel='Exposed terrain component',ylabel='Raw pose success fraction',title='Only four dependent groups');axes[1].legend(fontsize=8)
    axes[2].scatter([b[i]['elapsed_seconds'] for i in pairs],[a[i]['elapsed_seconds'] for i in pairs],s=16,alpha=.7)
    limits=[min(r['elapsed_seconds'] for r in rows),max(r['elapsed_seconds'] for r in rows)]
    axes[2].plot(limits,limits,color='grey',linestyle='--');axes[2].set(xlabel='SIFT + surface full inference (s)',ylabel='MatchAnything full inference (s)',title='Same-worker paired CPU measurements')
    fig.suptitle('Native512 exposed-development comparison — no accepted-pose or SOTA claim');fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=170);plt.close(fig)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--prefix',required=True)
    args=parser.parse_args();report(args.run,args.audit,args.prefix)
