"""Render completed measurements without selecting new checkpoints or thresholds."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compact_confidence import accepts,counts,scores
from dataset import sha256

NAMES={'loma_b':'LoMa-B','loma_r':'LoMa-R','romav2':'RoMa v2 (fast512)',
       'roma':'Original RoMa','matchanything':'MatchAnything4x','sift':'SIFT','original':'Original CNN, new confidence'}


def read(path):return json.loads(Path(path).read_text())


def accepted_rows(run,policy):
    rows=read(Path(run)/'results.json')
    for row in rows:
        calculated=accepts(policy,{'features':row['features']})
        if 'accepted' in row and row['accepted']!=calculated:raise ValueError('Published decision differs from frozen policy')
        row['accepted']=calculated
    return rows


def metrics(rows):
    result=counts(rows,[r['accepted'] for r in rows])
    result['raw_correct']=sum(r['correct'] for r in rows)
    result['raw_success']=result['raw_correct']/max(result['positives'],1)
    if not result['correct_accepted']+result['wrong_accepted']:result['accepted_error']=None
    return result


def paired_interval(a,b,bootstrap=2000):
    aa={r['pair_id']:r for r in a if not r['negative']}
    bb={r['pair_id']:r for r in b if not r['negative']}
    if aa.keys()!=bb.keys():raise ValueError('Paired uncertainty needs identical positive cases')
    groups=sorted({r['component'] for r in aa.values()})
    values=[]
    for group in groups:
        keys=[k for k,r in aa.items() if r['component']==group]
        delta=sum(int(aa[k]['accepted'] and aa[k]['correct'])-int(bb[k]['accepted'] and bb[k]['correct']) for k in keys)
        values.append([delta,len(keys)])
    values=np.array(values,float);rng=np.random.default_rng(20261030)
    sampled=values[rng.integers(len(values),size=(bootstrap,len(values)))].sum(1)
    intervals=100*np.percentile(sampled[:,0]/sampled[:,1],[2.5,97.5])
    return dict(gain_pp=float(100*values[:,0].sum()/values[:,1].sum()),
        exploratory_component_bootstrap95_pp=intervals.tolist(),components=len(groups),
        warning='Spatial components share terrain model/rendering pipeline; not guaranteed independent. Negative risk has no independent-pair bound.')


def core_data(protocol):
    selected=read('reports/compact_selection_v2.json');variant=selected['selected_variant']
    series={};all_stats={}
    for seed in selected['all_seeds']:
        model=f'runs/compact-v2-{variant}-{seed}'
        policy=read(protocol['policy_index'][model])
        rows=accepted_rows(model+'-test',policy)
        series[f'compact-{seed}']=rows;all_stats[f'compact-{seed}']=metrics(rows)
    for method in ('original','sift'):
        rows=accepted_rows(f'runs/compact-v2-{method}-test',read(protocol['policy_index'][method]))
        series[method]=rows;all_stats[method]=metrics(rows)
    frozen=[dict(r,accepted=bool(r['original_frozen_accepted'])) for r in series['original']]
    series['original-frozen']=frozen;all_stats['original-frozen']=metrics(frozen)
    return selected,series,all_stats


def modern_data(protocol,directory):
    result={}
    for method in ('loma_b','loma_r','romav2','roma','matchanything'):
        run=Path(directory)/method/'test'
        if not (run/'COMPLETE.json').exists():continue
        if method not in protocol['policy_index']:raise ValueError('Modern method has no frozen operating point')
        result[method]=accepted_rows(run,read(protocol['policy_index'][method]))
    return result


def same_host_timing(executions):
    results={}
    for path in Path(executions).rglob('timing_control.json'):
        baseline=read(path.with_name('predictions.json'))
        lookup={r['pair_id']:r for r in read(path)}
        if not baseline:continue
        method=baseline[0]['method']
        for row in baseline:
            if row['pair_id'] not in lookup:raise ValueError('CPU timing control case missing')
            control=lookup[row['pair_id']]
            results.setdefault(method,[]).append(dict(pair_id=row['pair_id'],
                baseline_ms=row['bidirectional_ms'],compact_ms=control['same_host_bidirectional_ms'],
                ratio=row['bidirectional_ms']/max(control['same_host_bidirectional_ms'],1e-9)))
    return {method:dict(cases=len(rows),median_baseline_ms=float(np.median([r['baseline_ms'] for r in rows])),
        median_compact_ms=float(np.median([r['compact_ms'] for r in rows])),
        median_paired_speedup=float(np.median([r['ratio'] for r in rows]))) for method,rows in results.items()}


def plots(series,common,policies,timing,folder):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'savefig.dpi':180,'axes.grid':True,'grid.alpha':.2})
    primary=series['compact-20261030']
    fig,ax=plt.subplots(figsize=(7,3.2))
    x=np.arange(3);pairs=((0,1),(0,2),(1,2))
    for offset,key,label,color in ((-.15,'original-frozen','Original frozen gate','#788899'),(.15,'compact-20261030','Compact v2','#137f83')):
        values=[]
        for a,b in pairs:
            rows=[r for r in series[key] if not r['negative'] and (r['light_a'],r['light_b'])==(a,b)]
            values.append(100*sum(r['accepted'] and r['correct'] for r in rows)/max(len(rows),1))
        ax.bar(x+offset,values,width=.28,label=label,color=color)
    ax.set(xticks=x,xticklabels=['1 vs2','1 vs3','2 vs3'],ylim=(0,105),ylabel='Correctly accepted positives (%)',xlabel='Source lighting-observation pair (Sun angles unavailable)')
    ax.legend(frameon=False,loc='lower left');fig.tight_layout();fig.savefig(folder/'compact_lighting_v2.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,3.2))
    for method,rows in common.items():
        policy=policies[method];prediction=scores(policy,rows)
        curve=[]
        for threshold in np.unique(prediction[prediction>=0])[::-1]:
            stat=counts(rows,prediction>=threshold)
            if stat['correct_accepted']+stat['wrong_accepted']:
                curve.append((100*stat['correct_acceptance'],100*stat['accepted_error']))
        if curve:
            xx,yy=np.array(curve).T;ax.plot(xx,yy,label=NAMES.get(method,'Compact v2'),linewidth=1.3)
        stat=metrics(rows)
        if stat['accepted_error'] is not None:ax.scatter(100*stat['correct_acceptance'],100*stat['accepted_error'],s=24)
    ax.axhline(1,color='#b55d43',ls='--',lw=1)
    ax.set_yscale('symlog',linthresh=1)
    ax.set(xlim=(0,105),ylim=(0,100),xlabel='Correctly accepted positives (%)',ylabel='Incorrect accepted outputs (%; nonlinear scale)')
    ax.set_yticks([0,1,5,20,100]);ax.set_yticklabels(['0','1','5','20','100'])
    ax.legend(frameon=False,fontsize=8,ncol=2);fig.tight_layout();fig.savefig(folder/'compact_risk_coverage_v2.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,3.2))
    plotted=False
    for method,stat in timing.items():
        if method not in common:continue
        quality=100*metrics(common[method])['correct_acceptance']
        ax.scatter(stat['median_baseline_ms'],quality,s=45)
        ax.annotate(NAMES.get(method,method),(stat['median_baseline_ms'],quality),xytext=(3,5),textcoords='offset points',fontsize=8)
        compact_quality=100*metrics(common['compact'])['correct_acceptance']
        ax.scatter(stat['median_compact_ms'],compact_quality,marker='x',color='#137f83',s=45)
        plotted=True
    if not plotted:
        ax.text(.5,.5,'Same-host CPU timing unavailable\nNo speedup is inferred from different machines.',ha='center',va='center',transform=ax.transAxes)
    else:ax.set_xscale('log')
    ax.set(ylim=(0,110),xlabel='Same-job median bidirectional inference (ms; log scale)',ylabel='Correctly accepted common positives (%)')
    fig.tight_layout();fig.savefig(folder/'compact_accuracy_runtime_v2.png');plt.close(fig)


def gallery(rows,manifest,folder):
    from compact_research import controlled_inputs
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={}
    choices=[('Accepted correct',next((r for r in rows if r['accepted'] and r['correct']),None)),
             ('Rejected nonoverlap',next((r for r in rows if r['negative'] and not r['accepted']),None)),
             ('Remaining failure',next((r for r in rows if not r['negative'] and not(r['accepted'] and r['correct'])),None))]
    fig,axes=plt.subplots(3,2,figsize=(6,8))
    records=[]
    for index,(label,row) in enumerate(choices):
        for ax in axes[index]:ax.axis('off')
        if row is None:
            axes[index,0].text(.1,.5,label+': none observed');continue
        a,b,*_=controlled_inputs(row,tiles,cache)
        axes[index,0].imshow(a,cmap='gray',vmin=0,vmax=255);axes[index,0].set_title(label+' / reference',fontsize=9)
        axes[index,1].imshow(b,cmap='gray',vmin=0,vmax=255);axes[index,1].set_title('Query',fontsize=9)
        records.append(dict(label=label,pair_id=row['pair_id'],accepted=row['accepted'],correct=row['correct'],error_px=row['error_px']))
    fig.tight_layout();fig.savefig(folder/'compact_failure_gallery_v2.png');plt.close(fig)
    return records


def build(protocol_path='reports/compact_freeze_v2.json',modern_directory='runs/compact-v2-modern-test',
          timing_directory='runs/compact-v2-modern-test-executions',output='reports'):
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    protocol=read(protocol_path);selected,series,full_stats=core_data(protocol)
    common_ids={c['pair_id'] for c in protocol['common_cases']}
    common={'compact':[r for r in series['compact-20261030'] if r['pair_id'] in common_ids]}
    primary_model=f'runs/compact-v2-{selected["selected_variant"]}-20261030'
    policies={'compact':read(protocol['policy_index'][primary_model])}
    for method in ('original','sift'):
        common[method]=[r for r in series[method] if r['pair_id'] in common_ids]
        policies[method]=read(protocol['policy_index'][method])
    common.update(modern_data(protocol,modern_directory))
    for method in common:
        if method not in policies:policies[method]=read(protocol['policy_index'][method])
    if any({r['pair_id'] for r in rows}!=common_ids for rows in common.values()):raise ValueError('Comparison case IDs differ')
    timing=same_host_timing(timing_directory)
    plots(series,common,policies,timing,folder)
    comparison=paired_interval(series['compact-20261030'],series['original-frozen'])
    primary=full_stats['compact-20261030']
    targets=dict(model_improvement=bool(comparison['gain_pp']>=10 and primary['accepted_error'] is not None and
        primary['accepted_error']<=.01 and primary['negative_acceptance']<=.01),
        positive_grouped_evidence=bool(comparison['exploratory_component_bootstrap95_pp'][0]>0))
    modern_methods=[m for m in common if m in ('loma_b','loma_r','romav2','roma','matchanything')]
    eligible=[m for m in modern_methods if metrics(common[m])['accepted_error'] is not None and
        metrics(common[m])['accepted_error']<=.01 and metrics(common[m])['negative_acceptance']<=.01]
    competitive=None
    if eligible:
        strongest=max(eligible,key=lambda m:metrics(common[m])['correct_acceptance'])
        candidate=metrics(common['compact']);baseline=metrics(common[strongest])
        delta=100*(candidate['correct_acceptance']-baseline['correct_acceptance'])
        cpu=timing.get(strongest,{}).get('median_paired_speedup')
        budget=candidate['accepted_error'] is not None and candidate['accepted_error']<=.01 and candidate['negative_acceptance']<=.01
        competitive=dict(strongest_completed_at_comparable_error=strongest,gain_pp=delta,
            paired_cpu_speedup=cpu,point_target_met=bool(budget and (delta>0 or (delta>=-3 and cpu is not None and cpu>=5))),
            uncertainty=paired_interval(common['compact'],common[strongest]),
            warning='Small common benchmark; no universal SOTA conclusion. A gain interval spanning zero is inconclusive.')
    targets['competitive_value']=None if competitive is None else competitive['point_target_met']
    sensors={}
    for method in ('compact','original','sift'):
        path=Path('runs')/('compact-v2-'+method+'-sensor')/'summary.json'
        if path.exists():sensors[method]=read(path)
    geometry={}
    for method in ('compact','original','sift'):
        path=Path('runs')/('compact-v2-'+method+'-3d-test')/'summary.json'
        if path.exists():geometry[method]=read(path)
    evidence=dict(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,
        protocol_sha256=sha256(protocol_path),dataset=read(protocol['manifest_path'])['components_by_split'],
        full=full_stats,common={method:metrics(rows) for method,rows in common.items()},
        paired_gain=comparison,targets=targets,competitive_comparison=competitive,same_host_cpu=timing,sensor_nominal=sensors,
        three_dimensional=geometry,
        global_sota_established=False,verified_real_navigation_accuracy=None,
        gallery=gallery(series['compact-20261030'],read(protocol['manifest_path']),folder),
        caveats=['Physical renders, not acquired lunar navigation ground truth.',
            'Common modern comparison has36 positives and36 negatives across12 spatial components.',
            'Confidence calibration has only12 positives and12 negatives across4 components; no1percent risk guarantee.',
            'Negative pairs reuse both component endpoints. Shared terrain model and renderer create further dependence.',
            'Sun-angle metadata absent; lighting-observation IDs cannot be interpreted as fixed angular changes.',
            'Risk-coverage curves are descriptive test curves; operating points were fixed on calibration data.'])
    (folder/'compact_results_v2.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    markdown=make_markdown(evidence)
    (folder/'RESEARCH_ITERATION_6.md').write_text(markdown)
    write_html(evidence,markdown,folder)
    write_pdf(evidence,folder)
    return evidence


def make_markdown(e):
    p=e['full']['compact-20261030'];gain=e['paired_gain'];ci=gain['exploratory_component_bootstrap95_pp']
    lines=['# Compact lunar matching: frozen south-pole study','',
        '**Question:** Can a compact illumination-aware matcher retain more correct alignments while rejecting incorrect outputs?',
        '',f"Selected configuration: **{e['selected']['selected_variant']}**; primary seed20261030, with both other predefined seeds reported.",
        '', '## Full frozen render evaluation','',
        '| Method | Correctly accepted positives | Incorrect accepted outputs | Accepted nonoverlap |',
        '|---|---:|---:|---:|']
    for key,label in [('original-frozen','Original CNN, frozen gate'),('original','Original CNN, new confidence'),
                      ('compact-20261030','Compact v2, seed20261030'),('compact-20261031','Compact v2, seed20261031'),
                      ('compact-20261032','Compact v2, seed20261032'),('sift','SIFT')]:
        r=e['full'][key];lines.append(f"| {label} | {r['correct_accepted']}/{r['positives']} ({100*r['correct_acceptance']:.2f}%) | {r['wrong_accepted']} | {r['negative_accepted']}/{r['negatives']} |")
    lines+=['',f"Primary-minus-original-frozen gain: **{gain['gain_pp']:+.2f} percentage points**; exploratory spatial-component bootstrap95% interval **{ci[0]:+.2f} to {ci[1]:+.2f}**. Twelve components share a terrain model/rendering pipeline; they are not guaranteed independent.",
        '',f"Predeclared10-point improvement/error-budget target met: **{e['targets']['model_improvement']}**. Positive grouped evidence: **{e['targets']['positive_grouped_evidence']}**.",
        '', '## Identical common-input comparison','',
        'All rows use the same36 positive and36 nonoverlap cases. Method-specific internal resolution and lunar training differ; this is not a reproduction of any paper benchmark.','',
        '| Method | Raw positive success | Correct acceptance | Incorrect accepted outputs | Nonoverlap accepted |',
        '|---|---:|---:|---:|---:|']
    for method,r in e['common'].items():
        lines.append(f"| {NAMES.get(method,'Compact v2')} | {r['raw_correct']}/{r['positives']} | {r['correct_accepted']}/{r['positives']} | {r['wrong_accepted']} | {r['negative_accepted']}/{r['negatives']} |")
    lines+=['','![Lighting observations](compact_lighting_v2.png)','',
        '![Descriptive risk and coverage](compact_risk_coverage_v2.png)','',
        '![Same-job CPU accuracy and runtime](compact_accuracy_runtime_v2.png)','',
        'CPU speedups are computed only from paired same-job controls; missing controls do not support speedup claims.',
        '', '## Three-dimensional matching','',
        'Raw correspondences are extracted before any planar fitting. Pose uses reference-map depth and predicted query pixels; query reference depth participates only in correspondence scoring, never in pose estimation. The secondary simulated pose threshold is2 degrees and1% of reference ray range.','']
    for method,r in e['three_dimensional'].items():
        lines.append(f"- {NAMES.get(method,'Compact v2')}: {r.get('poses_within_secondary_threshold',0)}/{r['cases']} poses meet the secondary criterion; {r['poses_estimated']}/{r['cases']} poses estimated. Match coverage: {r['correct_correspondences']}/{r['scorable_matches']} reference-scorable raw matches within3px, out of {r['returned_matches']} returned matches. These different correspondence coverages are not a method ranking.")
    lines+=['', '## Sensor transfer and failures','',
        'NASA POLAR uses new lighting configurations of previously inspected laboratory terrains. Nominal fixed-camera agreement has unknown drift/calibration uncertainty. It is not verified lunar navigation accuracy.','']
    for method,r in e['sensor_nominal'].items():
        lines.append(f"- {NAMES.get(method,'Compact v2')}: {r['accepted_nominal_within3px']}/{r['cases']} accepted within nominal3px; {r['accepted_nominal_outside3px']} accepted outside the nominal criterion.")
    lines+=['','![Success, rejection and failure examples](compact_failure_gallery_v2.png)','', '## Limits','']
    lines+=['- '+text for text in e['caveats']]
    lines+=['','No current-global-SOTA, NASA/JPL affiliation or flight-readiness claim is made. Earlier development and Tycho reserve results remain separate.',
        '', 'Sources: [MoonAnything](https://zenodo.org/records/18415119), [LoMa](https://www.davnords.com/loma), [RoMa v2](https://arxiv.org/abs/2511.15706), [NASA POLAR](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/).','']
    return '\n'.join(lines)


def write_html(e,markdown,folder):
    rows=''.join(f"<tr><td>{html.escape(NAMES.get(m,'Compact v2'))}</td><td>{r['correct_accepted']}/{r['positives']}</td><td>{r['wrong_accepted']}</td><td>{r['negative_accepted']}/{r['negatives']}</td></tr>" for m,r in e['common'].items())
    text='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compact lunar matcher - frozen study</title><style>body{font:17px/1.6 system-ui;background:#f2f4f7;color:#17283a;margin:auto;max-width:1000px;padding:35px}section{background:white;padding:24px;margin:20px 0;border-radius:14px}h1{font-size:34px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}small{color:#536475}</style><h1>Compact lunar matching</h1><p>Frozen south-pole render study. Real navigation accuracy remains unverified.</p><section><h2>Same inputs, measured outcomes</h2><table><tr><th>Method</th><th>Correct accepted positives</th><th>Incorrect accepted</th><th>Nonoverlap accepted</th></tr>'+rows+'</table><small>36 positives and36 negatives across12 spatial components. Original frozen-gate results and all three seeds are in the full report.</small></section>'
    for title,image in [('Lighting observations','compact_lighting_v2.png'),('Error versus coverage','compact_risk_coverage_v2.png'),('CPU tradeoff','compact_accuracy_runtime_v2.png'),('Success, rejection and remaining failure','compact_failure_gallery_v2.png')]:
        text+=f'<section><h2>{title}</h2><img src="{image}" alt="{title}"></section>'
    text+='<section><h2>Limits</h2><ul>'+''.join('<li>'+html.escape(c)+'</li>' for c in e['caveats'])+'</ul><a href="compact_brief_v2.pdf">Two-page research brief</a> | <a href="RESEARCH_ITERATION_6.md">Full report</a> | <a href="compact_results_v2.json">Measured results</a></section></html>'
    (folder/'compact_research_v2.html').write_text(text)


def write_pdf(e,folder):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
    story=[]
    def p(text,style='BodyText'):story.append(Paragraph(text,styles[style]));story.append(Spacer(1,7))
    def table(rows):
        t=Table(rows,hAlign='LEFT',colWidths=[205,100,90,90])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17364b')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef3f6'),colors.white])]))
        story.append(t);story.append(Spacer(1,10))
    p('Compact lunar image matching','Title')
    p('Frozen south-pole render study | Independent project; no NASA/JPL affiliation')
    p('<b>Question.</b> Can a small illumination-aware matcher accept more correct alignments while rejecting incorrect matches, with a useful CPU tradeoff against larger models?')
    p(f"<b>Design.</b> 28 spatial components:8 training,4 model selection,4 confidence calibration and12 final test across three polar sectors. Configuration: {e['selected']['selected_variant']}. Primary seed20261030; all three predefined seeds retained. Reference geometry comes from simulated depth and cameras.")
    rows=[['Full frozen render test','Correct accepted','Wrong accepted','Negative accepted']]
    for key,label in [('original-frozen','Original CNN / frozen gate'),('original','Original CNN / new confidence'),('compact-20261030','Compact v2 / primary'),('compact-20261031','Compact v2 / seed20261031'),('compact-20261032','Compact v2 / seed20261032')]:
        r=e['full'][key];rows.append([label,f"{r['correct_accepted']}/{r['positives']}",str(r['wrong_accepted']),f"{r['negative_accepted']}/{r['negatives']}"])
    table(rows)
    gain=e['paired_gain'];ci=gain['exploratory_component_bootstrap95_pp']
    p(f"<b>Measured gain.</b> {gain['gain_pp']:+.2f} percentage points versus the original frozen gate; exploratory grouped95% interval {ci[0]:+.2f} to {ci[1]:+.2f}. Predeclared10-point/error-budget target met: {e['targets']['model_improvement']}. Shared terrain/rendering dependence limits interpretation.")
    story.append(Image(str(folder/'compact_accuracy_runtime_v2.png'),width=480,height=220))
    p('Runtime comparisons use paired same-job CPU controls only. Common modern comparisons contain36 positives and36 negatives; these are not published-paper benchmark percentages.')
    story.append(PageBreak())
    p('Comparison, transfer and limits','Heading1')
    rows=[['Identical common test','Raw correct','Correct accepted','Wrong accepted']]
    for method,r in e['common'].items():rows.append([NAMES.get(method,'Compact v2'),f"{r['raw_correct']}/36",f"{r['correct_accepted']}/36",str(r['wrong_accepted'])])
    table(rows)
    story.append(Image(str(folder/'compact_lighting_v2.png'),width=480,height=195))
    story.append(Image(str(folder/'compact_risk_coverage_v2.png'),width=480,height=195))
    p('<b>Sensor transfer.</b> New NASA POLAR laboratory lighting configurations are scored against nominal fixed-camera identity. Reference drift/calibration uncertainty remains unknown. Real orbital navigation accuracy is unverified.')
    p('<b>Limits.</b> Calibration has only12 positives and12 negatives across4 components; an observed1% error budget is not a deployment guarantee. Negative pairs reuse both endpoints. Sun-angle metadata is absent. Risk curves are descriptive; operating points were fixed before test. No global state-of-the-art or flight-readiness claim.')
    p('Sources: MoonAnything DOI10.5281/zenodo.18415119; LoMa (ECCV2026); RoMa v2, arXiv2511.15706; NASA Ames POLAR. Repository contains case-level results, audits, licenses and reproduction commands.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536475'))
        canvas.drawString(42,24,'Independent lunar matching study | reproducible results, failures retained')
        canvas.drawRightString(553,24,str(doc.page))
    SimpleDocTemplate(str(folder/'compact_brief_v2.pdf'),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=37).build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default='reports/compact_freeze_v2.json')
    p.add_argument('--modern-directory',default='runs/compact-v2-modern-test');p.add_argument('--timing-directory',default='runs/compact-v2-modern-test-executions');p.add_argument('--output',default='reports')
    a=p.parse_args();build(a.protocol,a.modern_directory,a.timing_directory,a.output)
