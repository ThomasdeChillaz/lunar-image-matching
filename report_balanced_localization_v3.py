"""Report the supplemental balanced cohort without replacing primary findings."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(root='runs/localization-v3-balanced-development'):
    root=Path(root); out=Path('reports')
    audit=json.loads((out/'localization_balanced_audit_v3.json').read_text())
    if not audit['passed']:raise ValueError('Independent audit required')
    cohort=json.loads((root/'summary.json').read_text())
    runs=[json.loads(p.read_text()) for p in root.glob('*/training.json')]
    order={'descriptor_3d':0,'refinement':1,'reliability':2}
    runs.sort(key=lambda r:(r['seed']!=20261101,order[r['variant']],r['seed']))
    primary=next(r for r in runs if r['variant']==cohort['winner'] and r['seed']==20261101)
    with gzip.open(root/f'{primary["variant"]}-{primary["seed"]}'/'development_results.json.gz','rt') as f:
        rows=json.load(f)
    baseline_meta=json.loads((out/'localization_pose_refit_v3.json').read_text())
    with gzip.open(Path(baseline_meta['source_run'])/'results.json.gz','rt') as f:original=json.load(f)
    baseline={}
    for refit in baseline_meta['rows']:
        old=original[refit['source_row']]
        if old['experiment']=='deterministic_native':
            key=old['tile_id']+f'/lights-{old["light_a"]}-{old["light_b"]}'
            baseline[key]=refit['pose_correct']
    if set(baseline)!={r['pair_id'] for r in rows}:raise ValueError('Case sets differ')
    components=sorted({r['component'] for r in rows}); rng=np.random.default_rng(20261101)
    totals={c:(sum(int(r['pose_correct'])-int(baseline[r['pair_id']]) for r in rows if r['component']==c),
               sum(r['component']==c for r in rows)) for c in components}
    bootstrap=[]
    for _ in range(2000):
        chosen=rng.choice(components,len(components),replace=True)
        bootstrap.append(100*sum(totals[c][0] for c in chosen)/sum(totals[c][1] for c in chosen))
    interval=np.percentile(bootstrap,[2.5,97.5]).tolist()
    seeds=[r['evaluation']['correct_poses'] for r in runs if r['variant']==cohort['winner']]
    table=[dict(variant=r['variant'],seed=r['seed'],selected_step=r['selected_step'],
                correct_poses=r['evaluation']['correct_poses'],total_parameters=r['total_parameters'],
                positive_labels=r['positive_local_labels'],sampled_examples=r['sampled_examples'],
                last_bank_correct=r['history'][-1]['selection']['correct_poses'],
                last_bank_returned=r['history'][-1]['selection']['returned_matches'],
                strict_correct_poses=r['evaluation']['strict_correct_poses'],
                returned_matches=r['evaluation']['returned_matches'],
                scorable_matches=r['evaluation']['scorable_matches'],
                correct_correspondences_at_native_px=r['evaluation']['correct_correspondences_at_native_px'],
                final_weight_change_l2=r['final_weight_change_l2']) for r in runs]
    baseline_count=sum(baseline.values()); count=primary['evaluation']['correct_poses']
    improved=[r['pair_id'] for r in rows if r['pose_correct'] and not baseline[r['pair_id']]]
    regressed=[r['pair_id'] for r in rows if not r['pose_correct'] and baseline[r['pair_id']]]
    result=dict(scope='Supplemental development tuning after primary results, not fresh validation.',
                baseline_correct=baseline_count,cases=len(rows),primary_variant=cohort['winner'],
                primary_correct=count,seeds=seeds,paired_gain_pp=100*(count-baseline_count)/len(rows),
                exploratory_component_interval_pp=interval,components=len(components),
                improved_pair_ids=improved,regressed_pair_ids=regressed,runs=table,
                total_optimizer_steps=sum(r['completed_steps'] for r in runs),
                elapsed_cohort_seconds=cohort['elapsed_seconds'])
    (out/'localization_balanced_comparison_v3.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    names=['Frozen native\n+ LM']; values=[baseline_count]
    for r in runs:
        if r['seed']!=20261101:continue
        names.append(r['variant'].replace('_','\n')+ ('\n(initial retained)' if r['selected_step']==0 else ''))
        values.append(r['evaluation']['correct_poses'])
    axes[0].bar(names,values,color=['#647789','#17888e','#17888e','#be7954'])
    for i,v in enumerate(values):axes[0].text(i,v+1,f'{v}/72',ha='center')
    axes[0].set(ylabel='Correct raw poses / 72',ylim=(0,76),title='Balanced-sampling ablations')
    axes[1].bar([str(r['seed'])[-2:] for r in runs if r['variant']==cohort['winner']],seeds,color='#17888e')
    axes[1].axhline(baseline_count,color='#647789',linestyle='--',label='Frozen native + LM')
    for i,v in enumerate(seeds):axes[1].text(i,v+1,f'{v}/72',ha='center')
    axes[1].set(xlabel='Predefined seed suffix',ylabel='Correct raw poses / 72',ylim=(0,76),title='Selected configuration: every seed')
    axes[1].legend(frameon=False)
    fig.suptitle('Supplemental development study: four selected terrain components',fontsize=12)
    fig.savefig(out/'localization_balanced_outcomes_v3.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    for r in runs:
        if r['seed']==20261101:
            ax.plot([h['step'] for h in r['history']],[h['selection']['correct_poses'] for h in r['history']],marker='o',label=r['variant'])
    ax.set(xlabel='Optimizer step',ylabel='Correct poses / 16 selection-bank pairs',ylim=(-.5,16.5),title='All supplemental checkpoint selection outcomes')
    ax.legend(frameon=False);fig.savefig(out/'localization_balanced_selection_v3.png',dpi=180);plt.close(fig)
    lines='\n'.join(f'| {r["variant"]} | {r["seed"]} | {r["selected_step"]} | {r["correct_poses"]}/72 | {r["last_bank_correct"]}/16 | {r["positive_labels"]}/{r["sampled_examples"]} |' for r in table)
    fallbacks=[r for r in table if r['selected_step']==0]
    precision=100*primary['evaluation']['correct_correspondences_at_native_px']['2']/max(primary['evaluation']['scorable_matches'],1)
    fallback_text=(' Initial weights were retained for '+', '.join(f'{r["variant"]}/{r["seed"]}' for r in fallbacks)+
                   '; these entries are fallbacks, not successful trained improvements.') if fallbacks else ''
    text=f'''# Supplemental balanced-sampling study

This follow-up was chosen after the primary cohort exposed a low positive-label fraction and reliability overrejection. It is further tuning on the same development terrain, not preregistered independent confirmation. The [original five runs](LOCALIZATION_TRAINING_V3.md) remain unchanged: selected descriptor seeds 26/72, 25/72 and 26/72, against the deterministic native+LM baseline's 26/72.

Only the sampler changes: each batch has two geometrically valid local matches and two invalid proposals, rather than approximately 26% positives. Architecture, initialization, augmentation, 2,000-step/batch-four budget, physical lighting renders, pose estimator and 0.5 point-score cutoff remain the same. No calibration/test pixels are loaded. The random sample stream necessarily changes.

| Variant | Seed | Selected step | Full development poses | Last checkpoint bank poses | Positive labels / samples |
|---|---:|---:|---:|---:|---:|
{lines}

The configuration selected using the fixed 16-case bank is **{cohort['winner']}**. Its predefined primary seed recovers **{count}/72** raw poses, versus **{baseline_count}/72** for frozen native refinement + LM: **{result['paired_gain_pp']:+.2f} percentage points**. All three selected-configuration seeds give **{', '.join(str(s)+'/72' for s in seeds)}**. There are {len(improved)} improved and {len(regressed)} regressed pairs versus the baseline.{fallback_text}

The primary candidate returns {primary['evaluation']['returned_matches']:,} points; {primary['evaluation']['scorable_matches']:,} have co-visible geometric references. Among those scorable points, {precision:.2f}% are within 2 native pixels. It recovers {primary['evaluation']['strict_correct_poses']}/72 poses at the stricter 0.5-degree / 0.1%-range criterion. Co-visible point precision is conditional and must not hide invisible returned points or failed poses. Full 1/2/4/8/12-pixel counts for every seed are in localization_balanced_comparison_v3.json.

The post-selection descriptive component-bootstrap interval is [{interval[0]:+.2f}, {interval[1]:+.2f}] percentage points across only four development footprint components. Shared terrain/renderer and unknown trajectory dependence remain. A degenerate interval when case outcomes coincide does not imply equivalence elsewhere. No unbiased final-test interval, new SOTA result or reliable navigation claim follows from these data.

![Balanced ablations and all seeds](localization_balanced_outcomes_v3.png)
![All checkpoint selection outcomes](localization_balanced_selection_v3.png)

All five runs completed 2,000 steps, {cohort['elapsed_seconds']/60:.2f} wall-clock minutes including evaluation on two configured CPU threads. The two cohorts total 20,000 optimizer steps plus the separately retained 100-step pilot. Actual weight changes, failed checkpoints, initial/selected/final weights and executed source snapshots are retained. Model sizes including frozen coarse stay at 56,544 / 63,505 / 64,098 parameters.

These are raw pose results, not correctly accepted pose coverage: no calibrated pair-level acceptance policy exists. A 0.5 local matchability cutoff under balanced training is not a probability calibrated to the natural proposal prior. The default development command continues to return pose:null/transform:null/accepted:false, with raw correspondence proposals opt-in. Wrong-pose and nonoverlap acceptance targets, modern native-pose comparisons and fresh transfer evaluation remain unestablished.

Selection uses correct raw pose count on the same metadata-selected 16 pairs, then 2-pixel correspondence count and earliest step. Initial weights are eligible. All 72 pairs (8 geometries / 4 components, repeated lighting) are expanded development readouts. The primary seed is fixed; no best-of-seed headline is used. Success means rotation <=2 degrees and translation <=1% median reference range. Every eligible pair remains in the denominator.

Independent replay validates 234 geometric supervision cases, 360 candidate outcomes, source/weight hashes, complete optimizer budgets and selection/seed completeness for this cohort. See localization_balanced_audit_v3.json. Sources and measured artifacts are distributed with SHA256 checksums; [commands and scope](../LOCALIZATION_TRAINING.md) explain reproduction. The original fresh-data gate still governs any future generalization claim. No human review, paid compute or automatic outreach was used.
'''
    (out/'LOCALIZATION_BALANCED_V3.md').write_text(text)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':report()
