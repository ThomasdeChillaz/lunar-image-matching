"""Render the measured development training outcomes, including failed heads."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256


def report(root='runs/localization-v3-development'):
    root=Path(root);out=Path('reports')
    audit=json.loads((out/'localization_training_audit_v3.json').read_text())
    if not audit['passed']:raise ValueError('Independent training audit required')
    cohort=json.loads((root/'summary.json').read_text())
    models=[json.loads(p.read_text()) for p in root.glob('*/training.json')]
    models.sort(key=lambda d:(d['seed']!=20261101,('descriptor_3d','refinement','reliability').index(d['variant']),d['seed']))
    primary=next(d for d in models if d['variant']==cohort['winner'] and d['seed']==20261101)
    primary_path=root/f'{primary["variant"]}-{primary["seed"]}'
    with gzip.open(primary_path/'development_results.json.gz','rt') as f:primary_rows=json.load(f)
    old=json.loads((out/'localization_pose_refit_v3.json').read_text())
    with gzip.open(Path(old['source_run'])/'results.json.gz','rt') as f:old_rows=json.load(f)
    baseline={}
    for refined in old['rows']:
        original=old_rows[refined['source_row']]
        if original['experiment']=='deterministic_native':
            key=original['tile_id']+f'/lights-{original["light_a"]}-{original["light_b"]}'
            baseline[key]=dict(correct=refined['pose_correct'],component=original['component'])
    if set(baseline)!={r['pair_id'] for r in primary_rows}:raise ValueError('Baseline cases differ')
    components=sorted({r['component'] for r in primary_rows});rng=np.random.default_rng(20261101)
    totals={c:[sum(r['pose_correct']-baseline[r['pair_id']]['correct'] for r in primary_rows if r['component']==c),
                sum(r['component']==c for r in primary_rows)] for c in components}
    delta=[]
    for _ in range(2000):
        chosen=rng.choice(components,len(components),replace=True)
        delta.append(100*sum(totals[c][0] for c in chosen)/sum(totals[c][1] for c in chosen))
    interval=np.percentile(delta,[2.5,97.5]).tolist()
    baseline_count=sum(v['correct'] for v in baseline.values())
    seeds=[d['evaluation']['correct_poses'] for d in models if d['variant']==cohort['winner']]
    table=[]
    for d in models:
        table.append(dict(variant=d['variant'],seed=d['seed'],selected_step=d['selected_step'],
                          total_parameters=d['total_parameters'],correct_poses=d['evaluation']['correct_poses'],
                          last_bank_correct=d['history'][-1]['selection']['correct_poses'],
                          last_bank_returned=d['history'][-1]['selection']['returned_matches'],
                          positives=d['positive_local_labels'],sampled=d['sampled_examples'],
                          final_weight_change_l2=d['final_weight_change_l2']))
    result=dict(scope='Development-only, selected on these terrain groups; no fresh/final evaluation.',
                baseline_correct=baseline_count,cases=72,primary_variant=cohort['winner'],
                primary_correct=primary['evaluation']['correct_poses'],seeds=seeds,
                mean_seed_correct=float(np.mean(seeds)),paired_gain_pp=100*(primary['evaluation']['correct_poses']-baseline_count)/72,
                exploratory_component_interval_pp=interval,components=len(components),
                note='Post-selection descriptive bootstrap on four dependent development components, not an unbiased final-test interval.',
                runs=table,total_optimizer_steps=sum(d['completed_steps'] for d in models),
                elapsed_cohort_seconds=cohort['elapsed_seconds'])
    (out/'localization_training_comparison_v3.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    display=dict(descriptor_3d='3D-trained\ndescriptor',refinement='Learned\nrefinement',
                 reliability='Reliability\n(initial retained)')
    labels=['Frozen +\nnative + LM']+[display[d['variant']] for d in models if d['seed']==20261101]
    values=[baseline_count]+[d['evaluation']['correct_poses'] for d in models if d['seed']==20261101]
    axes[0].bar(labels,values,color=['#647789','#17888e','#17888e','#be7954'])
    for i,v in enumerate(values):axes[0].text(i,v+1,f'{v}/72',ha='center')
    axes[0].set(ylabel='Correct raw camera poses / 72',ylim=(0,76),title='Learning has not established an advantage')
    seed_labels=[str(d['seed'])[-2:] for d in models if d['variant']==cohort['winner']]
    axes[1].bar(seed_labels,seeds,color='#17888e');axes[1].axhline(baseline_count,color='#647789',linestyle='--',label='Frozen native baseline')
    for i,v in enumerate(seeds):axes[1].text(i,v+1,f'{v}/72',ha='center')
    axes[1].set(xlabel='Predefined seed suffix',ylabel='Correct raw camera poses / 72',ylim=(0,76),title='All predefined winner seeds')
    axes[1].legend(frameon=False)
    fig.suptitle('Development only: 8 geometries / 4 footprint components',fontsize=12)
    fig.savefig(out/'localization_training_outcomes_v3.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for d in models:
        if d['seed']!=20261101:continue
        axes[0].plot([h['step'] for h in d['history']],[h['selection']['correct_poses'] for h in d['history']],marker='o',label=d['variant'])
    axes[0].set(xlabel='Optimizer step',ylabel='Correct poses / 16 selection-bank pairs',ylim=(-.5,16.5),title='Checkpoint selection retains failures')
    axes[0].legend(frameon=False)
    strata=defaultdict(lambda:[0,0,0])
    for r in primary_rows:
        label='Same lighting ID' if r['light_a']==r['light_b'] else 'Different lighting IDs'
        c=strata[label];c[0]+=1;c[1]+=baseline[r['pair_id']]['correct'];c[2]+=r['pose_correct']
    x=np.arange(len(strata));width=.35
    axes[1].bar(x-width/2,[100*c[1]/c[0] for c in strata.values()],width,label='Frozen native',color='#647789')
    axes[1].bar(x+width/2,[100*c[2]/c[0] for c in strata.values()],width,label='Primary trained',color='#17888e')
    axes[1].set_xticks(x,list(strata));axes[1].set(ylabel='Correct raw pose rate (%)',ylim=(0,100),title='Lighting IDs are not Sun angles')
    axes[1].legend(frameon=False)
    fig.savefig(out/'localization_training_selection_v3.png',dpi=180);plt.close(fig)
    lines='\n'.join(f'| {r["variant"]} | {r["seed"]} | {r["selected_step"]} | {r["correct_poses"]}/72 | {r["last_bank_correct"]}/16 | {r["total_parameters"]:,} |' for r in table)
    text=f'''# Compact-v3 development training - iteration 8

The user requested training after the earlier fresh-data stop. This explicitly authorized development-only fitting on the original train groups. The historical data gate still governs fresh final validation. No calibration/test pixels were loaded and no old checkpoint/result was overwritten.

## Completed work and measured outcomes

Five runs completed 2,000 optimizer steps each: three ablations with seed 20261101, followed by the selected configuration with seeds20261102/20261103. Every run changed its final weights (independently replayed L2 norms), saved initial/selected/final checkpoints, retained failed selection outcomes and stayed below 100,000 total parameters including the frozen coarse network. A separately retained 100-step pilot preceded the common configuration.

| Candidate | Seed | Selected optimizer step | All development poses | Last-checkpoint selection poses | Total parameters |
|---|---:|---:|---:|---:|---:|
{lines}

The **unchanged frozen descriptor with deterministic native refinement and LM recovers {baseline_count}/72** on identical cases. The primary trained candidate recovers **{primary['evaluation']['correct_poses']}/72**, a difference of **{result['paired_gain_pp']:+.2f} percentage points**. The selected configuration's three outcomes are {', '.join(str(n)+'/72' for n in seeds)}; their mean is {np.mean(seeds):.2f}/72. Training has not established a consistent pose advantage over the simple baseline.

The exploratory component-resampling interval is [{interval[0]:+.2f}, {interval[1]:+.2f}] points. The primary candidate succeeds on exactly the same 26 cases as the baseline, making this resampling interval degenerate. It does not establish precise equivalence outside these observed cases. It is descriptive, after model selection, on only four selected development footprint components with shared renderer/unknown trajectory dependence, not a fresh-test interval or proof of independence. The model-improvement/70%-pose research targets have not been established.

![Ablations and seed variation](localization_training_outcomes_v3.png)
![Checkpoint outcomes and lighting strata](localization_training_selection_v3.png)

## What was learned

The full learned reliability head retained its initial checkpoint: later checkpoints recovered zero poses on the 16-case selection bank at the predeclared 0.5 point-score cutoff. This failure is visible in the table and curves. Its selected 26/72 result is the untrained-head fallback, not a successful newly learned rejection rule. The final trained weights are also distributed for diagnosis.

The prior engineering improvement remains important: native correspondence refinement and pose refitting recovered 26/72, compared with coarse+LM 3/72. That gain predates this training and is not attributed to the new network. The new descriptor-only training and pair-conditioned residual head did not improve the primary 72-case count under this budget.

For seed 20261101, {table[0]['positives']:,}/{table[0]['sampled']:,} sampled examples had a co-visible correct correspondence inside the 33px search window. All-negative batches have no spatial loss in the descriptor-only/refinement ablations; reliability still has its binary loss. The fixed0.5 point cutoff and this class balance are plausible contributors to overrejection, not proven sole causes. Last-batch losses are not smooth training curves and are not used as evidence of convergence.

The subsequent [balanced-sampling cohort](LOCALIZATION_BALANCED_V3.md) tests positive/negative balancing without replacing these results. Further development should measure point-score calibration and ranking rather than assume 0.5 is useful, and inspect depth/visibility uncertainty before attempting finer coordinate supervision. Compare these changes against the frozen native+LM baseline and retain every predefined seed. A fresh independently characterized reserve is still needed for a method paper's generalization claim.

## Data, geometry and selection

Training uses eighteen geometries from eight original spatial components; model selection uses eight other geometries from four components. All lighting variants stay with their geometry; batch/trajectory independence remains unverified. The native cache contains 162 training and 72 selection stereo/light cases, with 52,613 coarse train proposals. Labels derive from source cameras/ray depth and the existing visibility convention (0.05 m + 1% relative range tolerance); these tolerances are not independent sensor uncertainty estimates. Query depth/true query pose never enter refiner or pose inference. Map-aided pose fitting uses reference depth/pose and query intrinsics.

The 16-case selection bank was chosen by metadata before training and covers all eight selection geometries. It selects checkpoints/configuration by correct unfiltered poses, then 2-original-pixel correspondence count, then earliest step. The initial checkpoint is eligible. All 72 cases are an expanded development readout; both sets were previously inspected and neither is a new test. The primary seed was predefined; no best-of-seed headline is used.

The criterion remains rotation error <=2 degrees and translation <=1% of median reference range. Missing/incorrect poses remain in the72-pair denominator. This is a simulated diagnostic, not a mission requirement. Local score filtering is not calibrated pair acceptance. No new false-acceptance, deployment risk, SOTA or verified orbital-navigation claim is supported.

## Software verification and distribution

145 repository tests passed/four optional tests skipped, including native-crop/whole-image feature agreement, subpixel mode selection, finite supervised/no-match gradients, split leakage prevention and checkpoint tamper rejection. A fresh procedural smoke produced 20 summaries. Actual trained-candidate known-shift/blank/default-rejection controls are in localization_training_controls_v3.json. Independent audit checks 234 supervised geometry/light cases, 360 saved candidate development outcomes, checkpoint/ablation selection, five completed training budgets and all winner seeds.

See [training and inference commands](../LOCALIZATION_TRAINING.md). Download separate own-model, supervised-data and experiment archives from the development release, with their SHA256 sidecars. Weights/source images are excluded from Git. Source data: MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY 4.0. Own model/code weights are MIT. No third-party pretrained weights, private correspondence, human annotation or paid compute is included.

No new calibrated pair-level policy is fitted. The development command returns transform:null, pose:null, accepted:false and empty correspondences by default. --include-proposals explicitly requests native-pixel research correspondences; --checkpoint final selects final rather than development-selected weights. The existing production matching command and previous frozen policies remain unchanged.

The cohort took {cohort['elapsed_seconds']/60:.2f} wall-clock minutes including selection/evaluation on two configured CPU threads, within the 48-hour training cap. This is experiment duration, not a matcher runtime comparison. Sources, caches, checkpoints and case lists are hashed; reproduction does not depend on matching-derived correctness labels.
'''
    (out/'LOCALIZATION_TRAINING_V3.md').write_text(text)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':report()
