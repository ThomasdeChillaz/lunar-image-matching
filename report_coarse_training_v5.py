"""Report all corrected coarse-learning runs and a stronger simple fallback."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def paired_interval(candidate,baseline,seed=20261101):
    groups=defaultdict(list)
    for key,row in candidate.items():groups[row['component']].append(int(row['pose_correct'])-int(baseline[key]['pose_correct']))
    values=list(groups.values());rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(10000):
        draw=[values[i] for i in rng.integers(len(values),size=len(values))]
        samples.append(100*sum(map(sum,draw))/sum(map(len,draw)))
    return np.percentile(samples,[2.5,97.5]).tolist()


def report(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'summary.json').read_text())
    audit=json.loads((out/'coarse_training_audit_v5.json').read_text())
    if not audit['passed'] or audit['summary_sha256']!=sha256(root/'summary.json'):raise ValueError('Completed cohort audit required')
    source=Path('runs/localization-v5-axial-correction/results.json.gz')
    with gzip.open(source,'rt') as f:rows=json.load(f)
    sift={r['pair_id']:r for r in rows if r['method']=='sift_native' and r['solver']=='ap3p_reclassify'}
    frozen={r['pair_id']:r for r in rows if r['method']=='quarter_turn_bank' and r['selected']}
    manifest=json.loads(Path('reports/localization_training_data_v5.json').read_text())
    # Keep failed orientation-bank cases in the denominator, including no pose.
    for c in manifest['cases']:
        if c['split']=='model_selection' and c['pair_id'] not in frozen:
            frozen[c['pair_id']]=dict(c,pose=None,pose_correct=False,strict_pose_correct=False)
    expected=set(frozen)
    if len(expected)!=72 or set(sift)!=expected:raise ValueError('Common baseline cases differ')
    fallback={key:sift[key] if sift[key]['pose'] is not None else row for key,row in frozen.items()}
    baseline_count=sum(r['pose_correct'] for r in frozen.values());fallback_count=sum(r['pose_correct'] for r in fallback.values())
    table=[];models=[];candidates={};strata=[]
    for entry in summary['results']:
        folder=root/f"{entry['variant']}-{entry['seed']}";meta=json.loads((folder/'training.json').read_text());models.append(meta)
        with gzip.open(folder/'development_results.json.gz','rt') as f:part=json.load(f)
        candidate={r['pair_id']:r for r in part}
        if set(candidate)!=expected or len(part)!=72:raise ValueError('Candidate common cases differ')
        for mode in sorted({r['tile_id'].split('/')[0] for r in part}):
            for same in (True,False):
                ids=[r['pair_id'] for r in part if r['tile_id'].split('/')[0]==mode and (r['light_a']==r['light_b'])==same]
                strata.append(dict(variant=entry['variant'],seed=entry['seed'],mode=mode,same_light_id=same,cases=len(ids),
                    frozen_correct=sum(frozen[i]['pose_correct'] for i in ids),trained_correct=sum(candidate[i]['pose_correct'] for i in ids)))
        mixed={key:sift[key] if sift[key]['pose'] is not None else r for key,r in candidate.items()}
        n=meta['evaluation']['correct_poses'];nf=sum(r['pose_correct'] for r in mixed.values())
        estimated=meta['evaluation']['estimated_poses'];ef=sum((r.get('pose') if 'pose' in r else r['estimated_cam2world']) is not None for r in mixed.values())
        row=dict(variant=entry['variant'],seed=entry['seed'],selected_step=meta['selected_step'],
            correct=n,strict=meta['evaluation']['strict_poses'],estimated=estimated,wrong=estimated-n,
            sift_fallback_correct=nf,sift_fallback_estimated=ef,sift_fallback_wrong=ef-nf,
            raw_gain_vs_frozen_pp=100*(n-baseline_count)/72,
            descriptive_component_interval_pp=paired_interval(candidate,frozen),
            fallback_gain_vs_frozen_fallback_pp=100*(nf-fallback_count)/72,
            fallback_descriptive_component_interval_pp=paired_interval(mixed,fallback),
            completed_steps=meta['completed_steps'],final_weight_change_l2=meta['final_weight_change_l2'],
            trainable_parameters=meta['trainable_parameters'],total_parameters=meta['total_parameters'],elapsed_seconds=meta['elapsed_seconds'],
            selected_checkpoint_sha256=sha256(folder/'selected.pt'),results_sha256=sha256(folder/'development_results.json.gz'))
        table.append(row);candidates[(entry['variant'],entry['seed'])]=candidate
    result=dict(scope='Corrected axial-depth development only; eight geometries/four components, no final or calibrated acceptance claim.',
        winner=summary['winner'],runs=table,baseline_correct=baseline_count,baseline_fallback_correct=fallback_count,
        baseline_fallback_wrong=sum(r['pose'] is not None and not r['pose_correct'] for r in fallback.values()),
        total_optimizer_steps=sum(r['completed_steps'] for r in table),elapsed_cohort_seconds=summary['elapsed_seconds'],
        interval_note='Descriptive post-selection terrain-component bootstrap on only four exposed development components, not fresh-test confidence.',
        strata=strata,source_sha256=sha256(__file__),input_sha256={str(source):sha256(source),str(root/'summary.json'):sha256(root/'summary.json')})
    write_new(out/'coarse_training_comparison_v5.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    seeds=[r for r in table if r['variant']==summary['winner']]
    labels=['Frozen']+[str(r['seed'])[-2:] for r in seeds]
    x=np.arange(len(labels));width=.36
    axes[0].bar(x-width/2,[baseline_count]+[r['correct'] for r in seeds],width,label='Four orientations',color='#54758c')
    axes[0].bar(x+width/2,[fallback_count]+[r['sift_fallback_correct'] for r in seeds],width,label='SIFT-first fallback',color='#218f82')
    axes[0].set_xticks(x,labels);axes[0].set(ylim=(0,76),ylabel='Correct raw camera poses / 72',xlabel='Predefined seed suffix',title='Frozen and every selected-configuration seed')
    axes[0].legend(frameon=False,loc='lower left')
    for i,values in enumerate(([baseline_count]+[r['correct'] for r in seeds],[fallback_count]+[r['sift_fallback_correct'] for r in seeds])):
        for j,v in enumerate(values):axes[0].text(j+(-width/2 if i==0 else width/2),v+.7,str(v),ha='center',fontsize=9)
    for meta in models:
        axes[1].plot([h['step'] for h in meta['history']],[h['selection']['correct_poses'] for h in meta['history']],marker='o',label=f"{meta['variant']} / {str(meta['seed'])[-2:]}")
    axes[1].set(ylim=(-.5,16.5),xlabel='Optimizer step',ylabel='Correct poses / 16 selection pairs',title='Selection includes the initial checkpoint')
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Corrected-depth development training — raw poses, no calibrated acceptance')
    fig.savefig(out/'coarse_training_outcomes_v5.png',dpi=180);plt.close(fig)
    lines='\n'.join(f"| {r['variant']} | {r['seed']} | {r['selected_step']} | {r['correct']}/72 | {r['strict']}/72 | {r['wrong']} | {r['sift_fallback_correct']}/72 | {r['sift_fallback_wrong']} |" for r in table)
    primary=next(r for r in table if r['variant']==summary['winner'] and r['seed']==20261101)
    interval=primary['descriptive_component_interval_pp']
    text=f'''# Corrected axial-depth coarse-learning study

This study tests whether fine-tuning the compact coarse proposal encoder on true 3D pairs helps camera localization beyond the frozen four-orientation engineering baseline. Earlier ray-based results and training labels remain superseded by [the geometry correction](GEOMETRY_CORRECTION_V5.md). Only the corrected v5 data and geometry are used here.

## Completed experiment

All four runs finished 4,000 optimizer steps each ({result['total_optimizer_steps']:,} total): planar control and true-3D training at seed20261101, followed by the configuration selected on the fixed16-case bank at seeds20261102/03. Every run retains initial, selected, final and optimizer/RNG checkpoints, failed later checkpoints and executed source snapshots. The coarse network has28,272 trainable parameters, with a separate frozen28,272-parameter native encoder:56,544 combined.

| Training | Seed | Selected step | Correct raw poses | Stricter poses | Wrong raw poses | With SIFT fallback | Wrong fallback poses |
|---|---:|---:|---:|---:|---:|---:|---:|
{lines}

The frozen four-orientation baseline gives **{baseline_count}/72** correct raw poses. A stronger simple engineering control uses native SIFT whenever its pose estimator returns a pose, falling back to the frozen orientation bank otherwise: **{fallback_count}/72**, with **{result['baseline_fallback_wrong']} wrong poses**. The fallback uses no query truth or correctness labels for its decision, but was developed after looking at development evidence; it is not independent validation. Every learned candidate receives the identical fallback rule.

The primary selected candidate gives **{primary['correct']}/72**, a change of **{primary['raw_gain_vs_frozen_pp']:+.2f} percentage points** over the frozen orientation bank. Its descriptive component-bootstrap interval is [{interval[0]:+.2f},{interval[1]:+.2f}] points. These intervals follow selection on only four exposed development components with unknown trajectory and shared renderer dependence. They are not an unbiased final-test estimate. All predefined seeds are reported; no best-seed headline is used.

![All seeds, engineering fallback and selection history](coarse_training_outcomes_v5.png)

## Interpretation and next hypothesis

Pose success, not descriptor loss, selects checkpoints. The initial checkpoint is eligible; selecting step0 is a failure to improve through training, not a new learned result. Longer fitting can reduce pose recovery even when a final batch has lower contrastive loss. Native refinement and orientation search predate this training, so their gain is not attributed to it.

The matched training ablations use uniform reference points with exact corrected axial-depth/camera targets. The planar control uses same-camera cross-light pairs with known warps; true3D uses other-camera images. Both use +/-90degree rotation,0.7–1.4scale,translation and identical sensor perturbations. Real rendered lighting variation remains separate from exposure/noise augmentation. Training samples components uniformly, then a tile and lighting pair. No matcher-selected correspondence is treated as truth.

A focused next hypothesis is that this very wide orientation training wastes the compact encoder's capacity when inference already tries four orientations. Test a geometry-canonicalized training orientation against the same uncanonicalized control, with unchanged architecture, learning rate, fixed inference, selection bank and step budget. Training-only query truth may build supervision and choose data transformations; it must never enter deployed inference or branch selection. This is a hypothesis, not an explanation established by the current experiment.

## Scope and verification

Training uses18 geometries/12 corrected spatial components. Model selection uses8 geometries/four other components,72 repeated lighting pairs and the fixed16-case bank. No calibration/test pixels were loaded. The original fresh-final data gate remains unmet. Correctness is rotation<=2degrees and translation<=1% actual median Euclidean reference range; stricter means0.5degree/0.1%. Missing and wrong poses stay in the denominator.

These are **raw estimates, not correctly accepted pose coverage**. The70% accepted-pose target, <=1% accepted-pose error/nonoverlap rate,10-point learned improvement over the strongest simple baseline, positive fresh paired interval and modern native-pose advantage remain unestablished. No SOTA, flight or verified orbital-navigation claim is made. Runtime here includes selection/evaluation and cache reuse, so is not an inference speed comparison.

Independent replay verifies completed optimizer budgets, actual trained weight changes, source/cache/weight hashes, all checkpoints and predefined seeds,10,297 uniform supported training points, native correspondence metrics, physical range, camera-to-world pose errors and all288 saved candidate cases. The previously saved correction inputs remain unchanged. Cohort wall time: {summary['elapsed_seconds']/60:.2f} minutes on two configured CPU threads, below the48-hour training cap. No paid compute, human review or automatic outreach.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python train_coarse_localization_v5.py --output runs/new-v5-coarse --steps 4000 --interval 1000
python verify_coarse_training_v5.py --run runs/new-v5-coarse --output runs/new-v5-coarse-audit.json
```

Report generation additionally needs the frozen correction capsule's baseline correspondences. Use new output paths; completed artifacts are immutable. Own code/weights are MIT; source renders derive from MoonAnything LunarGeo (Grethen et al., doi:10.5281/zenodo.18415119), CC BY4.0.
'''
    (out/'COARSE_TRAINING_V5.md').write_text(text)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v5-coarse-development')
    p.add_argument('--output',default='reports');a=p.parse_args();report(a.run,a.output)
