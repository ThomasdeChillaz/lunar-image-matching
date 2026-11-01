"""Explain measured likelihood/geometry disagreement without model-gain claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import sha256
from lunargeo_dataset import write_new


def report(prefix):
    prefix=Path(prefix)
    root=Path('runs/localization-v17-pointwise-proposal-diagnosis')
    oracle_root=Path('runs/localization-v17-oracle-filter-diagnosis')
    diagnostic=json.loads((root/'summary.json').read_text());oracle=json.loads((oracle_root/'summary.json').read_text())
    audit_path=Path('reports/native_context_proposal_audit_v17.json');oa_path=Path('reports/native_oracle_filter_audit_v17.json')
    audit=json.loads(audit_path.read_text());oa=json.loads(oa_path.read_text())
    if not audit['passed'] or audit['summary_sha256']!=sha256(root/'summary.json') or not oa['passed'] or oa['summary_sha256']!=sha256(oracle_root/'summary.json'):
        raise ValueError('Complete independent diagnostic audits required')
    cohort=Path('runs/localization-v16-global-context-development');primary=[]
    for variant in ('pointwise','self_context','cross_context'):
        meta=json.loads((cohort/f'{variant}-20261101/training.json').read_text())
        final=next(v for v in meta['validation'] if v['step']==2000)
        primary.append(dict(variant=variant,selected_step=meta['selected_step'],selected_correct=meta['evaluation']['correct_poses'],
            final_trained_correct=final['selection']['correct_poses'],final_trained_wrong=final['selection']['estimated_poses']-final['selection']['correct_poses']))
    result=dict(scope='Development failure diagnosis; no positive model, acceptance or fresh-validation claim.',
        proposal_summary=diagnostic['summary'],primary_runs=primary,oracle_probes=oracle['summary'],
        diagnosis='Likelihood improves while usable proposals decrease on both training fixtures and exposed selection. This supports testing an objective change; it does not prove a unique cause or that the proposed loss will help.',
        sources=dict(proposal_audit=sha256(audit_path),oracle_audit=sha256(oa_path),source=sha256(__file__)),
        next_hypothesis='Search-window positives, genuine frozen-proposal anchors, and a trainable coarse encoder with unchanged frozen native fine descriptors. Core/tests only; no new fitting/result.',
        limits=['All72 selection pairs are previously exposed, across four terrain components.',
            'Proposal totals pool four correlated orientation branches, not independent landmarks/terrain tests.',
            'Training fixtures are metadata-selected first geometry per component,108 light pairs, not the full training population.',
            'Oracle filters/branch choices use query truth and are not deployable performance or a universal upper bound.',
            'New loss/model module is untrained and must pass actual initialization/pilot/source-freeze gates.'])
    write_new(prefix.with_suffix('.json'),result)
    lines=['# Why the context study did not improve lunar poses','',
        'All three primary configurations retained initialization at54/72. Their final trained checkpoints give47/72 pointwise,43/72 self-context and39/72 cross-context. These final checkpoints are failed experiments, not selected upgrades.','',
        '| Pointwise checkpoint | Training NLL | Selection NLL | Training reachable proposals | Selection reachable proposals |',
        '|---:|---:|---:|---:|---:|']
    for step in (0,500,2000):
        train=next(r for r in diagnostic['summary'] if r['step']==step and r['split']=='train')
        selection=next(r for r in diagnostic['summary'] if r['step']==step and r['split']=='model_selection')
        lines.append(f'| {step} | {train["mean_target_nll"]:.3f} | {selection["mean_target_nll"]:.3f} | {train["reachable"]:,} | {selection["reachable"]:,} |')
    lines+=['','Reachable means the independently projected visible point lies inside the actual rounded33px native query search. It does not guarantee successful refinement or pose. Totals pool four correlated orientation branches. Every proposal was checked against the actual matcher, and independent geometry replay covers all2,160 records. Likelihood aggregates replay; no independent neural reimplementation is claimed.','',
        'The likelihood improves while geometrically useful proposals decrease on both the training fixtures and selection cases. This is consistent with a mismatch between the current objective and useful matching. It does not identify a unique causal mechanism. A unit-controlled counterexample shows that bilinear cell labels can penalize an alternative coarse candidate from which native refinement could reach the true point.','',
        'The truth-dependent filtering probe examines all60 native-refined branches on the15 missing poses and the three wrong returned poses. Across the tested3/4/8/12px oracle filters, at most two additional cases yield any correct pose. Perfectly preserving the54 known-correct outputs and choosing oracle branches would produce56/72, but that requires query truth and is not a matcher result or universal upper bound. Independent replay covers all252 oracle fits. Filtering these saved point sets alone does not demonstrate the required gain.','',
        'The next hypothesis tests search-window positive sets, anchors for independently geometry-labelled genuine frozen proposals, and coarse-image feature learning while the native fine encoder stays frozen. The prepared model has81,696–82,208 total parameters. Its numerical/gradient controls pass; no new fitting or localization gain exists yet. See SEARCH_WINDOW_V18.md for the pending actual-forward and fitting gates.','',
        'Fresh final data, low-error calibrated acceptance/nonoverlap, remaining native pretrained comparisons, transfer, timing and the paper remain unfinished. No paid compute, human review or email.','']
    if prefix.with_suffix('.md').exists() or prefix.with_suffix('.png').exists():raise FileExistsError('Preserve report snapshots')
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for split,label in [('train','training fixtures'),('model_selection','selection')]:
        part=[r for r in diagnostic['summary'] if r['split']==split]
        axes[0].plot([r['step'] for r in part],[r['mean_target_nll'] for r in part],marker='o',label=label)
        axes[1].plot([r['step'] for r in part],[r['reachable']/r['pairs'] for r in part],marker='o',label=label)
    axes[0].set(title='Likelihood improves',xlabel='Pointwise optimizer updates',ylabel='Bilinear target NLL (lower is better)');axes[0].legend(fontsize=8)
    axes[1].set(title='Usable proposal support decreases',xlabel='Pointwise optimizer updates',ylabel='Reachable proposals /pair, four branches');axes[1].legend(fontsize=8)
    names=['frozen','pointwise','self context','cross context'];values=[54]+[r['final_trained_correct'] for r in primary]
    axes[2].bar(names,values,color=['grey','#4c78a8','#f58518','#54a24b']);axes[2].set(title='Final trained checkpoints regress',ylabel='Correct raw poses /72',ylim=(0,72))
    for i,value in enumerate(values):axes[2].text(i,value+.7,str(value),ha='center')
    axes[2].tick_params(axis='x',labelsize=8)
    fig.suptitle('Exposed-development diagnosis — initialization retained; no learned gain')
    fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',required=True);args=parser.parse_args();report(args.prefix)
