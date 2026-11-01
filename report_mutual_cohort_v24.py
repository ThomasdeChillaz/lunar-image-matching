"""Report every completed coupled-assignment seed after actual audit/controls."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from report_native_moonst3r_v29 import paired_interval,totals


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with gzip.open(path,'rt')as f:return json.load(f)


def report(root,audit_path,controls_root,output):
    root,controls_root,out=Path(root),Path(controls_root),Path(output)
    if out.exists()or out.with_suffix('.md').exists():raise FileExistsError('Preserve earlier complete reports')
    protocol=read(root/'protocol.json');summary=read(root/'summary.json');audit=read(audit_path);controls=read(controls_root/'summary.json')
    if (not audit['passed']or audit['partial']or audit['completed_runs']!=3 or audit['recorded_updates']!=6000
            or audit['replayed_pose_rows']!=1080 or audit['protocol_sha256']!=sha(root/'protocol.json')
            or audit['full_optimizer_retraining_performed']or audit['independent_new_loss_pilot_retraining_steps']!=100):
        raise ValueError('Full new-loss cohort audit required; do not overstate independent retraining')
    if (not controls['passed']or controls['selected_models']!=3 or controls['cohort_summary_sha256']!=sha(root/'summary.json')
            or not controls['records'][0]['actual_primary_forward_exact']or controls['records'][0]['actual_primary_forward_cases']!=72):
        raise ValueError('Actual selected-model and complete primary forward checks required')
    complete=read(root/'COMPLETE.json')
    if complete['summary_sha256']!=sha(root/'summary.json')or complete['protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('Completed study seal changed')
    base_path=Path('runs/localization-v11-surface-warp-development/results.json.gz');base_audit=read('reports/native_surface_audit_v11.json')
    if not base_audit['passed']or base_audit['results_sha256']!=sha(base_path):raise ValueError('Verified original54-case pipeline required')
    base=rows(base_path);ids={r['pair_id']for r in base};records=[]
    for saved in summary['results']:
        folder=root/f'{saved["variant"]}-{saved["seed"]}';meta=read(folder/'training.json');seal=read(folder/'COMPLETE.json')
        for name,digest in seal['sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Completed seed artifact changed')
        selected_path=folder/f'selection_step_{saved["selected_step"]}.json.gz';selected=rows(selected_path)
        final=rows(folder/'selection_step_2000.json.gz')
        if ({r['pair_id']for r in selected}!=ids or {r['pair_id']for r in final}!=ids or len(selected)!=72 or len(final)!=72):
            raise ValueError('All eligible pairs must remain in selected and final denominators')
        count=totals(selected);last=totals(final)
        if (meta['completed_steps']!=2000 or meta['total_parameters']!=81696 or meta['trainable_parameters']!=53424
                or count['correct']!=saved['evaluation']['correct_poses']or count['strict']!=saved['evaluation']['strict_poses']
                or meta['selected_rows_sha256']!=sha(selected_path)):
            raise ValueError('Audited selection/parameter count differs')
        records.append(dict(seed=saved['seed'],selected_step=saved['selected_step'],selected=count,final=last,
            validation=meta['validation'],paired_selected_minus_frozen=paired_interval(selected,base),
            per_component={g:totals([r for r in selected if r['component']==g])for g in sorted({r['component']for r in base})},
            checkpoint_sha256=sha(folder/'selected.pt'),selected_rows_sha256=sha(selected_path),
            final_rows_sha256=sha(folder/'selection_step_2000.json.gz'),fitting_seconds=saved['fitting_seconds']))
    if [r['seed']for r in records]!=[20261101,20261102,20261103]:raise ValueError('All predefined seeds required')
    if any(r['selected_step']!=0 or r['selected']!=totals(base) for r in records):raise ValueError('This negative-study report requires all selected initializations to equal the frozen baseline')
    # This historical primary has the same architecture/data/optimizer budget;
    # later old repeat seeds belonged to a different variant and are not paired.
    old=Path('runs/localization-v18-window-development/coarse_adaptation-20261101')
    old_meta=read(old/'training.json');old_protocol=Path('runs/localization-v18-window-development/protocol.json')
    if protocol['matched_control_protocol_sha256']!=sha(old_protocol):raise ValueError('Historical matched control changed')
    old_seal=read(old/'COMPLETE.json')
    for name,digest in old_seal['sha256'].items():
        if sha(old/name)!=digest:raise ValueError('Retained old-loss primary artifacts changed')
    old_final=totals(rows(old/'selection_step_2000.json.gz'))
    primary_meta=read(root/'coarse_adaptation-20261101'/'training.json')
    if [h['sample'] for h in old_meta['history']] != [h['sample'] for h in primary_meta['history']]:raise ValueError('Historical/new primary sampler or augmentation draws differ')
    result=dict(scope='Complete exposed-development coupled-assignment study. No learned improvement, calibrated acceptance, fresh generalization or SOTA claim.',
        completed_optimizer_steps=6000,total_parameters=81696,trainable_parameters=53424,
        frozen_surface=totals(base),stronger_simple_sift_surface_correct=57,runs=records,
        matched_old_loss_primary_final=old_final,matched_new_loss_primary_final=records[0]['final'],matched_primary_2000_sampler_and_augmentation_records_equal=True,
        any_trained_checkpoint_selected=any(r['selected_step']>0 for r in records),
        model_upgrade_adopted=False,accepted_pose_targets_proven=False,fresh_final_gate_passed=False,
        independent_saved_rows_replayed=1080,independently_retrained_pilot_steps=100,independently_retrained_full_cohort=False,
        actual_selected_model_controls=3,actual_primary_forward_cases=72,actual_primary_forward_exact=True,
        source_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),summary_sha256=sha(root/'summary.json'),
        audit_sha256=sha(audit_path),selected_controls_sha256=sha(controls_root/'summary.json'),
        limits=['72dependent pairs/eightgeometries/fourexposedterrain components; all model selection reuses them.',
            'Selected initialization is a negative finding, not a newly trained superior model.',
            'Raw wrong poses are not calibrated accepted errors; nonoverlap and1%risk targets remain unverified.',
            'Toy objective guarantees and independent numerical replay do not establish actual localization benefit.',
            'Fitting seconds are not isolated inference latency or a fair comparison to larger models.',
            'Original fresh-data shortfall, remaining author profiles, transfer, uncertainty, paper and compact-v3interface remain incomplete.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f"| {r['seed']} | {r['selected_step']} | {r['selected']['correct']} | {r['selected']['strict']} | {r['selected']['wrong']} | {r['final']['correct']} | {r['final']['strict']} | {r['final']['wrong']} |"for r in records)
    curves='\n'.join(f"- {r['seed']}: "+' / '.join(str(v['selection']['correct_poses'])for v in r['validation']) for r in records)
    out.with_suffix('.md').write_text(f'''# Coupled-assignment training: complete three-seed negative result

All **6,000updates** finished. Every predefined seed selects **initialization54/72**, with16strict poses and3wrong raw proposals. The new loss is not adopted as a model improvement. The stronger unchanged SIFT-first+surface baseline remains57/72. All counts describe72dependent exposed development pairs from eight geometries in four terrain components.

| Seed | Selected update | Selected correct /72 | Selected strict | Selected wrong raw | Trained endpoint correct /72 | Endpoint strict | Endpoint wrong raw |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

Correct raw poses at updates0/500/1000/1500/2000:

{curves}

## What the experiment shows

The added bidirectional reachable-window dual-softmax correction is standard assignment machinery, tested here with the same81,696parameter model, same geometry labels, training draws, optimizer and inference pipeline. The separate toy fixture showed that it penalizes one proposal-collapse counterexample. It **did not deliver a selected-model pose gain** in this study. This does not prove a universal failure of the method or identify the unique cause of the lunar regressions.

For the predefined primary seed, the matched historical old-loss2000-update endpoint recovers{old_final['correct']}/72, versus{records[0]['final']['correct']}/72 with the added correction. Both studies still select initialization. Other old repeat seeds used a different variant and are not substituted as matched controls. Modest strict-threshold changes do not replace the predeclared unfiltered-pose selection criterion.

## What was verified

The independent audit replays **1,080saved native checkpoint rows and6,000sampler/update records**, including geometry/window-label checks. Only the separate100-update pilot was independently retrained with exact model/optimizer/RNG/gradient replay; the full6,000-update optimizer history was **not** independently retrained. All three selected models pass actual transform/invalid-input controls. A fresh72-case primary neural forward exactly matches every saved row. Source, protocol, checkpoint and audit hashes are recorded in the JSON.

Initialization remains eligible under the predeclared rule; no best-seed headline or post-test tuning is used. The descriptive paired selected-minus-frozen intervals are zero because selected outputs match the frozen pipeline. They are not evidence of generalization or equivalence on fresh terrain.

Raw pose recovery is not calibrated acceptance. The selected pipeline still proposes three wrong poses; accepted-error/nonoverlap rates and the required10percentage-point correctly accepted gain remain unproven. A75%raw pose rate does not complete the70%correct-accepted target. Training time is not an inference-efficiency comparison.

## Next research decision

Preserve the complete negative result, retain the original matcher and prioritize verified additional3D terrain/lighting data and diagnostic evidence before another training change. The prospective northern scene study has no certified fresh benchmark or model result yet. The existing CRESENT terrain-registration discrepancy and original fresh-LunarGeo shortfall remain recorded. Real orbital accuracy, remaining modern author profiles, transfer, confidence calibration and the full research/paper targets remain unresolved. No email is sent.
''')
    print(json.dumps(dict(passed=True,selected=[r['selected']['correct']for r in records],selected_steps=[r['selected_step']for r in records],final=[r['final']['correct']for r in records],adopted=False),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/localization-v24-mutual-development');p.add_argument('--audit',default='reports/native_mutual_complete_audit_v24.json');p.add_argument('--controls',default='runs/localization-v24-mutual-selected-controls');p.add_argument('--output',default='reports/mutual_cohort_complete_v24.json');a=p.parse_args();report(a.run,a.audit,a.controls,a.output)
