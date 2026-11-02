"""Saved-result lighting/precision diagnosis; no new matching or fitting.

All returned points form the precision denominator. Geometrically nonvisible
returned points count as incorrect, rather than disappearing from precision.
The former scorable-only readout remains separately available. Neither is a
cross-paper comparison without matched images and ground-truth conventions.
"""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


THRESHOLDS=(1,2,4,5,8,12)
PATTERNS=((0,0),(0,3),(0,4),(1,5))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def read_rows(path):
    with gzip.open(path,'rt')as f:return json.load(f)


def point_counts(row):
    p=np.asarray(row['reference_points_native'],float).reshape(-1,2)
    q=np.asarray(row['query_points_native'],float).reshape(-1,2)
    target=np.asarray(row['target_points_native'],float).reshape(-1,2)
    visible=np.asarray(row['visible'],bool)
    if (p.shape!=q.shape or visible.shape!=(len(p),)or len(target)!=int(visible.sum())
            or len(p)!=row['returned']or int(visible.sum())!=row['scorable']):
        raise ValueError('Saved correspondence shapes/denominators differ')
    if any(not np.isfinite(v).all()for v in (p,q,target)):
        raise ValueError('Nonfinite saved native correspondences')
    distance=np.linalg.norm(q[visible]-target,axis=1)
    correct={str(t):int((distance<=t).sum())for t in THRESHOLDS}
    for t in (1,2,4,8,12):
        if correct[str(t)]!=row['correct_at_native_px'][str(t)]:
            raise ValueError('Existing independently replayed point count differs')
    return dict(returned=len(p),scorable=len(target),nonvisible_returned=len(p)-len(target),
        correct_at_native_px=correct)


def summarize(rows):
    if not rows:raise ValueError('Do not replace an empty denominator with zero success')
    points=[point_counts(r)for r in rows]
    returned=sum(r['returned']for r in points);scorable=sum(r['scorable']for r in points)
    correct={str(t):sum(r['correct_at_native_px'][str(t)]for r in points)for t in THRESHOLDS}
    estimated=sum(r['estimated_cam2world']is not None for r in rows)
    poses=sum(r['pose_correct']for r in rows)
    return dict(eligible_positive_pairs=len(rows),correct_raw_poses=poses,
        wrong_raw_poses=estimated-poses,no_pose=len(rows)-estimated,
        strict_raw_poses=sum(r['strict_pose_correct']for r in rows),
        returned_matches=returned,scorable_matches=scorable,
        nonvisible_returned_matches=returned-scorable,correct_at_native_px=correct,
        precision_all_returned={k:None if not returned else v/returned for k,v in correct.items()},
        precision_scorable_only={k:None if not scorable else v/scorable for k,v in correct.items()},
        pose_success_all_eligible=poses/len(rows),calibrated_acceptance_evaluated=False)


def lighting_decomposition(rows,cases):
    if [r['pair_id']for r in rows]!=[c['pair_id']for c in cases]:
        raise ValueError('All96 declared positive cases in original order required')
    for r,c in zip(rows,cases):
        if any(r[k]!=v for k,v in c.items()):raise ValueError('Case metadata changed')
    if len(rows)!=96 or len({r['component']for r in rows})!=8:
        raise ValueError('Complete eight-component denominator required')
    pattern={f'{a}-{b}':summarize([r for r in rows if(r['light_a'],r['light_b'])==(a,b)])
             for a,b in PATTERNS}
    if any(r['eligible_positive_pairs']!=24 for r in pattern.values()):
        raise ValueError('Each fixed lighting pattern must include all24geometries')
    changed=summarize([r for r in rows if r['light_a']!=r['light_b']])
    return dict(all=summarize(rows),by_lighting_pattern=pattern,changed_lighting=changed)


def training_loss_profile(history):
    if [r['step']for r in history]!=list(range(1,2001)):
        raise ValueError('Complete2000update history required for loss profile')
    for r in history:
        if (r['sample']['verified_nonoverlap']!=(r['step']%4==0)
                or not np.isfinite([r['loss'],r['gradient_norm']]).all()):
            raise ValueError('Recorded negative cadence or finite update history differs')
    out={}
    for lo,hi in ((1,500),(1501,2000)):
        groups={'nonoverlap':[],'same_light_positive':[],'changed_light_positive':[]}
        for r in history:
            if not lo<=r['step']<=hi:continue
            s=r['sample'];name=('nonoverlap'if s['verified_nonoverlap']else
                'same_light_positive'if s['light_a']==s['light_b']else'changed_light_positive')
            groups[name].append(r)
        out[f'{lo}-{hi}']={name:dict(updates=len(part),
            mean_loss=None if not part else float(np.mean([r['loss']for r in part])),
            mean_forward_positive_labels=None if not part else float(np.mean([r['labels']['forward']['positive']for r in part])))
            for name,part in groups.items()}
    return dict(windows=out,console_loss_sample_bias='Every100stepconsolelineisanonoverlapupdatebecause100isdivisibleby4. Neverinterpretitasoveralltrainingloss.',
        pose_improvement_claim=False)


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve previous diagnostic snapshots')
    manifest_path=Path('reports/north_training_data_v38.json');m=read(manifest_path)
    cohort=Path('runs/north-v39-expanded-control');protocol=read(cohort/'protocol.json')
    if (m['calibration_or_test_pixels_loaded']or protocol['calibration_or_test_pixels_loaded']
            or protocol['manifest_sha256']!=sha(manifest_path)):
        raise ValueError('Frozen development-only manifest required')
    for name,digest in protocol['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Running fitting/inference source changed')
    baseline=Path('runs/north-v39-full-baselines');b=read(baseline/'summary.json')
    audit=read('reports/north_full_baseline_audit_v39.json')
    if (not b['complete']or not audit['passed']or audit['replayed_rows']!=384
            or audit['summary_sha256']!=sha(baseline/'summary.json')
            or b['results_sha256']!=sha(baseline/'results.json.gz')):
        raise ValueError('Complete independently replayed baseline required')
    rows=[r for r in read_rows(baseline/'results.json.gz')if r['method']=='frozen_surface'and r['positive']]
    records=[dict(model='frozen_surface',seed=None,step=None,rows_path=str(baseline/'results.json.gz'),
        rows_sha256=sha(baseline/'results.json.gz'),
        results=lighting_decomposition(rows,m['cases']))]
    # Read only atomic, complete checkpoint files with their matching receipt.
    # In-progress per-case directories are deliberately excluded.
    loss_profiles=[]
    for seed in (20261101,20261102,20261103):
        folder=cohort/('coarse_adaptation-'+str(seed))
        if (folder/'COMPLETE.json').is_file():
            complete=read(folder/'COMPLETE.json');summary=read(folder/'summary.json')
            if sha(folder/'summary.json')!=complete['sha256']['summary.json']:
                raise ValueError('Complete seed history changed')
            loss_profiles.append(dict(seed=seed,summary_sha256=sha(folder/'summary.json'),
                **training_loss_profile(summary['history'])))
        for step in (0,500,1000,1500,2000):
            path=folder/f'selection_step_{step}.json.gz';receipt=folder/f'validation_step_{step}.json'
            if not(path.is_file()and receipt.is_file()):continue
            r=read(receipt)
            if r['step']!=step or r['results_sha256']!=sha(path):
                raise ValueError('Checkpoint receipt changed')
            part=read_rows(path);result=lighting_decomposition(part,m['cases'])
            total=result['all'];saved=r['selection']
            if (total['correct_raw_poses']!=saved['correct_poses']
                    or total['correct_at_native_px']['2']!=saved['correct_at_two_px']
                    or total['returned_matches']!=saved['returned']):
                raise ValueError('Saved checkpoint aggregate differs')
            records.append(dict(model='expanded_data_control',seed=seed,step=step,rows_path=str(path),
                rows_sha256=sha(path),receipt_sha256=sha(receipt),results=result))
    whole_complete=(cohort/'summary.json').is_file()
    result=dict(scope='Complete saved checkpoint decomposition by fixed lighting; interim while cohort unfinished. No new neural inference, model selection, pose estimation or acceptance calibration.',
        snapshot_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        manifest_sha256=sha(manifest_path),cohort_protocol_sha256=sha(cohort/'protocol.json'),
        complete_cohort_summary_present=whole_complete,records=records,completed_seed_loss_profiles=loss_profiles,
        precision_denominator='Allreturnedmatches; known nonvisible returns are incorrect. Scorable-only precision retained separately.',
        native_error_thresholds_px=list(THRESHOLDS),threshold5_used_for_selection=False,
        final_camera_pose_criterion='Unchanged2degrees/1percentmedianreference range; failed poses included.',
        stage_attribution_established=False,
        stage_limit='Saved final correspondences cannot determine whether a mismatch originated in coarse proposals or native refinement. A separate proposal-reachability diagnostic is required.',
        images_or_depth_arrays_loaded=0,neural_forwards=0,calibration_or_test_pixels_loaded=0,
        full_optimizer_or_neural_replay_performed=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    for record in records:
        a=record['results']['all'];c=record['results']['changed_lighting']
        print(record['model'],record['seed'],record['step'],'all',a['correct_raw_poses'],'/96',
            'changed',c['correct_raw_poses'],'/72','changed2pxPrecision',c['precision_all_returned']['2'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
