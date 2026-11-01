"""Saved-only localization diagnostics for all completed new-loss checkpoints."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve prior diagnostics')
    audit=json.loads(Path('reports/native_mutual_complete_audit_v24.json').read_text())
    if not audit['passed']or audit['replayed_pose_rows']!=1080:raise ValueError('Complete saved-row audit required')
    root=Path('runs/localization-v24-mutual-development');records=[];row_count=0
    for seed in (20261101,20261102,20261103):
        trajectory=[]
        for step in (0,500,1000,1500,2000):
            path=root/f'coarse_adaptation-{seed}'/f'selection_step_{step}.json.gz'
            with gzip.open(path,'rt')as f:rows=json.load(f)
            if len(rows)!=72:raise ValueError('Keep all eligible development cases')
            row_count+=len(rows)
            correct=sum(r['correct_at_native_px']['2']for r in rows)
            scorable=sum(r['scorable']for r in rows);returned=sum(r['returned']for r in rows)
            trajectory.append(dict(step=step,correct_raw_poses=sum(r['pose_correct']for r in rows),
                correct_at_2_native_px=correct,scorable_points=scorable,returned_points=returned,
                correct_2px_fraction_of_scorable=correct/scorable,
                verified_correct_2px_fraction_of_all_returned=correct/returned,
                source_rows_sha256=sha(path)))
        records.append(dict(seed=seed,trajectory=trajectory))
    result=dict(scope='Saved-only exposed-development diagnostics; no new inference or trained upgrade.',
        records=records,rows=row_count,source_sha256=sha(__file__),
        complete_audit_sha256=sha('reports/native_mutual_complete_audit_v24.json'),
        observation='All three trained endpoints lose both correct2px correspondence count and conditional2px precision relative to initialization, as well as raw pose count.',
        interpretation_limit='This is consistent with degraded matching, not solely pair-level rejection. It does not uniquely identify a cause, prove an architectural remedy or evaluate fresh terrain.',
        point_denominator='Scorable means the original independent geometry/mask metric support; all-returned fraction is reported separately. Points are dependent, not independent terrain tests.',
        acceptance_or_nonoverlap_evaluated=False,fresh_generalization_evaluated=False)
    out.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/mutual_precision_diagnostic_v34.json')
    r=report(p.parse_args().output)
    print(json.dumps(dict(rows=r['rows'],initial=r['records'][0]['trajectory'][0],
        endpoints=[x['trajectory'][-1]for x in r['records']]),indent=2))
