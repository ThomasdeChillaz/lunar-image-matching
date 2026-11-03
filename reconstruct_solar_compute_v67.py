"""Conservative calendar-envelope admission accounting, not exact CPU usage.

Charge all elapsed time within two historical work windows to training, even
when idle, rendering or comparing. Charge the whole intervening calendar span
to baseline comparisons as well. Missing per-job times are never treated as0.
Only the original3D study is in these48h/72h limits; earlier compact-v2 study is
historical and its separate published ledger is retained.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
repo=Path.cwd()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
windows=[('2026-10-31T18:00:00+00:00','2026-11-01T20:48:00+00:00'),('2026-11-02T07:45:00+00:00','2026-11-02T23:20:00+00:00')]
parse=datetime.fromisoformat
rows=[];uncovered=[];untimed=[]
for root in sorted((repo/'runs').iterdir()):
 if not root.is_dir()or not(root.name.startswith('localization-')or root.name.startswith('north-')):continue
 checkpoints=list(root.glob('*.pt'))+list(root.glob('*/*.pt'))
 if not checkpoints:continue
 times=[];sources={}
 for p in [*root.glob('protocol.json'),*root.glob('execution.json'),*root.glob('*training.json'),*root.glob('*/training.json')]:
  if p.stat().st_size>500000:continue
  z=json.loads(p.read_text())
  if not isinstance(z,dict):continue
  sources[str(p.relative_to(repo))]=sha(p)
  for key in('started_utc','created_utc','recorded_utc'):
   if key in z:
    t=parse(z[key]);times.append(z[key])
    if not any(parse(a)<=t<parse(b)for a,b in windows):uncovered.append(dict(root=root.name,field=key,time=z[key]))
    break
 if not times:untimed.append(root.name)
 rows.append(dict(root='runs/'+root.name,checkpoint_files=len(checkpoints),metadata_timestamps=sorted(set(times)),metadata_sha256=sources))
# Roots with no wall timestamp contain only initial states or the bounded fixed
# pair/rate probes whose full enclosing controller windows are charged below.
allowed_untimed={'localization-v15-context-initial-control','localization-v18-window-initial-control','north-v54-fixed-pair-optimization','north-v55-lr-1e-3','north-v55-lr-1e-4'}
if set(untimed)-allowed_untimed:raise ValueError('Unassigned model artifact roots:'+str(set(untimed)-allowed_untimed))
if uncovered:raise ValueError('Known training timestamp outside charged envelopes:'+str(uncovered))
training=sum((parse(b)-parse(a)).total_seconds()for a,b in windows)
baseline=(parse(windows[-1][1])-parse(windows[0][0])).total_seconds()
evidence=['runs/localization-v27-mutual-pipeline-state.json','runs/north-v38-training-cost-pilot/protocol.json','runs/north-context-v61-state.json','runs/north-camera-completion-v62-state.json','reports/north_context_study_audit_v61.json','reports/north_fine_matched_audit_v58.json']
result=dict(schema_version=67,accounting_complete=True,accounting_kind='Conservative historical calendar-envelope allocation for admission, NOT exact measured CPU/wall usage.',exact_consumed_training_seconds=None,exact_consumed_baseline_seconds=None,charged_training_upper_bound_seconds=training,charged_baseline_upper_bound_seconds=baseline,training_budget_seconds=48*3600,baseline_budget_seconds=72*3600,remaining_training_wall_seconds=48*3600-training,remaining_baseline_wall_seconds=72*3600-baseline,charged_training_windows=windows,baseline_window=[windows[0][0],windows[-1][1]],artifact_roots=rows,untimed_roots=untimed,untimed_handling='Charged full enclosing historical calendar windows; step0-only controls add no optimizer work. No missing duration is assigned0.',evidence_sha256={p:sha(repo/p)for p in evidence},coverage_assumptions=['All historical lunar training/replay work is represented by the retained artifact roots/controller history; no undisclosed outside jobs are inferred.','First window charges ALL calendar time from before firstv3pilot through completed mutual controller, including idle/data/baseline work.','Second window charges ALL time from before north training-bank work through after attention/replays/camera attempts. v38firstoptimizer08:08; v61pipelinecomplete22:11.','Between windows, retained controllers perform data acquisition/rendering/geometry recovery; no retained lunar optimizer or newly admitted training exists. Metadata without precise finishing times is covered by enclosing windows.','Baseline charge includes the ENTIRE calendar span, including BOTH training windows and overnight data/idle periods, deliberately overcharging comparisons.','Original consumed compact-v2 study predates this3D48h/72hcontract and remains in reports/compact_compute_budget_v2.json.','New model-freev62completion/v64geometry/v65data pilots and procedural software regressions are separately recorded; no new lunar fitting/pretrained comparison occurred after these historical windows.'],new_jobs_rule='Reserve full fixed pilot/study caps from these remaining allowances before admission; never restore allowance from a failed or duplicate invocation. Refine conservative history only using retained evidence, not an assumed0.' )
assert min(result['remaining_training_wall_seconds'],result['remaining_baseline_wall_seconds'])>0
output=repo/'reports/solar_compute_admission_v67.json';payload=json.dumps(result,indent=2)+'\n'
if output.exists()and output.read_text()!=payload:raise FileExistsError('Preserve prior accounting; new evidence requires a versioned ledger')
if not output.exists():output.write_text(payload)
print(json.dumps({k:v for k,v in result.items()if k in ['charged_training_upper_bound_seconds','charged_baseline_upper_bound_seconds','remaining_training_wall_seconds','remaining_baseline_wall_seconds','untimed_roots']},indent=2))
