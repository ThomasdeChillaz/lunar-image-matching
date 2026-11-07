"""Serial new-camera data acquisition under one immutable shared deadline.

No fitting, matcher or baseline is admitted. A stopped/expired unit is never
silently restarted. Explicit resume can reuse only sealed complete units and
only before the ORIGINAL shared deadline, with exactly the published sources.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import os,json,subprocess,hashlib,time,signal,sys
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit',required=True);parser.add_argument('--blender',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
blender=args.blender
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
commit=args.publication_commit
assert len(commit)==40 and all(c in '0123456789abcdef'for c in commit)
master_path='reports/solar_development_admission_v67.json';master=read(master_path);deadline=datetime.fromisoformat(master['absolute_deadline_utc'])
assert subprocess.check_output(['git','show',commit+':'+master_path])==Path(master_path).read_bytes()
for n,h in master['execution_source_sha256'].items():
 assert sha(n)==h and subprocess.check_output(['git','show',commit+':'+n])==Path(n).read_bytes(),n
assert master['compute_admitted']and not master['calibration_or_final_access']and master['shared_cap_seconds']==36000
assert len(master['units'])==96 and Counter(u['site']['split']for u in master['units'])=={'training':64,'model_selection':32}
assert len({u['site']['component']for u in master['units']})==24
assert sha(master['compute_ledger'])==master['compute_ledger_sha256']and read(master['compute_ledger'])['accounting_complete']
assert sha(master['pilot_geometry_audit'])==master['pilot_geometry_audit_sha256']and read(master['pilot_geometry_audit'])['passed']
root=Path('runs/solar-v67-development');state_path=Path('runs/solar-development-v67-state.json');resume=args.resume
if state_path.exists():
 assert resume,'Preserve existing controller state; explicit original-deadline resume only'
 old=read(state_path);assert old['state']!='running'and old['publication_commit']==commit
 assert not subprocess.run(['ps','-p',str(old['pid']),'-o','command='],text=True,capture_output=True).stdout.strip()
else:assert not resume
expected={str(p):sha(p)for p in Path('.').glob('*.py')};started=datetime.now(timezone.utc);records=[]
if started>=deadline:raise TimeoutError('Original published shared deadline has expired; no automatic extension')
root.mkdir(exist_ok=resume)
def state(stage,**extra):state_path.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),publication_commit=commit,master_protocol_sha256=sha(master_path),absolute_deadline_utc=deadline.isoformat(),completed_units=len(records),planned_units=96,planned_images=4608,matcher_forwards=0,optimizer_updates=0,**extra),indent=2)+'\n')
def unchanged():
 for n,h in expected.items():assert sha(n)==h,n
 if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Original shared acquisition deadline exhausted')
try:
 for index,unit in enumerate(master['units']):
  unchanged();folder=Path(unit['output']);seal=folder/'COMPLETE.json'
  if seal.exists():
   if not resume:raise FileExistsError('Existing data requires explicit original-deadline resume')
   saved=read(seal)
   for n,h in saved['sha256'].items():assert sha(folder/n)==h,n
   summary=read(folder/'summary.json');assert summary['complete']and summary['protocol_sha256']==sha(master_path)
   records.append(dict(unit_index=index,output=str(folder),summary_sha256=sha(folder/'summary.json'),seal_sha256=sha(seal),rendered_images=summary['rendered_images'],reused_complete_unit=True));continue
  if folder.exists():raise FileExistsError('Partial unit preserved; a separately declared amendment is required:'+str(folder))
  folder.parent.mkdir(parents=True,exist_ok=True);t=datetime.now(timezone.utc);unit_deadline=min(t+timedelta(seconds=900),deadline);tick=time.monotonic()
  with Path(f'runs/solar-v67-unit-{index:03d}.log').open('x')as log:
   child=subprocess.Popen([blender,'--background','--threads','2','--factory-startup','--python','render_solar_development_v67.py','--','--protocol',master_path,'--publication-commit',commit,'--unit-index',str(index)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'),start_new_session=True)
   awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   state('rendering fresh camera/light data',state='running',child_pid=child.pid,current_unit=index,component=unit['site']['component'],family=unit['camera_pairs'][0]['family'],unit_deadline_utc=unit_deadline.isoformat(),unit_started_utc=t.isoformat())
   try:
    while child.poll()is None:
     unchanged()
     if datetime.now(timezone.utc)>=unit_deadline or time.monotonic()-tick>=900:raise TimeoutError('Original per-unit900second allowance exhausted')
     time.sleep(.5)
    if child.returncode:raise subprocess.CalledProcessError(child.returncode,'render_solar_development_v67.py')
    unchanged();summary=read(folder/'summary.json');assert summary['complete']and summary['rendered_images']==48
   except BaseException:
    if child.poll()is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
   finally:
    if awake.poll()is None:awake.terminate()
  seal.write_text(json.dumps(dict(sha256={str(p.relative_to(folder)):sha(p)for p in sorted(folder.rglob('*'))if p.is_file()}),indent=2)+'\n')
  records.append(dict(unit_index=index,output=str(folder),summary_sha256=sha(folder/'summary.json'),seal_sha256=sha(seal),rendered_images=48,outer_calendar_seconds=(datetime.now(timezone.utc)-t).total_seconds(),reused_complete_unit=False))
  (root/'progress.json').write_text(json.dumps(dict(units=records,planned_units=96,complete=False,absolute_deadline_utc=deadline.isoformat()),indent=2)+'\n')
  print('SOLAR DEVELOPMENT UNIT COMPLETE',index+1,'/96',flush=True)
 unchanged();result=dict(complete=True,units=records,rendered_images=sum(x['rendered_images']for x in records),master_protocol_sha256=sha(master_path),source_publication_commit=commit,geometry_audit_pending=True,fitting_allowed=False,matcher_forwards=0,optimizer_updates=0,calibration_or_final_access=False,scope='New seeded4camera pairs per original16training/eightselection groups,24physical lights per view. Conditions are dependent geography; no matcher result or training permission until full geometry audit.')
 with(root/'summary.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 state('data rendering complete; full geometry audit/bank required before training',state='complete',fitting_allowed=False)
except BaseException as e:state('failed; all complete/partial data preserved',state='failed',error_type=type(e).__name__,error=str(e));raise
