"""Minimal externally charged fixed100-update pilot or exact replay.

No scientific imports, admission, decode, reservation or jobs at module import.
Root owns future cap/date/source publication and actual execution.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from reports.qualify_solar_coarse_teacher_v87 import sha,read,write,utc,now

CANONICAL='runs/solar-compute-reservations-v67.json'
EXTRA_SOURCES=('reports/qualify_solar_coarse_teacher_v87.py','solar_coarse_training_adapter_v86.py',
    'solar_distributional_response_v86.py','solar_coarse_bank_v88.py',
    'solar_distributional_training_v88.py','solar_distributional_pilot_v89.py')
FIXED=dict(seed=20261101,updates=100,batch_points=8,learning_rate=1e-4,weight_decay=1e-4,
    gradient_clip=1.,conditional_weight=1.,marginal_weight=.25,discrimination_weight=.10,
    discrimination_temperature=.10,cpu_threads=2)


def job_id(path,p):return sha(path)+'/distributional100_'+p['mode']


def validate_settings(p,c,started):
    if p.get('prepared_only')is not False or p.get('purpose')!='distributional_coarse100'or p.get('mode')not in('pilot','replay')or p.get('category')!='training'or p.get('cpu_threads')!=2 or p.get('fixed_settings')!=FIXED or p.get('calibration_or_final_access')is not False:raise ValueError('One published fixed100/seed/two-thread training-only pilot or replay required')
    cap=p['max_calendar_seconds'];limit=c['training_pilot_cap_seconds']if p['mode']=='pilot'else c['training_replay_cap_seconds']
    if type(cap)is not int or not 5<cap<=limit:raise ValueError('Explicit positive predeclared cap within original pilot/replay allowance required')
    if p['reservation_journal']!=CANONICAL or p['compute_ledger']!=c['compute_ledger']or p['compute_ledger_sha256']!=c['compute_ledger_sha256']:raise ValueError('Original one canonical training ledger required')
    for key in('teacher_directory','teacher_sha256','expected_frozen_descriptor_sha256'):
        expected=c[dict(teacher_directory='initializer',teacher_sha256='initializer_sha256',expected_frozen_descriptor_sha256='expected_frozen_descriptor_sha256')[key]]
        if p[key]!=expected:raise ValueError('Original qualified frozen teacher required')
    created,latest,absolute=map(utc,(p['created_utc'],p['latest_start_utc'],p['absolute_deadline_utc']));lease=p['publication_dispatch_lease_seconds']
    if type(lease)is not int or not 0<lease<=1800 or latest!=created+timedelta(seconds=lease)or absolute!=latest+timedelta(seconds=cap):raise ValueError('Original finite dispatch lease/deadline algebra required')
    if started<created or started>latest:raise TimeoutError('Original dispatch lease expired')
    if not p['output']or Path(p['output']).is_absolute()or '..'in Path(p['output']).parts:raise ValueError('Versioned repository-relative output required')
    return min(started+timedelta(seconds=cap),absolute)


def completed_pilot(protocol_path,root):
    """Completed original pilot evidence remains valid after its launch lease."""
    p=read(protocol_path);r=Path(root);s=read(r/'COMPLETE.json');w=read(r/'watchdog.json');a=read(r/'worker_summary.json')
    if p['mode']!='pilot'or not s.get('external_watchdog_passed')or not w.get('complete')or not a.get('complete')or any(x['protocol_sha256']!=sha(protocol_path)for x in(s,w,a))or s['worker_summary_sha256']!=sha(r/'worker_summary.json')or s['watchdog_sha256']!=sha(r/'watchdog.json'):raise ValueError('Original actual complete external100pilot required')
    elapsed=w['calendar_seconds']
    if type(elapsed)not in(int,float)or not math.isfinite(elapsed)or not 0<elapsed<=p['max_calendar_seconds']or w['cap_seconds']!=p['max_calendar_seconds']or w['child_returncode']!=0 or w['failure']is not None or not utc(p['created_utc'])<=utc(w['started_utc'])<=utc(p['latest_start_utc'])or utc(w['hard_deadline_utc'])!=min(utc(w['started_utc'])+timedelta(seconds=p['max_calendar_seconds']),utc(p['absolute_deadline_utc']))or utc(w['ended_utc'])>=utc(w['hard_deadline_utc']):raise ValueError('Original completed pilot did not respect its original startup-inclusive cap')
    verify_worker_evidence(p,r,a)
    return p,w,a


def verify_worker_evidence(p,r,a):
    if a.get('mode')!=p['mode']or a.get('updates')!=100 or a.get('bank_manifest_sha256')!=p['bank_manifest_sha256']or a.get('bank_prior_sha256')!=p['bank_prior_sha256']:raise ValueError('Fixed training evidence/data identity differs')
    if p['mode']=='pilot':
        source=r/'science';summary=read(source/'summary.json')
        required={'step_0.pt','step_100.pt','history_100.json','trace.jsonl','execution.json'}
        if sha(source/'summary.json')!=a['science_summary_sha256']or not summary.get('complete')or summary.get('format')!='distributional-coarse128-pilot-v88'or summary.get('bank_manifest_sha256')!=p['bank_manifest_sha256']or set(summary.get('artifacts',{}))!=required or summary['updates']!=100 or summary['seed']!=20261101 or summary['model_selection_performed']or summary['calibration_or_final_access']or summary['model_promoted']:raise ValueError('Actual100-update unselected pilot with allfivecheckpoint/trace artifacts required')
        for name,h in summary['artifacts'].items():
            if Path(name).name!=name or sha(source/name)!=h:raise ValueError('Pilot checkpoint/trace evidence changed')
        history=read(source/'history_100.json');trace=[json.loads(line)for line in(source/'trace.jsonl').read_text().splitlines()]
        if history!=trace or len(trace)!=100 or [x['step']for x in trace]!=list(range(1,101)):raise ValueError('All100 actual pilot trace records required')
    else:
        receipt=read(r/'replay.json');trace=[json.loads(line)for line in(r/'replay_trace.jsonl').read_text().splitlines()]
        if sha(r/'replay.json')!=a['replay_sha256']or sha(r/'replay_trace.jsonl')!=a['replay_trace_sha256']or receipt.get('passed')is not True or receipt.get('updates')!=100 or not receipt.get('exact_model_optimizer_rng_sampling_losses_gradients'):raise ValueError('Actual exact100 replay receipt required')
        expected=read(Path(p['original_pilot_run'])/'science'/'history_100.json')
        if trace!=expected or len(trace)!=100 or receipt['original_summary_sha256']!=sha(Path(p['original_pilot_run'])/'science'/'summary.json'):raise ValueError('Actual replay trace must exactly equal the original100 records')


def metadata_paths(p):
    paths={p['learning_contract'],p['bank_manifest'],p['bank_prior'],p['bank_protocol']}
    paths.update(str(Path(p['bank_run'])/n)for n in('COMPLETE.json','watchdog.json','summary.json'))
    if p['mode']=='replay':
        paths.add(p['original_pilot_protocol']);paths.update(str(Path(p['original_pilot_run'])/n)for n in('COMPLETE.json','watchdog.json','worker_summary.json','science/summary.json','science/execution.json','science/history_100.json','science/trace.jsonl'))
    return paths


def verify_sources(path,p,commit,deadline):
    from solar_learning_pipeline_v73 import local_source_closure
    required=local_source_closure();required.update({n:sha(n)for n in EXTRA_SOURCES})
    if p['source_sha256']!=required or set(p['metadata_sha256'])!=metadata_paths(p)or len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact source/metadata/publication identity required')
    def public(n):
        remaining=(deadline-now()).total_seconds()
        if remaining<=0:raise TimeoutError('Preflight exhausted original deadline')
        value=subprocess.check_output(['git','show',commit+':'+n],timeout=min(10.,remaining))
        if now()>=deadline:raise TimeoutError('Publication completed after original deadline')
        return value
    for n,h in {**p['source_sha256'],**p['metadata_sha256']}.items():
        if now()>=deadline:raise TimeoutError('Original preflight deadline exhausted')
        if sha(n)!=h or public(n)!=Path(n).read_bytes():raise ValueError('Source/input not unchanged and published:'+n)
    if public(str(path))!=Path(path).read_bytes():raise ValueError('Protocol not published before job')


def admission(path,commit,started):
    from solar_learning_pipeline_v73 import load_contract,remaining
    from solar_coarse_bank_v88 import validate_completed_bank
    p=read(path);c=load_contract(p['learning_contract']);deadline=validate_settings(p,c,started)
    proof=validate_completed_bank(p['bank_manifest']);m=proof['manifest']
    if p['bank_manifest_sha256']!=proof['manifest_sha256']or p['bank_complete_sha256']!=proof['complete_sha256']or p['bank_prior']!=m['view_prior']or p['bank_prior_sha256']!=m['view_prior_sha256']or p['bank_protocol']!=m['protocol_path']or p['bank_run']!=m['output']:raise ValueError('Exact completed full bank/prior required')
    if proof['protocol']['learning_contract']!=p['learning_contract']or set(m['eligible_unit_indices'])!=set(c['eligible_training_unit_indices'])or sorted(m['training_groups'])!=sorted(c['training_groups'])or m['teacher_descriptor_sha256']!=p['expected_frozen_descriptor_sha256']:raise ValueError('All original training-only camera/terrain/prior membership required')
    if any(p['source_sha256'].get(n)!=h for n,h in m['source_sha256'].items()):raise ValueError('Completed bank scientific sources changed')
    if p['mode']=='replay':
        original,_,_=completed_pilot(p['original_pilot_protocol'],p['original_pilot_run'])
        for key in('learning_contract','source_sha256','bank_manifest','bank_manifest_sha256','bank_prior','bank_prior_sha256','teacher_directory','teacher_sha256','expected_frozen_descriptor_sha256','fixed_settings'):
            if p[key]!=original[key]:raise ValueError('Exact replay must preserve original pilot source/data/settings:'+key)
    verify_sources(path,p,commit,deadline)
    if remaining(c)['training']<p['max_calendar_seconds']:raise ValueError('Original cumulative training allowance exceeded')
    return p,deadline


def prepare(settings_path,output):
    """Metadata only, requires actual full bank. Dates remain unset/blocked."""
    from solar_learning_pipeline_v73 import load_contract,local_source_closure,remaining
    from solar_coarse_bank_v88 import validate_completed_bank
    s=read(settings_path);c=load_contract(s['learning_contract']);proof=validate_completed_bank(s['bank_manifest']);m=proof['manifest']
    cap=s['max_calendar_seconds'];limit=c['training_pilot_cap_seconds']if s['mode']=='pilot'else c['training_replay_cap_seconds']
    if s['mode']not in('pilot','replay')or type(cap)is not int or not 5<cap<=min(limit,remaining(c)['training']):raise ValueError('Root must predeclare a cap within original remaining allowance')
    sources=local_source_closure();sources.update({n:sha(n)for n in EXTRA_SOURCES})
    p=dict(s,prepared_only=True,purpose='distributional_coarse100',category='training',cpu_threads=2,fixed_settings=FIXED,calibration_or_final_access=False,
        compute_ledger=c['compute_ledger'],compute_ledger_sha256=c['compute_ledger_sha256'],reservation_journal=CANONICAL,
        teacher_directory=c['initializer'],teacher_sha256=c['initializer_sha256'],expected_frozen_descriptor_sha256=c['expected_frozen_descriptor_sha256'],
        bank_manifest_sha256=proof['manifest_sha256'],bank_prior=m['view_prior'],bank_prior_sha256=m['view_prior_sha256'],bank_complete_sha256=proof['complete_sha256'],bank_run=m['output'],bank_protocol=m['protocol_path'],
        source_sha256=sources,created_utc=None,latest_start_utc=None,absolute_deadline_utc=None,publication_dispatch_lease_seconds=1800)
    if p['mode']=='replay':completed_pilot(p['original_pilot_protocol'],p['original_pilot_run'])
    p['metadata_sha256']={n:sha(n)for n in metadata_paths(p)};write(output,p);return p


def worker(path,commit,controller_pid):
    p=read(path);r=Path(p['output']);execution=read(r/'execution.json')
    if os.getppid()!=controller_pid or execution['controller_pid']!=controller_pid or execution['protocol_sha256']!=sha(path)or execution['publication_commit']!=commit:raise ValueError('Original admitted live controller required')
    from solar_learning_pipeline_v73 import journal_state,load_contract
    _,j,_=journal_state(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL);entries=[x for x in j['reservations']if x['job_id']==job_id(path,p)]
    if len(entries)!=1 or entries[0]!=read(r/'reservation.json')['reservation']or entries[0]['category']!='training'or entries[0]['seconds']!=p['max_calendar_seconds']:raise ValueError('Exact separate original pilot/replay charge required')
    with(r/'WORKER_STARTED').open('x')as f:f.write(str(os.getpid())+'\n')
    def clock():
        if now()>=utc(execution['work_deadline_utc']):raise TimeoutError('Original worker work allowance exhausted')
    clock();verify_sources(path,p,commit,utc(execution['work_deadline_utc']))
    import torch
    import numpy as np
    import cv2
    import platform
    from dataclasses import asdict
    from compact_lunar import load_checkpoint
    from solar_distributional_training_v88 import PilotSettings,CoarseTrainingBank,initialize,train_pilot,replay_pilot
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    settings=PilotSettings();settings.validate()
    if asdict(settings)!=p['fixed_settings']:raise ValueError('Actual fixed library mathematics/settings changed')
    c=load_contract(p['learning_contract']);clock()
    bank=CoarseTrainingBank(Path.cwd(),p['bank_manifest'],p['bank_manifest_sha256'],allowed_training_groups=c['training_groups'],eligible_unit_indices=c['eligible_training_unit_indices'],expected_teacher_sha256=p['expected_frozen_descriptor_sha256'],cache_units=4,clock=clock)
    for name,h in p['teacher_sha256'].items():
        if sha(Path(p['teacher_directory'])/name)!=h:raise ValueError('Original initializer inputs changed')
    frozen,_=load_checkpoint(p['teacher_directory']);model,opt,rng,digest=initialize(frozen,bank.prior,settings)
    if digest!=p['expected_frozen_descriptor_sha256']:raise ValueError('Actual original frozen descriptor differs')
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,platform=platform.platform(),processor=platform.processor(),torch_threads=2)
    if p['mode']=='replay'and read(Path(p['original_pilot_run'])/'worker_summary.json')['environment']!=environment:raise ValueError('Exact replay needs original actual software/hardware/two-thread environment')
    provenance=dict(split='training',protocol_path=str(path),protocol_sha256=sha(path),publication_commit=commit,source_sha256=p['source_sha256'],teacher_sha256=p['teacher_sha256'],bank_manifest_sha256=p['bank_manifest_sha256'],bank_prior_sha256=p['bank_prior_sha256'])
    result=dict(complete=True,mode=p['mode'],protocol_sha256=sha(path),updates=100,bank_manifest_sha256=p['bank_manifest_sha256'],bank_prior_sha256=p['bank_prior_sha256'],environment=environment,parameter_report=model.parameter_report,model_selection_performed=False)
    if p['mode']=='pilot':
        train_pilot(model,opt,rng,bank,r/'science',expected_coarse_digest=digest,provenance=provenance,settings=settings,clock=clock)
        result['science_summary_sha256']=sha(r/'science'/'summary.json')
    else:
        with(r/'replay_trace.jsonl').open('x')as f:
            def record(actual):clock();f.write(json.dumps(actual,sort_keys=True,allow_nan=False)+'\n');f.flush()
            replay=replay_pilot(model,opt,rng,bank,Path(p['original_pilot_run'])/'science',expected_coarse_digest=digest,settings=settings,clock=clock,record_callback=record)
        write(r/'replay.json',replay);result.update(replay_sha256=sha(r/'replay.json'),replay_trace_sha256=sha(r/'replay_trace.jsonl'))
    clock();verify_worker_evidence(p,r,result);write(r/'worker_summary.json',result)


def run(protocol_path,commit,python):
    # Clocks BEFORE protocol/admission/publication/hash imports. No reset later.
    started=now();tick=time.monotonic();p,deadline=admission(protocol_path,commit,started);cap=p['max_calendar_seconds']
    if now()>=deadline-timedelta(seconds=5)or time.monotonic()-tick>=cap-5:raise TimeoutError('Preflight exhausted original allowance')
    root=Path(p['output']);root.mkdir(parents=True,exist_ok=False);work_deadline=deadline-timedelta(seconds=5)
    child=None;failure=None;complete=False;charged=None;pending=[]
    old={s:signal.signal(s,lambda signum,frame:pending.append(signum))for s in(signal.SIGINT,signal.SIGTERM)}
    try:
        from solar_compute_budget_v67 import reserve
        charged=reserve(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL,job_id(protocol_path,p),'training',cap)
        write(root/'reservation.json',charged);write(root/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=commit,controller_pid=os.getpid(),started_utc=started.isoformat(),work_deadline_utc=work_deadline.isoformat(),hard_deadline_utc=deadline.isoformat(),cap_seconds=cap))
        if pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise TimeoutError('Startup exhausted allowance')
        with(root/'worker.log').open('x')as f:
            child=subprocess.Popen([python,'solar_distributional_pilot_v89.py','worker','--protocol',str(protocol_path),'--publication-commit',commit,'--controller-pid',str(os.getpid())],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
            while child.poll()is None:
                if pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise TimeoutError('Original work allowance exhausted')
                time.sleep(.1)
        if child.returncode or pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise RuntimeError('Worker failed/interrupted/late')
        a=read(root/'worker_summary.json')
        if not a.get('complete')or a['protocol_sha256']!=sha(protocol_path):raise ValueError('Actual pilot/replay completion differs')
        verify_worker_evidence(p,root,a)
        complete=True
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if child is not None and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=max(0.,min(3.,(deadline-now()).total_seconds()-1.)))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:child.wait(timeout=max(0.,min(1.,(deadline-now()).total_seconds())))
                except subprocess.TimeoutExpired:complete=False;failure=dict(type='CleanupFailure',message='Original cleanup bound exhausted')
        if now()>=deadline or time.monotonic()-tick>=cap:complete=False;failure=dict(type='TimeoutError',message='Original hard cap exhausted')
        write(root/'watchdog.json',dict(complete=complete,protocol_sha256=sha(protocol_path),publication_commit=commit,started_utc=started.isoformat(),ended_utc=now().isoformat(),calendar_seconds=time.monotonic()-tick,cap_seconds=cap,hard_deadline_utc=deadline.isoformat(),work_deadline_utc=work_deadline.isoformat(),compute_reservation=charged,child_returncode=None if child is None else child.returncode,failure=failure,artifacts={f.name:sha(f)for f in(root/'worker_summary.json',root/'replay.json',root/'replay_trace.jsonl',root/'worker.log')if f.exists()}))
        for s,h in old.items():signal.signal(s,h)
    if not complete:raise RuntimeError(str(failure))
    write(root/'COMPLETE.json',dict(external_watchdog_passed=True,protocol_sha256=sha(protocol_path),worker_summary_sha256=sha(root/'worker_summary.json'),watchdog_sha256=sha(root/'watchdog.json')))

if __name__=='__main__':
    sys.path.insert(0,str(Path.cwd()))
    a=argparse.ArgumentParser(description=__doc__);s=a.add_subparsers(dest='command',required=True)
    b=s.add_parser('prepare');b.add_argument('--settings',required=True);b.add_argument('--output',required=True)
    for name in('run','worker'):
        b=s.add_parser(name);b.add_argument('--protocol',required=True);b.add_argument('--publication-commit',required=True)
        b.add_argument('--python',default=sys.executable)if name=='run'else b.add_argument('--controller-pid',type=int,required=True)
    v=a.parse_args()
    if v.command=='prepare':prepare(v.settings,v.output)
    elif v.command=='run':run(v.protocol,v.publication_commit,v.python)
    else:worker(v.protocol,v.publication_commit,v.controller_pid)
