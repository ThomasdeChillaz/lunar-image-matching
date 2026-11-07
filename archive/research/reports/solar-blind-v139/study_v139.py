"""Source-frozen recovery within the existing user-approved40-minute supplement.

No new allowance is created. One remaining480-second reservation covers the full
generated cost gate and, only if it fits, every clean diagnostic case and replay.
"""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HOME=Path('reports/solar-blind-v139')
PREP=Path('reports/solar-blind-v138')
WORK=Path('runs/solar-blind-v139')
PRIVATE=Path('runs/solar-active-v137-allowance')
LEDGER=PRIVATE/'ledger.json'
JOURNAL=PRIVATE/'reservations.json'
PARENT_SUMMARY=Path('reports/solar-active-v137/allowance.json')
CAP=480


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def check(ok,message):
    if not ok:raise ValueError(message)
def write(path,value):
    path=Path(path)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def original_guard():
    summary=read(PARENT_SUMMARY)
    check(all(p.resolve()==(ROOT/p).absolute() for p in (LEDGER,JOURNAL,PRIVATE)),
          'No aliases of the original supplementary ledger or journal')
    check(sha('reports/solar_compute_admission_v67.json')==summary['original_ledger_sha256']
          and sha('runs/solar-compute-reservations-v67.json')==summary['original_journal_sha256'],
          'Original cumulative accounting cannot change or be refunded')
    check(sha(LEDGER)==summary['private_ledger_sha256'],'Approved parent allowance changed')
    parent=read(LEDGER)
    check(parent['user_approved'] and parent['accounting_complete'] and parent['paid_compute'] is False
        and parent['remaining_baseline_wall_seconds']==2400 and parent['remaining_training_wall_seconds']==0,
        'Existing approved2400-second supplement required; no new allowance inferred')
    return summary


def build():
    original_guard()
    journal=read(JOURNAL)
    check(journal['ledger_sha256']==sha(LEDGER),'Supplementary journal identity changed')
    check(sum(r['seconds'] for r in journal['reservations'])==1920 and len(journal['reservations'])==2,
          'Only the unspent480-second recovery envelope is available; no refunds or alias journals')
    old=read('reports/solar-active-v137/protocol.json');prep=read(PREP/'data-preparation.json')
    check(prep['prepared_only'] and prep['calibration_or_final_access'] is False,'Development preparation required')
    sources=dict(old['sources'])
    for p in (Path('reports/solar-active-v137/replay_saved.py'),Path('requirements-lock.txt')):
        sources[str(p)]=sha(p)
    for folder in (PREP,HOME):
        for p in folder.iterdir():
            if p.suffix in ('.py','.md'):sources[str(p)]=sha(p)
    inputs=dict(prep['source_files'])
    for p in (PREP/'inference-inputs.json',PREP/'scoring-inputs.json',PREP/'data-preparation.json',PARENT_SUMMARY,
              Path('reports/solar_full_geometry_audit_v70.json')):inputs[str(p)]=sha(p)
    for n,h in {**sources,**inputs}.items():check(sha(n)==h,'Changed frozen dependency: '+n)
    oracle=read('reports/solar_full_geometry_audit_v70.json')
    check(oracle['passed'] and oracle['complete_dataset'] and oracle['oracle_success']>=.95,'Original exact geometry gate required')
    inf,labels=read(PREP/'inference-inputs.json'),read(PREP/'scoring-inputs.json')
    check(len(inf)==len(labels)==8 and sum(x['positive'] for x in labels)==4,'Keep all eight clean diagnostic cases')
    render=read('runs/solar-clean-radiance-v106/render/COMPLETE.json')
    offline=dict(reference_render_export_seconds=sum(r['render_and_export_seconds'] for r in render['rows'] if r['role']=='reference'),
        full104_image_cohort_elapsed_seconds=render['elapsed_seconds'],incremental_render_seconds=0.,
        scope='Prior measured clean reference render/export; original terrain/camera preparation is separate historical cost')
    return dict(schema='existing_supplement_blind_recovery_v139',cap_seconds=CAP,offline_reference_preparation=offline,
        authorization_basis='Existing explicit user-approved2400-second supplement and resumed research goal; no new total allowance',
        recovery_diagnosis='Supplied matching illumination in v137 leaves the primary unknown-light question unresolved; use the published clean catalogue preparation.',
        prior_supplement_charges_seconds=1920,after_reservation_total_seconds=2400,
        parent_journal_prefix_digest=digest(journal['reservations']),
        original_accounting_unchanged=True,sources=sources,inputs=inputs,
        inference_manifest=str(PREP/'inference-inputs.json'),scoring_manifest=str(PREP/'scoring-inputs.json'),
        case_ids=[r['id'] for r in inf],methods=['direct','joint_no_extra','uniform','appearance','geometry','active'],
        comparison_cells=48,reference_images_per_case=24,extra_patch_budget=32,native_pixels=512,
        cpu_threads=2,opencv_threads=1,paid_compute=False,calibration_or_final_access=False,
        query_truth_inputs=False,training=False,accepted_pose_study=False,
        generated_controls='Full24-light/2048-reference-camera/32-extra-search six-arm flat and layered control; not a physical sunlight simulation',
        forecast='1.5*8*worst complete generated pair +30s; must fit the original remaining480-second deadline after timing',
        full_roster_or_no_lunar_inference=True,model_promotion=False,outreach=False)


def pins(protocol):
    for n,h in {**protocol['sources'],**protocol['inputs']}.items():check(sha(n)==h,'Changed source/input: '+n)


def published(commit,protocol):
    subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/main'],check=True)
    for n in (*protocol['sources'],str(HOME/'protocol.json')):
        check(subprocess.check_output(['git','show',commit+':'+n])==Path(n).read_bytes(),'Source differs from public freeze: '+n)


def check_ticket(ticket,protocol):
    original_guard();pins(protocol)
    check(ticket['parent_pid']==os.getppid() and ticket['protocol_sha256']==sha(HOME/'protocol.json'),
          'Original admitted controller and frozen protocol required')
    journal=read(JOURNAL);reservations=journal['reservations']
    check(journal['ledger_sha256']==sha(LEDGER) and digest(reservations[:2])==protocol['parent_journal_prefix_digest'] and len(reservations)==3
        and sum(r['seconds'] for r in reservations)==2400 and ticket['reservation']==reservations[-1],
        'Single original480-second recovery reservation required')
    deadline=datetime.fromisoformat(ticket['deadline'])
    check(deadline==datetime.fromisoformat(ticket['reservation']['reserved_utc'])+timedelta(seconds=CAP)
        and datetime.now(timezone.utc)<deadline,'Original recovery deadline expired')
    expected=PRIVATE/'scientific.lock'
    st=os.fstat(ticket['lease_fd']);actual=expected.stat()
    check((st.st_dev,st.st_ino)==(actual.st_dev,actual.st_ino),'Inherited serial scientific lease required')
    signal.setitimer(signal.ITIMER_REAL,max(.001,(deadline-datetime.now(timezone.utc)).total_seconds()))


def forecast_fits(timing,seconds_remaining):
    samples=timing.get('samples',[])
    check(len(samples)==2 and all(r.get('all_six_pose_controls_pass') and r.get('cameras')==2048
        and r.get('lights')==24 and r.get('extra_searches')==32
        and isinstance(r.get('elapsed_seconds'),(int,float)) and r['elapsed_seconds']>0 for r in samples)
        and {r.get('layered') for r in samples}=={False,True},
        'Complete flat/layered full-budget pilot required')
    value=1.5*8*max(r['elapsed_seconds'] for r in samples)+30.
    check(abs(value-timing['comparison_forecast_seconds'])<1e-8,'Timing forecast changed')
    return value<=seconds_remaining


def sealed(protocol):
    raw,seal=read(WORK/'predictions.json'),read(WORK/'PREDICTIONS_SEALED.json')
    check(seal['sha256']==sha(WORK/'predictions.json') and seal['rows']==len(raw['rows'])==len(protocol['case_ids'])==8
        and [r['id'] for r in raw['rows']]==protocol['case_ids']
        and raw['query_truth_used'] is False,'All eight truth-free predictions must be sealed before scoring')


def worker(operation,ticket_path):
    import resource
    ticket=read(ticket_path);protocol=read(HOME/'protocol.json');check_ticket(ticket,protocol);published(ticket['commit'],protocol)
    sys.path.insert(0,str(ROOT/HOME))
    import engine_v139 as engine
    engine.cv2.setNumThreads(1)
    def checked(name):
        check(name in protocol['inputs'] and sha(name)==protocol['inputs'][name],'Changed/unapproved data path: '+name)
        return Path(name)
    put=lambda name,value:write(WORK/name,value)
    get=lambda name:read(WORK/name)
    start=time.perf_counter()
    if operation=='timing':engine.timing(put)
    elif operation=='predict':engine.predict(protocol,put,checked)
    elif operation=='score':sealed(protocol);engine.score(protocol,put,checked,get)
    elif operation=='replay':sealed(protocol);engine.replay(protocol,put,checked,get)
    else:raise ValueError('Unknown operation')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(WORK/(operation+'-runtime.json'),dict(elapsed_seconds=time.perf_counter()-start,
        whole_worker_peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),
        memory_scope='Whole worker; shared caches and imports; not a per-model memory ranking'))


def execute(commit):
    from solar_compute_budget_v67 import reserve
    protocol=read(HOME/'protocol.json');pins(protocol);summary=original_guard();published(commit,protocol)
    check(not WORK.exists() and not WORK.is_symlink(),'Preserve every prior attempt; do not overwrite a recovery run')
    with (PRIVATE/'scientific.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=read(JOURNAL)
        check(digest(journal['reservations'])==protocol['parent_journal_prefix_digest'],'Parent charges changed before admission')
        charged=reserve(LEDGER,summary['private_ledger_sha256'],JOURNAL,sha(HOME/'protocol.json')+'/blind-recovery-v139','baseline',CAP)['reservation']
        deadline=datetime.fromisoformat(charged['reserved_utc'])+timedelta(seconds=CAP)
        WORK.mkdir();ticket=dict(commit=commit,parent_pid=os.getpid(),lease_fd=lease.fileno(),
            protocol_sha256=sha(HOME/'protocol.json'),reservation=charged,deadline=deadline.isoformat())
        write(WORK/'execution.json',ticket);start=time.perf_counter()
        try:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     VECLIB_MAXIMUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH=str(ROOT))
            for operation in ('timing','predict','score','replay'):
                remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
                check(remaining>0.,'Original recovery deadline expired')
                if operation=='predict':check(forecast_fits(read(WORK/'timing.json'),remaining),'Full comparison cannot fit; retain timing and do not substitute a smaller lunar subset')
                with (WORK/(operation+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker',operation,'--ticket',str(WORK/'execution.json')],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                    try:status=child.wait(timeout=remaining)
                    except BaseException:
                        os.killpg(child.pid,signal.SIGKILL);child.wait();raise
                check(status==0 and datetime.now(timezone.utc)<deadline,'Worker failed or original deadline elapsed')
                if operation=='predict':write(WORK/'PREDICTIONS_SEALED.json',dict(rows=8,sha256=sha(WORK/'predictions.json')))
            original_guard();pins(protocol)
            write(WORK/'COMPLETE.json',dict(ticket,complete=True,elapsed_seconds=time.perf_counter()-start))
        except BaseException as error:
            write(WORK/'FAILED.json',dict(ticket,complete=False,error=type(error).__name__,message=str(error)))
            raise


if __name__=='__main__':
    check(Path.cwd()==ROOT,'Run from the original repository root')
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('freeze');run=sub.add_parser('run');run.add_argument('--commit',required=True)
    job=sub.add_parser('worker');job.add_argument('operation',choices=('timing','predict','score','replay'));job.add_argument('--ticket',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='freeze':write(HOME/'protocol.json',build())
    elif args.action=='run':execute(args.commit)
    else:worker(args.operation,args.ticket)
