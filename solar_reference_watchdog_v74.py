"""Process-group watchdog for physical prep and metadata children, including startup.

Only the declared Blender reference-prep script or the two existing case/index
programs are admitted. Termination grace is inside the phase's charged hard cap.
"""
from pathlib import Path
from datetime import timedelta
import json,os,signal,subprocess,time
from solar_learning_pipeline_v73 import sha,utc,now,dump_new


def command(c,job,commit):
    if job['kind']=='reference':
        if job['program']!='prepare_solar_reference_v63.py':raise ValueError('No renderer/query-image program is allowed')
        return[c['blender_executable'],'--background','--threads','2','--factory-startup','--python-exit-code','3','--python',job['program'],'--','--protocol',job['protocol'],'--publication-commit',commit]
    if job['kind']!='python'or job['program']not in('index_solar_reference_assets_v69.py','build_solar_eval_data_v69.py'):raise ValueError('Whitelisted metadata program required')
    return[c['python_executable'],job['program'],'--protocol',job['protocol'],'--publication-commit',commit]


def execute(c,phase,job,commit,pending,phase_tick):
    if sha(job['protocol'])!=job['protocol_sha256']:raise ValueError('Child protocol changed after its phase freeze')
    logpath=Path(job['log']);receiptpath=Path(job['watchdog_receipt'])
    if logpath.exists()or receiptpath.exists():raise FileExistsError('Preserve prior child logs/receipts; never retry')
    logpath.parent.mkdir(parents=True,exist_ok=True);receiptpath.parent.mkdir(parents=True,exist_ok=True)
    started=now();tick=time.monotonic();phase_deadline=utc(phase['absolute_deadline_utc']);grace=5.
    work_deadline=min(started+timedelta(seconds=job['work_cap_seconds']),phase_deadline-timedelta(seconds=grace))
    if pending:raise InterruptedError('Stop requested before new child launch')
    if started>=work_deadline:raise TimeoutError('Original phase has insufficient work/cleanup time')
    argv=command(c,job,commit);child=None;awake=None;failure=None;complete=False
    try:
        if job['kind']=='reference':
            archive=Path(c['run_directory'])/'references'/f"u{job['unit_index']:03d}.npz"
            if archive.exists()or Path(str(archive)+'.json').exists():raise FileExistsError('Preserve prior reference payload/receipt; no retry')
            archive.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('x')as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            while child.poll()is None:
                if pending:raise InterruptedError('Stop requested; preserve reference artifacts')
                if now()>=work_deadline or time.monotonic()-tick>=job['work_cap_seconds']or time.monotonic()-phase_tick>=phase['max_calendar_seconds']-grace:raise TimeoutError('Original physical/metadata child allowance exhausted, including startup')
                time.sleep(.25)
            if pending or now()>=work_deadline:raise TimeoutError('Interrupted/late reference child exit')
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,argv)
            complete=True
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc))
        if child is not None and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=max(0.,min(3.,(phase_deadline-now()).total_seconds()-1.5)))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:child.wait(timeout=1.)
                except subprocess.TimeoutExpired:failure['cleanup_failure']='Child did not reap after SIGKILL; no unbounded cleanup wait is permitted'
        raise
    finally:
        if awake is not None and awake.poll()is None:
            awake.terminate()
            try:awake.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                awake.kill()
                try:awake.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
        receipt=dict(complete=complete,program=job['program'],kind=job['kind'],child_protocol_sha256=job['protocol_sha256'],
            publication_commit=commit,watchdog_source_sha256=sha(__file__),phase_protocol_sha256=phase['_protocol_sha256'],
            phase_protocol=phase['_protocol_path'],started_utc=started.isoformat(),ended_utc=now().isoformat(),work_deadline_utc=work_deadline.isoformat(),phase_hard_deadline_utc=phase_deadline.isoformat(),
            work_cap_seconds=job['work_cap_seconds'],calendar_seconds=time.monotonic()-tick,
            child_pid=None if child is None else child.pid,child_returncode=None if child is None else child.returncode,failure=failure,
            scope='External termination/startup receipt only, not a physical-label/accuracy claim; cleanup grace is inside charged phase cap.')
        dump_new(receiptpath,receipt)
    return receipt
