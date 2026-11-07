"""Same-deadline completion after the preserved checksum failure, then geometry gate.

No training or final reserve inference is automatically admitted. Each phase
uses its published protocol and original fixed allowance; failures preserve all
outputs and never restart merely because an observation timed out.
"""
import argparse,json,os,signal,subprocess,time,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def run(protocol_path,commit,python,blender):
    pending_signal=[]
    def interrupted(signum,frame):pending_signal.append(signum)
    def check_interrupted():
        if pending_signal:raise InterruptedError('Pipeline interrupted; terminate active phase and preserve outputs')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    p=read(protocol_path);state_path=Path(p['state_path'])
    if state_path.exists():raise FileExistsError('Preserve prior pipeline; no automaticrestart')
    if subprocess.check_output(['git','show',commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Pipeline admission not source-published')
    if sha(__file__)!=p['pipeline_source_sha256']or subprocess.check_output(['git','show',commit+':'+str(Path(__file__).name)])!=Path(__file__).read_bytes():raise ValueError('Executed pipeline source differs from freeze')
    for n,h in p['protocol_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Frozen child protocol changed:'+n)
    completed=[];queue=datetime.now(timezone.utc)
    def state(stage,**extra):state_path.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),queue_started_utc=queue.isoformat(),publication_commit=commit,pipeline_protocol_sha256=sha(protocol_path),completed_phases=completed,scientific_goal_complete=False,**extra),indent=2)+'\n')
    def phase(name,argv,cap,absolute=None):
        check_interrupted()
        start=datetime.now(timezone.utc);tick=time.monotonic();deadline=start+timedelta(seconds=cap)
        if absolute is not None:deadline=min(deadline,datetime.fromisoformat(absolute))
        if start>=deadline:raise TimeoutError('Original phase deadline already expired:'+name)
        with Path(p['log_prefix']+name+'.log').open('x')as log:
            child=None;awake=None
            try:
                child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'),start_new_session=True)
                check_interrupted()
                awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                state(name,state='running',child_pid=child.pid,phase_started_utc=start.isoformat(),phase_absolute_deadline_utc=deadline.isoformat(),phase_cap_seconds=cap)
                while child.poll()is None:
                    check_interrupted()
                    if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=cap:raise TimeoutError('Original serial-phase allowance exhausted:'+name)
                    time.sleep(.5)
                if child.returncode:raise subprocess.CalledProcessError(child.returncode,argv)
                check_interrupted()
                if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=cap:raise TimeoutError('Phase exited after its original allowance:'+name)
                completed.append(dict(phase=name,started_utc=start.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),observed_calendar_seconds=(datetime.now(timezone.utc)-start).total_seconds(),cap_seconds=cap))
            except BaseException:
                if child is not None and child.poll()is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                if awake is not None and awake.poll()is None:awake.terminate()
    try:
        contract=p['completion_contract'];c=read(contract)
        phase('missing_data_completion',[python,'run_solar_completion_v70.py','--protocol',contract,'--publication-commit',commit,'--blender',blender],50400,c['completion_absolute_deadline_utc'])
        summary=p['completion_summary']
        if not read(summary)['complete']:raise ValueError('Data completion has not finished')
        phase('full_geometry_audit',[python,'verify_solar_development_v68.py','--dataset-summary',summary,'--completion-contract',contract,'--output',p['geometry_audit_output'],'--publication-commit',commit],3600)
        audit=read(p['geometry_audit_output'])
        if not audit['passed']or not audit['complete_dataset']or audit['completed_units']!=96:raise ValueError('Full geometry gate failed; nolearning admitted')
        state('data and full geometry ready; source-published training bank/pilot admission still required',state='complete',training_started=False,goal_achieved=False)
    except BaseException as e:
        last=read(state_path)if state_path.exists()else{}
        state('failed; preserve every output and original deadline',state='failed',error_type=type(e).__name__,error=str(e),last_active_phase={k:last[k]for k in('stage','child_pid','phase_started_utc','phase_absolute_deadline_utc','phase_cap_seconds')if k in last})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--python',required=True);p.add_argument('--blender',required=True);a=p.parse_args();run(a.protocol,a.publication_commit,a.python,a.blender)
