"""Separate missing-only data completion; source-published before execution.

Ten complete units are copied byte-for-byte. Only original missing86 units are
rendered. Original failure/expiry and elapsed costs remain recorded separately.
A distinct14hour cap is fixed in the completion contract and never resets.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import argparse,json,os,shutil,signal,subprocess,time
from solar_completion_v68 import (RENDER_SOURCES,sha,read,verify_publication,verify_original,
                                  verify_seal,admit,require_renderer_summary)


def run(contract_path,commit,blender,resume=False):
    pending_signal=[]
    def interrupted(signum,frame):pending_signal.append(signum)
    def check_interrupted():
        if pending_signal:raise InterruptedError('Completion interrupted; terminate active renderer and preserve outputs')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    contract=verify_publication(contract_path,commit);master,snapshot=verify_original(contract)
    root=Path(contract['new_output_root']);state_path=Path(contract['state_path']);deadline=datetime.fromisoformat(contract['completion_absolute_deadline_utc'])
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Frozen NEWcompletion deadline expired; originalv67deadline remainsfailed')
    if state_path.exists():
        if not resume:raise FileExistsError('Preserve completion state; explicit same-deadline resume only')
        prior=read(state_path)
        if prior['state']=='running'or prior['publication_commit']!=commit:raise ValueError('Do not replace a live/differently frozen completion')
        if subprocess.run(['ps','-p',str(prior['pid']),'-o','command='],capture_output=True,text=True).stdout.strip():raise ValueError('Prior controller PID still exists; inspect before resume')
    elif resume:raise ValueError('No previous completion to resume')
    if root.exists()and not resume:raise FileExistsError('New completion output must not already exist')
    root.mkdir(parents=True,exist_ok=resume);rows=[];receipts=[];queue_start=datetime.now(timezone.utc)
    contract_digest=sha(contract_path)
    expected={n:contract['render_source_sha256'][n]for n in RENDER_SOURCES}
    def state(stage,**extra):
        state_path.write_text(json.dumps(dict(state=extra.pop('state','running'),stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),publication_commit=commit,completion_contract_sha256=sha(contract_path),completion_absolute_deadline_utc=deadline.isoformat(),original_shared_deadline_utc=contract['original_shared_deadline_utc'],original_deadline_met=False,complete_units=len(rows),planned_units=96,matcher_forwards=0,optimizer_updates=0,**extra),indent=2)+'\n')
    def unchanged():
        check_interrupted()
        if sha(contract_path)!=contract_digest:raise ValueError('Frozen completion contract changed')
        for n,h in expected.items():
            if sha(n)!=h:raise ValueError('Rendering dependency changed while completion live:'+n)
        for key in('original_master','original_failed_state','snapshot','schedule_replay'):
            if sha(contract[key])!=contract[key+'_sha256']:raise ValueError('Preserved admission input changed:'+key)
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Original NEWcompletion shared deadline exhausted')
    def destination(index):return root/Path(master['units'][index]['output']).relative_to(contract['original_root'])
    def record(index,reused,outer_seconds=None):
        folder=destination(index);summary=verify_seal(folder)
        if summary['protocol_sha256']!=contract['original_master_sha256']:raise ValueError('Unit physical schedule provenance changed')
        row=dict(unit_index=index,output=str(folder),summary_sha256=sha(folder/'summary.json'),seal_sha256=sha(folder/'COMPLETE.json'),rendered_images=48,reused_original_unit=reused,outer_calendar_seconds=outer_seconds)
        rows.append(row)
        (root/'progress.json').write_text(json.dumps(dict(units=rows,complete=False,expected_units=96,completion_contract_sha256=sha(contract_path),completion_absolute_deadline_utc=deadline.isoformat(),original_deadline_met=False),indent=2)+'\n')
    try:
        copy_start=time.monotonic();state('copying ten immutable completed units')
        for original in snapshot['complete_units']:
            unchanged();index=original['unit_index'];folder=destination(index)
            if folder.exists():
                if not resume:raise FileExistsError('Preserve previously copied output')
                if sha(folder/'COMPLETE.json')!=original['seal_sha256']:raise ValueError('Copied original seal changed')
            else:shutil.copytree(original['output'],folder,copy_function=shutil.copy2)
            if sha(folder/'COMPLETE.json')!=original['seal_sha256']:raise ValueError('Copied old seal not byte-identical')
            record(index,True)
        receipts.append(dict(phase='copy_complete_units',observed_calendar_seconds=time.monotonic()-copy_start,images_copied=480))
        for index in contract['missing_unit_indices']:
            unchanged();unit,folder,_=admit(contract,index,datetime.now(timezone.utc))
            if(folder/'COMPLETE.json').exists():
                if not resume:raise FileExistsError('Existing completed new unit requires explicit same-deadline resume')
                record(index,False);continue
            if folder.exists():raise FileExistsError('Partial NEWcompletion unit preserved; separately declared amendment required:'+str(folder))
            folder.parent.mkdir(parents=True,exist_ok=True);start=datetime.now(timezone.utc);tick=time.monotonic();unit_deadline=min(start+timedelta(seconds=900),deadline)
            log_path=Path(contract['log_prefix']+f'{index:03d}.log')
            if log_path.exists():raise FileExistsError('Preserve previous per-unit log')
            with log_path.open('x')as log:
                command=[blender,'--background','--threads','2','--factory-startup','--python-exit-code','3','--python','render_solar_completion_v68.py','--','--protocol',contract_path,'--publication-commit',commit,'--unit-index',str(index)]
                child=None;awake=None
                try:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'),start_new_session=True)
                    check_interrupted()
                    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(child.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    active=dict(unit_index=index,unit_started_utc=start.isoformat(),unit_absolute_deadline_utc=unit_deadline.isoformat(),child_pid=child.pid,command=command)
                    state('rendering only missing original units',**active)
                    while child.poll()is None:
                        unchanged()
                        if datetime.now(timezone.utc)>=unit_deadline or time.monotonic()-tick>=900:raise TimeoutError('Original NEWunit900second allowance exhausted')
                        time.sleep(.5)
                    if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
                    unchanged()
                    summary=require_renderer_summary(folder,contract_digest)
                except BaseException:
                    if child is not None and child.poll()is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
                finally:
                    if awake is not None and awake.poll()is None:awake.terminate()
            seal=folder/'COMPLETE.json'
            with seal.open('x')as stream:json.dump(dict(sha256={str(p.relative_to(folder)):sha(p)for p in sorted(folder.rglob('*'))if p.is_file()}),stream,indent=2);stream.write('\n')
            elapsed=(datetime.now(timezone.utc)-start).total_seconds();record(index,False,elapsed)
            receipts.append(dict(phase='new_unit',unit_index=index,started_utc=start.isoformat(),ended_utc=datetime.now(timezone.utc).isoformat(),observed_calendar_seconds=elapsed,internal_program_seconds=summary['calendar_seconds']))
            (root/'execution_receipts.json').write_text(json.dumps(dict(original_deadline_met=False,phases=receipts),indent=2)+'\n')
            print('MISSING-ONLY COMPLETION UNIT',index,'complete; total',len(rows),'/96',flush=True)
        unchanged();verify_original(contract)
        if len(rows)!=96 or {r['unit_index']for r in rows}!=set(range(96)):raise ValueError('All original96camera pairs required')
        result=dict(complete=True,units=rows,original_master=contract['original_master'],original_master_sha256=contract['original_master_sha256'],master_protocol_sha256=contract['original_master_sha256'],completion_contract=contract_path,completion_contract_sha256=sha(contract_path),source_publication_commit=commit,rendered_images=4608,reused_original_images=480,newly_rendered_images=4128,original_deadline_met=False,original_failure_preserved=True,original_observed_unit_outer_seconds=contract['original_observed_unit_outer_seconds'],new_completion_observed_queue_seconds=(datetime.now(timezone.utc)-queue_start).total_seconds(),new_completion_execution_phases=receipts,calibration_or_final_access=False,matcher_forwards=0,optimizer_updates=0,geometry_audit_pending=True,fitting_allowed=False,scope='Missing-only completion of original96unit camera/light schedule. Reused ten units retain originalbyte-identicalseals. Extra views/lights are conditions, not independent geography; full95%pose-control audit precedes any learning.')
        with(root/'summary.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
        state('complete data; full geometry audit required before learning',state='complete',fitting_allowed=False)
        return result
    except BaseException as exc:
        # Retain the active per-unit deadline/child in the terminal receipt.
        last=read(state_path)if state_path.exists()else{}
        state('failed; preserve all original/copied/new outputs',state='failed',error_type=type(exc).__name__,error=str(exc),last_active_unit={k:last[k]for k in('unit_index','unit_started_utc','unit_absolute_deadline_utc','child_pid')if k in last})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--publication-commit',required=True);parser.add_argument('--blender',required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args();run(a.protocol,a.publication_commit,a.blender,a.resume)
