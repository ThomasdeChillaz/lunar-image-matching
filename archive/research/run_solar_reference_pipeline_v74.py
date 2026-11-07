"""Serial reference/case dispatcher; exact dependent protocols stop for publication."""
from pathlib import Path
import argparse,fcntl,json,os,signal,time
from solar_learning_pipeline_v73 import sha,read,now,utc,dump_new,PublicationPending
from solar_reference_pipeline_v74 import (load_contract,prepare_next,verify_publication,completion,remaining,validate_reference,metadata_outputs,stage_paths,STAGES)


def state_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+'.new')
    with tmp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def run(contract_path,commit,*,executor=None,publication_verifier=verify_publication):
    c=load_contract(contract_path);journal=Path(c['stage_journal']);journal.parent.mkdir(parents=True,exist_ok=True)
    pending=[]
    def interrupted(signum,frame):pending.append(signum)
    prior_handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        with Path(str(journal)+'.lock').open('a')as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ValueError('Another physical dispatcher owns the stage journal')from exc
            state=read(journal)if journal.exists()else dict(contract_sha256=sha(contract_path),events=[],scientific_goal_complete=False)
            if state['contract_sha256']!=sha(contract_path):raise ValueError('Reference state belongs to another immutable contract')
            for _ in range(len(STAGES)+1):
                ready=prepare_next(contract_path)
                if ready.get('reference_preparation_complete'):
                    state.update(status='reference_case_assets_complete');state_write(journal,state);return ready
                stage=ready['stage'];path=ready['protocol']
                if any(e['stage']==stage and e['status']in('running','failed')for e in state['events']):raise ValueError('Partial/failed physical phase preserved; no automatic retry')
                try:c,p=publication_verifier(contract_path,path,commit)
                except PublicationPending as exc:
                    state.update(status='awaiting_source_publication',next_protocol=path,next_stage=stage);state_write(journal,state)
                    return dict(ready,status='awaiting_source_publication',reason=str(exc))
                if now()>utc(p['latest_start_utc']):raise TimeoutError('Original published phase dispatch lease expired; no reset')
                if p['max_calendar_seconds']>remaining(c):raise ValueError('Canonical baseline budget changed before dispatch')
                if pending:raise InterruptedError('Stop requested before new phase admission')
                from solar_compute_budget_v67 import reserve
                started=now();tick=time.monotonic()
                reservation=reserve(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'],sha(path)+'/reference_physics','baseline',p['max_calendar_seconds'])
                p['_protocol_sha256']=sha(path);p['_protocol_path']=path;root=Path(p['output']);root.mkdir(parents=True,exist_ok=False)
                event=dict(stage=stage,status='running',phase_protocol_sha256=sha(path),started_utc=started.isoformat(),reservation=reservation)
                state['events'].append(event);state_write(journal,state)
                try:
                    if executor is None:
                        from solar_reference_watchdog_v74 import execute
                    else:execute=executor
                    fresh=[]
                    for job in p['jobs']:
                        if pending or now()>=utc(p['absolute_deadline_utc'])or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original physical phase allowance exhausted')
                        for n,h in c['source_sha256'].items():
                            if sha(n)!=h:raise ValueError('Scientific source changed mid-phase:'+n)
                        execute(c,p,job,commit,pending,tick)
                        if job['kind']=='reference':fresh.append(validate_reference(c,job['unit_index']))
                    if stage.startswith('reference_')and stage not in('reference_index',):
                        rows=p['prior_reference_rows']+fresh
                        expected={'reference_one':1,'reference_four':4,'reference_full':len(c['selection_units'])}[stage]
                        if len(rows)!=expected or len({r['unit_index']for r in rows})!=expected:raise ValueError('No physical reference may be dropped or repeated')
                    else:rows=[]
                    outputs=metadata_outputs(c,stage)
                    if pending or now()>=utc(p['absolute_deadline_utc'])or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Late/interrupted physical phase completion')
                    summary=dict(complete=True,stage=stage,protocol_sha256=sha(path),publication_commit=commit,reference_rows=rows,
                        calendar_seconds=time.monotonic()-tick,started_utc=started.isoformat(),ended_utc=now().isoformat(),absolute_deadline_utc=p['absolute_deadline_utc'],max_calendar_seconds=p['max_calendar_seconds'],
                        source_sha256=c['source_sha256'],output_sha256=outputs,
                        scope='Measured reference preparation/metadata only; no neural matching, training, final reserve or accuracy result.')
                    dump_new(root/'summary.json',summary);dump_new(root/'COMPLETE.json',dict(complete=True,summary_sha256=sha(root/'summary.json'),protocol_sha256=sha(path)))
                    event.update(status='complete',ended_utc=now().isoformat());state_write(journal,state);completion(c,stage,True)
                except BaseException as exc:
                    event.update(status='failed',ended_utc=now().isoformat(),error_type=type(exc).__name__,error=str(exc));state.update(status='failed_preserved');state_write(journal,state)
                    dump_new(root/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),refund=False));raise
            raise ValueError('Finite physical stage list exhausted unexpectedly')
    finally:
        for s,handler in prior_handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True);p.add_argument('--publication-commit',required=True)
    a=p.parse_args();print(json.dumps(run(a.contract,a.publication_commit),indent=2))
