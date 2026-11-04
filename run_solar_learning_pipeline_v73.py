"""Run already-published learning stages serially; stop at publication barriers.

Scientific execution is delegated to the existing run_solar_job_v70 API. Its
original watchdog, child cleanup, source checks and canonical compute charges
remain authoritative. New dependent protocols are prepared but never published
or executed until their exact bytes are present in the supplied source commit.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from solar_learning_pipeline_v73 import (PublicationPending,completion,load_contract,now,
    prepare_next,protocol_path,read,remaining,sha,utc,verify_publication)


def command_for_stage(protocol,stage,commit,python):
    p=read(protocol);program='build_solar_training_bank_v68.py'if stage.startswith('teacher_')else'train_solar_response_v63.py'
    argv=[python,'run_solar_job_v70.py','--protocol',protocol,'--publication-commit',commit,'--python',python,'--program',program]
    if stage.endswith('_replay'):argv+=['--replay-pilot',p['output'],'--replay-output',p['exact_replay_output']]
    return argv


def write_state(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=Path(str(path)+'.new')
    with temporary.open('w')as f:json.dump(state,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def run(contract_path,commit,python=sys.executable,*,executor=None,publication_verifier=verify_publication):
    c=load_contract(contract_path);journal_path=Path(c['stage_journal']);journal_path.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(journal_path)+'.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Another learning dispatcher owns this serial pipeline')from exc
        state=read(journal_path)if journal_path.exists()else dict(contract_sha256=sha(contract_path),events=[],scientific_goal_complete=False)
        if state['contract_sha256']!=sha(contract_path):raise ValueError('Learning state belongs to a different frozen contract')
        for _ in range(11):
            ready=prepare_next(contract_path)
            if ready.get('learning_complete'):
                state.update(status='learning_stages_complete',scientific_goal_complete=False);write_state(journal_path,state);return ready
            if ready.get('awaiting_development_selection'):
                state.update(status='awaiting_development_selection');write_state(journal_path,state);return ready
            stage=ready['stage'];path=ready['protocol']
            if any(e['stage']==stage and e['status']in('running','failed')for e in state['events']):
                raise ValueError('A partial/failed learning invocation is preserved; no automatic restart')
            try:p=publication_verifier(contract_path,path,commit)
            except PublicationPending as exc:
                state.update(status='awaiting_source_publication',next_stage=stage,next_protocol=path);write_state(journal_path,state)
                return dict(ready,status='awaiting_source_publication',reason=str(exc),jobs_started_this_call=sum(e.get('dispatcher_commit')==commit for e in state['events']))
            if p['reservation_journal']!=c['reservation_journal']or p['compute_ledger_sha256']!=c['compute_ledger_sha256']:
                raise ValueError('Prepared child protocol changed canonical budget identities')
            mode='replay'if stage.endswith('_replay')else'run';cap=p['replay_cap_seconds']if mode=='replay'else p['max_calendar_seconds']
            if utc(p['replay_latest_start_utc']if mode=='replay'else p['latest_start_utc'])<now():
                raise TimeoutError('Original published learning-stage latest start expired; no reset')
            if cap>remaining(c)['training']:raise ValueError('Published scientific stage no longer fits cumulative budget')
            event=dict(stage=stage,status='running',protocol_sha256=sha(path),started_utc=now().isoformat(),dispatcher_commit=commit,
                argv=command_for_stage(path,stage,commit,python));state['events'].append(event);write_state(journal_path,state)
            try:
                if executor is None:
                    from run_solar_job_v70 import run as execute
                else:execute=executor
                program='build_solar_training_bank_v68.py'if stage.startswith('teacher_')else'train_solar_response_v63.py'
                # Exact existing API; no extra outer process can orphan its child.
                execute(path,commit,python,program,
                    replay_root=p['output']if mode=='replay'else None,
                    replay_output=p['exact_replay_output']if mode=='replay'else None)
                completed=completion(c,stage,True)
                event.update(status='complete',ended_utc=now().isoformat(),watchdog_receipt=completed['watchdog_receipt'],watchdog_receipt_sha256=completed['watchdog_receipt_sha256'])
                write_state(journal_path,state)
            except BaseException as exc:
                event.update(status='failed',ended_utc=now().isoformat(),error_type=type(exc).__name__,error=str(exc))
                state['status']='failed_preserved';write_state(journal_path,state);raise
        raise ValueError('Learning pipeline exceeded its finite declared stage count')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--python',default=sys.executable)
    a=p.parse_args();print(json.dumps(run(a.contract,a.publication_commit,a.python),indent=2,allow_nan=False))
