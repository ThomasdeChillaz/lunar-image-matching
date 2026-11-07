"""Cumulative fixed-cap reservations; failed/duplicate jobs never get free time."""
import fcntl,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone


def reserve(ledger_path,ledger_sha256,journal_path,job_id,category,seconds):
    if category not in('training','baseline')or type(seconds)is not int or seconds<=0 or not isinstance(job_id,str)or not job_id:raise ValueError('Explicit positive fixed-cap scientific reservation required')
    ledger_path=Path(ledger_path);raw=ledger_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ledger_sha256:raise ValueError('Budget ledger changed after protocol freeze')
    ledger=json.loads(raw)
    if not ledger['accounting_complete']:raise ValueError('Reconstruct consumed compute before new admission')
    available=ledger['remaining_'+category+'_wall_seconds']
    if available is None or available<0:raise ValueError('Known conservative remaining allowance required')
    journal=Path(journal_path);journal.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(journal)+'.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(journal.read_text())if journal.exists()else dict(ledger_sha256=ledger_sha256,reservations=[])
        if state['ledger_sha256']!=ledger_sha256:raise ValueError('Existing reservation ledger cannot silently change')
        if any(r['job_id']==job_id for r in state['reservations']):raise ValueError('Invocation already charged; preserve original run/deadline')
        charged=sum(r['seconds']for r in state['reservations']if r['category']==category)
        if charged+seconds>available:raise ValueError('Cumulative original compute allowance exceeded')
        entry=dict(job_id=job_id,category=category,seconds=seconds,reserved_utc=datetime.now(timezone.utc).isoformat(),refund_on_failure=False)
        state['reservations'].append(entry);temporary=Path(str(journal)+'.new')
        with temporary.open('w')as stream:json.dump(state,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,journal)
        return dict(reservation=entry,remaining_after_reservation=available-charged-seconds)
