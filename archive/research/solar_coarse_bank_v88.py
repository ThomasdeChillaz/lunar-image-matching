"""Source-only coarse128 bank preparation/worker. No fitting or job at import.

Future command: prepare metadata-only; publish; run one admitted pilot/full.
Actual scientific activation belongs to the root, not this prepared source.
"""
import argparse
from datetime import timedelta
import hashlib
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
SUFFIX='/coarse128_teacher_bank'
FORMAT='distributional-coarse128-bank-v88'
EXTRA_SOURCES=('solar_coarse_bank_v88.py','reports/qualify_solar_coarse_teacher_v87.py',
               'solar_coarse_training_adapter_v86.py','solar_distributional_response_v86.py')


def qualified_teacher(protocol_path,run_directory):
    """Completed receipt replay only: never reapply the old dispatch lease."""
    p=read(protocol_path);r=Path(run_directory);seal=read(r/'COMPLETE.json');w=read(r/'watchdog.json');a=read(r/'candidate.json')
    if (p['purpose']!='coarse_teacher_domain_qualification'or p['category']!='training'or p['max_calendar_seconds']!=600
            or seal.get('external_watchdog_passed')is not True or w.get('complete')is not True or a.get('complete')is not True or a.get('passed')is not True
            or seal['protocol_sha256']!=sha(protocol_path)or w['protocol_sha256']!=sha(protocol_path)or a['protocol_sha256']!=sha(protocol_path)
            or seal['candidate_sha256']!=sha(r/'candidate.json')or seal['watchdog_sha256']!=sha(r/'watchdog.json')or a['arrays_sha256']!=sha(r/'arrays.npz')):
        raise ValueError('Actual source-bound teacher-domain qualification required')
    elapsed=w['calendar_seconds']
    if type(elapsed)not in(int,float)or not math.isfinite(elapsed)or not 0<elapsed<=600 or w['cap_seconds']!=600 or w['child_returncode']!=0 or w['failure']is not None or utc(w['ended_utc'])>=utc(w['hard_deadline_utc']):raise ValueError('Qualification must finish under its original startup-inclusive cap')
    if a['teacher_descriptor_sha256']!=p['expected_frozen_descriptor_sha256']or not a['odd_refused_before_read']or not a['all_original_rows_retained']or not a['generated_control']['passed']or len(a['rows'])!=24:raise ValueError('Incomplete actual teacher domain/coordinate/even-light evidence')
    if any(not x['exact_teacher_identity']or not x['exact_conversion']or not x['exact_native_to_coarse_centres']or x['compared_rows']<=0 for x in a['rows']):raise ValueError('Exact teacher-domain comparison failed')
    return p,w,a


def completed_bank(protocol_path,run_directory):
    p=read(protocol_path);r=Path(run_directory);seal=read(r/'COMPLETE.json');w=read(r/'watchdog.json');a=read(r/'summary.json')
    if (seal.get('external_watchdog_passed')is not True or not w.get('complete')or not a.get('complete')
            or any(x['protocol_sha256']!=sha(protocol_path)for x in(seal,w,a))or seal['summary_sha256']!=sha(r/'summary.json')or seal['watchdog_sha256']!=sha(r/'watchdog.json')or seal['manifest_sha256']!=sha(r/'manifest.json')or a['manifest_sha256']!=sha(r/'manifest.json')):raise ValueError('Actual complete bank/pilot watchdog and manifest required')
    elapsed=w['calendar_seconds']
    if type(elapsed)not in(int,float)or not math.isfinite(elapsed)or not 0<elapsed<=p['max_calendar_seconds']or w['cap_seconds']!=p['max_calendar_seconds']or w['child_returncode']!=0 or w['failure']is not None or utc(w['ended_utc'])>=utc(w['hard_deadline_utc']):raise ValueError('Measured pilot exceeded its original allowance')
    if not utc(p['created_utc'])<=utc(w['started_utc'])<=utc(p['latest_start_utc'])or utc(w['hard_deadline_utc'])!=min(utc(w['started_utc'])+timedelta(seconds=p['max_calendar_seconds']),utc(p['absolute_deadline_utc'])):raise ValueError('Original completed lease/deadline was changed')
    return p,w,read(r/'manifest.json')


def validate_completed_bank(manifest_path):
    """Stable metadata-only FULL-bank gate shared by future trainer/replay.

    Does not decode data or reapply an expired completed job's launch lease.
    Individual unit array/seal hashes are checked by the data loader.
    """
    m=read(manifest_path);r=Path(m['output'])
    if Path(manifest_path).resolve()!=(r/'manifest.json').resolve():raise ValueError('Exact original bank manifest path required')
    p,w,checked=completed_bank(m['protocol_path'],r)
    if m!=checked or m.get('complete')is not True or p['phase']!='full_bank'or m['phase']!='full_bank'or m['format']!=FORMAT or m['array_schema']!='distributional-coarse128-teacher-v86'or m['split']!='training'or m.get('calibration_or_final_access')is not False or m['heldout_image_reads']!=0 or not m['all_historical_rows_retained']:raise ValueError('Complete training-only full bank required')
    if [r['unit_index']for r in m['records']]!=p['unit_indices']or set(p['unit_indices'])!=set(m['eligible_unit_indices'])or sorted(m['training_groups'])!=sorted(p['training_groups']):raise ValueError('All independently eligible original training cameras/groups required')
    if not m['view_prior']or sha(m['view_prior'])!=m['view_prior_sha256']:raise ValueError('Full training-derived view prior missing/changed')
    return dict(manifest=m,protocol=p,watchdog=w,manifest_sha256=sha(manifest_path),protocol_sha256=sha(m['protocol_path']),complete_path=str(r/'COMPLETE.json'),complete_sha256=sha(r/'COMPLETE.json'),watchdog_path=str(r/'watchdog.json'),watchdog_sha256=sha(r/'watchdog.json'))


def derive_plan(settings,contract,qualification_elapsed,pilot=None):
    ids=list(contract['eligible_training_unit_indices']);groups=list(contract['training_groups'])
    if not ids or len(set(ids))!=len(ids)or len(groups)!=16:raise ValueError('Original independently eligible training inventory required')
    components=contract['training_unit_components']
    # JSON dictionary keys may be decimal strings; existing contract uses them.
    def group(i):return components.get(str(i),components.get(i))
    if {group(i)for i in ids}!=set(groups):raise ValueError('Every original training group required')
    phase=settings['phase']
    if phase=='cost_pilot':
        count=settings['pilot_unit_count']
        if type(count)is not int or count not in(1,4):raise ValueError('Predeclared fresh1or4unit cost pilot required')
        ordered=sorted(ids)
        # Predeclared before any new bank forwards:64eligible gives positions
        #0,21,42,63, spanning terrain and camera variants without timing picks.
        positions=[0]if count==1 else [j*(len(ordered)-1)//3 for j in range(4)]
        selected=[ordered[j]for j in positions]
        if len({group(i)for i in selected})!=count:raise ValueError('Predeclared quantile pilot does not span distinct terrain groups')
        cap=math.ceil(2*qualification_elapsed*count)
    elif phase=='full_bank':
        if pilot is None:raise ValueError('Actual1or4unit cost pilot required before full bank')
        pp,pw,pm=pilot
        if pp['phase']!='cost_pilot'or len(pm['records'])not in(1,4)or pm['reused_records']!=0:raise ValueError('Fresh measured pilot required')
        selected=ids;cap=math.ceil(2*pw['calendar_seconds']*len(ids)/len(pm['records']))
    else:raise ValueError('Only cost_pilot/full_bank')
    if type(cap)is not int or cap<=5:raise ValueError('Measured cap must leave the fixed5s cleanup reserve; do not invent a larger cap')
    return selected,cap


def require_matching_pilot(p,pilot):
    pp,_,_=pilot
    for name in('source_sha256','qualification_protocol','qualification_run','bank_manifest','learning_contract','teacher_directory','teacher_sha256','expected_frozen_descriptor_sha256'):
        if pp[name]!=p[name]:raise ValueError('Measured pilot must match actual source/qualification/data/teacher:'+name)
    expected=[dict(protocol=p['pilot_protocol'],run=p['pilot_run'],complete_sha256=sha(Path(p['pilot_run'])/'COMPLETE.json'))]
    if p['cache_runs']!=expected:raise ValueError('Only the exact predeclared completed pilot cache is admitted')


def prepare(settings_path,output_protocol):
    """Metadata-only builder. Writes PREPARED settings with all UTCs unset."""
    from solar_learning_pipeline_v73 import load_contract,remaining,local_source_closure
    s=read(settings_path);c=load_contract(s['learning_contract'])
    qp,qw,qa=qualified_teacher(s['qualification_protocol'],s['qualification_run'])
    if qp['learning_contract']!=s['learning_contract']or qp['expected_frozen_descriptor_sha256']!=c['expected_frozen_descriptor_sha256']:raise ValueError('Qualified original teacher/contract required')
    for n in('solar_coarse_training_adapter_v86.py','solar_distributional_response_v86.py'):
        if qp['source_sha256'][n]!=sha(n):raise ValueError('Teacher-domain qualification did not cover current adapter')
    pilot=completed_bank(s['pilot_protocol'],s['pilot_run'])if s['phase']=='full_bank'else None
    ids,cap=derive_plan(s,c,qw['calendar_seconds'],pilot)
    sources=local_source_closure();sources.update({n:sha(n)for n in EXTRA_SOURCES})
    if remaining(c)['training']<cap:raise ValueError('Measured bank cap exceeds original cumulative training allowance')
    p=dict(s,prepared_only=True,purpose='coarse128_teacher_bank',category='training',cpu_threads=2,max_calendar_seconds=cap,
        factor_two_measured_projection_seconds=(2*qw['calendar_seconds']*len(ids))if not pilot else 2*pilot[1]['calendar_seconds']*len(ids)/len(pilot[2]['records']),
        created_utc=None,latest_start_utc=None,absolute_deadline_utc=None,publication_dispatch_lease_seconds=1800,
        unit_indices=ids,generated_training_units=c['generated_training_unit_count'],eligible_training_units=c['eligible_training_unit_count'],training_groups=c['training_groups'],
        qualification_protocol_sha256=sha(s['qualification_protocol']),qualification_complete_sha256=sha(Path(s['qualification_run'])/'COMPLETE.json'),qualification_watchdog_sha256=sha(Path(s['qualification_run'])/'watchdog.json'),
        compute_ledger=c['compute_ledger'],compute_ledger_sha256=c['compute_ledger_sha256'],reservation_journal=CANONICAL,
        original_master=c['original_master'],dataset_summary=c['dataset_summary'],geometry_audit=c['geometry_audit'],lighting_schedule=c['lighting_schedule'],
        bank_manifest=qp['bank_manifest'],teacher_directory=c['initializer'],teacher_sha256=c['initializer_sha256'],expected_frozen_descriptor_sha256=c['expected_frozen_descriptor_sha256'],
        source_sha256=sources,calibration_or_final_access=False,cache_runs=[]if not pilot else [dict(protocol=s['pilot_protocol'],run=s['pilot_run'],complete_sha256=sha(Path(s['pilot_run'])/'COMPLETE.json'))])
    if pilot:require_matching_pilot(p,pilot)
    pins=[p[k]for k in('learning_contract','original_master','dataset_summary','geometry_audit','lighting_schedule','bank_manifest','qualification_protocol')]
    pins += [str(Path(p['qualification_run'])/n)for n in('COMPLETE.json','watchdog.json','candidate.json')]
    if pilot:pins += [p['pilot_protocol'],str(Path(p['pilot_run'])/'COMPLETE.json'),str(Path(p['pilot_run'])/'watchdog.json'),str(Path(p['pilot_run'])/'summary.json'),str(Path(p['pilot_run'])/'manifest.json')]
    p['metadata_sha256']={n:sha(n)for n in pins};write(output_protocol,p);return p


def validate_frozen_plan(p,c,started):
    if p.get('prepared_only')is not False or p.get('purpose')!='coarse128_teacher_bank'or p.get('category')!='training'or p.get('cpu_threads')!=2 or p.get('calibration_or_final_access')is not False:raise ValueError('Published two-thread training-category development bank only')
    if p['reservation_journal']!=CANONICAL or p['compute_ledger']!=c['compute_ledger']or p['compute_ledger_sha256']!=c['compute_ledger_sha256']:raise ValueError('Original canonical budget required')
    for key in('original_master','dataset_summary','geometry_audit','lighting_schedule'):
        if p[key]!=c[key]or p['metadata_sha256'][p[key]]!=c[key+'_sha256']:raise ValueError('Original independently gated data lineage changed')
    if p['teacher_directory']!=c['initializer']or p['teacher_sha256']!=c['initializer_sha256']or p['expected_frozen_descriptor_sha256']!=c['expected_frozen_descriptor_sha256']or p['generated_training_units']!=c['generated_training_unit_count']or p['eligible_training_units']!=c['eligible_training_unit_count']or p['training_groups']!=c['training_groups']:raise ValueError('Original teacher/camera inventory changed')
    qp,qw,_=qualified_teacher(p['qualification_protocol'],p['qualification_run'])
    pilot=completed_bank(p['pilot_protocol'],p['pilot_run'])if p['phase']=='full_bank'else None
    ids,cap=derive_plan(p,c,qw['calendar_seconds'],pilot)
    if p['unit_indices']!=ids or p['max_calendar_seconds']!=cap:raise ValueError('Measured cap/original camera schedule changed')
    if p['phase']=='cost_pilot'and p['cache_runs']:raise ValueError('Pilot must extract fresh actual teacher targets')
    if pilot:require_matching_pilot(p,pilot)
    if (p['qualification_protocol_sha256']!=sha(p['qualification_protocol'])or p['qualification_complete_sha256']!=sha(Path(p['qualification_run'])/'COMPLETE.json')or p['qualification_watchdog_sha256']!=sha(Path(p['qualification_run'])/'watchdog.json')or qp['learning_contract']!=p['learning_contract']or qp['bank_manifest']!=p['bank_manifest']or qp['expected_frozen_descriptor_sha256']!=p['expected_frozen_descriptor_sha256']):raise ValueError('Actual qualification provenance changed')
    created,latest,absolute=map(utc,(p['created_utc'],p['latest_start_utc'],p['absolute_deadline_utc']));lease=p['publication_dispatch_lease_seconds']
    if type(lease)is not int or not 0<lease<=1800 or latest!=created+timedelta(seconds=lease)or absolute!=latest+timedelta(seconds=cap):raise ValueError('Finite frozen publication lease required')
    if started<created or started>latest:raise TimeoutError('Original dispatch lease expired')
    if Path(p['output']).is_absolute()or '..'in Path(p['output']).parts or not p['output']:raise ValueError('Versioned repository-relative output required')
    return min(started+timedelta(seconds=cap),absolute)


def verify_sources(protocol_path,p,commit,deadline):
    from solar_learning_pipeline_v73 import local_source_closure
    required=local_source_closure();required.update({n:sha(n)for n in EXTRA_SOURCES})
    if p['source_sha256']!=required or len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact source closure/publication required')
    metadata={p[k]for k in('learning_contract','original_master','dataset_summary','geometry_audit','lighting_schedule','bank_manifest','qualification_protocol')}
    metadata.update(str(Path(p['qualification_run'])/n)for n in('COMPLETE.json','watchdog.json','candidate.json'))
    if p['phase']=='full_bank':metadata.update([p['pilot_protocol'],*(str(Path(p['pilot_run'])/n)for n in('COMPLETE.json','watchdog.json','summary.json','manifest.json'))])
    if set(p['metadata_sha256'])!=metadata:raise ValueError('Exact data/qualification/pilot metadata pins required')
    def public(path):
        remaining=(deadline-now()).total_seconds()
        if remaining<=0:raise TimeoutError('Source preflight exhausted the original deadline')
        value=subprocess.check_output(['git','show',commit+':'+path],timeout=min(10.,remaining))
        if now()>=deadline:raise TimeoutError('Source preflight completed after original deadline')
        return value
    for n,h in {**p['source_sha256'],**p['metadata_sha256']}.items():
        if now()>=deadline:raise TimeoutError('Source preflight exhausted the original deadline')
        if sha(n)!=h or public(n)!=Path(n).read_bytes():raise ValueError('Unpublished or changed source/input:'+n)
    if public(str(protocol_path))!=Path(protocol_path).read_bytes():raise ValueError('Protocol not source-published')


def run(protocol_path,commit,python):
    # Clocks BEFORE protocol/admission/publication/hash imports. No reset later.
    started=now();tick=time.monotonic();p=read(protocol_path)
    from solar_learning_pipeline_v73 import load_contract,remaining
    c=load_contract(p['learning_contract']);deadline=validate_frozen_plan(p,c,started);cap=p['max_calendar_seconds'];verify_sources(protocol_path,p,commit,deadline)
    if remaining(c)['training']<cap:raise ValueError('Cumulative original training allowance exceeded')
    if now()>=deadline-timedelta(seconds=5)or time.monotonic()-tick>=cap-5:raise TimeoutError('Preflight exhausted original allowance')
    root=Path(p['output']);root.mkdir(parents=True,exist_ok=False);work_deadline=deadline-timedelta(seconds=5)
    child=None;failure=None;complete=False;charged=None;pending=[]
    old={s:signal.signal(s,lambda signum,frame:pending.append(signum))for s in(signal.SIGINT,signal.SIGTERM)}
    try:
        from solar_compute_budget_v67 import reserve
        charged=reserve(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL,sha(protocol_path)+SUFFIX,'training',cap)
        write(root/'reservation.json',charged);write(root/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=commit,controller_pid=os.getpid(),started_utc=started.isoformat(),work_deadline_utc=work_deadline.isoformat(),hard_deadline_utc=deadline.isoformat(),cap_seconds=cap))
        if pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise TimeoutError('Startup exhausted allowance')
        with(root/'worker.log').open('x')as f:
            child=subprocess.Popen([python,'solar_coarse_bank_v88.py','worker','--protocol',str(protocol_path),'--publication-commit',commit,'--controller-pid',str(os.getpid())],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
            while child.poll()is None:
                if pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise TimeoutError('Original work allowance exhausted')
                time.sleep(.1)
        if child.returncode or pending or now()>=work_deadline or time.monotonic()-tick>=cap-5:raise RuntimeError('Worker failed/interrupted/late')
        a=read(root/'summary.json')
        if not a.get('complete')or a['protocol_sha256']!=sha(protocol_path)or a['manifest_sha256']!=sha(root/'manifest.json')or a['unit_indices']!=p['unit_indices']:raise ValueError('Actual bank completion differs')
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
        write(root/'watchdog.json',dict(complete=complete,protocol_sha256=sha(protocol_path),publication_commit=commit,started_utc=started.isoformat(),ended_utc=now().isoformat(),calendar_seconds=time.monotonic()-tick,cap_seconds=cap,hard_deadline_utc=deadline.isoformat(),work_deadline_utc=work_deadline.isoformat(),compute_reservation=charged,child_returncode=None if child is None else child.returncode,failure=failure,artifacts={f.name:sha(f)for f in(root/'summary.json',root/'manifest.json',root/'worker.log')if f.exists()}))
        for s,h in old.items():signal.signal(s,h)
    if not complete:raise RuntimeError(str(failure))
    write(root/'COMPLETE.json',dict(external_watchdog_passed=True,protocol_sha256=sha(protocol_path),summary_sha256=sha(root/'summary.json'),manifest_sha256=sha(root/'manifest.json'),watchdog_sha256=sha(root/'watchdog.json')))


def unit_input_key(p,record,metadata,environment):
    payload=dict(format=FORMAT,unit_index=record['unit_index'],old_record_metadata_sha256=record['metadata_sha256'],
        original_geometry_audit_sha256=metadata['input']['geometry_audit_sha256'],source_unit_summary_sha256=metadata['input']['summary_sha256'],source_unit_seal_sha256=metadata['input']['seal_sha256'],
        teacher_descriptor_sha256=p['expected_frozen_descriptor_sha256'],teacher_sha256=p['teacher_sha256'],source_sha256=p['source_sha256'],environment=environment,
        qualification_protocol_sha256=p['qualification_protocol_sha256'],qualification_complete_sha256=p['qualification_complete_sha256'],definition='full128-native-view-v86-exact-even-only')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),payload


def worker(protocol_path,commit,controller_pid):
    p=read(protocol_path);root=Path(p['output']);execution=read(root/'execution.json')
    if os.getppid()!=controller_pid or execution['controller_pid']!=controller_pid or execution['protocol_sha256']!=sha(protocol_path)or execution['publication_commit']!=commit:raise ValueError('Admitted live controller required')
    from solar_learning_pipeline_v73 import journal_state
    _,j,_=journal_state(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL)
    entries=[x for x in j['reservations']if x['job_id']==sha(protocol_path)+SUFFIX]
    if len(entries)!=1 or entries[0]!=read(root/'reservation.json')['reservation']or entries[0]['category']!='training'or entries[0]['seconds']!=p['max_calendar_seconds']:raise ValueError('One exact original charge required')
    with(root/'WORKER_STARTED').open('x')as f:f.write(str(os.getpid())+'\n')
    def clock():
        if now()>=utc(execution['work_deadline_utc']):raise TimeoutError('Original bank worker deadline exhausted')
    clock();verify_sources(protocol_path,p,commit,utc(execution['work_deadline_utc']))
    import numpy as np
    import cv2
    import torch
    import platform
    from compact_lunar import load_checkpoint
    from solar_native_v63 import state_digest
    from solar_training_bank_v68 import EvenImageReader,unit_geometries
    from solar_coarse_training_adapter_v86 import load_training_metadata_index,build_coarse_training_unit,frozen_coarse_teacher,fit_balanced_training_view_prior
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    old_bank,metas,gate=load_training_metadata_index(Path.cwd(),p['bank_manifest'],p['original_master'],p['dataset_summary'],p['geometry_audit'])
    for n,h in p['teacher_sha256'].items():
        if sha(Path(p['teacher_directory'])/n)!=h:raise ValueError('Original frozen teacher input changed')
    teacher,_=load_checkpoint(p['teacher_directory']);teacher.eval().requires_grad_(False)
    if state_digest(teacher)!=p['expected_frozen_descriptor_sha256']:raise ValueError('Actual teacher weights differ')
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,platform=platform.platform(),torch_threads=2)
    if p['phase']=='full_bank'and read(Path(p['pilot_run'])/'manifest.json')['environment']!=environment:raise ValueError('Measured pilot must use the same actual software/hardware/two-thread environment')
    cache={}
    for entry in p['cache_runs']:
        if sha(Path(entry['run'])/'COMPLETE.json')!=entry['complete_sha256']:raise ValueError('Explicit immutable cache changed')
        cp,cw,cm=completed_bank(entry['protocol'],entry['run'])
        if cp['source_sha256']!=p['source_sha256']:raise ValueError('Cache source revision differs')
        for r in cm['records']:cache.setdefault(r['unit_index'],[]).append(r)
    source_records={r['unit_index']:r for r in old_bank['records']};metadata={m['unit_index']:m for m in metas};units={r['unit_index']:r for r in gate.dataset['units']}
    lighting=read(p['lighting_schedule'])['lighting'];records=[];prior_units=[];fresh=0;reused=0;(root/'records').mkdir()
    for index in p['unit_indices']:
        clock();original=source_records[index];meta=metadata[index];key,inputs=unit_input_key(p,original,meta,environment);hit=None
        for r in cache.get(index,[]):
            if r['input_key']==key:hit=r;break
        out=root/'records'/str(index);out.mkdir(exist_ok=False)
        if hit:
            import shutil
            source_meta=read(hit['metadata_path']);seal=read(hit['seal_path'])
            if sha(hit['path'])!=hit['sha256']or sha(hit['metadata_path'])!=hit['metadata_sha256']or sha(hit['seal_path'])!=hit['seal_sha256']or seal['npz_sha256']!=hit['sha256']or seal['metadata_sha256']!=hit['metadata_sha256']or source_meta['inputs']!=inputs:raise ValueError('Completed cache bytes/provenance changed')
            for src,dst in((hit['path'],out/'data.npz'),(hit['metadata_path'],out/'metadata.json'),(hit['seal_path'],out/'COMPLETE.json')):shutil.copyfile(src,dst)
            with np.load(out/'data.npz',allow_pickle=False)as z:views=z['target_view_basis'].copy()
            detail=source_meta['details'];samples=source_meta['samples'];reused+=1
        else:
            folder=Path(units[index]['output']);summary=read(folder/'summary.json');geometry,_,_=unit_geometries(folder,summary,gate.master['units'][index]);reader=EvenImageReader(folder,summary['images'])
            def encode(image,points):
                clock();return frozen_coarse_teacher(teacher,image,points,expected_descriptor_sha256=p['expected_frozen_descriptor_sha256'])
            a=build_coarse_training_unit(meta,geometry,reader,lighting[meta['component']],encode)
            if any(r['light']%2 for r in reader.decoded)or len(reader.decoded)!=24:raise ValueError('Exact even-only teacher inventory required')
            detail=a.pop('details');samples=len(a['world']);views=a['target_view_basis']
            if samples!=meta['samples']:raise ValueError('No historical physical row may be dropped')
            with(out/'data.npz').open('xb')as f:np.savez_compressed(f,**a)
            write(out/'metadata.json',dict(format=FORMAT,input_key=key,inputs=inputs,unit_index=index,component=meta['component'],split='training',samples=samples,details=detail,even_images_decoded=reader.decoded,heldout_image_reads=0,source_record_metadata_path=original['metadata_path'],source_record_metadata_sha256=original['metadata_sha256'],source_protocol_sha256=sha(protocol_path)))
            clock();write(out/'COMPLETE.json',dict(input_key=key,npz_sha256=sha(out/'data.npz'),metadata_sha256=sha(out/'metadata.json'),sealed_utc=now().isoformat(),original_worker_deadline_utc=execution['work_deadline_utc']));fresh+=1
        records.append(dict(unit_index=index,component=meta['component'],split='training',samples=samples,path=str(out/'data.npz'),sha256=sha(out/'data.npz'),metadata_path=str(out/'metadata.json'),metadata_sha256=sha(out/'metadata.json'),seal_path=str(out/'COMPLETE.json'),seal_sha256=sha(out/'COMPLETE.json'),input_key=key,reused=bool(hit)))
        prior_units.append(dict(target_view_basis=views,details=dict(unit_index=index,component=meta['component'],split='training')))
        print('COARSE128 TEACHER UNIT',index,'rows',samples,'cached',bool(hit),flush=True)
    prior_path=None;prior_sha=None
    if p['phase']=='full_bank':
        prior=fit_balanced_training_view_prior(prior_units,allowed_training_groups=list(gate.training_groups),eligible_unit_indices=list(gate.eligible_training_indices),source_sha256={**p['source_sha256'],'bank_manifest':sha(p['bank_manifest'])})
        prior_path=str(root/'view_prior.json');write(prior_path,prior.payload());prior_sha=sha(prior_path)
    manifest=dict(format=FORMAT,array_schema='distributional-coarse128-teacher-v86',complete=True,split='training',calibration_or_final_access=False,output=str(root),protocol_path=str(protocol_path),source_lights=list(range(0,24,2)),teacher_descriptor_sha256=p['expected_frozen_descriptor_sha256'],records=records,environment=environment,source_sha256=p['source_sha256'],metadata_sha256=p['metadata_sha256'],protocol_sha256=sha(protocol_path),qualification_complete_sha256=p['qualification_complete_sha256'],training_groups=sorted({r['component']for r in records}),eligible_unit_indices=list(gate.eligible_training_indices),phase=p['phase'],fresh_records=fresh,reused_records=reused,view_prior=prior_path,view_prior_sha256=prior_sha,heldout_image_reads=0,all_historical_rows_retained=True)
    write(root/'manifest.json',manifest);clock();write(root/'summary.json',dict(complete=True,phase=p['phase'],protocol_sha256=sha(protocol_path),manifest_sha256=sha(root/'manifest.json'),unit_indices=p['unit_indices'],fresh_records=fresh,reused_records=reused,samples=sum(r['samples']for r in records),heldout_image_reads=0))


if __name__=='__main__':
    sys.path.insert(0,str(Path.cwd()))
    a=argparse.ArgumentParser(description=__doc__);sub=a.add_subparsers(dest='command',required=True)
    b=sub.add_parser('prepare');b.add_argument('--settings',required=True);b.add_argument('--output',required=True)
    for name in('run','worker'):
        b=sub.add_parser(name);b.add_argument('--protocol',required=True);b.add_argument('--publication-commit',required=True)
        b.add_argument('--python',default=sys.executable)if name=='run'else b.add_argument('--controller-pid',type=int,required=True)
    v=a.parse_args()
    if v.command=='prepare':prepare(v.settings,v.output)
    elif v.command=='run':run(v.protocol,v.publication_commit,v.python)
    else:worker(v.protocol,v.publication_commit,v.controller_pid)
