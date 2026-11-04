"""One-shot600s coarse teacher-domain qualification; NOT a training runner.

Controller imports only metadata helpers. All NPZ/image/checkpoint decoding
and actual frozen-descriptor forwards occur in its admitted child worker.
No retry/refund, model update, new view prior, reserved terrain or publication.
"""
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CAP=600
CANONICAL='runs/solar-compute-reservations-v67.json'
SUFFIX='/coarse_teacher_domain_qualification'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc)
def write(p,value):
    with Path(p).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def utc(value):
    d=datetime.fromisoformat(value)
    if d.tzinfo is None:raise ValueError('Timezone-aware frozen UTC required')
    return d


def validate_settings(p,started,driver_path):
    if p.get('prepared_only')is not False or p.get('purpose')!='coarse_teacher_domain_qualification':raise ValueError('Prepared-only settings cannot launch')
    if p.get('category')!='training'or p.get('cpu_threads')!=2 or type(p.get('max_calendar_seconds'))is not int or p['max_calendar_seconds']!=CAP:raise ValueError('Original teacher category training, two threads, fixed600s required')
    if p.get('driver')!=driver_path or p.get('reservation_journal')!=CANONICAL or Path(CANONICAL).resolve()!=Path(CANONICAL).absolute():raise ValueError('Source-bound driver and one canonical journal required')
    if type(p.get('unit_index'))is not int or p['unit_index']<0 or p.get('lighting_indices')!=list(range(0,24,2)):raise ValueError('One predeclared training camera and all12even lights required')
    expected=dict(seed=20261101,native_translation=[8,-8],topk=8,threshold_native_px=16.,minimum_all_anchor_fraction=.8)
    if p.get('generated_control')!=expected:raise ValueError('Existing generated-control tolerances must be predeclared unchanged')
    created,latest,absolute=map(utc,(p['created_utc'],p['latest_start_utc'],p['absolute_deadline_utc']))
    lease=p.get('publication_dispatch_lease_seconds')
    if type(lease)is not int or not 0<lease<=1800 or latest!=created+timedelta(seconds=lease)or absolute!=latest+timedelta(seconds=CAP):raise ValueError('Finite original publication lease/deadline algebra required')
    if started<created or started>latest or started>=absolute:raise TimeoutError('Original publication lease expired or future-created protocol')
    if not isinstance(p.get('output'),str)or not p['output']or Path(p['output']).is_absolute()or '..'in Path(p['output']).parts:raise ValueError('Versioned repository-relative output required')
    if p.get('calibration_or_final_access')is not False:raise ValueError('Calibration/final access forbidden')
    return min(started+timedelta(seconds=CAP),absolute)


def verify_publication(protocol_path,p,commit,deadline=None):
    from solar_learning_pipeline_v73 import local_source_closure
    if len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact source-publication commit required')
    c=read(p['learning_contract'])
    # Existing49-source closure already owns original teacher/geometry/helpers;
    # explicit prototype imports add these three unpublished-until-now files.
    required=local_source_closure()
    required.update({n:sha(n)for n in(p['driver'],'solar_coarse_training_adapter_v86.py','solar_distributional_response_v86.py')})
    if p['source_sha256']!=required:raise ValueError('Actual transitive source closure must be frozen')
    metadata={p[k]for k in('learning_contract','original_master','dataset_summary','geometry_audit','bank_manifest','lighting_schedule')}
    if set(p['metadata_sha256'])!=metadata:raise ValueError('Exact metadata/source provenance required')
    deadline=deadline or now()+timedelta(seconds=CAP)
    def public(path):
        remaining=(deadline-now()).total_seconds()
        if remaining<=0:raise TimeoutError('Publication preflight exhausted original deadline')
        return subprocess.check_output(['git','show',commit+':'+path],timeout=min(10.,remaining))
    for path,h in {**p['source_sha256'],**p['metadata_sha256']}.items():
        if sha(path)!=h or public(path)!=Path(path).read_bytes():raise ValueError('Source/input not unchanged and published:'+path)
    if public(str(protocol_path))!=Path(protocol_path).read_bytes():raise ValueError('Qualification protocol not published')


def admission(protocol_path,commit,started):
    p=read(protocol_path);driver=str(Path(__file__).resolve().relative_to(Path.cwd().resolve()))
    deadline=validate_settings(p,started,driver);verify_publication(protocol_path,p,commit,deadline)
    from solar_learning_pipeline_v73 import load_contract,remaining
    c=load_contract(p['learning_contract'])
    if p['compute_ledger']!=c['compute_ledger']or p['compute_ledger_sha256']!=c['compute_ledger_sha256']or remaining(c)['training']<CAP:raise ValueError('Original cumulative training allowance cannot admit qualification')
    for key in('original_master','dataset_summary','geometry_audit','lighting_schedule'):
        if p[key]!=c[key]or p['metadata_sha256'][p[key]]!=c[key+'_sha256']:raise ValueError('Original learning/geometry data changed')
    a=read(p['geometry_audit']);b=read(p['bank_manifest']);m=read(p['original_master'])
    if a.get('passed')is not True or a.get('complete_dataset')is not True or a.get('completed_units')!=96 or a.get('expected_units')!=96 or a.get('oracle_success',0)<.95:raise ValueError('Actual full96 independent geometry gate required')
    if p['unit_index']not in c['eligible_training_unit_indices']or m['units'][p['unit_index']]['site']['split']!='training':raise ValueError('Predeclared unit must be eligible original training terrain')
    if p['expected_frozen_descriptor_sha256']!=b['teacher_descriptor_sha256']or p['teacher_directory']!=c['initializer']or p['teacher_sha256']!=c['initializer_sha256']:raise ValueError('Original descriptor/checkpoint identity required')
    if Path(p['output']).exists():raise FileExistsError('Preserve any prior attempt; no restart')
    return p,deadline


def run(protocol_path,commit,python):
    # Startup/admission and model import/load, work, serialization and cleanup
    # all consume the SAME original600s elapsed bound, never a fresh child cap.
    started=now();tick=time.monotonic();p,deadline=admission(protocol_path,commit,started)
    if now()>=deadline-timedelta(seconds=5)or time.monotonic()-tick>=CAP-5:raise TimeoutError('Admission/preflight exhausted original allowance; no child launch')
    root=Path(p['output']);root.mkdir(parents=True,exist_ok=False)
    work_deadline=deadline-timedelta(seconds=5);child=None;failure=None;complete=False;charged=None;pending=[]
    def interrupted(signum,frame):pending.append(signum)
    old={s:signal.signal(s,interrupted)for s in(signal.SIGINT,signal.SIGTERM)}
    try:
        from solar_compute_budget_v67 import reserve
        charged=reserve(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL,sha(protocol_path)+SUFFIX,'training',CAP)
        write(root/'reservation.json',charged)
        write(root/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=commit,controller_pid=os.getpid(),started_utc=started.isoformat(),work_deadline_utc=work_deadline.isoformat(),hard_deadline_utc=deadline.isoformat(),cap_seconds=CAP))
        if pending or now()>=work_deadline or time.monotonic()-tick>=CAP-5:raise TimeoutError('Startup exhausted original fixed allowance')
        with(root/'worker.log').open('x')as log:
            child=subprocess.Popen([python,p['driver'],'--protocol',str(protocol_path),'--publication-commit',commit,'--worker','--controller-pid',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
            while child.poll()is None:
                if pending or now()>=work_deadline or time.monotonic()-tick>=CAP-5:raise TimeoutError('Original scientific work allowance exhausted')
                time.sleep(.1)
        if child.returncode or pending or now()>=work_deadline or time.monotonic()-tick>=CAP-5:raise RuntimeError('Worker failed/interrupted/late; preserve all rows')
        result=read(root/'candidate.json')
        if result.get('complete')is not True or result.get('passed')is not True or result.get('protocol_sha256')!=sha(protocol_path)or result.get('arrays_sha256')!=sha(root/'arrays.npz'):raise ValueError('Actual teacher qualification failed or artifacts changed')
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
                except subprocess.TimeoutExpired:complete=False;failure=dict(type='CleanupFailure',message='Cleanup exceeded original allowance')
        if now()>=deadline or time.monotonic()-tick>=CAP:complete=False;failure=dict(type='TimeoutError',message='Original hard cap exhausted')
        receipt=dict(complete=complete,protocol_sha256=sha(protocol_path),publication_commit=commit,started_utc=started.isoformat(),ended_utc=now().isoformat(),calendar_seconds=time.monotonic()-tick,cap_seconds=CAP,hard_deadline_utc=deadline.isoformat(),work_deadline_utc=work_deadline.isoformat(),compute_reservation=charged,child_returncode=None if child is None else child.returncode,failure=failure,artifacts={f.name:sha(f)for f in(root/'candidate.json',root/'arrays.npz',root/'worker.log')if f.exists()})
        write(root/'watchdog.json',receipt)
        for s,handler in old.items():signal.signal(s,handler)
    if not complete:raise RuntimeError(str(failure))
    write(root/'COMPLETE.json',dict(external_watchdog_passed=True,protocol_sha256=sha(protocol_path),candidate_sha256=sha(root/'candidate.json'),watchdog_sha256=sha(root/'watchdog.json'),scope='Teacher-domain qualification only; no new model trained.'))
    return receipt


def worker(protocol_path,commit,controller_pid):
    p=read(protocol_path);root=Path(p['output']);execution=read(root/'execution.json')
    if os.getppid()!=controller_pid or execution['controller_pid']!=controller_pid or execution['protocol_sha256']!=sha(protocol_path)or execution['publication_commit']!=commit:raise ValueError('Worker requires its admitted original controller')
    from solar_learning_pipeline_v73 import journal_state
    _,journal,_=journal_state(p['compute_ledger'],p['compute_ledger_sha256'],CANONICAL)
    entries=[a for a in journal['reservations']if a['job_id']==sha(protocol_path)+SUFFIX]
    if len(entries)!=1 or entries[0]!=read(root/'reservation.json')['reservation']or entries[0]['category']!='training'or entries[0]['seconds']!=CAP:raise ValueError('Exact canonical original600s charge required')
    with(root/'WORKER_STARTED').open('x')as f:f.write(str(os.getpid())+'\n')
    def clock():
        if now()>=utc(execution['work_deadline_utc']):raise TimeoutError('Original worker deadline expired')
    clock();verify_publication(protocol_path,p,commit,utc(execution['work_deadline_utc']))
    import numpy as np
    import cv2
    import torch
    from compact_lunar import load_checkpoint
    from contrastive_terrain import sample_descriptors
    from solar_coarse_training_adapter_v86 import load_training_metadata_index,build_coarse_training_unit,frozen_coarse_teacher,native_to_coarse,coarse_support
    from solar_training_bank_v68 import unit_geometries,EvenImageReader
    from solar_frontend_v63 import reference_anchors
    from solar_native_v63 import state_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    bank,metas,gate=load_training_metadata_index(Path.cwd(),p['bank_manifest'],p['original_master'],p['dataset_summary'],p['geometry_audit'])
    selected=next(m for m in metas if m['unit_index']==p['unit_index']);record=next(u for u in gate.dataset['units']if u['unit_index']==p['unit_index']);folder=Path(record['output']);summary=read(folder/'summary.json')
    for name,h in p['teacher_sha256'].items():
        if sha(Path(p['teacher_directory'])/name)!=h:raise ValueError('Original checkpoint inputs changed')
    clock();teacher,_=load_checkpoint(p['teacher_directory']);teacher.eval().requires_grad_(False)
    if state_digest(teacher)!=p['expected_frozen_descriptor_sha256']:raise ValueError('Actual loaded teacher digest differs')
    geometry,_,_=unit_geometries(folder,summary,gate.master['units'][p['unit_index']])
    image_reads=[]
    def read_image(path):image_reads.append(str(path));clock();return Path(path).read_bytes()
    reader=EvenImageReader(folder,summary['images'],read_bytes=read_image)
    # Intentional odd request must fail before ANY image byte access.
    try:reader.get('reference',1)
    except ValueError:odd_refused_before_read=not image_reads
    else:odd_refused_before_read=False
    if not odd_refused_before_read:raise ValueError('Odd illumination reached source reader')
    teacher_inputs=[]
    def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    def adapter_teacher(image,points):
        clock();teacher_inputs.append(dict(image_sha256=array_sha(image),points_sha256=array_sha(points)))
        return frozen_coarse_teacher(teacher,image,points,expected_descriptor_sha256=p['expected_frozen_descriptor_sha256'])
    lighting=read(p['lighting_schedule'])['lighting'][selected['component']]
    arrays=build_coarse_training_unit(selected,geometry,reader,lighting,adapter_teacher)
    traces={k:v for k,v in arrays.items()if isinstance(v,np.ndarray)};rows=[]
    for light in p['lighting_indices']:
        for role_index,role in enumerate(('reference','query')):
            clock();native=reader.get(role,light);low=cv2.resize(native,(128,128),interpolation=cv2.INTER_AREA)
            point=((arrays['native_xy'][:,role_index]+.5)/4-.5).astype(np.float32)
            with torch.inference_mode():
                dense,_=teacher(torch.from_numpy(low.copy()).float()[None,None]/255)
                direct=sample_descriptors(dense,torch.from_numpy(point)[None])[0].cpu().numpy().copy()
            usable=arrays['available'][:,role_index,light]
            exact=np.array_equal(direct[usable],arrays['teacher'][usable,role_index,light])
            expected_input=dict(image_sha256=array_sha(low),points_sha256=array_sha(point[arrays['coarse_support'][:,role_index]]))
            converted=expected_input in teacher_inputs
            if role=='reference':converted=converted and np.array_equal(low,arrays['reference_images'][light//2])
            centres=np.array_equal(point,arrays['coarse_xy'][:,role_index])
            rows.append(dict(role=role,light=light,total_rows=len(point),compared_rows=int(usable.sum()),omitted_rows=int((~usable).sum()),exact_teacher_identity=exact,exact_native_to_coarse_centres=centres,exact_conversion=converted,max_absolute_difference=float(np.max(np.abs(direct[usable]-arrays['teacher'][usable,role_index,light])))if usable.any()else None))
            traces['direct_'+role+'_'+str(light)]=direct
    # Existing+8,-8 native generated control, top8/16px/80% ALL256 anchors.
    cfg=p['generated_control'];rng=np.random.default_rng(cfg['seed']);image=rng.integers(0,256,(512,512),dtype=np.uint8)
    query=cv2.warpAffine(image,np.array([[1.,0.,8.],[0.,1.,-8.]]),(512,512),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    anchors=reference_anchors();yy,xx=np.mgrid[:51,:51];candidates=np.column_stack([49.5+8*xx.ravel(),49.5+8*yy.ravel()])
    da=adapter_teacher(cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA),native_to_coarse(anchors))
    db=adapter_teacher(cv2.resize(query,(128,128),interpolation=cv2.INTER_AREA),native_to_coarse(candidates))
    scores=da@db.T;top=np.argsort(-scores,axis=1,kind='stable')[:,:8]
    correct=(np.linalg.norm(candidates[top]-(anchors+np.array([8.,-8.]))[:,None],axis=-1)<=16.).any(1)
    generated=dict(total_anchors=len(anchors),correct=int(correct.sum()),fraction=float(correct.mean()),passed=bool(correct.mean()>=.8),settings=cfg)
    traces.update(generated_reference_xy=anchors,generated_query_xy=candidates,generated_reference_features=da,generated_query_features=db,generated_top8=top,generated_correct=correct,generated_expected_query_support=coarse_support(np.ones((512,512),bool),native_to_coarse(anchors+np.array([8.,-8.]))))
    clock()
    with(root/'arrays.npz').open('xb')as f:np.savez_compressed(f,**traces)
    passed=odd_refused_before_read and generated['passed']and len(reader.decoded)==24 and len(arrays['world'])==selected['samples'] and all(r['total_rows']==selected['samples']and r['compared_rows']>0 and r['exact_teacher_identity']and r['exact_native_to_coarse_centres']and r['exact_conversion']for r in rows)
    write(root/'candidate.json',dict(complete=True,passed=bool(passed),protocol_sha256=sha(protocol_path),arrays_sha256=sha(root/'arrays.npz'),teacher_descriptor_sha256=state_digest(teacher),unit_index=p['unit_index'],odd_refused_before_read=odd_refused_before_read,decoded_images=reader.decoded,rows=rows,generated_control=generated,all_original_rows_retained=len(arrays['world'])==selected['samples'],unit_details=arrays['details'],scope='Teacher domain/coordinates only. No trained distributional model, lunar matching or pose claim.'))


if __name__=='__main__':
    sys.path.insert(0,str(Path.cwd()))
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--python',default=sys.executable);a.add_argument('--worker',action='store_true');a.add_argument('--controller-pid',type=int)
    v=a.parse_args()
    if v.worker:worker(v.protocol,v.publication_commit,v.controller_pid)
    else:run(v.protocol,v.publication_commit,v.python)
