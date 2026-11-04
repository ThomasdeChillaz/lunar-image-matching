"""Published, bounded native controls; never creates a qualification from mocks.

Historical bootstrap checks model lineage before constructing the existing raw
runtime, then executes HistoricalCompactAdapter.infer unchanged. Its own actual
control receipt cannot be a prerequisite for this first control. Separate
qualification uses the complete unchanged historical-profile validator.
"""
import argparse
import ast
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

CANONICAL_JOURNAL='runs/solar-compute-reservations-v67.json'
BASELINE_CEILING=66900
SELF='control_historical_native_v74.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def utc(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('Timezone-aware original deadline required')
    return result


def source_closure(seeds):
    def imports(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return
        if isinstance(node,ast.Import):
            for x in node.names:yield x.name.split('.')[0]+'.py'
        if isinstance(node,ast.ImportFrom)and node.module:yield node.module.split('.')[0]+'.py'
        for child in ast.iter_child_nodes(node):yield from imports(child)
    todo=list(seeds);seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        if not Path(name).is_file():raise ValueError('Missing control source: '+name)
        seen.add(name)
        if Path(name).suffix=='.py':todo.extend(x for x in imports(ast.parse(Path(name).read_text()))if Path(x).is_file())
    return {name:sha(name)for name in sorted(seen)}


def full_gate(p):
    for name in('original_master','dataset_summary','geometry_audit'):
        if sha(p[name])!=p[name+'_sha256']:raise ValueError('Full-data gate artifact changed: '+name)
    from solar_training_bank_v68 import verify_geometry_gate
    gate=verify_geometry_gate(p['original_master'],p['dataset_summary'],p['geometry_audit'])
    if gate.audit['audit_source_sha256']!=sha('verify_solar_development_v68.py'):
        raise ValueError('Independent full96 geometry audit source changed')
    return gate


def balance(p,check_history=True):
    if p['reservation_journal']!=CANONICAL_JOURNAL or Path(CANONICAL_JOURNAL).resolve()!=Path(CANONICAL_JOURNAL).absolute():
        raise ValueError('One canonical baseline journal required')
    if sha(p['compute_ledger'])!=p['compute_ledger_sha256']:raise ValueError('Frozen compute ledger changed')
    ledger=read(p['compute_ledger']);journal=read(CANONICAL_JOURNAL)if Path(CANONICAL_JOURNAL).exists()else dict(ledger_sha256=p['compute_ledger_sha256'],reservations=[])
    if not ledger['accounting_complete']or journal['ledger_sha256']!=p['compute_ledger_sha256']:raise ValueError('Known shared cumulative accounting required')
    if check_history and journal['reservations'][:len(p['initial_reservations'])]!=p['initial_reservations']:
        raise ValueError('Original canonical charges were removed or rewritten')
    ids=set();charged=0
    for r in journal['reservations']:
        if r['job_id']in ids or r.get('refund_on_failure')is not False or type(r['seconds'])is not int or r['seconds']<=0:raise ValueError('Invalid/duplicate/refunded shared reservation')
        ids.add(r['job_id'])
        if r['category']=='baseline':charged+=r['seconds']
    return min(BASELINE_CEILING,ledger['remaining_baseline_wall_seconds']-charged),journal


def bootstrap_profile(profile):
    """No native-control bypass for lunar inference; bootstrap is control-only."""
    import historical_compact_adapters_v73 as h
    if profile['backend']!='historical_compact'or profile['method']not in h.METHODS or profile['cpu_threads']!=2 or profile['native_size']!=512 or profile['calibration_or_final_access']:
        raise ValueError('Named historical native512/two-thread control profile required')
    c=profile['configuration'];method=profile['method']
    expected={'model_directory','checkpoint','input_profile'}if method==h.METHODS[0]else {'coarse_directory','selected_checkpoint','variant','input_profile','surface_iterations'}if method==h.METHODS[2]else {'model_directory','input_profile','surface_iterations'}
    if set(c)!=expected:raise ValueError('Historical configuration keys differ from the adapter contract')
    if method==h.METHODS[0]:
        if c['checkpoint']!='selected'or c['input_profile']!='raw_native512':raise ValueError('Original raw native profile required')
    elif c['input_profile']!='coarse128_native512'or c['surface_iterations']!=2 or method==h.METHODS[2]and c['variant']!='cross_adaptation':
        raise ValueError('Unchanged historical surface/context profile required')
    for name,digest in profile['artifact_sha256'].items():
        if sha(name)!=digest:raise ValueError('Historical model artifact changed: '+name)
    for name,digest in profile.get('historical_algorithm_source_sha256',{}).items():
        if sha(name)!=digest:raise ValueError('Historical algorithm changed; no silent requalification: '+name)
    return h._checkpoint_lineage(profile)


def prepare(settings_path,output_protocol):
    s=read(settings_path)
    if s.get('prepared_only',False):raise ValueError('Prepared settings cannot admit actual controls')
    cap=s['max_calendar_seconds'];lease=s['publication_dispatch_lease_seconds']
    if type(cap)is not int or not 1<=cap<=3600 or type(lease)is not int or lease<=0:raise ValueError('Explicit finite execution cap and dispatch lease required')
    p=dict(s,schema_version=74,cpu_threads=2,calibration_or_final_access=False,query_truth_used=False,
        original_master_sha256=sha(s['original_master']),dataset_summary_sha256=sha(s['dataset_summary']),geometry_audit_sha256=sha(s['geometry_audit']),compute_ledger_sha256=sha(s['compute_ledger']))
    full_gate(p)
    common={SELF,'solar_compute_budget_v67.py','solar_training_bank_v68.py','verify_solar_development_v68.py','synthetic.py','native_baseline_pose_v7.py','solar_evaluation_v70.py'}
    if s['control_kind']=='historical_compact':
        import historical_compact_adapters_v73 as h
        candidates=read(s['prepared_profiles'])['profiles'];base=next(x for x in candidates if x['method']==s['method'])
        bootstrap_profile(base)
        p.update(configuration=base['configuration'],configuration_sha256=h.value_digest(base['configuration']),artifact_sha256=base['artifact_sha256'],historical_algorithm_source_sha256=base.get('historical_algorithm_source_sha256',{}),base_profile=base)
        p['source_sha256']=source_closure(common|h.required_sources(s['method']))
    elif s['control_kind']=='legacy_matchanything':
        from control_native_matchanything_v17 import SOURCES
        lock=read('matchanything_model_lock.json');weight=lock['safe_weight_path']
        if sha(weight)!=lock['safe_weight_sha256']:raise ValueError('Existing safe MatchAnything weights required; downloads forbidden')
        p.update(method='matchanything',author_profile='native512',artifact_sha256={weight:sha(weight),'matchanything_model_lock.json':sha('matchanything_model_lock.json')},
            child_output=str(Path(s['output'])/'native'),source_sha256=source_closure(common|set(SOURCES)|{'native_pretrained_adapters_v73.py','dataset.py','matchers.py','geometry.py','lunargeo_dataset.py','remote_archive.py'}))
    else:raise ValueError('Explicit historical control or existing MatchAnything requalification required')
    p.update(initial_reservations=[])
    available,journal=balance(p,False);p['initial_reservations']=journal['reservations']
    if cap>available:raise ValueError('Control exceeds remaining shared baseline budget')
    created=datetime.now(timezone.utc);latest=created+timedelta(seconds=lease)
    p.update(created_utc=created.isoformat(),latest_start_utc=latest.isoformat(),absolute_deadline_utc=(latest+timedelta(seconds=cap)).isoformat(),queue_allowance_seconds=lease,
        image_seed=20261101,native_shape=[512,512],reference_to_query_affine=[[1.,0.,8.],[0.,1.,-4.]],
        reference_depth_m=10.,intrinsics=[[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]],
        expected_query_cam2world=[[1.,0.,0.,-.2],[0.,1.,0.,.1],[0.,0.,1.,0.],[0.,0.,0.,1.]],
        scope='Actual procedural native control only. No lunar image, calibration/final inference, model training or download.')
    if Path(s['output']).exists():raise FileExistsError('Preserve every actual control attempt')
    Path(s['output']).parent.mkdir(parents=True,exist_ok=True);write_new(output_protocol,p);return p


def admission(protocol,commit):
    p=read(protocol)
    if p.get('prepared_only',False)or p['schema_version']!=74 or p['cpu_threads']!=2 or p['calibration_or_final_access']or p['query_truth_used']:
        raise ValueError('Source-published control-only schema74 protocol required')
    if type(p['max_calendar_seconds'])is not int or not 1<=p['max_calendar_seconds']<=3600 or type(p['publication_dispatch_lease_seconds'])is not int or p['publication_dispatch_lease_seconds']<=0:
        raise ValueError('Explicit bounded integer control cap and finite dispatch lease required')
    if len(commit or'')!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact source publication revision required')
    for name in [protocol,*p['source_sha256']]:
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Control source/protocol not published: '+name)
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Frozen control source changed: '+name)
    for name,digest in p['artifact_sha256'].items():
        if sha(name)!=digest:raise ValueError('Control model/lock artifact changed: '+name)
    if p['native_shape']!=[512,512]or p['image_seed']!=20261101 or p['reference_to_query_affine']!=[[1.,0.,8.],[0.,1.,-4.]]or p['reference_depth_m']!=10. or p['intrinsics']!=[[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]]or p['expected_query_cam2world']!=[[1.,0.,0.,-.2],[0.,1.,0.,.1],[0.,0.,1.,0.],[0.,0.,0.,1.]]:
        raise ValueError('Analytic native control geometry cannot be relabeled from matches')
    created=utc(p['created_utc']);latest=utc(p['latest_start_utc'])
    if latest!=created+timedelta(seconds=p['publication_dispatch_lease_seconds'])or utc(p['absolute_deadline_utc'])!=latest+timedelta(seconds=p['max_calendar_seconds']):
        raise ValueError('Original queue/execution deadlines changed')
    full_gate(p);balance(p)
    if p['control_kind']=='historical_compact':
        model_profile=dict(p['base_profile'],configuration=p['configuration'],artifact_sha256=p['artifact_sha256'],source_sha256=p['source_sha256'])
        bootstrap_profile(model_profile)
    return p


def pose_arithmetic(pose):
    if pose is None:return dict(pose_returned=False,rotation_error_deg=None,translation_error_m=None)
    a=np.asarray(pose,float)
    if a.shape!=(4,4)or not np.isfinite(a).all()or not np.allclose(a[3],[0,0,0,1])or not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6)or not np.isclose(np.linalg.det(a[:3,:3]),1.,atol=1e-6):raise ValueError('Saved common pose is not a rigid transform')
    return dict(pose_returned=True,rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(a[:3,:3])-1)/2,-1,1)))),translation_error_m=float(np.linalg.norm(a[:3,3]-[-.2,.1,0.])))


def replay_outputs(output,profile,estimator):
    """Arithmetic plus common-estimator replay; no image/model is accessed."""
    import historical_compact_adapters_v73 as h
    checks=h.validate_control_outputs(output,profile)
    known=output['known_transform'];pose=np.asarray(known['estimated_cam2world'],float)if known['estimated_cam2world']is not None else None
    measured=pose_arithmetic(pose);p,q=h.validate_points(known['reference_xy'],known['query_xy']);replayed,info=estimator(p,q)
    if(pose is None)!=(replayed is None)or pose is not None and not np.allclose(pose,replayed,rtol=0,atol=1e-7):raise ValueError('Saved control pose differs from common reference geometry replay')
    passed=bool(measured['pose_returned']and measured['rotation_error_deg']<=.5 and measured['translation_error_m']<=.05 and info['inliers']>=6)
    if output['actual_pretrained_forward']is not True or output['measurement_origin']!='actual_torch_forward_hooks'or not output['neural_forward_counts']['known_transform']or any(n<=0 for n in output['neural_forward_counts']['known_transform'].values()):
        raise ValueError('Actual selected neural components must forward; requested settings are insufficient')
    return dict(checks,**measured,passed=passed,common_pose_replayed=True,query_truth_used=False)


def execute_suite(adapter,monitor,inputs,profile,estimator,check=lambda:None):
    """Production uses real images/hooks. Stub tests never write gate receipts."""
    cases={};counts={};cold=None
    for name in('cold_known_transform','known_transform','blank','invalid_support'):
        check();before=dict(monitor.counts);tick=time.perf_counter();error=None
        try:result=adapter.infer(inputs[name])
        except ValueError as exc:
            if name!='invalid_support':raise
            result=dict(reference_xy=[],query_xy=[],pose=None);error=dict(type=type(exc).__name__,message=str(exc))
        counts[name]={k:v-before[k]for k,v in monitor.counts.items()}
        case=dict(reference_xy=np.asarray(result.get('reference_xy',[])).reshape(-1,2).tolist(),query_xy=np.asarray(result.get('query_xy',[])).reshape(-1,2).tolist(),
            pose_returned=result.get('pose')is not None,estimated_cam2world=None if result.get('pose')is None else np.asarray(result['pose']).tolist(),elapsed_seconds=time.perf_counter()-tick,rejection=error)
        if name.endswith('known_transform'):case['reference_to_query_affine']=[[1.,0.,8.],[0.,1.,-4.]]
        cases[name]=case;check()
    output=dict(method=profile['method'],native_shape=[512,512],cpu_threads=2,configuration_sha256=profile['configuration_sha256'],
        artifact_sha256=profile['artifact_sha256'],source_sha256=profile['source_sha256'],actual_pretrained_forward=bool(monitor.origin=='actual_torch_forward_hooks'and counts['known_transform']and all(x>0 for x in counts['known_transform'].values())),
        query_truth_used=False,neural_forward_counts=counts,measurement_origin=monitor.origin,**cases)
    try:output['replay']=replay_outputs(output,profile,estimator)
    except ValueError as exc:output['replay']=dict(passed=False,error_type=type(exc).__name__,reason=str(exc),scope='Measured control failure; raw outputs are retained, no qualification.')
    return output


class ForwardMonitor:
    def __init__(self,runtime):
        self.origin='actual_torch_forward_hooks'
        matcher=runtime.matcher;modules={'native_descriptor':matcher.model}
        if hasattr(matcher,'proposal_matcher'):
            modules.update(coarse_descriptor=matcher.proposal_matcher.model,selected_context=matcher.proposal_matcher.context)
        self.counts={name:0 for name in modules};self.hooks=[]
        for name,module in modules.items():
            def record(module,args,result,key=name):self.counts[key]+=1
            self.hooks.append(module.register_forward_hook(record))
    def close(self):
        for hook in self.hooks:hook.remove()


def actual_inputs(p,root):
    """Called only by the published, admitted worker; never during preparation."""
    import cv2
    from PIL import Image
    from synthetic import texture
    image=texture(p['image_seed'],512);query=cv2.warpAffine(image,np.asarray(p['reference_to_query_affine'],float),(512,512))
    Image.fromarray(image).save(root/'reference.png');Image.fromarray(query).save(root/'query.png')
    k=np.asarray(p['intrinsics'],float)
    def inp(a,b,valid=True):return SimpleNamespace(reference=a,query=b,geometry=SimpleNamespace(depth=np.full((512,512),10. if valid else 0.),intrinsics=k,cam2world=np.eye(4)),query_intrinsics=k,reference_sun=np.array([1.,0.,0.]),known_query_sun=None,solar_lookup=None)
    positive=inp(image,query)
    return dict(cold_known_transform=positive,known_transform=positive,blank=inp(np.zeros_like(image),np.zeros_like(image)),invalid_support=inp(image,query,False))


def worker(protocol,commit):
    p=admission(protocol,commit);root=Path(p['output']);execution=read(root/'execution_admission.json')
    if execution['protocol_sha256']!=sha(protocol):raise ValueError('Original controller admission required')
    _,journal=balance(p);job_id=sha(protocol)+'/native_control'
    charges=[r for r in journal['reservations']if r['job_id']==job_id]
    if len(charges)!=1 or charges[0]['category']!='baseline'or charges[0]['seconds']!=p['max_calendar_seconds']:raise ValueError('Actual control requires exactly one canonical baseline charge')
    def check():
        if datetime.now(timezone.utc)>=utc(execution['actual_deadline_utc'])or time.monotonic()-execution['start_monotonic']>=p['max_calendar_seconds']:raise TimeoutError('Original native control allowance exhausted')
    check()
    import torch,cv2
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    if p['control_kind']=='legacy_matchanything':
        from control_native_matchanything_v17 import run
        run(p['child_output'],p.get('dependency_dir'),'native512',2,p['max_calendar_seconds']);check();return
    import historical_compact_adapters_v73 as h
    from solar_evaluation_v70 import common_pose
    profile=dict(p['base_profile'],source_sha256=p['source_sha256'],configuration=p['configuration'],artifact_sha256=p['artifact_sha256'],configuration_sha256=p['configuration_sha256'])
    tick=time.perf_counter();runtime=h._HistoricalRuntime(profile);load_seconds=time.perf_counter()-tick
    # Execute the exact public infer body, bypassing ONLY its circular demand
    # for this not-yet-measured control receipt. Never used by lunar inference.
    adapter=h.HistoricalCompactAdapter.__new__(h.HistoricalCompactAdapter);adapter.profile=profile;adapter.runtime=runtime;adapter.estimator=common_pose
    monitor=ForwardMonitor(runtime)
    try:
        inputs=actual_inputs(p,root);reference=inputs['known_transform']
        def estimate(a,b):return common_pose(a,b,reference)
        output=execute_suite(adapter,monitor,inputs,profile,estimate,check)
        output.update(model_load_seconds=load_seconds,reference_image_sha256=sha(root/'reference.png'),query_image_sha256=sha(root/'query.png'),
            scope='Actual analytic procedural native adapter control; not lunar performance or calibrated acceptance.')
        write_new(root/'outputs.json',output);check()
    finally:monitor.close()


def run(protocol,commit,python):
    p=admission(protocol,commit);root=Path(p['output'])
    if datetime.now(timezone.utc)>utc(p['latest_start_utc']):raise TimeoutError('Original native-control dispatch lease expired; no reset')
    if root.exists():raise FileExistsError('Preserve every completed/failed control; no automatic restart')
    available,_=balance(p)
    if p['max_calendar_seconds']>available:raise ValueError('Native control no longer fits canonical baseline budget')
    from solar_compute_budget_v67 import reserve
    reservation=reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],sha(protocol)+'/native_control','baseline',p['max_calendar_seconds'])
    root.mkdir();started=datetime.now(timezone.utc);tick=time.monotonic();deadline=min(started+timedelta(seconds=p['max_calendar_seconds']),utc(p['absolute_deadline_utc']))
    write_new(root/'execution_admission.json',dict(protocol_sha256=sha(protocol),reservation=reservation,start_monotonic=tick,started_utc=started.isoformat(),actual_deadline_utc=deadline.isoformat()))
    pending=[];child=None;complete=False;passed=False;failure=None
    def interrupted(signum,frame):pending.append(signum)
    previous={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        with(root/'worker.log').open('x')as log:
            child=subprocess.Popen([python,SELF,'worker','--protocol',protocol,'--publication-commit',commit],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
            while child.poll()is None:
                if pending:raise InterruptedError('Control interrupted; preserve outputs and charge')
                if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('External native-control watchdog exhausted original cap')
                time.sleep(.1)
            if pending or datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Native control completed after original cap or interruption')
            if child.returncode:raise subprocess.CalledProcessError(child.returncode,child.args)
            if p['control_kind']=='historical_compact':passed=bool(read(root/'outputs.json')['replay']['passed'])
            else:
                from native_pretrained_adapters_v73 import native_profile,validate_native_profile
                gate=validate_native_profile(native_profile('matchanything','matchanything','native512',p['child_output'],'.'))
                write_new(root/'legacy_replay.json',gate);passed=True
            if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Control validation exceeded original cap')
            complete=True
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
        if child is not None and child.poll()is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
        legacy_evidence={}
        if p['control_kind']=='legacy_matchanything':
            legacy_evidence={name:sha(Path(p['child_output'])/name)for name in('protocol.json','summary.json','controls.json','provenance.json')if(Path(p['child_output'])/name).exists()}
        write_new(root/'receipt.json',dict(complete=complete,passed=passed,method=p['method'],protocol_sha256=sha(protocol),outputs_sha256=sha(root/'outputs.json')if(root/'outputs.json').exists()else None,
            source_sha256=sha(SELF),cpu_threads=2,started_utc=started.isoformat(),absolute_deadline_utc=deadline.isoformat(),cap_seconds=p['max_calendar_seconds'],calendar_seconds=time.monotonic()-tick,
            child_returncode=None if child is None else child.returncode,failure=failure,child_native_control_evidence_sha256=legacy_evidence,
            scope='Actual native-control deadline/validation receipt; not lunar performance.'))
    return read(root/'receipt.json')


def validate_receipt(protocol,receipt_path):
    """Metadata-only outer deadline/charge check, including legacy controls."""
    p=read(protocol);r=read(receipt_path)
    if r['complete']is not True or r['passed']is not True or r['failure']is not None or r['method']!=p['method']or r['protocol_sha256']!=sha(protocol)or r['source_sha256']!=p['source_sha256'][SELF]or r['cpu_threads']!=2 or r['child_returncode']!=0:
        raise ValueError('Actual successful source-bound outer control receipt required')
    cap=p['max_calendar_seconds'];elapsed=r['calendar_seconds'];start=utc(r['started_utc']);deadline=utc(r['absolute_deadline_utc'])
    if r['cap_seconds']!=cap or start>utc(p['latest_start_utc'])or deadline!=min(start+timedelta(seconds=cap),utc(p['absolute_deadline_utc']))or isinstance(elapsed,bool)or not isinstance(elapsed,(int,float))or not math.isfinite(elapsed)or not 0<elapsed<=min(cap,(deadline-start).total_seconds()):
        raise ValueError('Outer control exceeded its original dispatch/deadline cap')
    _,journal=balance(p);charged=[x for x in journal['reservations']if x['job_id']==sha(protocol)+'/native_control']
    if len(charged)!=1 or charged[0]['category']!='baseline'or charged[0]['seconds']!=cap:
        raise ValueError('Outer control is not charged in the one canonical baseline journal')
    if p['control_kind']=='historical_compact':
        if r['outputs_sha256']!=sha(Path(p['output'])/'outputs.json'):raise ValueError('Actual native output arrays changed')
    else:
        expected={'protocol.json','summary.json','controls.json','provenance.json'}
        if set(r['child_native_control_evidence_sha256'])!=expected:raise ValueError('Complete actual two-thread inner control evidence required')
        for name,h in r['child_native_control_evidence_sha256'].items():
            if sha(Path(p['child_output'])/name)!=h:raise ValueError('Legacy inner control evidence changed')
    return dict(passed=True,protocol_sha256=sha(protocol),receipt_sha256=sha(receipt_path),canonical_baseline_charge_seconds=cap,
        scope='Outer clock/source/evidence/charge validation only; actual model measurements remain in saved control files.')



def validate_external_pretrained(profile):
    """Optional outer proof, checked before comparison network construction.

    The inner four-file schema stays unchanged. No dispatch, reservation,
    current-time lease check or additional execution allowance is performed.
    """
    evidence=profile.get('external_control')
    if evidence is None:return None
    expected={'protocol','protocol_sha256','receipt','receipt_sha256','publication_commit'}
    if not isinstance(evidence,dict)or set(evidence)!=expected:raise ValueError('Exact optional outer control pins required')
    for name in('protocol','receipt'):
        if sha(evidence[name])!=evidence[name+'_sha256']:raise ValueError('Outer control evidence changed: '+name)
    p=admission(evidence['protocol'],evidence['publication_commit'])
    if p['control_kind']!='legacy_matchanything'or p['method']!='matchanything'or profile.get('method')!='matchanything'or profile.get('author_profile')!='native512'or profile.get('cpu_threads')!=2:
        raise ValueError('Outer control belongs to another native pretrained method/profile')
    actual=validate_receipt(evidence['protocol'],evidence['receipt'])
    inner=profile.get('pretrained_evidence',{})
    for name in('protocol','summary','controls','provenance'):
        path=Path(p['child_output'])/(name+'.json')
        if Path(inner.get(name,'')).resolve()!=path.resolve()or inner.get(name+'_sha256')!=sha(path):
            raise ValueError('Outer receipt must bind the identical four inner pretrained files')
    return dict(actual,publication_commit=evidence['publication_commit'],source_sha256=p['source_sha256'],
        input_sha256={evidence[name]:evidence[name+'_sha256']for name in('protocol','receipt')})


def qualify(protocol,commit,output_profile):
    p=admission(protocol,commit)
    if p['control_kind']!='historical_compact':raise ValueError('Legacy MatchAnything evidence uses native_pretrained_adapters_v73.native_profile')
    root=Path(p['output']);validate_receipt(protocol,root/'receipt.json')
    from solar_evaluation_v70 import common_pose
    inp=SimpleNamespace(geometry=SimpleNamespace(depth=np.full((512,512),10.),intrinsics=np.asarray(p['intrinsics'],float),cam2world=np.eye(4)),query_intrinsics=np.asarray(p['intrinsics'],float))
    profile_inputs=dict(p['base_profile'],configuration_sha256=p['configuration_sha256'],source_sha256=p['source_sha256'])
    replay=replay_outputs(read(root/'outputs.json'),profile_inputs,lambda a,b:common_pose(a,b,inp))
    if not replay['passed']:raise ValueError('Independent common-geometry control replay failed')
    profile=dict(p['base_profile'],prepared_only=False,available=True,publication_commit=commit,source_sha256=p['source_sha256'])
    profile.pop('unavailable_reason',None);profile['native_control']={}
    for key,path in dict(protocol=protocol,outputs=str(root/'outputs.json'),receipt=str(root/'receipt.json')).items():profile['native_control'].update({key:path,key+'_sha256':sha(path)})
    from historical_compact_adapters_v73 import validate_historical_profile
    validate_historical_profile(profile);write_new(output_profile,profile);return profile


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);sub=cli.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--settings',required=True);p.add_argument('--protocol',required=True)
    for name in('run','worker','qualify'):
        p=sub.add_parser(name);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True)
        if name=='run':p.add_argument('--python',default=sys.executable)
        if name=='qualify':p.add_argument('--output-profile',required=True)
    a=cli.parse_args()
    result=prepare(a.settings,a.protocol)if a.action=='prepare'else run(a.protocol,a.publication_commit,a.python)if a.action=='run'else qualify(a.protocol,a.publication_commit,a.output_profile)if a.action=='qualify'else worker(a.protocol,a.publication_commit)
    if result is not None:print(json.dumps(result,indent=2,allow_nan=False))
