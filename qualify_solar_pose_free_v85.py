"""One source-published, parent-precharged 600s diagnostic; never a backend.

No Torch import/forward occurs during import or admission. The controller owns
one external process-group deadline. Failed/partial outputs and the full charge
remain. Native global feature extraction violates the principal active-patch
contract: every output is DIAGNOSTIC ONLY, unaccepted and pose-free.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


POSITIVE = 'u012-l00-00-positive'
NEGATIVE = 'u012-to-u020-l00-00-nonoverlap'
CHECKPOINTS = {'step0': 'runs/solar-learning-v79/sh_study/step_0.pt',
               'sh2000': 'runs/solar-learning-v79/sh_study/step_2000.pt',
               'pooled2000': 'runs/solar-learning-v79/pooled_study/step_2000.pt'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc)
def utc(value): return datetime.fromisoformat(value)
def write(path, value):
    with Path(path).open('x') as f: json.dump(jsonable(value), f, indent=2, allow_nan=False); f.write('\n')


def validate_job(protocol_path, commit, publication=True):
    p = read(protocol_path)
    if (p['purpose'] != 'pose_free_pretrained_diagnostic_qualification'
            or p['diagnostic_only'] is not True or p['calibration_or_final_access'] is not False
            or p['cpu_threads'] != 2 or type(p['max_calendar_seconds']) is not int or p['max_calendar_seconds'] != 600
            or p['cleanup_inside_cap_seconds'] != 5 or p['case_ids'] != [POSITIVE, NEGATIVE]
            or p['checkpoints'] != CHECKPOINTS or p['math_sun_budget'] != 292
            or not math.isfinite(p['per_extraction_cap_seconds'])
            or p['per_extraction_cap_seconds'] <= 0 or p['per_extraction_cap_seconds'] > 180):
        raise ValueError('One fixed diagnostic scope/600s total/292-Sun budget required')
    if (Path(p['output']).is_absolute() or '..' in Path(p['output']).parts
            or utc(p['latest_start_utc']) >= utc(p['absolute_deadline_utc'])):
        raise ValueError('Safe output and finite original dispatch/deadline required')
    from solar_learning_pipeline_v73 import CANONICAL_JOURNAL, load_contract, journal_state
    if p['reservation_journal'] != CANONICAL_JOURNAL: raise ValueError('One canonical journal required')
    contract = load_contract(p['learning_contract'])
    if sha(p['learning_contract']) != p['learning_contract_sha256']: raise ValueError('Learning contract changed')
    for pins in (p['source_sha256'],p['input_sha256']):
        for name,h in pins.items():
            if sha(name) != h: raise ValueError('Frozen diagnostic bytes changed: '+name)
    if p['source_sha256'].get(p['driver']) != sha(__file__): raise ValueError('Actual driver source not pinned')
    if publication:
        for name in (protocol_path, *p['source_sha256']):
            if subprocess.check_output(['git', 'show', commit+':'+name]) != Path(name).read_bytes():
                raise ValueError('Diagnostic protocol/source not published: '+name)
    manifest = read(p['manifest'])
    if p['input_sha256'].get(p['manifest']) != sha(p['manifest']): raise ValueError('Manifest must be pinned')
    by = {c['case_id']: c for c in manifest['cases']}
    pos, neg = by[POSITIVE], by[NEGATIVE]
    if (not pos['positive'] or neg['positive'] or neg['matched_positive_case'] != POSITIVE
            or pos['split'] != 'model_selection' or neg['split'] != 'model_selection'
            or pos['unit_index'] != 12 or neg['query_unit_index'] != 20):
        raise ValueError('Exact original positive and both negative endpoints required')
    for c in (pos, neg):
        for name, h in c['input_sha256'].items():
            if p['input_sha256'].get(name) != h: raise ValueError('All case inputs must retain original hashes')
    for name in CHECKPOINTS.values():
        if name not in p['input_sha256']: raise ValueError('Actual selected checkpoints must be pinned')
    _, journal, _ = journal_state(p['compute_ledger'], p['compute_ledger_sha256'], CANONICAL_JOURNAL)
    charge = read(p['reservation_receipt']); entry = charge['reservation']
    if (entry['job_id'] != sha(protocol_path)+'/pose_free_qualification' or entry['category'] != 'baseline'
            or entry['seconds'] != 600 or entry.get('refund_on_failure') is not False
            or [x for x in journal['reservations'] if x['job_id'] == entry['job_id']] != [entry]):
        raise ValueError('Parent must precharge the unchanged canonical reserve exactly once')
    return p, by


def proposal_recall(reference_xy, candidates, target_xy, visible, bank_xy, bank_valid):
    """Post-prediction diagnostic; no eligibility/point or case filtering."""
    import numpy as np
    target = np.asarray(target_xy); visible = np.asarray(visible, bool)
    finite = np.isfinite(target).all(1); valid = visible & finite
    errors = np.full(len(target), np.inf)
    errors[valid] = np.linalg.norm(np.asarray(candidates)[valid]-target[valid, None], axis=-1).min(1)
    in_image = valid & ((target >= 0)&(target <= 511)).all(1)
    nearest = np.full(len(target), np.inf)
    if valid.any():
        nearest[valid] = np.linalg.norm(target[valid, None]-np.asarray(bank_xy)[bank_valid][None], axis=-1).min(1)
    covered = valid & (nearest <= 16.)
    return dict(all_retained_reference_anchors=len(reference_xy), co_visible=int(valid.sum()),
        targets_inside_query_image=int(in_image.sum()), geometric_bank_coverage_at16=int(covered.sum()),
        recalled={str(t):int((errors <= t).sum()) for t in (1,2,4,8,12,16)},
        in_grid_recalled_at16=int(((errors <= 16.)&covered).sum()), in_grid_denominator=int(covered.sum()),
        scope='Candidate-window recall only. All retained anchors remain in the denominator; FOV/coverage reported separately. No camera pose or acceptance.')


def constant_score_control(frozen, decoded, query):
    """One real response, common query features; fail closed on NaNs/zero rows."""
    import numpy as np
    values=[np.asarray(x,np.float32)for x in(frozen,decoded,query)]
    if (any(x.ndim!=2 or x.shape[1]!=32 or not np.isfinite(x).all()for x in values)
            or values[0].shape!=values[1].shape or any((np.linalg.norm(x,axis=1)<=1e-8).any()for x in values)):
        raise ValueError('Finite nonzero32D common-feature control required')
    frozen,response,query=[x/np.linalg.norm(x,axis=1,keepdims=True)for x in values]
    native_scores=frozen@query.T;response_scores=response@query.T
    difference=float(np.max(np.abs(native_scores-response_scores)))
    if difference>2e-6:raise ValueError('Step0 constant-response/native score difference exceeds2e-6')
    top=np.sort(native_scores,axis=1)[:,-2:]
    return dict(max_score_difference=difference,tolerance=2e-6,
        near_top_ties_at_tolerance=int(((top[:,1]-top[:,0])<=2e-6).sum()),
        argmax_differences=int((native_scores.argmax(1)!=response_scores.argmax(1)).sum())),dict(
            frozen_reference=frozen,predicted_reference=response,query_descriptors=query)


def known_transform_gate(recall):
    """Predeclared80% top8/16px over ALL retained generated anchors."""
    denominator=recall['all_retained_reference_anchors'];correct=recall['recalled']['16']
    passed=denominator>0 and correct/denominator>=.8
    return dict(passed=bool(passed),correct=correct,denominator=denominator,
        minimum_fraction=.8,threshold_native_px=16,candidates_per_anchor=8,
        fraction=None if denominator==0 else correct/denominator,
        denominator_scope='ALL retained generated reference anchors; no FOV filtering')


def worker(protocol_path, commit):
    p, by = validate_job(protocol_path, commit); root = Path(p['output'])
    context = read(root/'watchdog_context.json')
    if context['controller_pid'] != os.getppid() or context['protocol_sha256'] != sha(protocol_path):
        raise ValueError('Actual original external watchdog parent required')
    deadline = utc(context['work_deadline_utc'])
    def check():
        if now() >= deadline: raise TimeoutError('Original diagnostic deadline exhausted')
    check()
    # Heavy dependencies/model startup are inside the external allowance.
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from localize_terrain import load_matcher
    from solar_frontend_v63 import ReferenceGeometry
    from solar_response_cache_v84 import GeometryOwnerGuardV84
    from solar_pose_free_features_v85 import extract_native_features, retrieve_from_extracted, ExtractionCaps, ExtractionUnavailable
    from solar_pose_free_retrieval_v85 import Limits
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    def image(path):
        with Image.open(path) as im: return np.asarray(im.convert('L')).copy()
    pos, neg = by[POSITIVE], by[NEGATIVE]
    reference = image(pos['inference_inputs']['reference']); positive = image(pos['inference_inputs']['query'])
    false = image(neg['inference_inputs']['query'])
    original_geometry = ReferenceGeometry.load(pos['inference_inputs']['reference_geometry'])
    original_guard = GeometryOwnerGuardV84(original_geometry)
    sun = np.asarray(pos['inference_inputs']['reference_sun_world'], float)
    rows = []; arrays = {}; load_rows = []
    def private_geometry():
        return ReferenceGeometry(**{name:np.array(getattr(original_geometry,name),copy=True)
            for name in original_geometry.__dataclass_fields__})
    def save_row(name, row, payload=None):
        check(); row.update(diagnostic_only=True, accepted=False, pose=None,
            principal_active_native_deferral_contract_met=False,estimated_sun_frame='reference_camera')
        if payload:
            path = root/(name+'.npz'); np.savez_compressed(path, **payload)
            row.update(arrays=str(path), arrays_sha256=sha(path))
        path=root/(name+'.json'); write(path,row); rows.append(dict(name=name,row=str(path),row_sha256=sha(path)))
    def load(key):
        check(); tick=time.perf_counter(); checkpoint=CHECKPOINTS[key]
        model=load_matcher(checkpoint,dict(model_version='solar-experimental-v63',checkpoint_sha256=sha(checkpoint),response=True)).response
        counts=dict(coarse=0,fine=0)
        def coarse(*args):counts['coarse']+=1
        def fine(*args):counts['fine']+=1
        hooks=[model.coarse.register_forward_hook(coarse),model.fine.register_forward_hook(fine)]
        load_rows.append(dict(key=key,checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),model_load_seconds=time.perf_counter()-tick,
            response_updates=0 if key=='step0' else 2000,step0_scope='Pretrained original CNN with untrained constant-response head'if key=='step0'else None))
        return model,counts,hooks
    def extract(model, query, physical):
        check(); cap=min(float(p['per_extraction_cap_seconds']),max(.01,(deadline-now()).total_seconds()))
        return extract_native_features(reference,query,private_geometry(),sun,model,
            caps=ExtractionCaps(max_calendar_seconds=cap,max_response_decodes=292),include_physical_evidence=physical)
    def assert_real(extracted,before,counts):
        meta=extracted.metadata; c=meta['encoder_counts']
        actual={k:counts[k]-before[k] for k in counts}
        if (meta['software_stub_only'] or actual['fine'] != c['reference_fine_forwards']
                or actual['coarse'] != c['reference_frozen_coarse_forwards']+c['query_frozen_coarse_forwards']
                or min(actual.values()) <= 0): raise ValueError('Actual pretrained forward hooks/counters disagree')
        return actual
    # Actual step0 single response against native descriptors from COMMON features.
    model,counts,hooks=load('step0'); before=dict(counts); tick=time.perf_counter()
    higher_nonzero=int(torch.count_nonzero(model.coefficients.weight[32:])+torch.count_nonzero(model.coefficients.bias[32:]))
    if higher_nonzero:raise ValueError('Declared step0 head is not constant across target Sun')
    rejected=False;before_invalid=dict(counts)
    try:extract_native_features(reference,np.zeros((511,512),np.uint8),private_geometry(),sun,model)
    except ValueError as exc:
        if 'Native512'not in str(exc):raise
        rejected=True
    if not rejected or counts!=before_invalid:raise ValueError('Wrong query shape was not rejected before neural forwards')
    save_row('invalid-query-shape',dict(passed=True,reason='Native512 shape rejected before encoders',actual_forward_hooks=dict(coarse=0,fine=0)))
    with extract(model,reference,False) as extracted:
        x=extracted.retrieval_inputs(); condition=np.column_stack([np.broadcast_to(x['known_reference_sun'],(len(extracted.reference_xy),3)),x['reference_normals'],x['reference_views']]).astype(np.float32)
        target=x['sun_vectors'][np.flatnonzero(x['coarse_sun_mask'])[0]]
        decoded=x['response_predictor'](x['reference_response_features'],condition,target,pooled=False)
        check_result,common=constant_score_control(x['frozen_reference_descriptors'],decoded,x['query_descriptors'][x['query_valid']])
        check_result['coarse_fine_digest_equal']=extracted.metadata['model_identity']['coarse_sha256']==extracted.metadata['model_identity']['fine_sha256']
        check_result['higher_harmonic_nonzeros']=higher_nonzero
        if not check_result['coarse_fine_digest_equal']:raise ValueError('Step0 constant-response/native basis control failed')
        actual=assert_real(extracted,before,counts)
        save_row('step0-common-features',dict(passed=True,controls=check_result,actual_forward_hooks=actual,
            actual_pretrained_forward=True,extraction=extracted.metadata,calendar_seconds=time.perf_counter()-tick),
            dict(reference_xy=extracted.reference_xy,**common,query_native_xy=x['query_native_xy'][x['query_valid']]))
    # Generated image translation: native-only, no physical shadow labels.
    from synthetic import texture
    generated=texture(20261101,512);affine=np.array([[1.,0.,8.],[0.,1.,-8.]])
    translated=cv2.warpAffine(generated,affine,(512,512),flags=cv2.INTER_LINEAR)
    before=dict(counts);tick=time.perf_counter()
    with extract_native_features(generated,translated,private_geometry(),sun,model,
            caps=ExtractionCaps(max_calendar_seconds=min(p['per_extraction_cap_seconds'],max(.01,(deadline-now()).total_seconds()))),include_physical_evidence=False) as extracted:
        result=retrieve_from_extracted(extracted,arm='native',limits=Limits(max_scored_suns=292))
        target=np.column_stack([extracted.reference_xy,np.ones(len(extracted.reference_xy))])@affine.T
        recall=proposal_recall(extracted.reference_xy,result['candidates']['query_native_xy'],target,np.ones(len(target),bool),extracted.inputs['query_native_xy'],extracted.inputs['query_valid'])
        gate=known_transform_gate(recall)
        save_row('generated-translation-native',dict(passed=gate['passed'],gate=gate,recall=recall,affine_reference_to_query=affine.tolist(),retrieval=result,
            actual_forward_hooks=assert_real(extracted,before,counts),calendar_seconds=time.perf_counter()-tick),
            dict(reference_xy=extracted.reference_xy,query_candidates=result['candidates']['query_native_xy'],target_query_xy=target))
        if not gate['passed']:raise ValueError('Predeclared generated-translation80% top8/16px gate failed; Moon runs remain unqualified')
    for h in hooks:h.remove()
    del model,extracted,result,x
    # Fresh extraction for each real arm. A single global600s budget covers all.
    for key,arm,name,query,positive_case in [('sh2000','native','u012-native',positive,True),
            ('sh2000','sh','u012-sh',positive,True),('sh2000','native','u012-to-u020-native',false,False),
            ('pooled2000','pooled','u012-pooled',positive,True)]:
        model,counts,hooks=load(key);before=dict(counts);tick=time.perf_counter()
        if min(reference.std(),query.std())<1:
            save_row(name,dict(status='retained_blank_guard_failure',reason='blank_or_uninformative_image',actual_forward_hooks=dict(coarse=0,fine=0)))
            for h in hooks:h.remove()
            del model;continue
        with extract(model,query,True) as extracted:
            result=retrieve_from_extracted(extracted,arm=arm,limits=Limits(max_scored_suns=292))
            # Scoring labels are first opened AFTER candidate prediction.
            recall=None
            if positive_case:
                with np.load(pos['scoring_labels'],allow_pickle=False) as labels:
                    ids=extracted.anchor_indices;target=labels['query_xy'][ids];visible=labels['visible'][ids]
                recall=proposal_recall(extracted.reference_xy,result['candidates']['query_native_xy'],target,visible,extracted.inputs['query_native_xy'],extracted.inputs['query_valid'])
            save_row(name,dict(recall=recall,retrieval=result,actual_forward_hooks=assert_real(extracted,before,counts),
                actual_pretrained_forward=True,
                case_binding={key:(pos if positive_case else neg)[key]for key in('case_id','component','query_component','unit_index','query_unit_index','light_a','light_b')},
                negative_case=not positive_case,negative_expectation='Proposals retained; no rejection policy or pose backend exists',
                calendar_seconds=time.perf_counter()-tick),dict(reference_xy=extracted.reference_xy,query_candidates=result['candidates']['query_native_xy']))
        for h in hooks:h.remove()
        del model,extracted,result
    # Driver applies the unchanged std<1 rule before extraction; zero forwards.
    blank=np.zeros_like(reference)
    if blank.std()>=1:raise ValueError('Blank fixture invalid')
    save_row('blank',dict(status='retained_blank_guard_failure',reason='blank_or_uninformative_image',actual_forward_hooks=dict(coarse=0,fine=0)))
    original_guard.check();check()
    summary=dict(complete=True,software_controls_passed=True,protocol_sha256=sha(protocol_path),rows=rows,model_loads=load_rows,
        software=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,python=sys.version.split()[0],cpu_threads=torch.get_num_threads(),device='cpu',dtype='float32'),
        source_publication_commit=commit,scope='Actual pretrained diagnostic only; no accepted pose, principal active-patch claim, checkpoint selection or accuracy threshold invented.')
    write(root/'summary.json',summary)


def jsonable(value):
    # Applied only to worker result persistence; no object/tensor serialization.
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def controller(protocol_path, commit):
    started=now();tick=time.monotonic()
    p,_=validate_job(protocol_path,commit);root=Path(p['output'])
    if root.exists():raise FileExistsError('Preserve previous qualification; no restart')
    if now()>utc(p['latest_start_utc']):raise TimeoutError('Original dispatch lease expired')
    deadline=min(started+timedelta(seconds=600),utc(p['absolute_deadline_utc']));work=deadline-timedelta(seconds=5)
    if now()>=work:raise TimeoutError('Original qualification deadline exhausted')
    root.mkdir(parents=True,exist_ok=False)
    write(root/'watchdog_context.json',dict(controller_pid=os.getpid(),protocol_sha256=sha(protocol_path),work_deadline_utc=work.isoformat(),absolute_deadline_utc=deadline.isoformat()))
    child=None;failure=None;complete=False
    def stop(signum,frame):raise InterruptedError('Stop requested; preserve every diagnostic artifact')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        with (root/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--protocol',protocol_path,'--publication-commit',commit,'--worker'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
            code=child.wait(timeout=max(0.,(work-now()).total_seconds()))
        if code or now()>=work:raise RuntimeError('Qualification failed or missed original work deadline')
        summary=read(root/'summary.json')
        if not summary['complete'] or summary['protocol_sha256']!=sha(protocol_path):raise ValueError('Actual complete qualification summary required')
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
                except subprocess.TimeoutExpired:failure=dict(type='CleanupFailure',message='No unbounded wait permitted');complete=False
        if now()>=deadline or time.monotonic()-tick>=600:complete=False;failure=dict(type='TimeoutError',message='Original total600s exhausted')
        receipt=dict(complete=complete,protocol_sha256=sha(protocol_path),publication_commit=commit,started_utc=started.isoformat(),ended_utc=now().isoformat(),original_absolute_deadline_utc=p['absolute_deadline_utc'],absolute_deadline_utc=deadline.isoformat(),work_deadline_utc=work.isoformat(),cap_seconds=600,calendar_seconds=time.monotonic()-tick,child_pid=None if child is None else child.pid,child_returncode=None if child is None else child.returncode,failure=failure,summary_sha256=sha(root/'summary.json')if(root/'summary.json').exists()else None,scope='External startup-inclusive diagnostic watchdog; charges are parent-owned and never refunded')
        write(root/('COMPLETE.json'if complete else'FAILED.json'),receipt)
    if not complete:raise SystemExit(json.dumps(failure))
    return receipt


if __name__=='__main__':
    # The published driver may live under reports/; use the one invoked checkout.
    sys.path.insert(0,str(Path.cwd()))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);parser.add_argument('--publication-commit',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.protocol,args.publication_commit)if args.worker else controller(args.protocol,args.publication_commit)
