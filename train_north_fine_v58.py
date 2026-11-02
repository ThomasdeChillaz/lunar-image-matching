"""Matched fine-encoder query-phase ablation with a frozen proposal network."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib,json,re,shutil,subprocess
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch

from benchmark_north_control_v38 import inference_inputs,score_case,totals
from compact_lunar import CompactMatcher
from evaluate_geometry_warp_v11 import control
from geometry_warp_native_v11 import infer
from localization_refiner_v3 import decode_peak
from native_refinement_v7 import objective
from north_fine_data_v58 import VARIANTS,prepare,sample_batch,physical_samples,make_fine,FineMatcher,CheckpointFeatureCache
from north_training_bundle_v38 import load_bundle,read,sha
from north_coarse_reachability_v53 import verify_ended
from train_balanced_localization_v3 import save_torch
from train_north_control_v39 import SOURCES as OLD_SOURCES,read_rows,save_rows,counts,initial_equivalence,check_seal
from train_search_window_v18 import configure,record_once,weights_once,selection_key
from lunargeo_dataset import write_new

SEED=20261101
BANK_SEED=20261108
MANIFEST='reports/north_training_data_v38.json'
STEPS=2000
INTERVAL=500
CAPS={'pilot':600,'study':5400}
SOURCES=list(dict.fromkeys(OLD_SOURCES+['north_fine_data_v58.py','train_north_fine_v58.py',
    'verify_north_fine_v58.py','test_north_fine_v58.py','native_crop_alignment_v57.py',
    'native_refinement_v7.py','localization_refiner_v3.py','north_coarse_reachability_v53.py']))


def clock_check(protocol,tick):
    now=datetime.now(timezone.utc)
    if (now<datetime.fromisoformat(protocol['created_utc'])
            or now>=datetime.fromisoformat(protocol['absolute_deadline_utc'])
            or perf_counter()-tick>=protocol['max_calendar_seconds']):
        raise TimeoutError('Original fine-study allowance exhausted; never reset it')


def inputs_sha(arrays):
    return [hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()for a in arrays]


def fit_step(fine,optimizer,tiles,data,prepared,rng,variant,step):
    before=perf_counter();fine.train()
    arrays,metadata=sample_batch(tiles,data,prepared,rng,variant)
    batch=[torch.from_numpy(x)for x in arrays]
    optimizer.zero_grad(set_to_none=True)
    local,no_match=fine(*batch[:3])
    loss=objective(local,no_match,batch[3],torch.ones(len(local),dtype=torch.bool),False)
    if not torch.isfinite(loss):raise ValueError('Nonfinite native fine objective')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(fine.parameters(),5.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite native fine gradient')
    optimizer.step()
    return dict(step=step,loss=float(loss),gradient_norm=float(norm),samples=metadata,
        input_sha256=inputs_sha(arrays),fitting_seconds=perf_counter()-before)


def snapshot(fine,optimizer,rng,step,history):
    return dict(model=fine.state_dict(),optimizer=optimizer.state_dict(),step=step,history=history,
        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        fitting_seconds=sum(h['fitting_seconds']for h in history))


def bank_metrics(fine,tiles,data,prepared,variant):
    arrays,meta=sample_batch(tiles,data,prepared,np.random.default_rng(BANK_SEED),variant)
    batch=[torch.from_numpy(x)for x in arrays];fine.eval()
    with torch.inference_mode():
        logits,no_match=fine(*batch[:3])
        loss=objective(logits,no_match,batch[3],torch.ones(len(logits),dtype=torch.bool),False)
        error=torch.linalg.vector_norm(decode_peak(logits)+16-batch[3],dim=1).numpy()
    return dict(loss=float(loss),correct_at_native_px={str(t):int((error<=t).sum())for t in (1,2,4,8,12)},
        examples=8,input_sha256=inputs_sha(arrays),samples=meta,training_only=True)


def admission(mode,variant):
    verify_ended('runs/native-crop-v57-state.json')
    a=read('reports/native_crop_audit_v57.json')
    if not a['passed']or not a['aligned_controls_passed']or a['replayed_points']!=32:
        raise ValueError('Actual native crop/full-image control and independent replay required')
    result=dict(crop_audit_sha256=sha('reports/native_crop_audit_v57.json'))
    if mode=='study':
        for v in VARIANTS:
            path='reports/north_fine_'+v+'_pilot_audit_v58.json';p=read(path)
            if not p['passed']or not p['exact_optimizer_retraining']or p['replayed_updates']!=100:
                raise ValueError('Both actual100update pilots must exactly replay')
        pilot=Path('runs/north-fine-v58-pilot-'+variant);s=read(pilot/'summary.json')
        if not s['post_control']['passed']or s['fine_weight_change_l2']<=0:
            raise ValueError('Pilot must preserve controls and actually update the fine encoder')
        bank=read('runs/north-v38-baseline-cost-bank/summary.json')
        estimate=2*(s['fitting_seconds']/100*STEPS+bank['results']['frozen_surface']['median_measured_inference_ms']/1000*576+30)
        if not np.isfinite(estimate)or estimate>=CAPS['study']:
            raise ValueError('Measured fine training/evaluation exceeds declared90minute cap')
        result.update(pilot_root=str(pilot),pilot_summary_sha256=sha(pilot/'summary.json'),
            pilot_latest_sha256=sha(pilot/'latest.pt'),projected_seconds_with_factor2=float(estimate),reused_pilot_updates=100)
    return result


def evaluate(matcher, bundle, cases, folder, features, protocol, tick):
    """Save each case once; an interrupted evaluation resumes that checkpoint."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    checkpoint_cases = [c['pair_id'] for c in cases]
    record_once(folder / 'protocol.json', dict(case_ids=checkpoint_cases,
        cohort_protocol_sha256=sha(Path(protocol['root']) / 'protocol.json')))
    rows = []
    for case in cases:
        clock_check(protocol, tick)
        name = hashlib.sha256(case['pair_id'].encode()).hexdigest() + '.json'
        path = folder / name
        if path.exists():
            row = read(path)
            if any(row[k] != v for k, v in case.items()):
                raise ValueError('Saved evaluation case changed')
        else:
            inputs = inference_inputs(bundle, case)
            ids = ((case['tile_id'], 2 * case['light_a']),
                   (case.get('query_tile_id', case['tile_id']), 2 * case['light_b'] + 1))
            before = perf_counter()
            error = None
            try:
                chosen, branches = infer(matcher, *inputs, features, ids)
                branch = chosen if chosen is not None else branches[0]
                p, q = branch['p'], branch['q']
                pose = None if chosen is None else branch['pose']
            except (RuntimeError, ValueError, cv2.error) as exc:
                p = q = np.empty((0, 2)); pose = None; branch = {}
                error = dict(type=type(exc).__name__, message=str(exc))
            elapsed_ms = (perf_counter() - before) * 1000
            row = dict(case, method='native_fine_sampling_control',
                **score_case(bundle, case, p, q, pose),
                elapsed_ms=elapsed_ms, timing_scope='Frozen fine-map cache reused; not isolated latency.',
                inference_error=error, quarter_turn=branch.get('quarter_turn'),
                pose_info=branch.get('info', {}), geometry_history=branch.get('geometry_history', []))
            write_new(path, row)
        rows.append(row)
    if len(rows) != len(cases) or len({r['pair_id'] for r in rows}) != len(cases):
        raise ValueError('Incomplete or duplicate selection denominator')
    return rows


def run(mode,variant,output,commit,resume=False):
    if mode not in CAPS or variant not in VARIANTS:raise ValueError('Declared mode/variant required')
    started=datetime.now(timezone.utc);tick=perf_counter();root=Path(output)
    if root.exists()and not resume:raise FileExistsError('Preserve attempts; explicit same-deadline resume required')
    if resume and not root.exists():raise FileNotFoundError('No existing fine attempt')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full public source required')
    for name in SOURCES:
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Source differs from public freeze:'+name)
    qa_path='reports/north_fine_software_v58.json';qa=read(qa_path)
    if not qa['passed']or not qa['full_repository_regression']or not qa['fresh_smoke']:raise ValueError('Current source-bound full software checks required')
    for name,digest in qa['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Source changed after software checks')
    gate=admission(mode,variant);configure();steps=100 if mode=='pilot'else STEPS
    static=dict(root=str(root),mode=mode,variant=variant,seed=SEED,steps=steps,interval=INTERVAL,
        max_calendar_seconds=CAPS[mode],cpu_threads=2,publication_commit=commit,
        source_sha256={n:sha(n)for n in SOURCES},manifest_sha256=sha(MANIFEST),
        software_sha256=sha(qa_path),gate=gate,total_parameters=56544,trainable_parameters=28272,
        batch_size=8,learning_rate=1e-4,weight_decay=1e-4,gradient_clip=5.,
        frozen_coarse_proposals=True,initialization_eligible=True,
        query_phase_intervention_only=True,full_image_appearance_before_crop=True,
        fresh_fine_cache_per_checkpoint=True,model_promoted=False,
        scope='Matched native fine-encoder query-phase control. Exact BVH3D positive patches, original frozen coarse proposals, unchanged native inference/pose estimator. No no-match head, accepted-risk policy, final-test or SOTA claim.')
    if root.exists():
        protocol=read(root/'protocol.json')
        if any(protocol[k]!=v for k,v in static.items()):raise ValueError('Changed fine experiment on resume')
        if(root/'COMPLETE.json').exists():
            for n,h in read(root/'COMPLETE.json')['sha256'].items():
                if sha(root/n)!=h:raise ValueError('Sealed fine experiment changed')
            return read(root/'summary.json')
    else:
        root.mkdir(parents=True)
        protocol=dict(static,created_utc=started.isoformat(),absolute_deadline_utc=(started+timedelta(seconds=CAPS[mode])).isoformat())
        write_new(root/'protocol.json',protocol)
    try:
        clock_check(protocol,tick);train=load_bundle(MANIFEST,train_only=True);prepared=prepare(train[1],train[2])
        if len(prepared[0])!=16 or len(prepared[1])!=48:raise ValueError('Complete original training split required')
        coarse=CompactMatcher(train[0]['coarse_model']);coarse.model.eval().requires_grad_(False)
        coarse_initial={k:v.clone()for k,v in coarse.model.state_dict().items()}
        torch.manual_seed(SEED);fine=make_fine(coarse);optimizer=torch.optim.AdamW(fine.parameters(),lr=1e-4,weight_decay=1e-4)
        rng=np.random.default_rng(SEED)
        if not(root/'initial.pt').exists():
            save_torch(fine.state_dict(),root/'initial.pt')
            save_torch(snapshot(fine,optimizer,rng,0,[]),root/'initial_state.pt')
            if mode=='pilot':save_torch(snapshot(fine,optimizer,rng,0,[]),root/'latest.pt')
            else:
                pilot_initial=torch.load(Path(gate['pilot_root'])/'initial.pt',map_location='cpu',weights_only=True)
                if pilot_initial.keys()!=fine.state_dict().keys()or any(not torch.equal(v,pilot_initial[k])for k,v in fine.state_dict().items()):
                    raise ValueError('Pilot and study initial fine weights differ')
                shutil.copy2(Path(gate['pilot_root'])/'latest.pt',root/'latest.pt')
        initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(v,initial[k])for k,v in fine.state_dict().items()):raise ValueError('Original fine initializer changed')
        initial_control=control(FineMatcher(coarse,fine));record_once(root/'initial_control.json',initial_control)
        if not initial_control['passed']:raise ValueError('Actual initialized fine wrapper control failed')
        before_bank=bank_metrics(fine,*train[1:],prepared,variant);record_once(root/'bank_initial.json',before_bank)
        selection=None;cache=CheckpointFeatureCache();validation=[]
        if mode=='study':
            selection=load_bundle(MANIFEST,selection_only=True)
            weights_once(root/'checkpoint_step_0.pt',initial)
            path=root/'selection_step_0.json.gz'
            if not path.exists():
                rows=evaluate(FineMatcher(coarse,fine),selection,selection[0]['cases'],root/'partial-selection-0',cache.for_weights(sha(root/'checkpoint_step_0.pt')),protocol,tick)
                initial_equivalence(rows,read_rows('runs/north-v39-full-baselines/results.json.gz'));save_rows(path,rows)
            rows=read_rows(path);record_once(root/'validation_step_0.json',dict(step=0,selection=counts(rows),results_sha256=sha(path)))
        state=torch.load(root/'latest.pt',map_location='cpu',weights_only=True)
        fine.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);cache.invalidate()
        history=state['history'];step=state['step']
        if len(history)!=step or step>steps:raise ValueError('Checkpoint/history boundary differs')
        while True:
            clock_check(protocol,tick);check_seal(protocol)
            if mode=='study'and step%INTERVAL==0 and step>0:
                cp=root/f'checkpoint_step_{step}.pt';weights_once(cp,fine.state_dict());path=root/f'selection_step_{step}.json.gz'
                if not path.exists():
                    rows=evaluate(FineMatcher(coarse,fine),selection,selection[0]['cases'],root/f'partial-selection-{step}',cache.for_weights(sha(cp)),protocol,tick);save_rows(path,rows)
                rows=read_rows(path);record_once(root/f'validation_step_{step}.json',dict(step=step,selection=counts(rows),results_sha256=sha(path)))
                print('FINE POSE',variant,step,counts(rows),flush=True)
            if step==steps:break
            step+=1;history.append(fit_step(fine,optimizer,train[1],train[2],prepared,rng,variant,step))
            save_torch(snapshot(fine,optimizer,rng,step,history),root/'latest.pt')
            if step%50==0:print('FINE UPDATE',mode,variant,step,'loss',history[-1]['loss'],flush=True)
        after_bank=bank_metrics(fine,*train[1:],prepared,variant)
        if any(not torch.equal(v,coarse_initial[k])for k,v in coarse.model.state_dict().items()):raise ValueError('Frozen proposal network changed')
        delta=float(sum(float((v-initial[k]).square().sum())for k,v in fine.state_dict().items())**.5)
        result=dict(complete=True,mode=mode,variant=variant,completed_steps=steps,history=history,
            fine_weight_change_l2=delta,fitting_seconds=sum(h['fitting_seconds']for h in history),
            initial_bank=before_bank,final_bank=after_bank,coarse_weights_unchanged=True,
            protocol_sha256=sha(root/'protocol.json'),model_promoted=False,
            calibration_or_test_pixels_loaded=0,model_selection_pixels_loaded=0 if mode=='pilot'else 1,
            scope=static['scope'])
        if mode=='pilot':
            result['post_control']=control(FineMatcher(coarse,fine))
        else:
            validation=[read(root/f'validation_step_{s}.json')for s in range(0,STEPS+1,INTERVAL)]
            winner=max(validation,key=lambda x:selection_key(x['selection'],x['step']))
            weights_once(root/'final.pt',fine.state_dict());fine.load_state_dict(torch.load(root/f'checkpoint_step_{winner["step"]}.pt',map_location='cpu',weights_only=True),strict=True);cache.invalidate()
            weights_once(root/'selected.pt',fine.state_dict())
            neg=evaluate(FineMatcher(coarse,fine),selection,selection[0]['negative_cases'],root/'partial-selected-negatives',cache.for_weights(sha(root/'selected.pt')),protocol,tick)
            if not(root/'selected_negatives.json.gz').exists():save_rows(root/'selected_negatives.json.gz',neg)
            elif read_rows(root/'selected_negatives.json.gz')!=neg:raise ValueError('Saved selected negatives differ')
            result.update(validation=validation,selected_step=winner['step'],selection=winner['selection'],
                selected_negative_counts=totals(neg),selected_control=control(FineMatcher(coarse,fine)),
                reused_pilot_updates=100,new_training_updates=1900)
        clock_check(protocol,tick);result['calendar_seconds']=(datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds()
        record_once(root/'summary.json',result)
        write_new(root/'COMPLETE.json',dict(sha256={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p!=root/'COMPLETE.json'}))
        print(json.dumps({k:v for k,v in result.items()if k not in ('history','validation')},indent=2),flush=True);return result
    except BaseException as exc:
        record_once(root/('FAILED-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),dict(error_type=type(exc).__name__,error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=list(CAPS),required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--output',required=True)
    p.add_argument('--publication-commit',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.mode,a.variant,a.output,a.publication_commit,a.resume)
