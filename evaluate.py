"""Frozen lunar evaluation; exact controls and provisional map references stay separate."""
import argparse,csv,hashlib,json,platform,subprocess,sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from dataset import validate_manifest,load_pair,common_points,sha256
from geometry import transform
from synthetic import texture,make_pair
from matchers import ncc,sift,preprocess


def endpoint_statistics(matrix,reference_points,query_points):
    if matrix is None or not len(reference_points):return None,None
    residual=np.linalg.norm(transform(reference_points,matrix)-np.asarray(query_points),axis=1)
    if not np.isfinite(residual).all():return None,None
    return float(np.median(residual)),float(np.sqrt(np.mean(residual**2)))


def summarize(rows,thresholds):
    summaries=[]
    keys=['dataset_kind','split','error_kind','method','preprocessing','applicability']
    groups=sorted({tuple(r[k] for k in keys) for r in rows})
    for group in groups:
        subset=[r for r in rows if tuple(r[k] for k in keys)==group]
        errors=[r['error_px'] for r in subset if r['error_px'] is not None]
        summary=dict(zip(keys,group));summary.update(pairs=len(subset),registered=sum(r['estimated'] for r in subset),
            failed_registrations=sum(not r['estimated'] for r in subset),unscorable=sum(r['error_px'] is None for r in subset),
            median_error_registered_px=float(np.median(errors)) if errors else None,
            median_runtime_ms=float(np.median([r['runtime_ms'] for r in subset])))
        for threshold in thresholds:
            summary[f'within_{threshold:g}px_rate']=sum(r['error_px'] is not None and r['error_px']<=threshold for r in subset)/len(subset)
        summaries.append(summary)
    return summaries


def validate_config(config):
    if config['schema_version']!=1 or config['image_size']!=512 or config['repeats']<1:
        raise ValueError('Unsupported config; imported checkpoints use the frozen 512px grid')
    if not config['methods'] or len(set(config['methods']))!=len(config['methods']) or any(m not in ('ncc','sift','disk_lightglue') for m in config['methods']):
        raise ValueError('Unsupported or duplicate methods')
    if config['preprocessing']!=['none','clahe']:
        raise ValueError('Use the prespecified none/CLAHE ablation')
    if any(not np.isfinite(t) or t<=0 for t in config['thresholds_px']):raise ValueError('Thresholds must be positive')


def run(manifest,config,output,root='.',split='all',model_cache='data/models'):
    validate_config(config);assets=validate_manifest(manifest,root)
    pairs=[p for p in manifest['pairs'] if split=='all' or p['split']==split]
    if not pairs:raise ValueError('No pairs selected')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    cv2.setNumThreads(1);cv2.setRNGSeed(config['seed']);np.random.seed(config['seed'])
    methods={'ncc':ncc,'sift':sift};learned=None
    if 'disk_lightglue' in config['methods']:
        from learned import LearnedMatcher
        learned=LearnedMatcher(model_cache,config['max_keypoints'],config['device']);methods['disk_lightglue']=learned
    # Warmup uses a procedural image, never a held-out observation.
    warm=texture(config['seed'],256)
    for name in config['methods']:
        methods[name](warm,warm)
    if learned:learned.sync()
    output.mkdir(parents=True)
    (output/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    (output/'frozen_config.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    code_files=['dataset.py','evaluate.py','experiment_report.py','geometry.py','matchers.py','learned.py','synthetic.py','fetch_models.py']
    source_hashes={n:sha256(Path(__file__).with_name(n)) for n in code_files}
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision=None
    meta=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,
        source_sha256=source_hashes,git_revision=revision,manifest_sha256=sha256(output/'frozen_manifest.json'),
        config_sha256=sha256(output/'frozen_config.json'),device=config['device'],opencv_threads=1,
        learned_initialization_ms=learned.initialization_ms if learned else None,
        timing='Warmed CPU/device inference including preprocessing, feature extraction, matching and geometry; excludes I/O, download and learned model loading. One repeat: descriptive, not a precise speed benchmark.',
        reference_warning='Map-grid error is reference disagreement with unquantified map uncertainty.')
    if learned:
        import torch
        meta['torch']=torch.__version__;meta['model_lock']=json.loads(Path(__file__).with_name('model_lock.json').read_text())
    (output/'environment.json').write_text(json.dumps(meta,indent=2))
    rows=[];gallery=[];controlled_regions=set()

    def evaluate_pair(pair,a,b,ma,mb,points,targets,kind,error_kind,case,incidence_difference,angle=0):
        for name in config['methods']:
            for preparation in config['preprocessing']:
                timings=[];result=None
                for repetition in range(config['repeats']):
                    cv2.setRNGSeed(config['seed'])
                    if learned:learned.sync()
                    start=perf_counter()
                    aa=preprocess(a,preparation=='clahe');bb=preprocess(b,preparation=='clahe')
                    aa[~ma]=0;bb[~mb]=0
                    result=methods[name](aa,bb,ma,mb)
                    if learned:learned.sync()
                    timings.append((perf_counter()-start)*1000)
                error,rmse=endpoint_statistics(result.matrix,points,targets)
                row=dict(pair_id=pair['pair_id'],region_id=pair['region_id'],split=pair['split'],dataset_kind=kind,
                    case=case,error_kind=error_kind,reference_uncertainty_px=pair.get('reference_uncertainty_px') if kind=='real_repeat_observation' else 0,
                    method=name,preprocessing=preparation,applicability='out_of_model_rotation' if name=='ncc' and angle else 'in_model',
                    estimated=result.matrix is not None,error_px=error,rmse_px=rmse,scored_points=len(points),
                    within_primary_threshold=error is not None and error<=config['primary_threshold_px'],
                    reason=result.reason or ('no_valid_reference_points' if not len(points) else ''),
                    runtime_ms=float(np.median(timings)),support=result.support,
                    support_type='template_pixels' if name=='ncc' else 'geometric_inliers',
                    incidence_difference_deg=incidence_difference,transform=None if result.matrix is None else result.matrix.tolist())
                rows.append(row)
                if kind=='real_repeat_observation' and preparation=='none':
                    gallery.append(dict(row=row,reference=a,query=b,matrix=result.matrix,
                        reference_points=result.reference_points,query_points=result.query_points))
                print(f'{kind} {pair["pair_id"]} {name}/{preparation}: '+('failed' if error is None else f'{error:.2f}px'),flush=True)

    for pair in pairs:
        a,b,ma,mb,grid=load_pair(pair,assets,root,config['image_size'])
        if pair['reference_type']=='independent_checkpoints':
            points=np.asarray([p['reference'] for p in pair['checkpoints']],float)
            targets=np.asarray([p['query'] for p in pair['checkpoints']],float)
            valid=(ma[np.rint(points[:,1]).astype(int).clip(0,511),np.rint(points[:,0]).astype(int).clip(0,511)] &
                   mb[np.rint(targets[:,1]).astype(int).clip(0,511),np.rint(targets[:,0]).astype(int).clip(0,511)])
            points,targets=points[valid],targets[valid];error_kind='independent_checkpoint_error'
        else:
            points=common_points(ma,mb);targets=points.copy();error_kind='map_reference_disagreement'
        inc=abs(assets[pair['reference']]['incidence_deg']-assets[pair['query']]['incidence_deg'])
        evaluate_pair(pair,a,b,ma,mb,points,targets,'real_repeat_observation',error_kind,'repeat',inc)
        if pair['region_id'] not in controlled_regions:
            controlled_regions.add(pair['region_id'])
            for i,case in enumerate(config['controls']):
                q,truth=make_pair(a,case,config['seed']+i)
                qm=cv2.warpAffine(ma.astype(np.uint8),truth,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
                q[~qm]=0
                points=common_points(ma,ma);targets=transform(points,truth)
                indices=np.rint(targets).astype(int)
                valid=(indices[:,0]>=0)&(indices[:,0]<512)&(indices[:,1]>=0)&(indices[:,1]<512)
                valid &= qm[indices[:,1].clip(0,511),indices[:,0].clip(0,511)]
                ctrl=dict(pair,pair_id=pair['region_id']+'_control_'+case['name'])
                evaluate_pair(ctrl,a,q,ma,qm,points[valid],targets[valid],'real_image_synthetic_transform','known_transform_error',case['name'],None,case['angle'])
    (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary=summarize(rows,config['thresholds_px']);(output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    from experiment_report import render_results
    render_results(rows,summary,gallery,config,meta,output)
    return rows,summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='lunar_manifest.json');p.add_argument('--config',default='evaluation_config.json');p.add_argument('--output',required=True);p.add_argument('--root',default='.');p.add_argument('--split',choices=['train','dev','test','all'],default='all');p.add_argument('--model-cache',default='data/models');a=p.parse_args()
    run(json.loads(Path(a.manifest).read_text()),json.loads(Path(a.config).read_text()),a.output,a.root,a.split,a.model_cache)
