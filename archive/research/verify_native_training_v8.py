"""Independent uniform-positive sampler and native training replay."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from verify_axial_results_v5 import projected
from verify_coarse_training_v5 import replay_row,aggregate


def audit(root,output):
    import torch
    from compact_lunar import CompactMatcher
    from native_refinement_v7 import NativeRefiner
    from train_balanced_localization_v3 import load_bundle
    from native_sampling_v8 import prepare_uniform,training_batch
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if protocol['calibration_or_test_pixels_loaded']!=0:raise ValueError('Forbidden evaluation scope')
    for p,h in protocol['source_sha256'].items():
        if sha256(p)!=h:raise ValueError('Executed native source changed: '+p)
    manifest,tiles,data=load_bundle('reports/localization_training_data_v5.json')
    stats=prepare_uniform(tiles,data)
    if stats!=protocol['uniform_supervision']:raise ValueError('Uniform supervision count differs')
    for key,t in tiles.items():
        if t['split']!='train':continue
        z=data[key];p=z['uniform_reference_native'];target,visible=projected(p,z['depth_a'],z['depth_b'],*t['cameras'])
        if not visible.all() or not np.allclose(target,z['uniform_target_native'],atol=1e-6,rtol=0):raise ValueError('Independent uniform positive truth differs')
    if sha256('reports/localization_training_data_v5.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    train={t['component'] for t in tiles.values() if t['split']=='train'};selection={t['component'] for t in tiles.values() if t['split']=='model_selection'}
    if train&selection or any(t['split'] not in ('train','model_selection') for t in tiles.values()):raise ValueError('Pixel/split leakage')
    labels=0
    for c in manifest['cases']:
        z=data[c['tile_id']];p=z[f'p{c["key"]}'];target,visible=projected(p,z['depth_a'],z['depth_b'],*tiles[c['tile_id']]['cameras'])
        if visible.tolist()!=z[f'visible{c["key"]}'].tolist() or not np.allclose(target[visible],z[f'target{c["key"]}'][visible],rtol=0,atol=4e-5):
            raise ValueError('Independent corrected cached label differs')
        labels+=1
    paths=sorted(root.glob('*/training.json'));expected_runs=1 if protocol['pilot'] else 5
    if len(paths)!=expected_runs:raise ValueError('Missing native ablation or seed')
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'};coarse=CompactMatcher(manifest['coarse_model']);records=[];replayed=0
    for path in paths:
        folder=path.parent;meta=json.loads(path.read_text());complete=json.loads((folder/'COMPLETE.json').read_text())
        for name,h in [('training.json',complete['training_sha256']),('selected.pt',complete['checkpoint_sha256']),('final.pt',complete['final_sha256']),('development_results.json.gz',complete['results_sha256'])]:
            if sha256(folder/name)!=h:raise ValueError('Completed native artifact changed')
        for p,h in protocol['source_sha256'].items():
            if meta['source_sha256'][p]!=h or sha256(folder/'source'/p)!=h:raise ValueError('Native source snapshot differs')
        if meta['completed_steps']!=protocol['steps'] or meta['calibration_or_test_pixels_loaded']!=0:raise ValueError('Training budget/scope differs')
        torch.manual_seed(meta['seed']);model=NativeRefiner(meta['variant']);model.initialize_encoder(coarse)
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True);selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)
        final=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()):raise ValueError('Seeded native initialization differs')
        model.load_state_dict(selected)
        n=sum(p.numel() for p in model.parameters())
        if n!=meta['trainable_parameters'] or n+28272!=meta['total_parameters'] or n+28272>=100000:raise ValueError('Parameter cap differs')
        if latest['step']!=protocol['steps'] or any(not torch.equal(v,latest['model'][k]) for k,v in final.items()):raise ValueError('Final optimizer checkpoint differs')
        if {int(s['step']) for s in latest['optimizer']['state'].values()}!={protocol['steps']}:raise ValueError('Optimizer step budget differs')
        change=float(sum((final[k]-v).square().sum() for k,v in initial.items()).sqrt())
        if change<=0 or not np.isclose(change,meta['final_weight_change_l2']):raise ValueError('Actual native weight change differs')
        history=meta['history'];best=max((h['selection']['correct_poses'],h['selection']['correct_at_two_px'],-h['step']) for h in history)
        if list(best)!=meta['selection_key'] or -best[2]!=meta['selected_step']:raise ValueError('Native checkpoint selection differs')
        steps=[0]+list(range(protocol['interval'],protocol['steps']+1,protocol['interval']))
        if steps[-1]!=protocol['steps']:steps.append(protocol['steps'])
        if [h['step'] for h in history]!=steps:raise ValueError('Missing native validation checkpoint')
        if meta['selected_step']==0 and any(not torch.equal(v,selected[k]) for k,v in initial.items()):raise ValueError('Selected initial checkpoint changed')
        rng=np.random.default_rng(meta['seed']);kinds=Counter();positives=0
        for _ in range(protocol['steps']):
            _,batch_kinds,n_positive=training_batch(manifest,tiles,data,rng,8);kinds.update(batch_kinds);positives+=n_positive
        if dict(kinds)!=meta['sampled_kinds'] or positives!=meta['positive_labels'] or meta['sampled_examples']!=protocol['steps']*8:
            raise ValueError('Exact balanced training sampling differs')
        if rng.bit_generator.state!=latest['numpy_rng']:raise ValueError('Recorded sampler RNG state differs')
        with gzip.open(folder/'development_results.json.gz','rt') as f:rows=json.load(f)
        if len(rows)!=len(expected) or {r['pair_id'] for r in rows}!=expected:raise ValueError('Native eligible cases missing or repeated')
        for row in rows:
            tile=tiles[row['tile_id']]
            if row['component']!=tile['component']:raise ValueError('Corrected group differs')
            replay_row(row,tile,data[row['tile_id']]);branches=row['all_branches']
            valid=[b for b in branches if b['estimated']]
            selected_branch=max(valid,key=lambda b:(b['inliers'],-b['median_inlier_residual'],-b['quarter_turn'])) if valid else branches[0]
            if row['quarter_turn']!=selected_branch['quarter_turn']:raise ValueError('Truth-free native branch choice differs')
            if row['no_match_count']+row['returned']!=row['supported_matches'] or len(row['point_reliability'])!=row['returned']:
                raise ValueError('Local rejection counts differ')
            replayed+=1
        if aggregate(rows)!=meta['evaluation']:raise ValueError('Native aggregate differs')
        bank=[r for r in rows if r['pair_id'] in manifest['selection_bank']]
        if aggregate(bank,True)!=next(h['selection'] for h in history if h['step']==meta['selected_step']):raise ValueError('Selected native bank differs')
        records.append(dict(variant=meta['variant'],seed=meta['seed'],selected_step=meta['selected_step'],selection_key=meta['selection_key'],evaluation=meta['evaluation']))
    order={v:i for i,v in enumerate(protocol['variants'])};primary=[r for r in records if r['seed']==20261101]
    winner=max(primary,key=lambda r:(tuple(r['selection_key']),-order[r['variant']]))['variant'];expected_pairs={(v,20261101) for v in protocol['variants']}
    if not protocol['pilot']:expected_pairs|={(winner,s) for s in (20261102,20261103)}
    if {(r['variant'],r['seed']) for r in records}!=expected_pairs or summary['winner']!=winner:raise ValueError('Native configuration/seed selection differs')
    for r in records:
        saved=next(s for s in summary['results'] if (s['variant'],s['seed'])==(r['variant'],r['seed']))
        if any(saved[k]!=r[k] for k in ('selected_step','selection_key','evaluation')):raise ValueError('Native cohort summary differs')
    result=dict(passed=True,scope='Corrected native development replay; no fresh-test or calibrated-acceptance claim.',trained_runs=len(records),optimizer_steps=protocol['steps']*len(records),
        supervision_cases=labels,development_rows=replayed,selected_variant=winner,runs=records,source_sha256=sha256(__file__),
        summary_sha256=sha256(root/'summary.json'),calibration_or_test_pixels_loaded=0)
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.output)
