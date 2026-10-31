"""Additional aggregate/interval replay; no inference, fitting or metric changes."""
import argparse
import json
from pathlib import Path

import numpy as np

from compact_confidence import accepts
from dataset import sha256
from lunargeo_dataset import write_new


def read(path):
    return json.loads(Path(path).read_text())


def interval(a,b):
    a={r['pair_id']:r for r in a if not r['negative']}
    b={r['pair_id']:r for r in b if not r['negative']}
    assert a.keys()==b.keys()
    groups=sorted({r['component'] for r in a.values()})
    delta=[];sizes=[]
    for group in groups:
        keys=[k for k,r in a.items() if r['component']==group]
        delta.append(sum(int(a[k]['accepted'] and a[k]['correct'])-
                         int(b[k]['accepted'] and b[k]['correct']) for k in keys))
        sizes.append(len(keys))
    draws=np.random.default_rng(20261030).integers(len(groups),size=(2000,len(groups)))
    samples=[100*sum(delta[i] for i in ids)/sum(sizes[i] for i in ids) for ids in draws]
    return 100*sum(delta)/sum(sizes),np.percentile(samples,[2.5,97.5])


def check_interval(a,b,reported):
    point,bounds=interval(a,b)
    assert np.isclose(point,reported['gain_pp'],atol=1e-10)
    assert np.allclose(bounds,reported['exploratory_component_bootstrap95_pp'],atol=1e-10)


def audit(output,index_path='reports/compact_run_index_v2.json',report_path='reports/compact_results_v2.json'):
    from compact_research import controlled_inputs
    from geometry import evaluation_points,transform
    from replay_compact_v2 import verify_aggregate
    protocol=read('reports/compact_freeze_v2.json')
    report=read(report_path)
    index=read(index_path)
    checked=[]
    for method in ('compact','original','sift'):
        root=Path(f'runs/compact-v2-{method}-3d-test')
        rows=read(root/'results.json')
        summary=read(root/'summary.json')
        expected=dict(cases=len(rows),poses_estimated=sum(r['pose_estimated'] for r in rows),
            returned_matches=sum(r['returned_matches'] for r in rows),
            scorable_matches=sum(r['reference_scorable_matches'] for r in rows),
            correct_correspondences=sum(r['correct_correspondences'] for r in rows),
            poses_within_secondary_threshold=sum(r['pose_within_secondary_threshold'] for r in rows))
        assert all(r['pose_estimated']==(r['estimated_cam2world'] is not None) for r in rows)
        assert all(not r['pose_within_secondary_threshold'] or r['pose_estimated'] for r in rows)
        for key,value in expected.items():
            assert summary[key]==value and report['three_dimensional'][method][key]==value
        checked.append(str(root))
        root=Path(f'runs/compact-v2-{method}-sensor')
        rows=read(root/'results.json');summary=read(root/'summary.json')
        within=lambda r:r['nominal_error_px'] is not None and r['nominal_error_px']<=3
        expected=dict(cases=len(rows),raw_nominal_within3px=sum(within(r) for r in rows),
            accepted_nominal_within3px=sum(r['accepted'] and within(r) for r in rows),
            accepted_nominal_outside3px=sum(r['accepted'] and not within(r) for r in rows))
        for key,value in expected.items():
            assert summary[key]==value and report['sensor_nominal'][method][key]==value
        if method=='original':
            for suffix,predicate in [('within3px',within),('outside3px',lambda r:not within(r))]:
                value=sum(r['original_frozen_accepted'] and predicate(r) for r in rows)
                key='original_gate_accepted_nominal_'+suffix
                assert summary[key]==value and report['sensor_nominal'][method][key]==value
        checked.append(str(root))

    def accepted(item):
        policy=read(protocol['policy_index'][item['policy_key']])
        return [dict(r,accepted=bool(r['original_frozen_accepted']) if item.get('original_gate')
                     else accepts(policy,{'features':r['features']})) for r in read(Path(item['run'])/'results.json')]
    full={name:accepted(item) for name,item in index['aggregate_full'].items()}
    common_ids={c['pair_id'] for c in protocol['common_cases']}
    common={name:[r for r in accepted(item) if r['pair_id'] in common_ids]
            for name,item in index['aggregate_common'].items()}
    check_interval(full['compact-20261030'],full['original-frozen'],report['paired_gain'])
    for name,other in common.items():
        if name!='compact':check_interval(common['compact'],other,report['paired_common_comparisons'][name])

    # Reconstruct compact control correctness from its saved matrices, rather
    # than trusting the report's derived same-host accuracy flags.
    cases={c['pair_id']:c for c in protocol['common_cases']}
    manifest=read(protocol['manifest_path']);tiles={t['tile_id']:t for t in manifest['tiles']}
    policy=read(protocol['policy_index']['runs/compact-v2-expanded-20261030'])
    cache={};controls={};timings={}
    for path in Path('runs/compact-v2-modern-test-executions').rglob('timing_control.json'):
        completion=read(path.with_name('COMPLETE.json'))
        assert sha256(path)==completion['timing_control_sha256']
        baseline=read(path.with_name('predictions.json'));lookup={r['pair_id']:r for r in read(path)}
        for row in baseline:
            r=lookup[row['pair_id']];case=cases[row['pair_id']]
            assert r['checkpoint_sha256']==protocol['models']['runs/compact-v2-expanded-20261030']
            a,b,ma,mb,truth=controlled_inputs(case,tiles,cache);error=None
            if truth is not None and r['transform'] is not None:
                points=evaluation_points(a.shape,truth);target=transform(points,truth)
                ia=np.rint(points).astype(int);ib=np.rint(target).astype(int).clip(0,127)
                valid=ma[ia[:,1],ia[:,0]]&mb[ib[:,1],ib[:,0]]
                if valid.any():
                    matrix=np.array(r['transform'])
                    predicted=points[valid]@matrix[:,:2].T+matrix[:,2]
                    error=float(np.median(np.linalg.norm(predicted-target[valid],axis=1)))
            controls.setdefault(row['method'],[]).append(dict(case,correct=error is not None and error<=3,
                accepted=accepts(policy,{'features':r['features']})))
            timings.setdefault(row['method'],[]).append([row['bidirectional_ms'],r['same_host_bidirectional_ms']])
    for method,rows in controls.items():
        assert len({r['pair_id'] for r in rows})==len(rows)
        expected=report['same_host_cpu'][method]
        verify_aggregate(rows,[r['accepted'] for r in rows],expected['compact_same_host_quality'])
        t=np.array(timings[method])
        assert np.allclose([np.median(t[:,0]),np.median(t[:,1]),np.median(t[:,0]/t[:,1])],
            [expected['median_baseline_ms'],expected['median_compact_ms'],expected['median_paired_speedup']])
    comparison=report['competitive_comparison']
    interval_checks=len(common)
    if comparison:
        strongest=comparison['strongest_completed_at_comparable_error']
        check_interval(common['compact'],common[strongest],comparison['uncertainty'])
        interval_checks+=1
        if comparison['same_host_accuracy_uncertainty'] is not None:
            check_interval(controls[strongest],common[strongest],comparison['same_host_accuracy_uncertainty'])
            interval_checks+=1
    assert report['verified_real_navigation_accuracy'] is None and not report['global_sota_established']
    result=dict(passed=True,report_sha256=sha256(report_path),
        summary_tables_checked=checked,paired_intervals_checked=interval_checks,
        same_host_control_cases=sum(len(r) for r in controls.values()),
        independent_control_geometry_and_cpu_medians=True,new_inference_or_fitting=False)
    write_new(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--index',default='reports/compact_run_index_v2.json')
    parser.add_argument('--report',default='reports/compact_results_v2.json')
    args=parser.parse_args()
    print(json.dumps(audit(args.output,args.index,args.report),indent=2))
