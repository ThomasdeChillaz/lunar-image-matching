"""Independent saved-array arithmetic audit. No matcher/model/ray imports."""
from datetime import datetime,timezone
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def arrays(path):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}


def exact_label_indices(prediction_reference_xy,label_reference_xy):
    """Coordinate join; archive-anchor ordinals are NOT label-row identifiers."""
    p=np.asarray(prediction_reference_xy);labels=np.asarray(label_reference_xy)
    for value in(p,labels):
        if value.ndim!=2 or value.shape[1]!=2 or not np.isfinite(value).all():
            raise ValueError('Finite native reference Nx2 coordinates required')
        if len(np.unique(value,axis=0))!=len(value):raise ValueError('Duplicate reference coordinates are ambiguous')
    lookup={tuple(x):i for i,x in enumerate(labels)}
    missing=[tuple(x)for x in p if tuple(x)not in lookup]
    if missing:raise ValueError('Missing exact reference-coordinate labels: '+str(missing[:3]))
    return np.asarray([lookup[tuple(x)]for x in p],dtype=np.int64)


def counts(points,candidates,target,visible):
    assert candidates.shape==(len(points),8,2)and np.isfinite(candidates).all()
    assert ((candidates>=0)&(candidates<=511)).all()
    errors=np.full(len(points),np.inf);visible=np.asarray(visible,bool)&np.isfinite(target).all(1)
    errors[visible]=np.sqrt(np.sum((candidates[visible]-target[visible,None])**2,axis=-1)).min(1)
    # Independent closed-form inverse quarter turns for the published fixed bank.
    axis=49.5+8*np.arange(51);y,x=np.meshgrid(axis,axis,indexing='ij');x=x.ravel();y=y.ravel()
    bank=np.concatenate([np.column_stack([x,y]),np.column_stack([511-y,x]),
        np.column_stack([511-x,511-y]),np.column_stack([y,511-x])])
    nearest=np.full(len(points),np.inf)
    nearest[visible]=np.sqrt(np.sum((target[visible,None]-bank[None])**2,axis=-1)).min(1)
    covered=visible&(nearest<=16.)
    inside=visible&((target>=0)&(target<=511)).all(1)
    return dict(all_retained_reference_anchors=len(points),co_visible=int(visible.sum()),
        targets_inside_query_image=int(inside.sum()),geometric_bank_coverage_at16=int(covered.sum()),
        recalled={str(t):int(np.count_nonzero(errors<=t))for t in(1,2,4,8,12,16)},
        in_grid_recalled_at16=int(np.count_nonzero((errors<=16)&covered)),in_grid_denominator=int(covered.sum())),errors


def replay(repo_root,output_directory):
    REPO=Path(repo_root).resolve();OUT=Path(output_directory).resolve()
    if OUT==REPO or REPO in OUT.parents:raise ValueError('A fresh output directory outside the repository is required')
    if OUT.exists():raise FileExistsError('Preserve previous correction artifacts; choose a new outside directory')
    OUT.mkdir(parents=True,exist_ok=False)
    ROOT=REPO/'runs/solar-pose-free-qualification-v85'
    PROTOCOL=REPO/'reports/solar_pose_free_qualification_job_v85.json'
    tick=time.perf_counter();p=read(PROTOCOL);summary=read(ROOT/'summary.json');complete=read(ROOT/'COMPLETE.json')
    assert complete['complete']is True and complete['failure']is None and complete['child_returncode']==0
    assert complete['protocol_sha256']==summary['protocol_sha256']==sha(PROTOCOL)
    assert complete['summary_sha256']==sha(ROOT/'summary.json')and complete['calendar_seconds']<=600
    assert summary['source_publication_commit']==complete['publication_commit']=='4f0ed7bf6b1d47b30dd68e0389810359354da506'
    assert not(ROOT/'FAILED.json').exists()
    source_hashes={name:sha(REPO/name)for name in p['source_sha256']}
    assert source_hashes==p['source_sha256']
    rows={entry['name']:read(REPO/entry['row'])for entry in summary['rows']}
    for entry in summary['rows']:assert sha(REPO/entry['row'])==entry['row_sha256']
    for row in rows.values():
        assert row['diagnostic_only']is True and row['accepted']is False and row['pose']is None
        assert row['principal_active_native_deferral_contract_met']is False
        if row.get('arrays'):assert sha(REPO/row['arrays'])==row['arrays_sha256']
    a=arrays(REPO/rows['step0-common-features']['arrays'])
    native=a['frozen_reference']@a['query_descriptors'].T
    predicted=a['predicted_reference']@a['query_descriptors'].T
    difference=float(np.max(np.abs(native-predicted)));top=np.sort(native,axis=1)[:,-2:]
    constant=dict(max_score_difference=difference,near_top_ties_at_tolerance=int(np.count_nonzero(top[:,1]-top[:,0]<=2e-6)),argmax_differences=int(np.count_nonzero(native.argmax(1)!=predicted.argmax(1))))
    original=rows['step0-common-features']['controls']
    assert difference<=2e-6 and np.isclose(difference,original['max_score_difference'],rtol=0,atol=1e-12)
    assert all(constant[k]==original[k]for k in('near_top_ties_at_tolerance','argmax_differences'))
    generated=rows['generated-translation-native'];a=arrays(REPO/generated['arrays'])
    affine=np.array([[1.,0.,8.],[0.,1.,-8.]])
    target=np.column_stack([a['reference_xy'],np.ones(len(a['reference_xy']))])@affine.T
    np.testing.assert_array_equal(target,a['target_query_xy'])
    generated_counts,generated_errors=counts(a['reference_xy'],a['query_candidates'],target,np.ones(len(target),bool))
    assert all(generated_counts[k]==generated['recall'][k]for k in generated_counts)
    n=len(target);correct=generated_counts['recalled']['16'];gate=n>0 and correct/n>=.8
    assert gate is True and generated['passed']is True and generated['gate']['passed']is True
    assert generated['gate']['denominator']==n and generated['gate']['correct']==correct
    manifest=read(REPO/p['manifest']);case=next(c for c in manifest['cases']if c['case_id']=='u012-l00-00-positive')
    assert sha(REPO/case['scoring_labels'])==case['input_sha256'][case['scoring_labels']]
    # Access only native correspondence labels, AFTER original predictions.
    with np.load(REPO/case['scoring_labels'],allow_pickle=False)as z:
        label_p=z['reference_xy'].copy();label_q=z['query_xy'].copy();label_visible=z['visible'].copy()
    lunar={};archive={};association={};mismatches={}
    with np.load(REPO/case['reference_geometry'],allow_pickle=False)as z:
        world=z['world_variants'][:,0].copy();k=z['intrinsics'].copy();pose=z['cam2world'].copy()
        physical_shapes=dict(world_variants=list(z['world_variants'].shape),normals=list(z['normals'].shape),solar_states=list(z['solar_states'].shape))
    axis=49.5+16*np.rint(np.linspace(0,25,16)).astype(int);y,x=np.meshgrid(axis,axis,indexing='ij');anchor_grid=np.column_stack([x.ravel(),y.ravel()])
    camera=(world-pose[:3,3])@pose[:3,:3];h=camera@k.T;projection=h[:,:2]/h[:,2,None]
    projection_error=np.linalg.norm(projection-anchor_grid,axis=1)
    assert projection_error.max()<=.02 and len(world)==physical_shapes['normals'][0]==physical_shapes['solar_states'][0]==256
    archive['reference_archive_projection_error_px']=projection_error
    for name in('u012-native','u012-sh','u012-pooled'):
        row=rows[name];a=arrays(REPO/row['arrays']);ids=exact_label_indices(a['reference_xy'],label_p)
        result,errors=counts(a['reference_xy'],a['query_candidates'],label_q[ids],label_visible[ids])
        mismatches[name]=[key for key in result if result[key]!=row['recall'][key]]
        wrong,_=counts(a['reference_xy'],a['query_candidates'],label_q[np.arange(len(ids))],label_visible[np.arange(len(ids))])
        assert all(wrong[key]==row['recall'][key]for key in wrong)
        association[name]=dict(reference_inventory_count=len(label_p),prediction_anchor_count=len(ids),
            nonidentity_label_rows=int((ids!=np.arange(len(ids))).sum()),first20_correct_label_indices=ids[:20].tolist(),
            original_scoring_reproduced_by_wrong_first256_label_rows=True)
        extraction=row['retrieval']['native_feature_extraction'];c=extraction['encoder_counts'];hooks=row['actual_forward_hooks']
        assert hooks['coarse']==c['reference_frozen_coarse_forwards']+c['query_frozen_coarse_forwards']and hooks['fine']==c['reference_fine_forwards']
        assert c['query_frozen_coarse_forwards']==4 and row['actual_pretrained_forward']is True and extraction['software_stub_only']is False
        lunar[name]=dict(**result,recall_at16= result['recalled']['16']/result['all_retained_reference_anchors'],
            median_nearest_candidate_error_px=float(np.median(errors)),global_score=row['retrieval']['global_score'],
            selected_sun_parameters=row['retrieval']['shared_sun_parameters'],scored_suns=row['retrieval']['diagnostics']['unique_scored_suns'],
            response_decodes=c['response_decodes'],actual_forward_hooks=hooks,calendar_seconds=row['calendar_seconds'])
        archive[name+'_nearest_candidate_error_px']=errors
        archive[name+'_correct_label_indices']=ids
        archive[name+'_reference_xy']=a['reference_xy']
        archive[name+'_scoring_query_xy']=label_q[ids]
        archive[name+'_scoring_visible']=label_visible[ids]
    false=rows['u012-to-u020-native'];assert false['negative_case']is True and false['recall']is None
    assert rows['blank']['reason']=='blank_or_uninformative_image'and rows['blank']['actual_forward_hooks']==dict(coarse=0,fine=0)
    assert rows['invalid-query-shape']['passed']is True and rows['invalid-query-shape']['actual_forward_hooks']==dict(coarse=0,fine=0)
    # Source-order/API audit, not execution or a matcher replay.
    features=(REPO/'solar_pose_free_features_v85.py').read_text();math=(REPO/'solar_pose_free_retrieval_v85.py').read_text();driver=(REPO/'qualify_solar_pose_free_v85.py').read_text()
    public=next(n for n in ast.parse(features).body if isinstance(n,ast.FunctionDef)and n.name=='extract_native_features')
    arguments=[x.arg for x in public.args.args+public.args.kwonlyargs]
    assert not set(arguments)&{'query_pose','query_depth','query_sun','query_intrinsics','true_query_pose'}
    assert 'secondary_known_query_sun_world'not in driver and 'query_cam2world'not in driver
    assert driver.index("result=retrieve_from_extracted(extracted,arm=arm")<driver.index("np.load(pos['scoring_labels']")
    assert 'candidates = _suppress(winning_scores' in math and 'response_predictor(features, condition, sun_vectors[si].copy()'in math
    archive['generated_nearest_candidate_error_px']=generated_errors
    np.savez_compressed(OUT/'corrected_errors_v86.npz',**archive)
    result=dict(correction_audit_passed=True,original_lunar_metrics_verified=False,software_controls_replayed=True,
        raw_candidates_unchanged=True,observed_utc=datetime.now(timezone.utc).isoformat(),source_publication_commit=complete['publication_commit'],
        protocol_sha256=sha(PROTOCOL),summary_sha256=sha(ROOT/'summary.json'),complete_sha256=sha(ROOT/'COMPLETE.json'),
        rows_replayed=len(rows),constant_head=constant,generated=dict(counts=generated_counts,gate_passed=gate,fraction=correct/n),
        corrected_lunar=lunar,original_lunar={name:rows[name]['recall']for name in lunar},mismatched_metric_fields=mismatches,
        label_association=association,physical_input_association=dict(affected_by_scoring_bug=False,reference_anchor_grid_count=256,
            source_label_grid_count=len(label_p),archive_shapes=physical_shapes,max_reference_reprojection_error_px=float(projection_error.max()),
            evidence='Reference archive uses the SAME16x16 subsampled grid as feature adapter; independently projected existing world anchors agree.676-row scoring label inventory is separate and was only accessed after prediction.'),
        false_control=dict(proposals_retained=True,accepted=False,pose=None,global_score=false['retrieval']['global_score'],case_binding=false['case_binding']),
        source_audit=dict(no_query_truth_API=True,query_labels_open_after_prediction=True,global_Sun_before_NMS=True,
            native_dense_vs_canonical_domain_mismatch_explicit=True,reference_view_response_is_not_target_view_synthesis=True),
        interpretation='Software controls passed; real single-case retrieval is poor and learned SH/pooled do not improve native recall here. No camera pose, rejection or lunar accuracy superiority established.',
        audit_scope='Independent saved-array/source-label arithmetic only. No model imports/forwards, matching, fresh images, new rays, fitting, new eligibility or scientific job.',
        audit_calendar_seconds=time.perf_counter()-tick,correction_source_sha256=sha(__file__),error_archive_sha256=sha(OUT/'corrected_errors_v86.npz'))
    with(OUT/'CORRECTION_V86.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo-root',default=str(Path.cwd()));parser.add_argument('--output-directory',required=True);args=parser.parse_args()
    result=replay(args.repo_root,args.output_directory);print(json.dumps({k:result[k]for k in('correction_audit_passed','original_lunar_metrics_verified','rows_replayed','constant_head','generated','corrected_lunar','audit_calendar_seconds')},indent=2))
