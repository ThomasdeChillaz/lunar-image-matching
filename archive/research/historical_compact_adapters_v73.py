"""Explicit historical CNN controls; no new-response-model substitution.

Loading a profile requires source-pinned checkpoint lineage and a new native512
adapter control. This module imports no torch and runs no model at import time.
All final poses use the evaluator's common reference-geometry PnP estimator.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

METHODS=('historical_original_native512','historical_frozen_surface','historical_north_context')
ORIGINAL_SHA='ab555480bf44c878f509ffacbe2778f35bbd48d1f1b4b04902fb6e858d9ca416'
EXPANDED_SHA='c92f6917659e075688eb55387d4dbd5c8136b717772a962f1267933e338b2c65'
CONTEXT_SHA='ec7b9f77f695bcbc6c4377aa3f1b49687bf2f60bc50b04389ea7e5f09d1c5f09'
COMMON_SOURCES={'historical_compact_adapters_v73.py','solar_evaluation_v70.py','native_baseline_pose_v7.py',
 'localization_robust_pose_v4.py','localization_pose_refit_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py',
 'contrastive_terrain.py','dataset.py','matchers.py','geometry.py'}
SURFACE_SOURCES={'compact_lunar.py','geometry_warp_native_v11.py','coarse_pipeline_v5.py',
 'localization_diagnostics_v3.py','rotation_bank_pose_v4.py'}
SURFACE_SOURCES|={'localization_refiner_v3.py','lunargeo_dataset.py','prepare_localization_training_v3.py','remote_archive.py','train_balanced_localization_v3.py'}
CONTEXT_SOURCES={'search_window_learning_v18.py','context_supervision_v15.py','global_context_v15.py'}


class BaselineUnavailable(RuntimeError):
    """An unavailable/unfinished comparison is never replaced with another model."""


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def value_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def required_sources(method):
    if method not in METHODS:raise ValueError('Unknown historical method; no substitution')
    result=set(COMMON_SOURCES)
    if method=='historical_original_native512':result.add('raw_correspondences_v2.py')
    else:result|=SURFACE_SOURCES
    if method=='historical_north_context':result|=CONTEXT_SOURCES
    return result


def validate_images(inp):
    for image in(inp.reference,inp.query):
        if not isinstance(image,np.ndarray)or image.shape!=(512,512)or image.dtype!=np.uint8:
            raise ValueError('Actual native512 uint8 grayscale image content required')
    depth=np.asarray(inp.geometry.depth)
    if depth.shape!=(512,512)or not np.isfinite(depth).all()or not(depth>0).all():raise ValueError('Common valid native reference geometry required')


def validate_points(a,b):
    a=np.asarray(a,float).reshape(-1,2);b=np.asarray(b,float).reshape(-1,2)
    if a.shape!=b.shape or not np.isfinite(a).all()or not np.isfinite(b).all():raise ValueError('Finite paired native correspondences required')
    if len(a)and(not((a>=0)&(a<=511)).all()or not((b>=0)&(b<=511)).all()):raise ValueError('Historical adapter coordinates left native image content')
    return a,b


def validate_control_outputs(output,profile):
    """Independent arithmetic replay of the actual saved procedural predictions."""
    if (output['method']!=profile['method']or output['native_shape']!=[512,512]or output['cpu_threads']!=2 or
        output['configuration_sha256']!=value_digest(profile['configuration'])or
        output['artifact_sha256']!=profile['artifact_sha256']or output['source_sha256']!=profile['source_sha256']or
        output['actual_pretrained_forward']is not True or output['query_truth_used']is not False):
        raise ValueError('Native adapter control belongs to another model/source/configuration')
    known=output['known_transform'];a,b=validate_points(known['reference_xy'],known['query_xy'])
    matrix=np.asarray(known['reference_to_query_affine'],float)
    expected_affine=np.array([[1.,0,8.],[0,1.,-4.]])
    if matrix.shape!=(2,3)or not np.array_equal(matrix,expected_affine)or len(a)<6:
        raise ValueError('Predeclared analytic native affine control required; matcher-fitted labels forbidden')
    expected=a@matrix[:,:2].T+matrix[:,2]
    error=np.linalg.norm(b-expected,axis=1)
    if not np.isfinite(error).all()or np.median(error)>1.:raise ValueError('Native known-transform control failed')
    for kind in('blank','invalid_support'):
        p,q=validate_points(output[kind]['reference_xy'],output[kind]['query_xy'])
        if len(p)or len(q)or output[kind]['pose_returned']:raise ValueError('Invalid/blank native control must return no correspondences or pose')
    return dict(passed=True,known_matches=len(a),median_native_error_px=float(np.median(error)))


def _check_artifact(path,digest):
    if sha(path)!=digest:raise ValueError('Historical artifact changed:'+str(path))


def _checkpoint_lineage(profile):
    c=profile['configuration'];method=profile['method'];artifacts=profile['artifact_sha256']
    directory=Path(c['coarse_directory']if method=='historical_north_context'else c['model_directory'])
    for name in('selected.pt','training.json','COMPLETE.json'):
        path=str(directory/name)
        if path not in artifacts:raise ValueError('All original frozen checkpoint artifacts must be pinned')
    expected=ORIGINAL_SHA if method=='historical_original_native512'else EXPANDED_SHA
    if sha(directory/'selected.pt')!=expected:raise ValueError('This is not the named immutable historical checkpoint; no model substitution')
    seal=read(directory/'COMPLETE.json');meta=read(directory/'training.json')
    checkpoint_key='selected_sha256'if method=='historical_original_native512'else'checkpoint_sha256'
    if seal[checkpoint_key]!=sha(directory/'selected.pt')or seal['training_sha256']!=sha(directory/'training.json'):
        raise ValueError('Historical model completion seal differs')
    source='contrastive_terrain.py'if method=='historical_original_native512'else'compact_lunar.py'
    if meta['source_sha256'][source]!=sha(source):raise ValueError('Frozen model implementation differs from its training record')
    if method!='historical_original_native512'and meta['variant']!='expanded':raise ValueError('The historical surface/context controls used the expanded backbone')
    if method=='historical_north_context':
        selected=Path(c['selected_checkpoint']);seed_root=selected.parent
        for name in('selected.pt','summary.json','COMPLETE.json','selected_controls.json'):
            if str(seed_root/name)not in artifacts:raise ValueError('Selected v61 checkpoint and actual historical evidence must be pinned')
        if sha(selected)!=CONTEXT_SHA:raise ValueError('Selected v61 immutable checkpoint identity differs')
        complete=read(seed_root/'COMPLETE.json');summary=read(seed_root/'summary.json');controls=read(seed_root/'selected_controls.json')
        for name in('selected.pt','summary.json','selected_controls.json'):
            if complete['sha256'][name]!=sha(seed_root/name):raise ValueError('Completed v61 selected evidence changed')
        if (summary['seed']!=20261101 or summary['completed_steps']!=2000 or summary['selected_step']!=1500 or
            summary['selected_checkpoint_sha256']!=sha(selected)or not summary['actual_selected_control_passed']or
            not controls['passed']or summary['calibrated_acceptance_evaluated']):
            raise ValueError('Expected completed, selected and historically controlled v61 model required')
        if c['variant']!='cross_adaptation':raise ValueError('Do not replace the selected attention control with pointwise or response weights')
        return dict(coarse_checkpoint_sha256=sha(directory/'selected.pt'),selected_checkpoint_sha256=sha(selected),
                    historical_correct_raw_poses=summary['selection']['correct_raw_poses'],historical_positive_cases=summary['selection']['positive_cases'])
    return dict(selected_checkpoint_sha256=sha(directory/'selected.pt'))


def validate_historical_profile(profile):
    if profile.get('prepared_only',False)or not profile.get('available',True):raise BaselineUnavailable(profile.get('unavailable_reason','Historical profile awaits native adapter controls and public source freeze'))
    method=profile['method']
    if profile['backend']!='historical_compact'or method not in METHODS or profile['cpu_threads']!=2 or profile['native_size']!=512 or profile['calibration_or_final_access']:
        raise ValueError('Explicit development-only native512 historical profile required')
    c=profile['configuration']
    if method=='historical_original_native512':
        if set(c)!={'model_directory','checkpoint','input_profile'}or c['checkpoint']!='selected'or c['input_profile']!='raw_native512':raise ValueError('Explicit unchanged original-native profile required')
    else:
        expected={'coarse_directory','selected_checkpoint','variant','input_profile','surface_iterations'}if method=='historical_north_context'else{'model_directory','input_profile','surface_iterations'}
        if set(c)!=expected or c['input_profile']!='coarse128_native512'or c['surface_iterations']!=2:raise ValueError('Published two-iteration compact surface profile required')
    commit=profile['publication_commit']
    if len(commit)!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact public adapter source revision required')
    if not required_sources(method)<=set(profile['source_sha256']):raise ValueError('Historical source closure is incomplete')
    for name,h in profile['source_sha256'].items():
        _check_artifact(name,h)
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Historical adapter source not published:'+name)
    for name,h in profile['artifact_sha256'].items():_check_artifact(name,h)
    for name,h in profile.get('historical_algorithm_source_sha256',{}).items():
        if sha(name)!=h:raise ValueError('Historical inference algorithm changed; require a separately named requalification:'+name)
    lineage=_checkpoint_lineage(profile)
    evidence=profile['native_control']
    for key in('protocol','outputs','receipt'):_check_artifact(evidence[key],evidence[key+'_sha256'])
    protocol=read(evidence['protocol']);output=read(evidence['outputs']);receipt=read(evidence['receipt'])
    if subprocess.check_output(['git','show',commit+':'+evidence['protocol']])!=Path(evidence['protocol']).read_bytes():raise ValueError('Native-control protocol was not published')
    if (protocol['method']!=method or protocol['configuration_sha256']!=value_digest(profile['configuration'])or
        protocol['artifact_sha256']!=profile['artifact_sha256']or protocol['source_sha256']!=profile['source_sha256']):
        raise ValueError('Native-control protocol model/source profile differs')
    measured=validate_control_outputs(output,profile)
    if (not receipt['complete']or not receipt['passed']or receipt['protocol_sha256']!=evidence['protocol_sha256']or
        receipt['outputs_sha256']!=evidence['outputs_sha256']or receipt['method']!=method):
        raise ValueError('A completed measured native control is required before lunar inference')
    return dict(method=method,lineage=lineage,native_control=measured,native_control_receipt_sha256=evidence['receipt_sha256'],
        source_sha256=profile['source_sha256'],artifact_sha256=profile['artifact_sha256'],
        original_checkpoint_trained_at_128px=method=='historical_original_native512',
        input_profile='Full native512 content; original fully convolutional raw matching has no implicit resize.'if method=='historical_original_native512'else'Published coarse128 proposal search and native512 refinement/surface normalization.',
        acceptance='Raw proposals only; old 2D rejection rules are not transplanted into 3D acceptance.',
        illumination='Same reference/Sun metadata available in InferenceInput; these historical algorithms do not use Sun inference.',
        cache='Fresh per-call native feature cache; model loading is reported separately, no cross-case frozen-feature timing advantage.')


class _HistoricalRuntime:
    def __init__(self,profile):
        import torch
        torch.set_num_threads(2)
        self.profile=profile;c=profile['configuration'];method=profile['method']
        if method=='historical_original_native512':
            from contrastive_terrain import ContrastiveTerrainMatcher
            self.matcher=ContrastiveTerrainMatcher(c['model_directory'],checkpoint='selected',device='cpu')
        else:
            from compact_lunar import CompactMatcher
            coarse=CompactMatcher(c['coarse_directory']if method=='historical_north_context'else c['model_directory'])
            coarse.model.eval().requires_grad_(False)
            if method=='historical_north_context':
                from search_window_learning_v18 import SearchWindowModel,SearchWindowMatcher
                model=SearchWindowModel(coarse,variant='cross_context',train_coarse=True)
                model.load_state_dict(torch.load(c['selected_checkpoint'],map_location='cpu',weights_only=True),strict=True)
                model.eval().requires_grad_(False);self.matcher=SearchWindowMatcher(coarse,model)
            else:self.matcher=coarse
    def infer_points(self,inp):
        if self.profile['method']=='historical_original_native512':
            from raw_correspondences_v2 import raw_points
            valid=np.ones((512,512),bool)
            p,q=raw_points(self.matcher,'original',inp.reference,inp.query,valid,valid)
            return p,q,dict(branch_search='none; original native raw mutual CNN correspondences',surface_iterations=0)
        from geometry_warp_native_v11 import infer
        chosen,branches=infer(self.matcher,inp.reference,inp.query,inp.geometry.depth,inp.geometry.intrinsics,
             inp.geometry.cam2world,inp.query_intrinsics,features={},feature_ids=('reference','query'))
        branch=chosen if chosen is not None else(branches[0]if branches else None)
        if branch is None:return np.empty((0,2)),np.empty((0,2)),dict(reason='no_historical_branch')
        return branch['p'],branch['q'],dict(internal_selected_pose_returned=chosen is not None,
            quarter_turn=branch.get('quarter_turn'),geometry_history=branch.get('geometry_history',[]),
            internal_pose_used_only_for_refinement_and_branch_selection=True)


class HistoricalCompactAdapter:
    def __init__(self,profile,runtime_factory=None,estimator=None):
        self.profile=profile;self.provenance=validate_historical_profile(profile)
        self.runtime=(runtime_factory or _HistoricalRuntime)(profile)
        if estimator is None:
            from solar_evaluation_v70 import common_pose
            estimator=common_pose
        self.estimator=estimator
    def infer(self,inp):
        validate_images(inp)
        p,q,details=self.runtime.infer_points(inp);p,q=validate_points(p,q)
        pose,info=self.estimator(p,q,inp)
        return dict(reference_xy=p,query_xy=q,pose=pose,reason=info.get('reason'),common_estimator=info,
                    historical_control=details,sun_metadata_used=False,query_truth_used=False)


def make_historical_compact(profile):return HistoricalCompactAdapter(profile)


def historical_profile_inputs(profile):
    """All non-case evidence the central comparison protocol must pin."""
    result=dict(profile['artifact_sha256']);e=profile['native_control']
    for key in('protocol','outputs','receipt'):result[e[key]]=e[key+'_sha256']
    return result
