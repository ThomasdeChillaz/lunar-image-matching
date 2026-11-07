"""Strict native512 MatchAnything/ELoFTR and MOONSt3R comparison adapters.

Reuses the reviewed safe-weight implementations; never downloads, switches
methods, crops the principal field of view, or fits an image-plane transform.
An actual pretrained known-transform/invalid-input receipt is required BEFORE
constructing the network. Stub tests cannot create scientific gate evidence.
All returned points go to the caller's common reference-geometry pose solver.
"""
import hashlib,importlib,json,math
from pathlib import Path
import numpy as np

METHODS={
    'matchanything':dict(display='MatchAnything-EfficientLoFTR',module='native_matchanything_v17',factory='NativeMatchAnything',lock='matchanything_model_lock.json',control='control_native_matchanything_v17.py',profiles=('native512','training_size832')),
    'moonst3r':dict(display='MOONSt3R',module='native_moonst3r_v19',factory='NativeMOONSt3R',lock='moonst3r_model_lock_v19.json',control='control_native_moonst3r_v19.py',profiles=('common_native512_full_field_of_view',)),
}
EVIDENCE_NAMES=('protocol','summary','controls','provenance')
COMMON_CONTROL_SOURCES={'native_baseline_pose_v7.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py','localization_diagnostics_v3.py','synthetic.py','loftr_matcher.py','eloftr_matcher.py'}
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text())


def specification(profile):
    if profile.get('backend')!='native_pretrained'or profile.get('method')not in METHODS:raise ValueError('Explicit native_pretrained MatchAnything or MOONSt3R method required; no fallback')
    spec=METHODS[profile['method']]
    if profile.get('author_profile')not in spec['profiles']:raise ValueError('Unsupported author/common-input profile; no silent crop, resize or method substitution')
    if profile.get('cpu_threads',2)!=2:raise ValueError('Principal comparison uses exactly two CPU threads')
    return spec


def _evidence_paths(profile):
    evidence=profile.get('pretrained_evidence',{})
    required={key for name in EVIDENCE_NAMES for key in(name,name+'_sha256')}
    if set(evidence)!=required:raise ValueError('Four exact hash-pinned ACTUAL pretrained protocol/summary/controls/provenance files required')
    paths={}
    for name in EVIDENCE_NAMES:
        path=Path(evidence[name])
        if sha(path)!=evidence[name+'_sha256']:raise ValueError('Actual pretrained evidence changed:'+name)
        paths[name]=path
    if len({p.resolve().parent for p in paths.values()})!=1:raise ValueError('Pretrained evidence must belong to one preserved actual control run')
    for failure in('FAILED.json','FAILURE.json','TIMEOUT.json'):
        if(paths['summary'].parent/failure).exists():raise ValueError('Failed/incomplete pretrained attempt cannot admit scientific inference')
    return paths


def _finite_number(value):return isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)


def replay_native_control(controls):
    """Independent arithmetic on saved raw native points and camera-to-world pose.

    This does NOT execute a pretrained network and cannot replace the actual
    forward provenance. It rejects a successful-looking aggregate with missing,
    resized, inconsistent, or inaccurate underlying known-transform arrays.
    """
    if controls.get('passed')is not True or controls.get('blank_empty')is not True or controls.get('invalid_mask_empty')is not True:raise ValueError('Known-transform, blank and invalid-mask ACTUAL controls must all pass')
    if controls.get('known_translation_native')!=[8.,-4.]or controls.get('image_seed')!=20261101:raise ValueError('Pinned native512 control seed/known transform differs')
    p,q=[np.asarray(controls.get(name,[]),float)for name in('reference_points_native','query_points_native')]
    if p.ndim!=2 or p.shape[1:]!=(2,)or q.shape!=p.shape or len(p)<6 or not np.isfinite(p).all()or not np.isfinite(q).all()or not((p>=0)&(p<=511)&(q>=0)&(q<=511)).all():raise ValueError('Complete finite native512 known-transform correspondences required')
    pose=np.asarray(controls.get('estimated_cam2world'),float)
    if pose.shape!=(4,4)or not np.isfinite(pose).all()or not np.allclose(pose[3],[0,0,0,1],rtol=0,atol=1e-8)or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),rtol=0,atol=1e-6)or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-6):raise ValueError('Rigid native known-transform camera-to-world pose required')
    median=float(np.median(np.linalg.norm(q-p-[8.,-4.],axis=1)))
    translation=float(np.linalg.norm(pose[:3,3]-[-.2,.1,0.]))
    rotation=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1.,1.))))
    values=dict(matches=len(p),median_native_error_px=median,translation_error_m=translation,rotation_error_deg=rotation)
    for name,value in values.items():
        saved=controls.get(name)
        if not _finite_number(saved)or not np.isclose(value,saved,rtol=0,atol=1e-8):raise ValueError('Saved native control arithmetic differs:'+name)
    if median>1. or translation>.05 or rotation>.5:raise ValueError('Actual pretrained native control fails unchanged accuracy criteria')
    if type(controls.get('inliers'))is not int or not 6<=controls['inliers']<=len(p):raise ValueError('Actual native pose control needs at least six inliers')
    if not _finite_number(controls.get('warm_match_seconds'))or controls['warm_match_seconds']<=0:raise ValueError('Actual measured pretrained forward timing required')
    from native_baseline_pose_v7 import propose_pose
    intrinsics=np.array([[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]])
    replay,info=propose_pose(p,q,np.full((512,512),10.),intrinsics,np.eye(4),intrinsics)
    if replay is None or not np.allclose(replay,pose,rtol=0,atol=1e-7)or info['inliers']!=controls['inliers']:raise ValueError('Saved native control pose differs from common reference-geometry estimator replay')
    return dict(values,blank_empty=True,invalid_mask_empty=True,common_pose_replayed=True,scope='Independent saved-array arithmetic and pose-estimator replay only; actual pretrained forward comes from linked source/provenance receipt.')


def required_native_sources(profile,protocol=None):
    spec=specification(profile)
    result=COMMON_CONTROL_SOURCES|{'native_pretrained_adapters_v73.py',spec['module']+'.py',spec['control'],spec['lock']}
    result|={'matchanything_matcher.py'}if profile['method']=='matchanything'else{'contrastive_terrain.py','dataset.py','lunargeo_dataset.py'}
    if protocol is not None:result|=set(protocol['source_sha256'])
    return result


def _semantic_provenance(value,profile,lock,lock_sha):
    spec=specification(profile);internal=832 if profile['author_profile']=='training_size832' else 512
    if value.get('method')!=spec['display']or value.get('profile')!=profile['author_profile']or value.get('native_input_size')!=512 or value.get('internal_input_size')!=internal:raise ValueError('Actual pretrained method/profile/native content mismatch')
    if value.get('source_lock_sha256')!=lock_sha or value.get('source_revision')!=lock['source_revision']or value.get('cpu_threads')!=2 or value.get('trainable_parameters')!=0:raise ValueError('Actual pretrained source/lock/two-thread/frozen profile differs')
    if profile['method']=='matchanything':
        fields=('method','profile','native_input_size','internal_input_size','source_revision','source_repository','source_lock_sha256','weight_sha256','npe','reparameterization','source_license','cpu_threads','trainable_parameters','environment')
        if value.get('weight_sha256')!=lock['safe_weight_sha256']or value.get('npe')!=[832,832,internal,internal]or value.get('reparameterization')is not False or value.get('source_repository')!=lock['source_repository']or value.get('source_license')!=lock['source_license']:raise ValueError('MatchAnything safe weights/author NPE/config/license differ')
    else:
        fields=('method','profile','native_input_size','internal_input_size','source_revision','source_repository','source_lock_sha256','safe_weight_sha256','weight_repository','weight_revision','dependencies','preprocessing','matching','query_geometry_in_interface','cpu_threads','total_parameters','trainable_parameters','environment')
        if value.get('safe_weight_sha256')!=lock['safe_weight_sha256']or value.get('weight_repository')!=lock['weight_repository']or value.get('weight_revision')!=lock['weight_revision']or value.get('dependencies')!=lock['dependencies']or value.get('query_geometry_in_interface')is not False:raise ValueError('MOONSt3R safe weight revision/runtime/common-reference geometry differs')
    if any(name not in value for name in fields):raise ValueError('Complete actual pretrained semantic provenance required')
    return {name:value[name]for name in fields}



def validate_external_comparison(profile,comparison_protocol):
    """Bind any optional outer watchdog to the actual comparison freeze."""
    if profile.get('external_control')is None:return None
    from control_historical_native_v74 import validate_external_pretrained
    external=validate_external_pretrained(profile)
    for name,h in external['source_sha256'].items():
        if comparison_protocol['source_sha256'].get(name)!=h:
            raise ValueError('Outer native control source must be pinned to comparison: '+name)
    for name,h in external['input_sha256'].items():
        if comparison_protocol['input_sha256'].get(name)!=h:
            raise ValueError('Outer native control protocol/receipt must be pinned to comparison: '+name)
    return external


def validate_native_profile(profile,source_root=None):
    """Fail closed without loading a network, a weight file, or a lunar image."""
    spec=specification(profile);external=None
    if profile.get('external_control')is not None:
        from control_historical_native_v74 import validate_external_pretrained
        external=validate_external_pretrained(profile)
    paths=_evidence_paths(profile)
    protocol,summary,controls,provenance=[read(paths[name])for name in EVIDENCE_NAMES]
    root=Path(source_root or profile.get('repository_root')or Path(__file__).resolve().parent).resolve()
    expected=required_native_sources(profile)-{'native_pretrained_adapters_v73.py'}
    pins=protocol.get('source_sha256',{})
    if not expected<=set(pins):raise ValueError('ACTUAL native control source closure incomplete')
    for name,h in pins.items():
        path=Path(name)
        if path.is_absolute()or'..'in path.parts:raise ValueError('Control source names must stay in the repository')
        if sha(root/path)!=h:raise ValueError('Actual pretrained control source changed:'+name)
    if protocol.get('cpu_threads')!=2:raise ValueError('A successful actual TWO-thread native control is required for this comparison')
    if protocol.get('calibration_or_test_pixels_loaded')!=0 or protocol.get('lunar_images_loaded',0)!=0:raise ValueError('Actual pretrained gate must use only procedural images')
    if summary.get('passed')is not True or summary.get('source_sha256')!=pins[spec['control']]:raise ValueError('Successful source-bound ACTUAL pretrained summary required')
    if not _finite_number(protocol.get('max_seconds'))or not _finite_number(summary.get('elapsed_seconds'))or not 0<summary['elapsed_seconds']<=protocol['max_seconds']:raise ValueError('Actual pretrained control exceeded its declared deadline')
    if spec['module']=='native_moonst3r_v19'and summary.get('actual_native_pretrained_forward_attempted')is not True:raise ValueError('Actual MOONSt3R forward evidence required')
    for name in('protocol','controls'):
        if name+'_sha256'in summary and summary[name+'_sha256']!=sha(paths[name]):raise ValueError('Actual summary payload hash differs:'+name)
    if summary.get('controls')!=controls:raise ValueError('Actual summary and raw saved native control differ')
    lock=read(root/spec['lock']);semantic=_semantic_provenance(provenance,profile,lock,sha(root/spec['lock']))
    replay=replay_native_control(controls)
    return dict(method=profile['method'],author_profile=profile['author_profile'],source_root=str(root),source_sha256=pins,
        evidence_sha256={name:sha(path)for name,path in paths.items()},semantic_provenance=semantic,control_replay=replay,
        isolated_latency_established=False,primary_query_truth_inputs=False,external_control=external)


def native_profile(name,method,author_profile,control_root,repository_root='.',**settings):
    """Prepare concrete evidence pins, never invent control success."""
    control_root=Path(control_root);evidence={}
    for field in EVIDENCE_NAMES:
        path=control_root/(field+'.json');evidence[field]=str(path);evidence[field+'_sha256']=sha(path)
    return dict(name=name,backend='native_pretrained',method=method,author_profile=author_profile,repository_root=str(repository_root),cpu_threads=2,pretrained_evidence=evidence,**settings)


def _construct_model(profile,gate):
    spec=specification(profile);module=importlib.import_module(spec['module'])
    if Path(module.__file__).resolve()!=(Path(gate['source_root'])/(spec['module']+'.py')).resolve():raise ValueError('Unpinned native adapter imported from another source tree')
    factory=getattr(module,spec['factory'])
    if profile['method']=='matchanything':return factory(root=gate['source_root'],dependency_dir=profile.get('dependency_dir'),profile=profile['author_profile'],threads=2)
    source=profile.get('source',str(Path(gate['source_root'])/'data/models/moonst3r-upstream'))
    weight=profile.get('weight',str(Path(gate['source_root'])/'data/models/moonst3r/model.safetensors'))
    return factory(source=source,weight=weight,dependency_dir=profile.get('dependency_dir'),threads=2)


def _native_inputs(reference,query,reference_mask,query_mask):
    images=[np.asarray(im)for im in(reference,query)]
    if any(im.shape!=(512,512)or im.dtype!=np.uint8 for im in images):raise ValueError('Original native512 uint8 grayscale images required')
    masks=[np.ones((512,512),bool)if mask is None else np.asarray(mask,bool)for mask in(reference_mask,query_mask)]
    if any(mask.shape!=(512,512)for mask in masks):raise ValueError('Native mask shape differs')
    usable=all(mask.any()for mask in masks)and min(im[mask].std()for im,mask in zip(images,masks))>=1
    return images,masks,usable


class NativePretrainedAdapter:
    def __init__(self,profile):
        self.profile=profile;self.gate=validate_native_profile(profile)
        self.model=_construct_model(profile,self.gate)
        root=Path(self.gate['source_root']);lock=read(root/METHODS[profile['method']]['lock'])
        loaded=_semantic_provenance(self.model.provenance,profile,lock,sha(root/METHODS[profile['method']]['lock']))
        if loaded!=self.gate['semantic_provenance']:raise ValueError('Loaded model differs from actual pretrained native control configuration/runtime')
        self.provenance=dict(self.model.provenance,adapter_revision='native-pretrained-v73',pretrained_gate=self.gate,
            correspondence_convention='reference-to-query native512 integer-pixel centres; subpixel positions retained',
            principal_pose='Only external common reference-geometry AP3P; model-predicted pose/depth is not used',
            sun_metadata_used=False,accepted_policy=None,parameter_cap_scope='The100k cap applies to the compact candidate, not pretrained comparison models')

    def points(self,reference,query,reference_mask=None,query_mask=None):
        images,masks,usable=_native_inputs(reference,query,reference_mask,query_mask)
        if not usable:return np.empty((0,2)),np.empty((0,2))
        p,q=self.model.points(*images,*masks);self.model.sync()
        p,q=[np.asarray(value,float)for value in(p,q)]
        if p.size==q.size==0:return np.empty((0,2)),np.empty((0,2))
        if p.ndim!=2 or p.shape[1:]!=(2,)or q.shape!=p.shape or not np.isfinite(p).all()or not np.isfinite(q).all()or not((p>=0)&(p<=511)&(q>=0)&(q<=511)).all():raise ValueError('Native pretrained adapter returned invalid/unmapped original-pixel correspondences')
        return p,q

    def sync(self):self.model.sync()


def make_native_pretrained(profile):return NativePretrainedAdapter(profile)
