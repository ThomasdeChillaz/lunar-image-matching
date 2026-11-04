"""Unintegrated native feature adapter for the v85 pose-free math prototype.

There is no top-level Torch import. Actual execution requires an already
loaded LightingResponseModel and an external published/budgeted experiment.
No query camera, depth, pose, Sun or labels enter this API. Native half-pixel
query descriptor interpolation is NOT equal to encoding a resampled teacher
patch, and same-reference-view relighting does not synthesize a target view.
"""
from dataclasses import dataclass
import hashlib
import math
import sys
import time

import cv2
import numpy as np

from lunar_geometry_v2 import camera_check
from solar_physics_v63 import soft_shadow_observation


GRID_AXIS = 49.5+8*np.arange(51)
QUERY_ENTRIES = 4*51*51
QUARTER_ANGLES = (0.,90.,180.,270.)


class ExtractionUnavailable(RuntimeError):
    """An eligible case failed extraction; future evaluation must retain it."""


@dataclass(frozen=True)
class ExtractionCaps:
    reference_patch_batch: int = 32
    query_image_batch: int = 1
    max_reference_encoder_forwards: int = 16
    max_query_encoder_forwards: int = 4
    max_response_decodes: int = 292
    max_explicit_array_bytes: int = 128*1024*1024
    max_calendar_seconds: float = 300.


def preflight_extraction(n_reference, n_suns, caps=ExtractionCaps()):
    ints = (n_reference, n_suns, caps.reference_patch_batch, caps.query_image_batch,
            caps.max_reference_encoder_forwards, caps.max_query_encoder_forwards,
            caps.max_response_decodes, caps.max_explicit_array_bytes)
    if any(type(v) is not int for v in ints) or not 1 <= n_reference <= 256 or not 1 <= n_suns <= 1885:
        raise ValueError('Finite bounded integer feature-extraction dimensions required')
    if not 1 <= caps.reference_patch_batch <= 32 or caps.query_image_batch != 1:
        raise ValueError('Reference batches<=32 and query-image batch exactly1 required')
    if (min(caps.max_reference_encoder_forwards, caps.max_query_encoder_forwards,
            caps.max_response_decodes, caps.max_explicit_array_bytes) <= 0
            or not math.isfinite(caps.max_calendar_seconds) or caps.max_calendar_seconds <= 0):
        raise ValueError('Positive finite extraction limits required')
    reference_calls = 2*math.ceil(n_reference/caps.reference_patch_batch)
    dense_bytes = 32*512*512*4
    fused_bytes = 2*n_suns*(32*32+32)*4
    # Include map/index/support temporaries, concatenate overlap, rotations and
    # intensity-observation temporaries; this still excludes encoder internals.
    estimated_bytes = (dense_bytes+fused_bytes+2*QUERY_ENTRIES*(32*4+2*8+8+1+8)
                       +80*n_reference*33*33+caps.reference_patch_batch*32*33*33*4
                       +64*512*512+1024*1024)
    if reference_calls > caps.max_reference_encoder_forwards or 4 > caps.max_query_encoder_forwards:
        raise ValueError('Declared encoder-forward allowance insufficient')
    if estimated_bytes > caps.max_explicit_array_bytes:
        raise ValueError('Declared explicit-array allowance insufficient')
    return dict(reference_encoder_forwards_bound=reference_calls, query_encoder_forwards_bound=4,
                response_decodes_cap=caps.max_response_decodes, dense_feature_output_bytes=dense_bytes,
                two_mode_fused_decoder_bytes_bound=fused_bytes, estimated_explicit_array_bytes=estimated_bytes,
                scope='Explicit arrays only; excludes encoder internals/allocator/BLAS workspace. External watchdog and measured RSS remain required.')


def _rss():
    if sys.platform not in ('darwin', 'linux'):
        return dict(available=False, bytes=None, reason='unsupported_platform')
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(available=True, bytes=int(value if sys.platform == 'darwin' else value*1024),
                    scope='Current worker PROCESS-LIFETIME high-water RSS, including prior models/caches; not isolated adapter/model RAM; excludes children')
    except Exception as exc:
        return dict(available=False, bytes=None, reason=type(exc).__name__)


def _digest_arrays(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.asarray(value)
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class _Guard:
    def __init__(self, caps, arrays):
        self.caps = caps; self.arrays = arrays
        self.started = time.perf_counter(); self.original_digest = _digest_arrays(arrays)
    def check(self):
        if time.perf_counter()-self.started >= self.caps.max_calendar_seconds:
            raise TimeoutError('Original cooperative feature-adapter calendar allowance exhausted')
    def finish(self):
        if _digest_arrays(self.arrays) != self.original_digest:
            raise ValueError('Reference/query extraction inputs changed during this case')
        self.check()


class _TorchRuntime:
    is_test_stub = False
    def __init__(self, model, caps, guard):
        import torch
        from solar_response_v63 import LightingResponseModel
        from solar_native_v63 import state_digest
        if not isinstance(model, LightingResponseModel):
            raise ValueError('An existing LightingResponseModel is required; no added encoder')
        if torch.get_num_threads() > 2:
            raise ValueError('Configure at most two CPU threads before extraction')
        self.torch = torch; self.model = model; self.caps = caps; self.guard = guard
        self._state_digest = state_digest
        params = list(model.parameters())
        if (not params or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in params)
                or model.training or model.coarse.training or model.fine.training):
            raise ValueError('Float32 CPU eval-mode model required')
        total = sum(p.numel() for p in params)
        if total >= 100000 or any(p.requires_grad for p in model.coarse.parameters()):
            raise ValueError('Frozen original coarse encoder and combined<100000 parameters required')
        self.identity = dict(model_sha256=state_digest(model), coarse_sha256=state_digest(model.coarse),
                             fine_sha256=state_digest(model.fine), total_parameters=total,
                             parameter_bytes=sum(p.numel()*p.element_size() for p in params),
                             device='cpu', dtype='float32', extra_encoder_parameters=0)
        self._versions = self._version_key(); self._decoders = {}
        self.counts = dict(reference_fine_forwards=0, reference_frozen_coarse_forwards=0,
                           query_frozen_coarse_forwards=0, response_decodes=0, decoder_builds=0)
    def _version_key(self):
        tensors = list(self.model.named_parameters())+list(self.model.named_buffers())
        return (tuple((n, id(v), v._version, tuple(v.shape), str(v.dtype), str(v.device)) for n,v in tensors),
                tuple((n, m.training) for n,m in self.model.named_modules()))
    def check(self):
        self.guard.check()
        if self._version_key() != self._versions:
            raise ValueError('Existing response model changed during extraction/retrieval')
    def reference_features(self, patches):
        self.check(); torch = self.torch
        if len(patches) > self.caps.reference_patch_batch:
            raise ValueError('Reference patch batch cap exceeded')
        with torch.inference_mode():
            x = torch.from_numpy(patches.copy()).float()[:,None]/255
            learned = self.model.patch_features(x).cpu().numpy().copy()
            dense,_ = self.model.coarse(x)
            if dense.shape != (len(patches),32,33,33):
                raise ValueError('Frozen teacher must produce native32D features')
            frozen = dense[:,:,16,16].cpu().numpy().copy()
        self.counts['reference_fine_forwards'] += 1
        self.counts['reference_frozen_coarse_forwards'] += 1
        self.check(); return learned, frozen
    def query_features(self, image, points):
        self.check(); torch = self.torch
        from contrastive_terrain import sample_descriptors
        if self.counts['query_frozen_coarse_forwards'] >= self.caps.max_query_encoder_forwards:
            raise ValueError('Frozen query-forward cap exceeded')
        with torch.inference_mode():
            x = torch.from_numpy(image.copy()).float()[None,None]/255
            dense,_ = self.model.coarse(x)
            if dense.shape != (1,32,512,512):
                raise ValueError('One native512 frozen query feature map required')
            p = torch.from_numpy(points.astype(np.float32))[None]
            sampled = sample_descriptors(dense,p)[0].cpu().numpy().copy()
        self.counts['query_frozen_coarse_forwards'] += 1
        self.check(); return sampled
    def bind(self, features, condition, suns):
        self._features = features.copy(); self._condition = condition.copy(); self._suns = suns.copy()
    def decode(self, base, condition, target_sun, *, pooled):
        self.check(); torch = self.torch
        from solar_response_v63 import FusedLightingDecoder
        if (type(pooled) is not bool or not np.array_equal(base,self._features)
                or not np.array_equal(condition,self._condition)):
            raise ValueError('Response requires the extracted reference features and known-reference conditioning')
        target = np.asarray(target_sun,float)
        found = np.flatnonzero(np.all(self._suns == target,axis=1)) if target.shape == (3,) else []
        if len(found) != 1:
            raise ValueError('Response target must be one exact generic catalogue Sun')
        if self.counts['response_decodes'] >= self.caps.max_response_decodes:
            raise ValueError('Response decode cap exhausted')
        if pooled not in self._decoders:
            self._decoders[pooled] = FusedLightingDecoder(self.model,self._suns,pooled)
            self.counts['decoder_builds'] += 1
        with torch.inference_mode():
            result = self._decoders[pooled](torch.from_numpy(np.array(base,copy=True)),
                       torch.from_numpy(np.array(condition,copy=True)),np.full(len(base),int(found[0]),np.int64))
            result = result.cpu().numpy().copy()
        self.counts['response_decodes'] += 1
        self.check(); return result
    def finish(self):
        self.check()
        if self._state_digest(self.model) != self.identity['model_sha256']:
            raise ValueError('Response model bytes changed, including unversioned data mutation')


def _make_runtime(model, caps, guard):
    return _TorchRuntime(model,caps,guard)


def _make_geometry_guard(geometry):
    # Reuse the qualified v84 live-owner guard lazily. Stub controls load its
    # exact array-only class definition without importing Torch or forwarding.
    from solar_response_cache_v84 import GeometryOwnerGuardV84
    return GeometryOwnerGuardV84(geometry)


def _make_rotation_bank(image):
    # The official rotation module transitively imports Torch. Keep even this
    # dependency lazy so importing/stub-testing this adapter does not do so.
    from solar_rotation_v71 import RotationBank
    return RotationBank(image,QUARTER_ANGLES)


def reference_anchor_grid():
    axis = 49.5+16*np.rint(np.linspace(0,25,16)).astype(int)
    y,x = np.meshgrid(axis,axis,indexing='ij')
    return np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)


@dataclass
class ExtractedNativeFeatures:
    inputs: dict
    anchor_indices: np.ndarray
    reference_xy: np.ndarray
    _runtime: object
    _guard: object
    _metadata: dict
    _geometry_guard: object = None
    def retrieval_inputs(self):
        return dict(self.inputs,response_predictor=self._runtime.decode)
    @property
    def metadata(self):
        return dict(self._metadata, encoder_counts=dict(self._runtime.counts),
                    elapsed_calendar_seconds=time.perf_counter()-self._guard.started,
                    worker_lifetime_peak_rss=_rss())
    def finish(self):
        try:
            self._runtime.finish(); self._guard.finish()
        finally:
            if self._geometry_guard is not None: self._geometry_guard.check()
        return self.metadata
    def __enter__(self): return self
    def __exit__(self,*exc): self.finish()


def _extract_native_features(reference_image, query_image, reference_geometry,
                            known_reference_sun, response_model, *,
                            caps=ExtractionCaps(), include_physical_evidence=True):
    """Reference-only geometry and known-reference illumination; query IMAGE only.

    Both native-control reference descriptors and query descriptors come from
    model.coarse. Learned reference base features come from model.fine. Same
    bank/coordinates/support are used in every retrieval arm. Geometry support
    exclusions are reported separately from zero features and image darkness.
    The returned object must be finished after retrieval (or used as context).
    """
    started = time.perf_counter(); rss_before = _rss()
    for image in (reference_image,query_image):
        if not isinstance(image,np.ndarray) or image.shape != (512,512) or image.dtype != np.uint8:
            raise ValueError('Native512 uint8 grayscale image arrays required')
    if min(reference_image.std(),query_image.std()) < 1:
        raise ExtractionUnavailable('blank_or_uninformative_image')
    if type(include_physical_evidence) is not bool:
        raise ValueError('Explicit physical-evidence mode required')
    geometry = reference_geometry
    k,pose = camera_check(geometry.intrinsics,geometry.cam2world)
    depth = np.asarray(geometry.depth)
    normals = np.asarray(geometry.normals,float)
    worlds = np.asarray(geometry.world_variants,float)
    suns_world = np.asarray(geometry.sun_vectors,float)
    parameters = np.asarray(geometry.sun_parameters,float)
    coarse_mask = np.asarray(geometry.coarse_sun_mask)
    ref_sun = np.asarray(known_reference_sun,float)
    if (depth.shape != (512,512) or normals.shape != (256,3) or worlds.shape != (256,7,3)
            or suns_world.ndim != 2 or suns_world.shape[1] != 3 or not np.isfinite(worlds).all()
            or not np.isfinite(normals).all() or parameters.shape != (len(suns_world),2)
            or coarse_mask.shape != (len(suns_world),) or coarse_mask.dtype != bool):
        raise ValueError('Reference geometry/native anchor/Sun catalogue schema mismatch')
    if (ref_sun.shape != (3,) or not np.isfinite(ref_sun).all()
            or not np.isclose(np.linalg.norm(ref_sun),1.,atol=1e-6,rtol=0)
            or not np.isfinite(suns_world).all()
            or not np.allclose(np.linalg.norm(suns_world,axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Unit known-reference and generic world-Sun directions required')
    anchors = reference_anchor_grid()
    yy,xx = np.mgrid[-16:17,-16:17]
    offsets = np.stack([xx,yy],-1)
    maps = anchors[:,None,None,:]+offsets[None]
    indices = np.rint(maps).astype(int)
    inside = ((indices>=0)&(indices<=511)).all(-1)
    safe = indices.clip(0,511)
    sampled_depth = depth[safe[...,1],safe[...,0]]
    support = (inside & np.isfinite(sampled_depth) & (sampled_depth>0)).all((1,2))
    chosen = np.flatnonzero(support)
    if not len(chosen):
        raise ExtractionUnavailable('insufficient_reference_surface_support')
    bound = preflight_extraction(int(len(chosen)),int(len(suns_world)),caps)
    reference_world = worlds[chosen,0]
    camera = (reference_world-pose[:3,3])@pose[:3,:3]
    h = camera@k.T
    with np.errstate(divide='ignore',invalid='ignore'):
        projection = h[:,:2]/h[:,2,None]
    if ((camera[:,2]<=0).any() or not np.isfinite(projection).all()
            or np.max(np.linalg.norm(projection-anchors[chosen],axis=1)) > .02):
        raise ValueError('Exact reference anchors violate native pixel conventions')
    r = pose[:3,:3]
    views = pose[:3,3]-reference_world
    lengths = np.linalg.norm(views,axis=1,keepdims=True)
    if (lengths <= 1e-10).any() or not np.allclose(np.linalg.norm(normals[chosen],axis=1),1.,atol=1e-6,rtol=0):
        raise ValueError('Known reference view and unit normals required')
    view_reference = (views/lengths)@r
    normal_reference = normals[chosen]@r
    sun_reference = ref_sun@r
    suns_reference = suns_world@r
    physical = None
    if include_physical_evidence:
        state = np.asarray(geometry.solar_states)
        materialized = np.asarray(geometry.solar_materialized_mask)
        if (state.shape != (256,len(suns_world)) or not np.isin(state,[0,1,2]).all()
                or materialized.dtype != bool or materialized.shape != (len(suns_world),)
                or not materialized.all()):
            raise ValueError('Physical retrieval requires all generic columns actually computed; unknown coverage remains distinct')
        physical = state[chosen].copy()
    tracked = dict(reference=reference_image,query=query_image,depth=depth,
                   intrinsics=np.asarray(geometry.intrinsics),pose=np.asarray(geometry.cam2world),
                   normals=np.asarray(geometry.normals),worlds=np.asarray(geometry.world_variants),
                   suns=np.asarray(geometry.sun_vectors),parameters=np.asarray(geometry.sun_parameters),
                   coarse_mask=np.asarray(geometry.coarse_sun_mask),reference_sun=np.asarray(known_reference_sun))
    if include_physical_evidence:
        tracked.update(solar_states=np.asarray(geometry.solar_states),materialized=np.asarray(geometry.solar_materialized_mask))
    guard = _Guard(caps,tracked)
    # Keep one original elapsed origin; preflight/geometry work is not free time.
    guard.started = started; guard.check()
    runtime = _make_runtime(response_model,caps,guard)
    phases = {}; snapshots = []
    try:
        tick = time.perf_counter()
        patch_maps = maps[chosen].astype(np.float32)
        patches = np.stack([cv2.remap(reference_image,m[:,:,0],m[:,:,1],cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT) for m in patch_maps])
        learned_parts = []; frozen_parts = []
        for lo in range(0,len(chosen),caps.reference_patch_batch):
            guard.check()
            learned,frozen = runtime.reference_features(patches[lo:lo+caps.reference_patch_batch])
            if (learned.shape != (len(patches[lo:lo+caps.reference_patch_batch]),32)
                    or frozen.shape != learned.shape or not np.isfinite(learned).all() or not np.isfinite(frozen).all()):
                raise ValueError('Separate finite learned/frozen reference centre features required')
            learned_parts.append(learned); frozen_parts.append(frozen)
        learned = np.concatenate(learned_parts).astype(np.float32)
        frozen = np.concatenate(frozen_parts).astype(np.float32)
        phases['reference_patch_and_encoders_seconds'] = time.perf_counter()-tick
        snapshots.append(dict(phase='after_reference',rss=_rss()))
        tick = time.perf_counter(); bank = _make_rotation_bank(query_image)
        y,x = np.meshgrid(GRID_AXIS,GRID_AXIS,indexing='ij')
        grid = np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)
        qfeatures = []; qxy = []; qvalid = []; qrotation = []
        for rotation,image in enumerate(bank.images):
            guard.check()
            valid = np.array([bank.native_support(point,rotation) for point in grid],bool)
            features = runtime.query_features(image,grid)
            if features.shape != (len(grid),32) or not np.isfinite(features).all():
                raise ValueError('Finite frozen native sampled query features required')
            features = features.astype(np.float32,copy=True); features[~valid] = np.nan
            qfeatures.append(features); qxy.append(bank.transform(grid,rotation,True))
            qvalid.append(valid); qrotation.append(np.full(len(grid),rotation,np.int64))
            snapshots.append(dict(phase='after_query_rotation_'+str(rotation),rss=_rss()))
        query_features = np.concatenate(qfeatures)
        query_xy = np.concatenate(qxy); query_valid = np.concatenate(qvalid)
        rotation_indices = np.concatenate(qrotation)
        phases['query_rotation_and_encoders_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter(); shadow = None; observation = None
        if include_physical_evidence:
            shadow_image,observation = soft_shadow_observation(query_image)
            shadow = cv2.remap(shadow_image.astype(np.float32),query_xy[:,0,None].astype(np.float32),
                               query_xy[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel()
        phases['query_intensity_observation_seconds'] = time.perf_counter()-tick
        condition = np.column_stack([np.broadcast_to(sun_reference,(len(chosen),3)),
                                     normal_reference,view_reference]).astype(np.float32)
        runtime.bind(learned,condition,suns_reference)
        inputs = dict(reference_response_features=learned,frozen_reference_descriptors=frozen,
                      known_reference_sun=sun_reference,reference_normals=normal_reference,
                      reference_views=view_reference,query_descriptors=query_features,
                      query_native_xy=query_xy,query_rotation_indices=rotation_indices,query_valid=query_valid,
                      sun_vectors=suns_reference,sun_parameters=parameters.copy(),coarse_sun_mask=coarse_mask.copy())
        if include_physical_evidence:
            inputs.update(solar_states=physical,query_soft_shadow=shadow)
        for value in inputs.values(): value.setflags(write=False)
        metadata = dict(prototype_only=True,software_stub_only=bool(runtime.is_test_stub),
                        model_identity=dict(runtime.identity),preflight=bound,phase_seconds=phases,
                        reference_anchors_inventory=256,reference_anchors_retained=int(len(chosen)),
                        reference_support_mask=support.tolist(),reference_support_omissions=int((~support).sum()),
                        support_is_not_query_occlusion_or_shadow=True,grid_axis_start=49.5,grid_stride=8,
                        grid_count_per_rotation=2601,quarter_turn_degrees=list(QUARTER_ANGLES),query_entries=QUERY_ENTRIES,
                        valid_query_entries=int(query_valid.sum()),query_image_batch=1,reference_patch_batch=caps.reference_patch_batch,
                        query_bank_sha256=_digest_arrays(dict(descriptors=query_features,xy=query_xy,rotation=rotation_indices,valid=query_valid)),
                        zero_norm_reference_frozen=int((np.linalg.norm(frozen,axis=1)<=1e-8).sum()),
                        zero_norm_reference_learned=int((np.linalg.norm(learned,axis=1)<=1e-8).sum()),
                        zero_norm_valid_query=int((np.linalg.norm(query_features[query_valid],axis=1)<=1e-8).sum()),
                        physical_evidence_observation=observation,physical_is_image_measurement_not_ground_truth=True,
                        worker_rss_before=rss_before,worker_rss_snapshots=snapshots,
                        limitations=['Half-pixel interpolation of dense descriptors is not encoding a subpixel-resampled33px teacher patch',
                                     'Native query features lack surface/view canonicalization',
                                     'Response predicts same-reference-view relighting, not an unknown query view',
                                     'Latitude extension outside training is unsupervised',
                                     'Cooperative cap requires the unchanged external job watchdog; this creates no compute allowance'])
        result = ExtractedNativeFeatures(inputs,chosen.copy(),anchors[chosen].copy(),runtime,guard,metadata)
        result.finish()
        return result
    except BaseException:
        runtime.finish(); guard.finish()
        raise


def extract_native_features(reference_image, query_image, reference_geometry,
                            known_reference_sun, response_model, *,
                            caps=ExtractionCaps(), include_physical_evidence=True):
    """Bind ORIGINAL live geometry before preflight; check every exit and use."""
    geometry_guard = _make_geometry_guard(reference_geometry)
    try:
        result = _extract_native_features(reference_image, query_image, reference_geometry,
            known_reference_sun, response_model, caps=caps,
            include_physical_evidence=include_physical_evidence)
        result._geometry_guard = geometry_guard
        return result
    finally:
        geometry_guard.check()


def retrieve_from_extracted(extracted, *, arm='sh', limits=None):
    """Prototype integration with mandatory model/input-byte checks after use."""
    from solar_pose_free_retrieval_v85 import retrieve_candidates,Limits
    try:
        result = retrieve_candidates(**extracted.retrieval_inputs(),arm=arm,
                                     limits=Limits() if limits is None else limits)
    finally:
        metadata = extracted.finish()
    result['native_feature_extraction'] = metadata
    result['software_stub_only'] = metadata['software_stub_only']
    return result
