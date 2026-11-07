"""Inactive inference-only response cache; frozen scientific code is unchanged.

This module is not selected by any factory. A future caller MUST call finish()
before emitting a case result, including when inference fails. Full byte checks
at that boundary catch writes through .data that bypass Tensor version counters.
No query truth, new model parameters, changed batching, or approximate keys.
"""
from collections import OrderedDict
import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from lunar_geometry_v5 import map_points_z
from solar_frontend_v63 import reference_anchors
from solar_native_v63 import state_digest
from solar_response_v63 import FusedLightingDecoder


def _array_digest(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _snapshot(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


class GeometryOwnerGuardV84:
    """Bind live attributes, including the original pre-reference-Sun owner."""
    names = ('depth', 'intrinsics', 'cam2world', 'normals', 'sun_vectors',
             'reference_sun', 'world_variants')

    def __init__(self, owner):
        self.owner = owner
        self.bindings = tuple((name, getattr(owner, name),
            _array_digest(getattr(owner, name))) for name in self.names)

    def check(self):
        for name, original, digest in self.bindings:
            current = getattr(self.owner, name, None)
            if current is not original or _array_digest(current) != digest:
                raise RuntimeError('Response geometry attribute replaced or mutated: '+name)


class ImmutableModelIdentityV84:
    """Private identity, not a second learned model or a mutable-model cache.

    Normal in-place updates, replacement, device/dtype and mode changes fail on
    the next call. A full boundary digest also covers otherwise invisible .data
    writes. Concurrent model mutation is forbidden; callers may not emit a
    result until the final boundary check succeeds.
    """
    def __init__(self, model):
        self.model = model
        if any(module.training for module in model.modules()):
            raise ValueError('An eval-only per-case response model is required')
        self.digest = state_digest(model)
        self.fine_digest = state_digest(model.fine)
        self.signature = self._signature()
        self.full_checks = 1

    def _signature(self):
        values = [('parameter', n, v) for n, v in self.model.named_parameters()]
        values += [('buffer', n, v) for n, v in self.model.named_buffers()]
        tensors = tuple((kind, name, id(v), v.data_ptr(), tuple(v.shape), tuple(v.stride()),
            str(v.dtype), str(v.device), v.requires_grad, v._version) for kind, name, v in values)
        modes = tuple((name, id(module), module.training) for name, module in self.model.named_modules())
        return tensors, modes

    def check(self):
        if self._signature() != self.signature:
            raise RuntimeError('Response model mutated during the immutable case')

    def full_check(self):
        self.check(); self.full_checks += 1
        if state_digest(self.model) != self.digest:
            raise RuntimeError('Response model bytes mutated during the immutable case')


class _ImmutableFusedDecoder:
    """Original fused arithmetic and batch shapes; head is already immutable."""
    def __init__(self, model, target_suns, pooled):
        # Original implementation computes the identical fusion once. Repeated
        # decoder calls omit only its repeated head hashing, not its arithmetic.
        original = FusedLightingDecoder(model, target_suns, pooled)
        self.model = model; self.target_suns = original.target_suns
        self.weight = original.weight; self.bias = original.bias

    def __call__(self, base, condition, target_indices):
        ids = torch.as_tensor(target_indices, dtype=torch.long, device=base.device)
        if ids.shape != (len(base),) or not len(ids) or ids.min() < 0 or ids.max() >= len(self.target_suns):
            raise ValueError('One valid target Sun column per patch required')
        with torch.inference_mode():
            adjusted = self.model.adjusted_features(base, condition)
            value = torch.einsum('boi,bi->bo', self.weight[ids], adjusted)+self.bias[ids]
            return F.normalize(value, dim=-1, eps=1e-8)


class PairResponseCacheV84:
    """Same selected-measurement response API, with exact per-case reuse.

    Original two-patch encoder batches and each requested Sun-tuple decode batch
    are retained. Cache storage is private and bounded. There is no cross-case
    warm state. finish() is a required precondition for publishing any result.
    """
    def __init__(self, model, reference, query, geometry, k_query, anchors,
                 graph_sun_ids, pooled=False, capacity=512):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive bounded response cache capacity required')
        self.model = model; self.identity = ImmutableModelIdentityV84(model)
        self.pooled = bool(pooled); self.capacity = capacity; self.closed = False
        self._geometry_guard = GeometryOwnerGuardV84(geometry)
        names = self._geometry_guard.names
        originals = {'reference': reference, 'query': query, 'k_query': k_query,
            'anchors': anchors, 'graph_sun_ids': graph_sun_ids}
        originals.update({'geometry.'+name: getattr(geometry, name) for name in names})
        self._input_bindings = tuple((name, value, _array_digest(value)) for name, value in originals.items())
        self.reference = _snapshot(reference); self.query = _snapshot(query)
        self.k_query = _snapshot(k_query)
        self.geometry = SimpleNamespace(**{name: _snapshot(getattr(geometry, name)) for name in names})
        self.anchors = _snapshot(np.asarray(anchors, int)); self.graph_sun_ids = _snapshot(np.asarray(graph_sun_ids, int))
        self._points = reference_anchors()[self.anchors].copy()
        r = self.geometry.cam2world[:3, :3]
        self.decoder = _ImmutableFusedDecoder(model, self.geometry.sun_vectors@r, pooled)
        self.reference_decoder = _ImmutableFusedDecoder(model, (self.geometry.reference_sun@r)[None], pooled)
        self.cache = OrderedDict(); self.pairs = OrderedDict(); self.features = OrderedDict()
        self.surfaces = OrderedDict(); self.contexts = OrderedDict()
        self.reference_predictions = OrderedDict(); self.reference_self_predictions = OrderedDict()
        self.cost = dict(refresh_requests=0, cache_hits=0, canonicalizations=0,
            patch_encoder_calls=0, patches_encoded=0, reference_surface_builds=0,
            reference_surface_cache_hits=0, geometric_pair_cache_hits=0,
            canonical_feature_cache_hits=0, reference_prediction_cache_hits=0,
            model_full_digest_builds=1, immutable_boundary_checks=0)

    def _check_open(self):
        if self.closed: raise RuntimeError('Per-case response cache is closed')
        self.identity.check()

    def cache_token(self):
        self._check_open()
        return self.identity.digest

    def assert_unchanged(self):
        """Full byte check, required before a case result is allowed to escape."""
        self._check_open(); self.identity.full_check()
        self._geometry_guard.check()
        for name, value, digest in self._input_bindings:
            if _array_digest(value) != digest:
                raise RuntimeError('Response case input mutated: '+name)
        self.cost['immutable_boundary_checks'] += 1

    def finish(self):
        self.assert_unchanged(); self.closed = True
        return dict(self.cost)

    def __enter__(self):
        self._check_open(); return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()
        return False

    @staticmethod
    def _put(cache, key, value, capacity):
        cache[key] = value; cache.move_to_end(key)
        while len(cache) > capacity: cache.popitem(last=False)

    def _surface(self, i):
        if i in self.surfaces:
            self.cost['reference_surface_cache_hits'] += 1
            return self.surfaces[i]
        point = np.asarray(self._points[i], float)
        yy, xx = np.mgrid[-16:17, -16:17]
        source = np.stack([point[0]+xx, point[1]+yy], -1).reshape(-1, 2)
        g = self.geometry
        world, valid = map_points_z(source, g.depth, g.intrinsics, g.cam2world)
        source_map = source.reshape(33, 33, 2).astype(np.float32)
        reference_patch = cv2.remap(self.reference, source_map[:, :, 0], source_map[:, :, 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = dict(source=source, world=world, valid=valid, reference_xy=source_map,
            reference_patch=reference_patch)
        # At most256 declared anchors, and only acquired anchors are prepared.
        self.surfaces[i] = result; self.cost['reference_surface_builds'] += 1
        return result

    def canonical_pair(self, i, query_center, trial_pose):
        """Exact original maps/support. Exposed for procedural equivalence QA."""
        self._check_open()
        for image in (self.reference, self.query):
            if image.shape != (512, 512) or image.dtype != np.uint8:
                raise ValueError('Native512uint8 grayscale canonicalization inputs required')
        g = self.geometry
        if g.depth.shape != (512, 512): raise ValueError('Native reference axial depth required')
        for value, shape in ((g.intrinsics, (3, 3)), (g.cam2world, (4, 4)),
                             (self.k_query, (3, 3)), (trial_pose, (4, 4))):
            if np.asarray(value).shape != shape or not np.isfinite(value).all():
                raise ValueError('Finite calibrated reference/trial geometry required')
        surface = self._surface(i); source = surface['source']; world = surface['world']
        target = np.full((len(source), 2), np.nan)
        cam = (world-np.asarray(trial_pose)[:3, 3])@np.asarray(trial_pose)[:3, :3]
        with np.errstate(divide='ignore', invalid='ignore'):
            h = cam@np.asarray(self.k_query).T; pixels = h[:, :2]/h[:, 2, None]
        inside = (cam[:, 2] > 0)&np.isfinite(pixels).all(1)&((pixels >= 0)&(pixels <= 511)).all(1)
        ids = np.flatnonzero(surface['valid']); target[ids[inside]] = pixels[inside]
        if query_center is not None:
            centre = np.asarray(query_center, float)
            if centre.shape != (2,) or not np.isfinite(centre).all(): raise ValueError('Finite measured query centre required')
            if not np.isfinite(target[16*33+16]).all(): return None, dict(reason='missing_canonical_centre')
            target += centre-target[16*33+16]
        supported = np.isfinite(target).all(1)&((source >= 0)&(source <= 511)).all(1)&((target >= 0)&(target <= 511)).all(1)
        if not supported.all(): return None, dict(reason='incomplete_canonical_surface_or_image_support')
        query_map = target.reshape(33, 33, 2).astype(np.float32)
        patch = cv2.remap(self.query, query_map[:, :, 0], query_map[:, :, 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return [surface['reference_patch'], patch], dict(reason=None, reference_xy=surface['reference_xy'],
            query_xy=query_map, mapping='Reference surface coordinates projected to native query pixels under the trial pose.')

    def _canonical_record(self, i, c, query_xy, trial_pose):
        key = (int(i), int(c), np.asarray(query_xy, float).tobytes(), np.asarray(trial_pose, float).tobytes())
        if key in self.pairs:
            self.cost['geometric_pair_cache_hits'] += 1; self.pairs.move_to_end(key)
            return self.pairs[key]
        self.cost['canonicalizations'] += 1
        patches, _ = self.canonical_pair(i, query_xy, trial_pose)
        result = None if patches is None else dict(patches=patches, features=None)
        self._put(self.pairs, key, result, self.capacity)
        return result

    def _pair_features(self, record):
        if record['features'] is not None: return record['features']
        patches = record['patches']
        feature_key = (self.identity.fine_digest, hashlib.sha256(patches[0].tobytes()+patches[1].tobytes()).hexdigest())
        if feature_key in self.features:
            features = self.features[feature_key]; self.cost['canonical_feature_cache_hits'] += 1
        else:
            dtype = self.model.coefficients.weight.dtype; device = self.model.coefficients.weight.device
            with torch.inference_mode():
                # Preserve original batch2 shape/order exactly.
                features = self.model.patch_features(torch.as_tensor(np.stack(patches).copy(), dtype=dtype, device=device)[:, None]/255)
            self.cost['patch_encoder_calls'] += 1; self.cost['patches_encoded'] += 2
            self._put(self.features, feature_key, features, 512)
        record['features'] = features
        return features

    def _context(self, i, trial_pose):
        g = self.geometry; anchor = int(self.anchors[i]); r = g.cam2world[:3, :3]
        world = g.world_variants[anchor, 0]; view = trial_pose[:3, 3]-world
        reference_view = g.cam2world[:3, 3]-world
        if min(np.linalg.norm(reference_view), np.linalg.norm(view)) < 1e-10:
            raise ValueError('Exact reference anchor must be separated from camera centres')
        view = view/np.linalg.norm(view)@r
        if i not in self.contexts:
            normal = g.normals[anchor]@r; reference_sun = g.reference_sun@r
            reference_view = reference_view/np.linalg.norm(reference_view)@r
            dtype = self.model.coefficients.weight.dtype; device = self.model.coefficients.weight.device
            ref_condition = torch.tensor(np.concatenate([reference_sun, normal, reference_view]), dtype=dtype, device=device)[None]
            self.contexts[i] = normal, ref_condition
        normal, ref_condition = self.contexts[i]
        return normal, ref_condition, view

    def _scores(self, i, selected, features, context):
        normal, ref_condition, view = context
        g = self.geometry; r = g.cam2world[:3, :3]
        suns = g.sun_vectors[selected]@r; n = len(suns)
        dtype = self.model.coefficients.weight.dtype; device = self.model.coefficients.weight.device
        query_condition = torch.tensor(np.column_stack([suns, np.repeat(normal[None], n, 0), np.repeat(view[None], n, 0)]), dtype=dtype, device=device)
        ref_feature_bytes = features[:1].detach().cpu().contiguous().numpy().tobytes()
        ref_key = (int(i), ref_feature_bytes, tuple(selected))
        self_key = (int(i), ref_feature_bytes)
        with torch.inference_mode():
            if ref_key in self.reference_predictions:
                reference_at_query = self.reference_predictions[ref_key]; self.cost['reference_prediction_cache_hits'] += 1
            else:
                reference_at_query = self.decoder(features[:1].expand(n, -1), ref_condition.expand(n, -1), selected)
                self._put(self.reference_predictions, ref_key, reference_at_query, self.capacity)
            if self_key in self.reference_self_predictions:
                reference_at_reference = self.reference_self_predictions[self_key]; self.cost['reference_prediction_cache_hits'] += 1
            else:
                reference_at_reference = self.reference_decoder(features[:1], ref_condition, np.zeros(1, int))
                self._put(self.reference_self_predictions, self_key, reference_at_reference, self.capacity)
            query_at_query = self.decoder(features[1:].expand(n, -1), query_condition, selected)
            query_at_reference = self.reference_decoder(features[1:].expand(n, -1), query_condition, np.zeros(n, int))
            scores = .5*((reference_at_query*query_at_query).sum(-1)+(query_at_reference*reference_at_reference.expand(n, -1)).sum(-1))
        return scores.cpu().numpy()

    def __call__(self, i, c, query_xy, trial_pose, sun_indices):
        self._check_open(); self.cost['refresh_requests'] += 1
        if trial_pose is None: return None
        sun_indices = np.asarray(sun_indices, int)
        # Original exact score key is retained; only lower-level pure inputs
        # omit Sun tuples. No rounded pose, location, or lighting cache key.
        key = (self.identity.digest, int(i), int(c), np.asarray(query_xy, float).tobytes(),
            np.asarray(trial_pose, float).tobytes(), tuple(sun_indices))
        if key not in self.cache:
            selected = self.graph_sun_ids[sun_indices]
            # Original response_evidence validates the selected global columns
            # after canonical support, so keep the same ordering for None cases.
            pair = self._canonical_record(i, c, query_xy, trial_pose)
            if pair is None: value = None
            else:
                context = self._context(i, trial_pose)
                if selected.ndim != 1 or not len(selected) or (selected < 0).any() or (selected >= len(self.geometry.sun_vectors)).any():
                    raise ValueError('Explicit valid target Sun columns required')
                value = self._scores(i, selected, self._pair_features(pair), context)
            self._put(self.cache, key, value, self.capacity)
        else:
            self.cost['cache_hits'] += 1; self.cache.move_to_end(key)
        value = self.cache[key]
        return None if value is None else value.copy()
