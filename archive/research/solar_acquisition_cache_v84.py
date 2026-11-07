"""Unreferenced, per-case memoization of the frozen acquisition projection.

No solver/frontend integration or global patch is performed here. The cached
block preserves the original expanded action shape, einsum, order and dtypes.
Only exact repeated projections are reused; query measurements, solar choices
and hypothesis weights are recomputed after every acquisition.
"""
from collections import OrderedDict

import numpy as np

from solar_physics_v63 import SolarState


class AcquisitionProjectionCacheV84:
    """Bounded exact projection cache owned by ONE case/copied solver call.

    Construct after the solver has copied its CandidateGraph. World geometry,
    action identities and intrinsics must remain unchanged for its lifetime.
    Mutable query coordinates/solar evidence/beam states are deliberately not
    frozen. A later case receives a new instance, never a process-global cache.
    """

    def __init__(self, graph, actions, intrinsics, capacity=4):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Positive integer projection-cache capacity required")
        self._graph = graph
        self._actions_source = actions
        self._intrinsics_source = intrinsics
        self._world = np.array(graph.world_variants, copy=True)
        self._actions = np.asarray(actions, int).copy()
        self._intrinsics = np.asarray(intrinsics).copy()
        self.capacity = capacity
        self._entries = OrderedDict()
        self._requests = 0
        self._projection_computations = 0
        self._projection_hits = 0
        self._evictions = 0

    def assert_unchanged(self, graph=None, actions=None, intrinsics=None):
        graph = self._graph if graph is None else graph
        actions = np.asarray(self._actions_source if actions is None else actions, int)
        intrinsics = np.asarray(self._intrinsics_source if intrinsics is None else intrinsics)
        if graph is not self._graph:
            raise ValueError("Projection cache cannot cross case/graph instances")
        world = np.asarray(graph.world_variants)
        if (world.dtype != self._world.dtype or world.shape != self._world.shape
                or not np.array_equal(world, self._world)):
            raise ValueError("Reference world variants changed during acquisition")
        if (actions.dtype != self._actions.dtype or actions.shape != self._actions.shape
                or not np.array_equal(actions, self._actions)):
            raise ValueError("Action identities changed during acquisition")
        if (intrinsics.dtype != self._intrinsics.dtype or intrinsics.shape != self._intrinsics.shape
                or not np.array_equal(intrinsics, self._intrinsics)):
            raise ValueError("Query intrinsics changed during acquisition")

    def projection(self, graph, beams, actions, intrinsics):
        """Return original xy/front arrays; only the exact static block is cached."""
        self.assert_unchanged(graph, actions, intrinsics)
        actions = np.asarray(actions, int)
        i, c = actions.T
        # Keep these expressions byte-for-byte equivalent to the frozen block.
        poses = np.stack([np.eye(4) if b['pose'] is None else b['pose'] for b in beams])
        live = np.array([b['pose'] is not None for b in beams])
        key = (poses.dtype.str, poses.shape, poses.tobytes(), live.tobytes())
        self._requests += 1
        if key in self._entries:
            self._projection_hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        world = graph.world_variants[i]
        camera = np.einsum('havc,hcd->havd', world[None]-poses[:, None, None, :3, 3],
                           poses[:, :3, :3], optimize=True)
        homogeneous = camera @ np.asarray(intrinsics).T
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = homogeneous[..., :2]/homogeneous[..., 2, None]
        front = live[:, None, None] & (camera[..., 2] > 0) & np.isfinite(xy).all(-1)
        # Downstream arithmetic reads these arrays. Protect retained cache bytes.
        xy.setflags(write=False)
        front.setflags(write=False)
        self._entries[key] = (xy, front)
        self._projection_computations += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._evictions += 1
        return xy, front

    @property
    def stats(self):
        projection_bytes = sum(xy.nbytes+front.nbytes for xy, front in self._entries.values())
        return dict(revision="acquisition-projection-cache-v84", requests=self._requests,
                    projection_computations=self._projection_computations,
                    projection_cache_hits=self._projection_hits, evictions=self._evictions,
                    entries=len(self._entries), capacity=self.capacity,
                    resident_projection_bytes=projection_bytes,
                    immutable_guard_snapshot_bytes=(self._world.nbytes+self._actions.nbytes
                                                    + self._intrinsics.nbytes))

    def finish(self):
        self.assert_unchanged()
        return self.stats


def acquisition_predictions_v84(graph, beams, actions, intrinsics,
                                solar_mode='shared', cache=None):
    """Frozen prediction arithmetic with optional exact per-case projection reuse.

    Future copied-solver integration supplies one cache instance across calls.
    Nothing installs this function into the existing solver automatically.
    """
    actions = np.asarray(actions, int)
    i, c = actions.T
    h = len(beams)
    if cache is None:
        cache = AcquisitionProjectionCacheV84(graph, actions, intrinsics)
    xy, front = cache.projection(graph, beams, actions, intrinsics)
    # Everything below depends on CURRENT mutable evidence/assignments/weights.
    delta = (xy-graph.query_xy[i, c][None, :, None, :])/16.
    predictions = np.full((h, len(actions), 7, 4), np.nan)
    predictions[..., :2] = np.where(front[..., None], delta, np.nan)
    for j, b in enumerate(beams):
        independent = b['assignment'].get('independent_sun_indices')
        sun = np.full(len(i), b['sun_index'], int) if independent is None else independent[i]
        states = graph.solar_variants[i, sun]
        if solar_mode != 'off' and not graph.solar_materialized_mask[sun].all():
            raise ValueError('Active solar predictions require computed rays')
        observable = front[j] & ((xy[j] >= 0) & (xy[j] <= 511)).all(-1) & (states != 0) & (solar_mode != 'off')
        predictions[j, :, :, 2] = np.where(observable, (states == SolarState.SHADOW).astype(float), np.nan)
    predictions[:, :, :, 3] = np.where(front, np.exp(-.5*np.sum(delta*delta, -1)), np.nan)
    scores = np.asarray([b['score'] for b in beams])
    weights = np.exp(scores-scores.max())
    return predictions, weights, np.isfinite(predictions)
