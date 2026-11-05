"""Exact v117 footprint z-buffer with one reduction per footprint origin."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

from lunar_geometry_v5 import project_z


ORIGINAL_SHA256 = "86bb0a8e56da2a1cf1ed161f39454dc50a53ce715f1719bd55e8427e7d24de95"


def _load_original():
    # Use the same module identity as the v117 worker and its ReferenceCloud.
    name = "warp_candidates"
    module = sys.modules.get(name)
    source = (Path(module.__file__) if module is not None else
              Path(__file__).resolve().parents[1] / "solar-warp-v117" / "warp_candidates.py")
    if sha256(source.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Exact frozen v117 projection and ranking source required")
    if module is None:
        spec = spec_from_file_location(name, source)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


original = _load_original()


def _origin_winners(bins, source, depth, shape):
    height, width = shape
    bucket = bins[:, 1] * width + bins[:, 0]
    nearest = np.full(shape, np.inf)
    np.minimum.at(nearest.ravel(), bucket, depth)
    at_minimum = depth == nearest.ravel()[bucket]
    # v117 supports at most 4096 points, so the sentinel fits in its int16
    # output and is larger than every original source index.
    winners = np.full((height, width), original.MAX_SOURCE_SAMPLES, dtype=np.int16)
    np.minimum.at(winners.ravel(), bucket[at_minimum], source[at_minimum])
    return nearest, winners


def _merge_origins(nearest, winners):
    """Each detector bin sees the four possible footprint origins beside it."""
    depth = nearest[:-1, :-1].copy()
    result = winners[:-1, :-1].copy()
    for ys, xs in ((slice(1, None), slice(None, -1)),
                   (slice(None, -1), slice(1, None)),
                   (slice(1, None), slice(1, None))):
        other_z, other_source = nearest[ys, xs], winners[ys, xs]
        better = (other_z < depth) | ((other_z == depth) & (other_source < result))
        np.copyto(depth, other_z, where=better)
        np.copyto(result, other_source, where=better)
    return result


def detector_winners(world, intrinsics, pose, shape, stride, *, footprint=False):
    """Return byte-identical source indices using the original axial projection."""
    height, width = shape
    world = np.asarray(world, float)
    if (world.ndim != 2 or world.shape[1] != 3 or len(world) > original.MAX_SOURCE_SAMPLES
            or type(stride) is not int or stride <= 0 or height % stride or width % stride):
        raise ValueError("Detector stride must divide both detector dimensions")
    xy, z, front = project_z(world, intrinsics, pose)
    inside = (front & np.isfinite(xy).all(axis=1) & (xy >= -.5).all(axis=1)
              & (xy < [width - .5, height - .5]).all(axis=1))
    source = np.flatnonzero(inside)
    low_shape = height // stride, width // stride
    if footprint:
        # Exact v117 origin formula. The one-cell padding includes origin -1.
        # Every sample in one origin covers the same (possibly clipped) bins,
        # so only its closest sample can win in any of the four output bins.
        origins = np.floor((xy[source] + .5) / stride - .5).astype(int) + 1
        padded = low_shape[0] + 1, low_shape[1] + 1
        depth, winners = _origin_winners(origins, source, z[source], padded)
        result = _merge_origins(depth, winners)
    else:
        bins = np.floor((xy[source] + .5) / stride).astype(int)
        _, result = _origin_winners(bins, source, z[source], low_shape)
    result[result == original.MAX_SOURCE_SAMPLES] = -1
    return result


class FastProjectionBank(original.ProjectionBank):
    """Drop-in v117 cache, preserving its reference identity checks and reuse."""

    def get(self, index):
        if index not in self.indices:
            self.indices[index] = tuple(
                detector_winners(world, self.intrinsics, self.poses[index], self.shape, self.stride, footprint=True)
                for world in self.worlds
            )
        return self.indices[index]


def rank_pose_bank(cloud, query_luminance, query_intrinsics, poses, *, cell_pixels=32,
                   detector_stride=4, region=None, min_cells_per_quadrant=2,
                   min_occupancy=.125, min_spatial_rms=1e-9, evidence=None, projection_bank=None):
    """Run the frozen ranker with exact faster projection when none is supplied."""
    if projection_bank is None and isinstance(cloud, original.ReferenceCloud):
        projection_bank = FastProjectionBank(cloud, query_intrinsics, poses,
                                             np.shape(query_luminance), detector_stride)
    return original.rank_pose_bank(
        cloud, query_luminance, query_intrinsics, poses, cell_pixels=cell_pixels,
        detector_stride=detector_stride, region=region, min_cells_per_quadrant=min_cells_per_quadrant,
        min_occupancy=min_occupancy, min_spatial_rms=min_spatial_rms, evidence=evidence,
        projection_bank=projection_bank,
    )
