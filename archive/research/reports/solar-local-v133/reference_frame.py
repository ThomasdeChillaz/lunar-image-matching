"""Use the same frozen AP3P solver in the reference camera's coordinate frame.

This rigid change of coordinates does not change geometry or image endpoints.
It avoids solving an almost pi-rotation on the generated horizontal-plane case.
Every matcher receives the same optional frame control; old outputs are retained.
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"solar-local-v132"))
import native_radiance as native


def solve(p, q, depth, k, reference_pose, query_k):
    p, q = np.asarray(p, float).reshape(-1, 2), np.asarray(q, float).reshape(-1, 2)
    # Form local points directly from the depth and calibrated rays. Avoid a
    # round trip through million-metre body coordinates and subtracting them.
    local, valid = native.control.pipeline.map_points_z(p, depth, k, np.eye(4))
    p, q = p[valid], q[valid]
    world = local@reference_pose[:3, :3].T+reference_pose[:3, 3]
    pose, info = native.control.pipeline.estimate(local, q, query_k, native.control.pipeline.SOLVER)
    result = None if pose is None else reference_pose@pose
    return dict(reference_xy=p.tolist(), query_xy=q.tolist(), reference_world=world.tolist(),
                pose=None if result is None else result.tolist(), estimator=info,
                estimator_coordinate_frame="reference_camera", estimator_threshold_native_px=3.)


def reestimate(row, depth, k, reference_pose, query_k):
    tick = perf_counter()
    result = solve(row["reference_xy"], row["query_xy"], depth, k, reference_pose, query_k)
    return dict(row, **result, inference_seconds=row["inference_seconds"]+perf_counter()-tick)


def run(reference, query, depth, k, reference_pose, query_k, seeds, library, **kwargs):
    result = native.run(reference, query, depth, k, reference_pose, query_k, seeds, library, **kwargs)
    result["methods"].update({"framed_"+name: reestimate(row, depth, k, reference_pose, query_k)
                              for name, row in list(result["methods"].items())})
    return result
