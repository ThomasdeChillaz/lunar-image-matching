"""Acquire only the four prospectively assigned terrain/cost pilots.

No calibration, selection or test rasters may be read by this entry point.
Source-window hashes identify acquired samples, not the complete remote TIFF.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from north_render_plan_v31 import RADIUS, SOURCE, enu_frame

PLAN = Path('reports/north_render_prospective_plan_v31.json')
PLAN_SHA = '1487368cc7c81df3e539522e49fce5b911d05a74c3c3eca29b4bd919ae03e73b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pilot_plan():
    if sha(PLAN) != PLAN_SHA:
        raise ValueError('Prospective terrain assignment changed')
    plan = json.loads(PLAN.read_text())
    sites = [s for s in plan['sites'] if s['split'] == 'renderer_cost_pilot']
    if [s['component'] for s in sites] != [f'np31-pilot-{i}' for i in range(4)]:
        raise ValueError('Only the four predeclared pilots are permitted')
    return plan, sites


def source_identity():
    with urlopen(Request(SOURCE, method='HEAD', headers={'User-Agent': 'lunar-terrain-study'}), timeout=60) as response:
        identity = {k: response.headers.get(k) for k in ('ETag', 'Last-Modified', 'Content-Length')}
    if not identity['ETag'] or not identity['Content-Length']:
        raise ValueError('Stable source identity headers required')
    return identity


def local_vertices(height, site, offsets):
    """Convert the declared source nodes through spherical body coordinates."""
    if height.shape != (2049, 2049) or not np.isfinite(height).all():
        raise ValueError('Complete valid source window required; no inpainting')
    indices = np.asarray(offsets, dtype=int) + 1024
    h = height[np.ix_(indices, indices)].astype(np.float64)
    x, y = np.meshgrid(site['x'] + np.asarray(offsets)*10., site['y'] - np.asarray(offsets)*10.)
    rho = np.hypot(x, y)
    latitude = np.pi/2 - 2*np.arctan(rho/(2*RADIUS))
    longitude = np.arctan2(x, -y)
    radius = RADIUS+h
    world = np.stack([radius*np.cos(latitude)*np.cos(longitude),
                      radius*np.cos(latitude)*np.sin(longitude), radius*np.sin(latitude)], -1)
    origin, rows = enu_frame(site['x'], site['y'], float(height[1024, 1024]))
    rows, origin = np.asarray(rows), np.asarray(origin)
    vertices = (world-origin) @ rows.T
    # Retain float64 body transforms; only local render vertices become float32.
    return vertices.astype(np.float32), origin, rows


def grid_faces(size):
    if size < 2:
        raise ValueError('At least two nodes required')
    a = np.arange((size-1)*size).reshape(size-1, size)[:, :-1].ravel()
    # Row direction is descending projected northing. Flip below if needed.
    return np.concatenate([np.stack([a, a+1, a+size], 1),
                           np.stack([a+1, a+size+1, a+size], 1)]).astype(np.int32)


def acquire(output):
    import rasterio
    from rasterio.windows import Window
    plan, sites = pilot_plan()
    if not json.loads(Path('reports/north_camera_control_audit_v31.json').read_text())['passed']:
        raise ValueError('Actual renderer/independent pose controls required')
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve previous acquisition or failure')
    root.mkdir(parents=True)
    identity = source_identity()
    protocol = dict(plan_sha256=PLAN_SHA, source_url=SOURCE, source_identity=identity,
                    components=sites, source_sha256={p: sha(p) for p in
                    ['north_terrain_pilot_v32.py', 'north_render_plan_v31.py']},
                    scope='Only four source-assigned pilots; no fresh benchmark or model evaluation.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    try:
        with rasterio.Env(GDAL_NUM_THREADS='2', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF'):
            with rasterio.open(SOURCE) as src:
                if (src.shape != (43280, 43280) or src.count != 1 or src.dtypes != ('float32',)
                        or list(src.transform)[:6] != plan['source_grid_transform']):
                    raise ValueError('Remote terrain header differs from prospective source')
                for site in sites:
                    row, col, h, w = site['native_window']
                    height = src.read(1, window=Window(col, row, w, h))
                    if height.shape != (2049, 2049) or not np.isfinite(height).all():
                        raise ValueError('Pilot source has missing terrain; do not replace it')
                    folder = root/site['component']; folder.mkdir()
                    np.save(folder/'source_height.npy', height, allow_pickle=False)
                    vertices, origin, rows = local_vertices(height, site, plan['mesh_native_offset_indices'])
                    flat = vertices.reshape(-1, 3); faces = grid_faces(vertices.shape[0])
                    a, b, c = flat[faces[:1_000]].transpose(1, 0, 2)
                    if np.cross(b-a, c-a)[:, 2].mean() < 0:
                        faces = faces[:, ::-1].copy()
                    core = np.abs(np.asarray(plan['mesh_native_offset_indices'])) <= 256
                    clearance = float(vertices[np.ix_(core, core)][:, :, 2].max())
                    np.savez_compressed(folder/'mesh.npz', vertices=flat, faces=faces,
                                        body_origin=origin, enu_rows=rows, core_max_height=clearance)
                    record = dict(component=site['component'], split=site['split'],
                                  source_height_sha256=sha(folder/'source_height.npy'),
                                  mesh_sha256=sha(folder/'mesh.npz'), native_window=site['native_window'],
                                  source_pixels=int(height.size), source_minimum_m=float(height.min()),
                                  source_maximum_m=float(height.max()), vertices=len(flat), triangles=len(faces),
                                  core_max_local_height_m=clearance)
                    records.append(record)
                    print(json.dumps(record), flush=True)
        if source_identity() != identity:
            raise ValueError('Remote source changed during acquisition; preserve outputs')
        result = dict(complete=True, records=records, protocol_sha256=sha(root/'protocol.json'),
                      publisher_full_file_checksum_verified=False,
                      source_height_pixels_decoded=sum(r['source_pixels'] for r in records),
                      calibration_or_test_pixels_decoded=0, fresh_test_gate_passed=False)
        (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    except Exception as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc),
                                                       completed_components=len(records)), indent=2)+'\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/north-render-v31/pilots-v32')
    acquire(p.parse_args().output)
