"""Acquire only the prospectively assigned16training/eightselection supports.

Transport/meshing reuse the validated pilot implementations. Calibration/final
windows and the four excluded pilots cannot be acquired by this entry point.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from north_render_plan_v31 import SOURCE
from north_terrain_pilot_v32 import (PLAN_SHA, source_identity, sha,
                                     local_vertices, grid_faces)
from download_north_source_v33 import EXPECTED_BYTES, EXPECTED_ETAG
from cache_north_development_v35 import verify_cache
from north_development_plan_v35 import development_plan,prerequisite_gate


def acquire(output, source_root):
    import rasterio
    from rasterio.windows import Window
    gates=prerequisite_gate();plan, sites = development_plan()
    if not json.loads(Path('reports/north_camera_control_audit_v31.json').read_text())['passed']:
        raise ValueError('Actual renderer/independent pose controls required')
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve previous acquisition or failure')
    root.mkdir(parents=True)
    local = Path(source_root)
    source_proof = verify_cache(local)
    identity = source_identity()
    if identity['ETag'] != EXPECTED_ETAG or int(identity['Content-Length']) != EXPECTED_BYTES:
        raise ValueError('Source version changed from failed remote acquisition')
    protocol = dict(plan_sha256=PLAN_SHA, source_url=SOURCE, source_identity=identity, development_range_cache_summary_sha256=sha(local/'summary.json'), complete_source_download=False,
                    components=sites, source_sha256={p: sha(p) for p in
                    ['acquire_north_development_v35.py', 'north_development_plan_v35.py', 'north_terrain_pilot_v32.py', 'north_render_plan_v31.py', 'download_north_source_v33.py', 'cache_north_development_v35.py']},
                    prerequisite_sha256=gates,scope='Only original16training/eightselection supports; no calibration/final/pilot pixels or model evaluation.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    try:
        with rasterio.Env(GDAL_NUM_THREADS='2', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF'):
            with rasterio.open(local/'development-only.tif') as src:
                if (src.shape != (43280, 43280) or src.count != 1 or src.dtypes != ('float32',)
                        or list(src.transform)[:6] != plan['source_grid_transform']):
                    raise ValueError('Remote terrain header differs from prospective source')
                for site in sites:
                    row, col, h, w = site['native_window']
                    height = src.read(1, window=Window(col, row, w, h))
                    if height.shape != (2049, 2049) or not np.isfinite(height).all():
                        raise ValueError('Development source has missing terrain; do not replace it')
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
    p.add_argument('--output', default='data/north-render-v31/development-v35')
    p.add_argument('--source-root', default='data/north-render-v31/range-cache-dev-v35')
    a=p.parse_args(); acquire(a.output,a.source_root)
