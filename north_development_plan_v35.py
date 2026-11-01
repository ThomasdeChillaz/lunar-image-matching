"""Only the 16 training and eight selection sites from the original split."""
import json
from pathlib import Path

import numpy as np

from north_terrain_pilot_v32 import pilot_plan,sha


def development_plan():
    plan,_=pilot_plan()  # Verifies the immutable, publicly assigned 44-site plan.
    sites=[s for s in plan['sites'] if s['split'] in ('training','model_selection')]
    counts={k:sum(s['split']==k for s in sites)for k in ('training','model_selection')}
    if counts!={'training':16,'model_selection':8}:
        raise ValueError('Exactly the original development components are required')
    return plan,sites


def prerequisite_gate():
    if sha('reports/north_terrain_pilot_audit_v34.json')!='e55eda8f9dd89ed237c4cb5b9e83193573a1ef5840c1e599e715151c3976672f':raise ValueError('Public M34 prerequisite changed')
    if sha('reports/north_historical_exposure_v34.json')!='c9c7be7803286337ec42aae29c369a802ae44602bcdc6b08aedabc48786ab0a4':raise ValueError('Public M34 prerequisite changed')
    pilot=Path('reports/north_terrain_pilot_audit_v34.json')
    audit=json.loads(pilot.read_text())
    if (not audit['passed']or audit['correct_oracle_poses']!=12 or audit['eligible_geometry_pairs']!=12
            or audit['measured_raw_depth_convention']!='axial'
            or audit['full44component_render_linear_projection_seconds']>21600):
        raise ValueError('Actual terrain/depth/oracle/cost pilot required')
    exposure=Path('reports/north_historical_exposure_v34.json');e=json.loads(exposure.read_text())
    if not e['passed']or not e['own_project_historical_pixel_supports_disjoint']or e['probe_support_intersections']:
        raise ValueError('Named historical project input separation required')
    return {str(pilot):sha(pilot),str(exposure):sha(exposure)}


def quarter_turn_points(points,k,size=512):
    """Native integer-centred coordinates under np.rot90, including fractions."""
    q=np.asarray(points,dtype=float).copy()
    if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all():
        raise ValueError('Finite x,y point coordinates required')
    for _ in range(k%4):q=np.column_stack([q[:,1],size-1-q[:,0]])
    return q


def proposal_grid_native():
    y,x=np.mgrid[12:116:4,12:116:4]
    return (np.column_stack([x.ravel(),y.ravel()]).astype(float)+.5)*4-.5


def world_rays(points,intrinsics,pose):
    p=np.asarray(points,float);k=np.asarray(intrinsics,float);pose=np.asarray(pose,float)
    camera=np.column_stack([(p[:,0]-k[0,2])/k[0,0],(p[:,1]-k[1,2])/k[1,1],np.ones(len(p))])
    rays=camera@pose[:3,:3].T
    return rays/np.linalg.norm(rays,axis=1)[:,None]
