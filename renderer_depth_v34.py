"""Explicit depth semantics from the already measured analytic renderer controls."""
import hashlib
import json
from pathlib import Path

import numpy as np


def control_convention():
    path=Path('reports/north_camera_control_summary_v34.json')
    audit=json.loads(Path('reports/north_camera_control_audit_v31.json').read_text())
    if not audit['passed']or hashlib.sha256(path.read_bytes()).hexdigest()!=audit['renderer_summary_sha256']:
        raise ValueError('Actual independently audited camera-control summary required')
    control=json.loads(path.read_text());conventions={r['depth_convention']for r in control['records']}
    if not control['passed']or len(control['records'])!=2 or len(conventions)!=1:
        raise ValueError('Both actual analytic camera controls must agree')
    convention=conventions.pop()
    if convention not in ('axial','ray'):raise ValueError('Unknown measured renderer convention')
    return convention


def canonical_depth(position,pose,renderer_depth,convention):
    """Validate the raw pass, returning separate camera-Z and Euclidean range.

The convention comes from frozen analytic controls, never from matcher scores
or choosing whichever definition happens to fit the pilot.
"""
    position=np.asarray(position,dtype=float);pose=np.asarray(pose,dtype=float)
    raw=np.asarray(renderer_depth,dtype=float)
    if (position.ndim!=3 or position.shape[-1]!=3 or raw.shape!=position.shape[:2]
            or pose.shape!=(4,4) or not np.isfinite(position).all()or not np.isfinite(pose).all()
            or not np.isfinite(raw).all()):raise ValueError('Finite compatible geometry required')
    delta=position-pose[:3,3];axial=(delta@pose[:3,:3])[...,2];ray=np.linalg.norm(delta,axis=-1)
    if (axial<=0).any()or (ray<=0).any()or (raw<=0).any():raise ValueError('All pilot rays must hit forward terrain')
    if convention=='axial':expected=axial
    elif convention=='ray':expected=ray
    else:raise ValueError('Explicit measured depth convention required')
    relative=np.abs(raw-expected)/expected
    if relative.max()>.0005:raise ValueError('Raw renderer pass disagrees with fixed analytic convention')
    return axial,ray,float(relative.max())
