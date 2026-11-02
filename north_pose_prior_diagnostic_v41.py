"""Metadata-only diagnostic of the fixed northern camera design.

Fits training-camera templates without query-view IDs. The best-of-bank result
is explicitly an ORACLE using query truth for scoring, never a usable matcher.
No image/depth array is read and no neural model or pose estimator is run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from north_development_plan_v35 import development_plan
from north_render_plan_v31 import camera_to_world


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_cameras(plan, sites, acquired):
    records={r['component']:r for r in acquired['records']};rows=[]
    for site in sites:
        clearance=records[site['component']]['core_max_local_height_m'];cameras=[]
        for spec in plan['camera_specs']:
            position=np.asarray(spec['position_offset'],float).copy();position[2]+=clearance
            cameras.append(np.asarray(camera_to_world(position.tolist(),spec['target'])))
        for query in cameras[1:]:
            rows.append(dict(component=site['component'],split=site['split'],
                reference=cameras[0],query=query))
    return rows


def fit_training_templates(rows):
    """All template fitting uses training extrinsics; no image or view identity."""
    clusters={}
    for row in rows:
        if row['split']!='training':
            raise ValueError('Template fitting must receive training metadata only')
        a,b=np.asarray(row['reference']),np.asarray(row['query'])
        if not np.allclose(a[:3,:3],np.diag([1.,-1.,-1.]),atol=1e-10,rtol=0):
            raise ValueError('This diagnostic is restricted to the fixed nadir reference design')
        if not np.allclose(a[:2,3],0,atol=1e-10,rtol=0)or abs(b[2,2])<1e-6:
            raise ValueError('Camera design differs from declared reference/target-plane assumptions')
        offset=b[:3,3]-a[:3,3]
        target=b[:3,3]-b[2,3]/b[2,2]*b[:3,2]
        key=tuple(np.round(offset,6))
        clusters.setdefault(key,[]).append((offset,target))
    if len(clusters)!=3:
        raise ValueError('Expected exactly three training-derived camera translation templates')
    bank=[]
    for key,values in sorted(clusters.items()):
        bank.append(dict(position_offset=np.median([v[0]for v in values],axis=0).tolist(),
            target=np.median([v[1]for v in values],axis=0).tolist(),training_examples=len(values)))
    return bank


def proposals(reference,bank):
    """Inference accepts only the given reference pose and fitted template bank."""
    a=np.asarray(reference,float)
    return [np.asarray(camera_to_world((a[:3,3]+np.asarray(t['position_offset'])).tolist(),t['target']))
            for t in bank]


def errors(predicted,truth):
    rotation=np.degrees(np.arccos(np.clip((np.trace(predicted[:3,:3].T@truth[:3,:3])-1)/2,-1,1)))
    translation=np.linalg.norm(predicted[:3,3]-truth[:3,3])
    return float(rotation),float(translation)


def evaluate_metadata(rows,bank):
    outcomes=[]
    for row in rows:
        candidates=proposals(row['reference'],bank)
        scored=[errors(p,row['query'])for p in candidates]
        # Oracle choice is evaluation-only. It MUST NOT be reported as a matcher.
        winner=min(range(len(scored)),key=lambda i:(scored[i][1],scored[i][0]))
        angle,translation=scored[winner]
        single_angle,single_translation=scored[0]
        outcomes.append(dict(component=row['component'],split=row['split'],
            oracle_template_index=winner,oracle_rotation_deg=angle,oracle_translation_m=translation,
            oracle_near_exact=angle<=1e-4 and translation<=1e-6,
            first_fixed_template_close=single_angle<=2 and single_translation<=.01))
    return outcomes


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve every prior diagnostic')
    source=Path('data/north-render-v31/development-v35/summary.json')
    plan,sites=development_plan();acquired=json.loads(source.read_text())
    if not acquired['complete']or len(acquired['records'])!=24:
        raise ValueError('Complete acquired development metadata required')
    rows=metadata_cameras(plan,sites,acquired)
    train=[r for r in rows if r['split']=='training']
    held=[r for r in rows if r['split']=='model_selection']
    if len(train)!=48 or len(held)!=24:raise ValueError('Original development split required')
    bank=fit_training_templates(train);outcomes=evaluate_metadata(held,bank)
    result=dict(scope='Metadata-only camera-design bias diagnostic. Best-of-three is an oracle, not image-based localization or verified renderer accuracy.',
        passed=True,source_sha256=sha(__file__),acquired_summary_sha256=sha(source),
        plan_source_sha256=sha('north_render_plan_v31.py'),template_training_geometries=48,
        template_training_components=16,selection_geometries=24,selection_components=8,
        learned_templates=bank,oracle_near_exact_geometries=sum(r['oracle_near_exact']for r in outcomes),
        first_fixed_template_close_geometries=sum(r['first_fixed_template_close']for r in outcomes),
        maximum_oracle_rotation_deg=max(r['oracle_rotation_deg']for r in outcomes),
        maximum_oracle_translation_m=max(r['oracle_translation_m']for r in outcomes),
        outcomes=outcomes,images_or_depth_arrays_loaded=0,neural_forwards=0,
        query_view_or_case_ids_used_in_proposals=False,query_pose_used_for_oracle_scoring=True,
        calibration_or_test_metadata_loaded=0,calibration_or_test_pixels_loaded=0,
        inference_claim=False,query_pose_generalization_established=False,
        interpretation='The known reference pose and three training-derived templates can reconstruct intended selection camera poses. Broad unseen-pose claims need a separately frozen camera/roll transfer study and an image-selected pose-prior comparator.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('outcomes','learned_templates')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/north_pose_prior_diagnostic_v41.json');a=p.parse_args();run(a.output)
