"""Procedural source closure/evidence/diagnostic tests; no actual model execution."""
import ast
import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE if (HERE/'solar_frontend_v63.py').is_file() else HERE.parents[1]/'outputs'/'lunar-image-matching'
if str(REPO) not in sys.path: sys.path.append(str(REPO))
sys.path.insert(0, str(HERE))
from solar_evaluation_v70 import (InferenceInput, checkpoint_profiles, checkpoint_ranking, completed_training_evidence,
    evaluate_case, sha, validate_profile_training)
from run_solar_evaluation_v70 import REQUIRED_SOURCES


def write(path, value): path.write_text(json.dumps(value)); return path


def fixture(tmp_path):
    root = tmp_path/'run'; root.mkdir(); protocol_path=tmp_path/'training.json'; receipt_path=tmp_path/'watchdog.json'
    started=datetime(2026, 11, 4, tzinfo=timezone.utc)
    protocol=dict(updates=2000, seed=20261101, pooled=False, output=str(root), max_calendar_seconds=40,
        absolute_deadline_utc=(started+timedelta(seconds=60)).isoformat(), watchdog_receipts=dict(run=str(receipt_path)),
        source_sha256={'run_solar_job_v70.py':sha(REPO/'run_solar_job_v70.py')})
    write(protocol_path, protocol)
    checkpoints={}
    for step in (0,100,500,1000,1500,2000):
        path=root/f'step_{step}.pt'; path.write_bytes(f'PROCEDURAL CHECKPOINT IDENTITY ONLY {step}'.encode()); checkpoints[path.name]=sha(path)
    history=write(root/'history_2000.json', {'procedural':True})
    summary=dict(complete=True, updates=2000, pooled=False, seed=20261101, protocol_sha256=sha(protocol_path),
        checkpoint_sha256=checkpoints, all_history_sha256=sha(history), calendar_seconds=9.)
    write(root/'summary.json', summary)
    receipt=dict(complete=True, mode='run', program='train_solar_response_v63.py', child_returncode=0, failure=None,
        protocol_sha256=sha(protocol_path), source_sha256=protocol['source_sha256']['run_solar_job_v70.py'],
        started_utc=started.isoformat(), absolute_deadline_utc=(started+timedelta(seconds=40)).isoformat(), cap_seconds=40, calendar_seconds=10.)
    write(receipt_path, receipt)
    return root, protocol_path, receipt_path, protocol, summary, receipt


def test_all_five_checkpoints_including_initialization_need_completed_outer_watchdog(tmp_path):
    root, protocol, receipt, _, _, _=fixture(tmp_path)
    profiles=checkpoint_profiles(root, training_protocol=protocol)
    assert [p['training_step'] for p in profiles]==[0,500,1000,1500,2000]
    for p in profiles: validate_profile_training(p,'checkpoint_selection')
    assert profiles[0]['training_step']==0
    with pytest.raises(ValueError, match='external watchdog evidence'):
        checkpoint_profiles(root)


def test_completed_study_can_honestly_select_initialization_as_negative_finding(tmp_path):
    root, protocol, _, _, _, _=fixture(tmp_path)
    profiles=checkpoint_profiles(root,training_protocol=protocol)
    rows=[dict(profile=p['name'],case_id='same_eligible_case',track='estimated_sun',positive=True,
        metrics=dict(correct=p['training_step']==0,correspondence={'2':dict(correct=5)})) for p in profiles]
    result=checkpoint_ranking(rows,profiles,['same_eligible_case'])
    assert result['selected']['step']==0 and result['automatically_promoted'] is False


@pytest.mark.parametrize('mutation', ['failed_watchdog','late_watchdog','other_protocol','other_program','other_source','failure_record','late_summary'])
def test_summary_complete_cannot_promote_late_or_unrelated_checkpoint(tmp_path, mutation):
    root, protocol_path, receipt_path, protocol, summary, receipt=fixture(tmp_path)
    if mutation=='failed_watchdog': receipt['complete']=False
    elif mutation=='late_watchdog': receipt['calendar_seconds']=41
    elif mutation=='other_protocol': receipt['protocol_sha256']='0'*64
    elif mutation=='other_program': receipt['program']='other.py'
    elif mutation=='other_source': receipt['source_sha256']='0'*64
    elif mutation=='failure_record': receipt['failure']={'error':'timeout'}
    elif mutation=='late_summary': summary['calendar_seconds']=11
    write(receipt_path,receipt); write(root/'summary.json',summary)
    with pytest.raises(ValueError): checkpoint_profiles(root,training_protocol=protocol_path)


def test_hash_pinned_evidence_and_checkpoint_identity_are_checked(tmp_path):
    root, protocol, receipt, _, _, _=fixture(tmp_path)
    profile=checkpoint_profiles(root,training_protocol=protocol)[0]
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        completed_training_evidence(profile)


def test_initialization_cost_control_is_explicit_and_never_promotable():
    p=dict(backend='solar',initialization_only_cost_control=True)
    validate_profile_training(p,'cost_pilot')
    for purpose in ('checkpoint_selection','principal_ablations','additional_controls','modern_comparison'):
        with pytest.raises(ValueError,match='restricted to cost pilots'): validate_profile_training(p,purpose)
    for purpose in ('cost_pilot','checkpoint_selection','principal_ablations','additional_controls'):
        with pytest.raises(ValueError,match='external watchdog evidence'):
            validate_profile_training(dict(backend='solar'),purpose)
    validate_profile_training(dict(backend='sift'),'additional_controls')


def test_required_source_closure_contains_new_physics_rotation_and_imported_local_dependencies():
    required={'solar_rotation_v71.py','solar_sun_search_v68.py','solar_lookup_v70.py','solar_lookup_worker_v70.py',
        'solar_map_v63.py','dataset.py','matchers.py','geometry.py','north_terrain_pilot_v32.py','north_render_plan_v31.py',
        'localization_diagnostics_v3.py','lunargeo_dataset.py','remote_archive.py','train_balanced_localization_v3.py',
        'localization_refiner_v3.py','prepare_localization_training_v3.py'}
    assert required<=REQUIRED_SOURCES
    def top_level_imports(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): return
        if isinstance(node,ast.Import):
            for name in node.names: yield name.name.split('.')[0]+'.py'
        if isinstance(node,ast.ImportFrom) and node.module: yield node.module.split('.')[0]+'.py'
        for child in ast.iter_child_nodes(node): yield from top_level_imports(child)
    for name in REQUIRED_SOURCES:
        path=HERE/name if (HERE/name).exists() else REPO/name
        for child in top_level_imports(ast.parse(path.read_text())):
            if (REPO/child).exists(): assert child in REQUIRED_SOURCES, (name,child)


def test_requested_response_with_no_usable_pose_is_not_reported_as_learned_evidence():
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    provider=SimpleNamespace(stats=dict(requests=3,computed_columns=5,cold_start_seconds=1.,physical_key='same'))
    inp=InferenceInput(np.zeros((512,512),np.uint8),np.zeros((512,512),np.uint8),SimpleNamespace(depth=np.full((512,512),10.),intrinsics=k),k,np.array([1.,0,0]),None,provider)
    class Backend:
        profile=dict(configuration=dict(response=True))
        def infer(self, data):
            provider.stats.update(requests=4,computed_columns=7)
            return dict(pose=None,reference_xy=[],query_xy=[],reason='no_pose_hypothesis',response_refresh_calls=0,
                rounds=[dict(round=1,live_poses=0,sun_indices=[np.int64(2)])],hypotheses=[dict(pose=None),dict(pose=None)])
    row,_=evaluate_case(Backend(),inp,dict(positive=False),lambda c,p:(dict(positive=False),'procedural'),lambda p,q,i:(None,dict(reason='insufficient_matches')))
    assert row['response_requested'] is True and row['response_usage_confirmed'] is None
    assert row['response_refresh_calls']==0 and row['returned_hypothesis_count']==2 and row['live_hypothesis_count']==0
    assert row['reason']=='no_pose_hypothesis' and row['rounds'][0]['sun_indices']==[2]
    assert row['solar_lookup_stats']['numeric_delta']['requests']==1
    assert row['solar_lookup_stats']['numeric_delta']['computed_columns']==2
    assert row['solar_lookup_stats']['numeric_delta']['cold_start_seconds']==0
    json.dumps(row,allow_nan=False)
