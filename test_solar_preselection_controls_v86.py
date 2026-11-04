"""Synthetic metadata-only controls; no images, models or scientific jobs."""
import copy
from datetime import timedelta
from pathlib import Path

import pytest

import solar_evaluation_protocol_v74 as freeze
from test_solar_evaluation_protocol_v74 import workspace,write,NOW,completed_cost_pilot


def controls(tmp_path,monkeypatch,quantized=True):
    settings,contract,manifest,journal=workspace(tmp_path,monkeypatch)
    positives=[]
    for case in manifest['cases']:
        if case['positive'] and (case['light_a'],case['light_b'])==(0,0) and all(case['component']!=p['component']for p in positives):
            positives.append(case)
        if len(positives)==4:break
    settings.update(mode=freeze.PRESELECTION_MODE,include_quantization_aware_control=quantized,
        pilot_case_ids=[key for p in positives for key in(p['case_id'],p['case_id']+'/negative')],
        output_directory='runs/preselection-controls')
    write('metadata/settings.json',settings)
    return settings,contract,manifest,journal


def test_preselection_builds_only_frozen_native_controls_without_selection_or_pooled_study(tmp_path,monkeypatch):
    s,c,m,j=controls(tmp_path,monkeypatch)
    monkeypatch.setattr(freeze.learning,'selection',lambda *a:pytest.fail('Frozen controls requested learned checkpoint selection'))
    Path('reports/learning/pooled_study.json').unlink()
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/frozen-controls.json',NOW)
    assert p['purpose']=='cost_pilot' and p['target_purpose']=='additional_controls'
    assert len(p['profiles'])==4 and len(p['case_ids'])==384 and len(p['pilot_case_ids'])==8
    assert p['profiles'][0]['name']=='SIFT2048' and p['profiles'][0]['backend']=='sift'
    solar=p['profiles'][1:]
    assert [r['configuration']['pose_initialization']for r in solar]==['strict3','strict3','quantization_aware']
    assert [(r['configuration']['solar_mode'],r['configuration']['active'])for r in solar]==[('off',False),('shared',True),('shared',True)]
    assert all(r['training_step']==0 and r['training_seed']==20261101 and Path(r['checkpoint']).name=='step_0.pt'for r in solar)
    assert all(not r['configuration']['response']and not r['configuration']['pooled'] and r['configuration']['budget']==128 and r['configuration']['proposal_angles']==[0.,90.,180.,270.]for r in solar)
    assert solar[-1]['configuration']['inference_revision']=='solar-v73-quantized-init'
    assert {r['checkpoint']for r in solar}<=set(p['input_sha256'])
    assert not any('/step_500.pt'in k or 'selection.json'in k for k in p['input_sha256'])
    assert p['automatic_model_promotion']is False and p['accepted_results']is False
    assert not Path(p['output']).exists()


def test_optional_quantized_control_is_explicit_and_cache_revision_matches_policy(tmp_path,monkeypatch):
    s,c,m,j=controls(tmp_path,monkeypatch,False)
    s['inference_configuration']['inference_revision']='solar-v84-exact-cache'
    profiles,_,selection=freeze.profiles_from_artifacts(s,c)
    assert selection is None and len(profiles)==3
    assert all(p['configuration']['inference_revision']=='solar-v84-exact-cache'for p in profiles if p['backend']=='solar')
    s['include_quantization_aware_control']=True
    profiles,_,_=freeze.profiles_from_artifacts(s,c)
    assert profiles[-1]['configuration']['inference_revision']=='solar-v84-quantized-init-exact-cache'


@pytest.mark.parametrize('defect',['missing_option','boolean_option','sixteen_rotations','quantized_base','oversized_pilot','secondary_track'])
def test_preselection_cannot_expand_its_profile_or_execution_contract(tmp_path,monkeypatch,defect):
    s,c,m,j=controls(tmp_path,monkeypatch)
    if defect=='missing_option':s.pop('include_quantization_aware_control')
    elif defect=='boolean_option':s['include_quantization_aware_control']=1
    elif defect=='sixteen_rotations':s['inference_configuration']['proposal_angles']=[float(x)*22.5 for x in range(16)]
    elif defect=='quantized_base':s['inference_configuration'].update(pose_initialization='quantization_aware',inference_revision='solar-v73-quantized-init')
    elif defect=='oversized_pilot':s['pilot_cap_seconds']=601
    else:s['tracks']=['estimated_sun','known_sun']
    with pytest.raises(ValueError):freeze.profiles_from_artifacts(s,c)


def test_preselection_retains_all_matched_pilot_endpoints(tmp_path,monkeypatch):
    s,c,m,j=controls(tmp_path,monkeypatch)
    s['pilot_case_ids']=s['pilot_case_ids'][:4];write('metadata/settings.json',s)
    with pytest.raises(ValueError,match='four positives'):freeze.prepare('metadata/settings.json','cost_pilot','reports/not-created.json',NOW)
    assert not Path('reports/not-created.json').exists()


@pytest.mark.parametrize('defect',['trained_checkpoint','enable_response','remove_sift','settings_changed','pilot_subset','maximum_changed'])
def test_runtime_rejects_control_substitution_not_just_builder_metadata(tmp_path,monkeypatch,defect):
    s,c,m,j=controls(tmp_path,monkeypatch);p=freeze.prepare('metadata/settings.json','cost_pilot','reports/controls.json',NOW)
    if defect=='trained_checkpoint':
        profiles,_=freeze._study_profiles(c,'sh');trained=next(x for x in profiles if x['training_step']==2000)
        p['profiles'][1].update(checkpoint=trained['checkpoint'],training_step=2000)
        p['profiles'][1]['configuration']['checkpoint_sha256']=trained['configuration']['checkpoint_sha256']
    elif defect=='enable_response':p['profiles'][2]['configuration']['response']=True
    elif defect=='remove_sift':p['profiles']=p['profiles'][1:]
    elif defect=='pilot_subset':p['pilot_case_ids']=p['pilot_case_ids'][:4]
    elif defect=='maximum_changed':p['baseline_admission']['maximum_full_calendar_seconds']+=1
    else:s['include_quantization_aware_control']=False;write('metadata/settings.json',s)
    with pytest.raises(ValueError,match='preselection|Preselection'):freeze.validate_runtime(p,'reports/controls.json',NOW)


def test_learned_ablations_still_require_completed_full_checkpoint_selection(tmp_path,monkeypatch):
    s,c,m,j=controls(tmp_path,monkeypatch);s['mode']='additional_controls_sh'
    def unavailable(*a):raise ValueError('Synthetic missing full learned selection')
    monkeypatch.setattr(freeze.learning,'selection',unavailable)
    with pytest.raises(ValueError,match='missing full learned selection'):freeze.profiles_from_artifacts(s,c)


def test_frozen_control_full_study_reuses_existing_measured_admission_without_selection(tmp_path,monkeypatch):
    s,c,m,j=controls(tmp_path,monkeypatch)
    monkeypatch.setattr(freeze.learning,'selection',lambda *a:pytest.fail('Frozen controls selected a learned checkpoint'))
    pilot,_=completed_cost_pilot(s,c,m,j,monkeypatch)
    p=freeze.prepare('metadata/settings.json','full','reports/full-controls.json',NOW+timedelta(minutes=1))
    assert p['purpose']=='additional_controls'and p['profiles']==pilot['profiles']
    assert len(p['case_ids'])==384 and len(p['profiles'])==4 and p['max_calendar_seconds']==231
    assert p['cost_pilot_summary_sha256']==freeze.sha(s['cost_pilot_evidence']['summary'])
    assert not Path(p['output']).exists()
