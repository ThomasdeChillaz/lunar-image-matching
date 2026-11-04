"""Source-frozen generic physics choices; metadata/synthetic arrays, no rays or NN."""
from pathlib import Path
import json
import numpy as np
import pytest
import solar_reference_pipeline_v74 as p
import solar_portable_evidence_v75 as portable
from test_solar_reference_pipeline_v74 import setup


def settings(s,mode):
    value=json.loads(Path('settings.json').read_text());value['reference_precompute']=mode;Path('settings.json').write_text(json.dumps(value));return value


def test_new_protocol_requires_explicit_choice_before_pilot(setup):
    value=json.loads(Path('settings.json').read_text());value.pop('reference_precompute',None);Path('settings.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='explicitly freeze'):p.initialize('settings.json',setup.contract)
    assert not Path(setup.contract).exists()


@pytest.mark.parametrize('mode',['coarse','full'])
def test_physical_choice_bound_in_every_protocol_and_contract(setup,mode):
    settings(setup,mode);c=p.initialize('settings.json',setup.contract)
    assert c['reference_precompute']==mode
    for i in c['selection_units']:
        pre,lookup=p.physical_protocols(c,i)
        assert pre['precompute']==mode and pre['query_truth_used']is False and lookup['query_truth_used']is False
    assert p.load_contract(setup.contract)['reference_precompute']==mode
    altered=json.loads(Path(setup.contract).read_text());altered['reference_precompute']='full'if mode=='coarse'else'coarse';Path(setup.contract).write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='physical protocols changed'):p.load_contract(setup.contract)


def test_full_physics_requires_exact_generic_catalogue_and_computed_columns(tmp_path):
    from solar_sun_search_v68 import sun_search_catalogue
    catalogue=sun_search_catalogue(85.);s=len(catalogue)
    z=dict(depth=np.ones((512,512)),intrinsics=np.eye(3),cam2world=np.eye(4),normals=np.tile([0.,0.,1.],(256,1)),sun_vectors=np.array([x['sun_enu']for x in catalogue]),solar_states=np.zeros((256,s),np.uint8),solar_variants=np.zeros((256,s,7),np.uint8),reference_sun=np.array([0.,0.,1.]),world_variants=np.zeros((256,7,3)),sun_parameters=np.array([[x['hour_angle_deg'],x['subsolar_latitude_deg']]for x in catalogue]),coarse_sun_mask=np.array([x['coarse']for x in catalogue],bool),solar_materialized_mask=np.ones(s,bool))
    # UNKNOWN coverage is valid computed information, not an uncomputed column.
    np.savez_compressed(tmp_path/'full.npz',**z);assert portable.full_physics(tmp_path,'full.npz',85.)
    z['solar_materialized_mask'][np.flatnonzero(~z['coarse_sun_mask'])[0]]=False;np.savez_compressed(tmp_path/'coarse.npz',**z)
    with pytest.raises(ValueError,match='EVERY generic'):portable.full_physics(tmp_path,'coarse.npz',85.)
    z['solar_materialized_mask'][:]=True;z['sun_parameters'][0,0]+=.01;np.savez_compressed(tmp_path/'query-specific.npz',**z)
    with pytest.raises(ValueError,match='generic Sun catalogue'):portable.full_physics(tmp_path,'query-specific.npz',85.)
