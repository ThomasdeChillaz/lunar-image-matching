"""Analytic solar geometry and archive routing; no lunar neural inference."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from compact_lunar import CompactDescriptor
from solar_frontend_v63 import ReferenceGeometry,SolarMatcher,reference_anchors
from solar_sun_search_v68 import sun_search_catalogue,bounded_sun_indices
from solar_physics_v63 import sun_candidates,lighting_schedule
from solar_assignment_v63 import height_variants
from lunar_geometry_v5 import world_points_z


def geometry():
    catalogue=sun_search_catalogue(80.);suns=np.array([r['sun_enu']for r in catalogue]);parameters=np.array([[r['hour_angle_deg'],r['subsolar_latitude_deg']]for r in catalogue]);coarse=np.array([r['coarse']for r in catalogue])
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[:3,3]=[0,0,100]
    world=world_points_z(reference_anchors(),np.full(256,100.),k,pose)
    return ReferenceGeometry(np.full((512,512),100.),k,pose,np.tile([0.,0.,1.],(256,1)),suns,np.ones((256,len(suns)),np.uint8),np.ones((256,len(suns),7),np.uint8),suns[0],height_variants(world,pose),parameters,coarse),k


def test_coarse_initialization_exactly_matches_existing_geometry_and_actual24_lights_are_supported():
    catalogue=sun_search_catalogue(80.);coarse={(r['hour_angle_deg'],r['subsolar_latitude_deg']):r['sun_enu']for r in catalogue if r['coarse']}
    old={(r['hour_angle_deg'],r['subsolar_latitude_deg']):r['sun_enu']for r in sun_candidates(80.)}
    assert coarse.keys()==old.keys()
    for k in coarse:np.testing.assert_allclose(coarse[k],old[k],atol=1e-12,rtol=0)
    suns=np.array([r['sun_enu']for r in catalogue]);mask=np.array([r['coarse']for r in catalogue])
    for r in lighting_schedule(80.):
        hits=np.flatnonzero(np.linalg.norm(suns-r['sun_enu'],axis=1)<1e-8)
        assert len(hits)==1 and not mask[hits[0]]


def test_refinement_is_bounded_without_target_label_or_image():
    g,k=geometry();centre=np.array([40.,1.5]);ids=bounded_sun_indices(g.sun_parameters,centre,0)
    assert (np.abs(g.sun_parameters[ids,0]-40)<=2.5).all()and(np.abs(g.sun_parameters[ids,1]-1.5)<=.25).all()
    assert ((g.sun_parameters[ids]-[41.25,1.5])**2).sum(1).min()==0
    with pytest.raises(ValueError):bounded_sun_indices(g.sun_parameters,[300,1.5],0)


def test_primary_search_uses_coarse_mask_known_track_uses_exact_column_and_reference_metadata_is_separate(monkeypatch):
    import solar_frontend_v63 as frontend
    g,k=geometry();p=reference_anchors();known=np.array(lighting_schedule(80.)[17]['sun_enu']);seen=[]
    monkeypatch.setattr(frontend,'candidates',lambda *args:(p.copy(),np.repeat(p[:,None],8,1),np.ones((256,8)),np.zeros((256,8),int)))
    def capture(graph,*args,**kwargs):
        seen.append(graph);return dict(pose=None,reason='software_capture',reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)))
    monkeypatch.setattr(frontend,'solve',capture)
    torch.manual_seed(13);model=CompactDescriptor(False);matcher=SolarMatcher(model,model)
    a=np.random.default_rng(13).integers(0,256,(512,512),dtype=np.uint8)
    primary=matcher.localize(a,a,g,k);secondary=matcher.localize(a,a,g,k,known_sun=known,reference_sun=known)
    assert len(seen[0].sun_vectors)==len(g.sun_vectors)and not seen[0].coarse_sun_mask.all()
    assert len(seen[1].sun_vectors)==1 and seen[1].coarse_sun_mask.all()
    assert not primary['known_sun_supplied']and secondary['known_sun_supplied']
    np.testing.assert_allclose(seen[1].sun_vectors[0],known,atol=1e-12)
    np.testing.assert_array_equal(g.reference_sun,g.sun_vectors[0])
    np.testing.assert_array_equal(secondary['reference_sun_world'],known)


def test_legacy_archive_stays_discrete_and_extra_query_truth_is_forbidden(tmp_path):
    g,k=geometry();values={n:getattr(g,n)for n in g.__dataclass_fields__ if n not in('sun_parameters','coarse_sun_mask')};p=tmp_path/'legacy.npz';np.savez(p,**values)
    legacy=ReferenceGeometry.load(p);assert legacy.sun_parameters.shape==(0,2)and legacy.coarse_sun_mask.all()
    np.savez(p,**values,query_sun=g.sun_vectors[0])
    with pytest.raises(ValueError,match='query truth'):ReferenceGeometry.load(p)
