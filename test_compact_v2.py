import numpy as np
import pytest

from compact_confidence import calibrate,fit_confidence,scores,accepts
from lunar_geometry_v2 import (world_points,project,correspondences,resize_intrinsics,
                               footprint_components,estimate_pose,pose_error)


def test_ray_range_is_not_z_depth_and_camera_roundtrip():
    k=np.array([[50.,0,32],[0,50,32],[0,0,1]])
    pose=np.eye(4);pose[:3,3]=[1000,2000,3000]
    pixels=np.array([[32,32],[62,52]],float)
    world=world_points(pixels,np.array([10.,10.]),k,pose)
    np.testing.assert_allclose(np.linalg.norm(world-pose[:3,3],axis=1),10)
    assert world[1,2]-pose[2,3]<10
    xy,r,front=project(world,k,pose)
    np.testing.assert_allclose(xy,pixels,atol=1e-10)
    np.testing.assert_allclose(r,10)
    assert front.all()


def test_two_depth_cache_requires_identical_depths_across_all_lighting_variants():
    from compact_research import validate_manifest
    tiles=[]
    for i,split in enumerate(['train','model_selection','calibration']+['test']*12):
        tiles.append(dict(tile_id=str(i),component=str(i),split=split,
            source_hashes=[dict(camera=f'{i}-camera-{j%2}',depth=f'{i}-depth-{j%2}') for j in range(6)]))
    manifest=dict(tiles=tiles)
    assert validate_manifest(manifest) is manifest
    tiles[0]['source_hashes'][4]['depth']='different-depth-under-another-light'
    with pytest.raises(ValueError,match='identical source depths'):
        validate_manifest(manifest)


def test_depth_visibility_rejects_occlusion_without_query_match_coordinates():
    k=np.array([[30.,0,16],[0,30,16],[0,0,1]])
    yy,xx=np.mgrid[:32,:32]
    plane=10*np.sqrt(1+((xx-16)/30)**2+((yy-16)/30)**2)
    occluded=plane.copy();occluded[10:20,10:20]*=.5
    points=np.array([[4.,4.],[16.,16.]])
    a,b,world,valid=correspondences(plane,occluded,k,np.eye(4),k,np.eye(4),points=points)
    np.testing.assert_array_equal(valid,[True,False])
    np.testing.assert_allclose(a,b,atol=1e-5)
    assert len(world)==1
    empty=correspondences(plane,occluded,k,np.eye(4),k,np.eye(4),points=np.empty((0,2)))
    assert empty[0].shape==(0,2) and empty[2].shape==(0,3) and len(empty[3])==0


def test_intrinsic_resize_uses_pixel_centers():
    k=np.array([[100.,0,63.5],[0,100,63.5],[0,0,1]])
    small=resize_intrinsics(k,(128,128),(64,64))
    np.testing.assert_allclose(small,[[50,0,31.5],[0,50,31.5],[0,0,1]])


def test_connected_footprints_and_shared_trajectory_stay_together():
    rows=[dict(pair_group=str(i),world_bounds=[[a,0,0],[a+2,2,2]],source_groups=[str(s)])
          for i,(a,s) in enumerate([(0,0),(3,1),(100,1),(200,2)])]
    assert footprint_components(rows,buffer_m=2)==[['0','1','2'],['3']]


def test_pnp_large_world_coordinates_and_invalid_camera():
    rng=np.random.default_rng(4)
    world=rng.uniform([-3,-3,10],[3,3,15],(40,3))+[1e6,2e6,3e6]
    pose=np.eye(4);pose[:3,3]=[1e6,2e6,3e6]
    k=np.array([[100.,0,64],[0,100,64],[0,0,1]])
    xy,*_=project(world,k,pose)
    found=estimate_pose(world,xy,k)
    assert found is not None
    assert pose_error(found,pose)['translation_m']<1e-3
    invalid=pose.copy();invalid[0,0]=2
    with pytest.raises(ValueError):project(world,k,invalid)


def rows(split):
    return [dict(split=split,correct=i<20,negative=i>=30,
             features=[1,np.log1p(50 if i<20 else 6),1,.8 if i<20 else .1,0,0,0,0 if i<20 else 9]) for i in range(40)]


def test_confidence_splits_calibration_and_truth_free_inference():
    policy=fit_confidence(rows('model_selection'))
    gate=calibrate(policy,rows('calibration'))
    assert gate['counts']['correct_accepted']==20
    assert gate['counts']['wrong_accepted']==0
    inference={'features':rows('calibration')[0]['features']}
    assert accepts(gate,inference)
    inference.update(correct=False,negative=True,split='test',ground_truth_motion='forbidden')
    assert accepts(gate,inference)
    with pytest.raises(ValueError):fit_confidence(rows('test'))
    with pytest.raises(ValueError):calibrate(policy,rows('model_selection'))


def test_confidence_reject_all_is_explicit_and_missing_fit_cannot_pass():
    policy=fit_confidence(rows('model_selection'))
    bad=rows('calibration')
    for r in bad:r['correct']=False
    gate=calibrate(policy,bad)
    assert gate['counts']['correct_accepted']==0
    assert not accepts(gate,{'features':[0,0,0,0,0,0,0,0]})


def test_compact_parameter_limit_and_training_gradient():
    torch=pytest.importorskip('torch')
    from compact_lunar import CompactDescriptor
    model=CompactDescriptor()
    assert sum(p.numel() for p in model.parameters())==28305
    d,q=model(torch.rand(2,1,48,48))
    assert d.shape==(2,32,48,48) and q.shape==(2,1,48,48)
    (d.square().mean()+q.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_released_files_are_grouped_by_camera_geometry_not_six_filenames():
    import io,zipfile
    from lunargeo_dataset import source_triplets
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for i in range(12):
            # Geometries are interleaved so naive chunks of six mix terrain.
            group=0 if i//2 in (0,3,4) else 1
            pose=np.eye(4);pose[0,3]=100*group+(i%2)
            camera=io.BytesIO();np.savez(camera,intrinsics=np.eye(3),cam2world=pose)
            stem=f'folder/im_{i:05d}'
            z.writestr(stem+'.npz',camera.getvalue())
            z.writestr(stem+'.jpg',b'not decoded by metadata grouping')
            z.writestr(stem+'.exr',b'not decoded by metadata grouping')
    with zipfile.ZipFile(io.BytesIO(blob.getvalue())) as z:
        result,diag=source_triplets(z,True)
    assert len(result)==2 and diag['complete_three_light_geometries']==2
    names={tuple(f['camera'] for f in frames) for _,_,frames in result}
    assert tuple(f'folder/im_{i:05d}.npz' for i in (0,1,6,7,8,9)) in names


def test_all_compact_training_objectives_are_finite_with_occlusions():
    torch=pytest.importorskip('torch')
    from compact_lunar import CompactDescriptor
    from train_compact_v2 import sample_pair,objective
    from synthetic import texture
    rng=np.random.default_rng(7)
    a=texture(7,128)
    tile=dict(images=np.stack([a,a,a]),masks=np.ones((3,128,128),bool))
    sample=sample_pair(tile,rng,radius=27)
    assert sample is not None
    for multi in (False,True):
        model=CompactDescriptor()
        loss=objective(model,sample,negative=texture(8,128),hard=True,multiscale=multi)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.matchability.weight.grad is not None


def test_sift_evidence_adapter_preserves_original_geometry():
    import cv2
    from matchers import sift
    from raw_correspondences_v2 import EvidenceMatcher
    from synthetic import texture
    a=texture(23,128)
    b=cv2.warpAffine(a,np.array([[1.,0,8.],[0,1.,-4.]]),(128,128))
    cv2.setRNGSeed(20261030);old=sift(a,b)
    wrapper=EvidenceMatcher(sift,'sift')
    cv2.setRNGSeed(20261030);new=wrapper(a,b)
    assert old.matrix is not None
    np.testing.assert_array_equal(old.matrix,new.matrix)
    np.testing.assert_array_equal(old.reference_points,new.reference_points)
    assert wrapper.last_evidence['raw_matches']>=new.support


def test_3d_adapter_never_filters_matches_with_a_planar_model(monkeypatch):
    import cv2,matchers
    from raw_correspondences_v2 import EvidenceMatcher,raw_points
    from synthetic import texture
    def forbidden(*args):raise AssertionError('3D extraction called planar fitting')
    monkeypatch.setattr(matchers,'fit_geometry',forbidden)
    a=texture(23,128);mask=np.ones(a.shape,bool)
    wrapped=EvidenceMatcher(matchers.sift,'sift')
    p,q=raw_points(wrapped,'sift',a,a,mask,mask)
    assert len(p)>=6
    np.testing.assert_allclose(p,q,atol=1e-6)


def test_official_weight_acquisition_retries_timeout_and_rejects_changed_cache(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    import hashlib,io,json,socket
    import baseline_weights_v2 as acquire
    monkeypatch.chdir(tmp_path)
    (tmp_path/'reports').mkdir()
    blob=b'isolated-test-asset';digest=hashlib.sha256(blob).hexdigest()
    (tmp_path/'reports/compact_baseline_lock_v2.json').write_text(json.dumps(
        {'modern':{'loma_b':{'weights':{'dad.pth':digest}}}}))
    monkeypatch.setattr(torch.hub,'get_dir',lambda:str(tmp_path/'cache'))
    previous_timeout=socket.getdefaulttimeout();calls=[]
    def download(url,timeout):
        calls.append((url,timeout))
        if len(calls)==1:raise TimeoutError('simulated stalled header')
        return io.BytesIO(blob)
    monkeypatch.setattr(acquire.urllib.request,'urlopen',download)
    try:
        acquire.acquire('loma_b',tmp_path/'attempts.json')
        assert len(calls)==2 and calls[1][0].endswith('?download=1')
        assert [r['status'] for r in json.loads((tmp_path/'attempts.json').read_text())['attempts']]==['failed','verified']
        path=tmp_path/'cache/checkpoints/dad.pth';assert path.read_bytes()==blob
        path.write_bytes(b'changed')
        with pytest.raises(ValueError,match='Existing official weight differs'):
            acquire.acquire('loma_b',tmp_path/'changed-attempts.json')
        assert len(calls)==2
    finally:socket.setdefaulttimeout(previous_timeout)
