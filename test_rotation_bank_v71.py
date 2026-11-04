"""Procedural rotation-control geometry and support; no lunar NN comparison."""
import cv2
import numpy as np
import pytest
import torch
from compact_lunar import CompactDescriptor
from solar_rotation_v71 import RotationBank,QUARTER_ANGLES,EXPANDED_ANGLES,proposal_angles
from solar_frontend_v63 import candidates,reference_anchors
from rotation_bank_pose_v4 import quarter_turn
from contrastive_terrain import sample_descriptors


def historical_candidates(encoder,reference,query):
    p=reference_anchors();yy,xx=np.mgrid[12:116:4,12:116:4];low=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32);native=(low+.5)*4-.5
    images=[reference]+[np.rot90(query,r).copy()for r in range(4)]
    batch=np.stack([cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)for im in images])
    with torch.inference_mode():
        dense,_=encoder(torch.from_numpy(batch).float()[:,None]/255)
        a=sample_descriptors(dense[:1],torch.from_numpy(((p+.5)/4-.5).astype(np.float32))[None])[0]
        b=sample_descriptors(dense[1:],torch.from_numpy(low)[None].expand(4,-1,-1)).reshape(-1,32)
        score=a@b.T;rank=torch.argsort(score,dim=1,descending=True,stable=True)[:,:8]
        values=score.gather(1,rank).numpy();ids=rank.numpy()
    rotations=ids//len(low);indices=ids%len(low);q=np.empty((256,8,2))
    for r in range(4):
        use=rotations==r;q[use]=quarter_turn(native[indices[use]],r,512,True)
    return p,q,values,rotations


def test_historical_quarter_rotation_bank_and_default_proposals_preserved_exactly():
    torch.manual_seed(45);model=CompactDescriptor(False).eval();rng=np.random.default_rng(45)
    reference=rng.integers(0,256,(512,512),dtype=np.uint8);query=rng.integers(0,256,(512,512),dtype=np.uint8)
    actual=candidates(model,reference,query);expected=historical_candidates(model,reference,query)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    bank=RotationBank(query)
    for r in range(4):np.testing.assert_array_equal(bank.images[r],np.rot90(query,r))


def test_sixteen_rotation_affines_roundtrip_native_pixel_centres_and_exact_quarters():
    image=np.indices((512,512))[0].astype(np.uint8);bank=RotationBank(image,EXPANDED_ANGLES)
    points=np.array([[0.,0.],[511.,511.],[100.25,287.125],[255.5,255.5]])
    for i,angle in enumerate(bank.angles):
        transformed=bank.transform(points,i);back=bank.transform(transformed,i,True)
        np.testing.assert_allclose(back,points,atol=2e-13,rtol=0)
        np.testing.assert_allclose(bank.transform([[255.5,255.5]],i),[[255.5,255.5]],atol=1e-13)
        if angle%90==0:
            np.testing.assert_array_equal(transformed,quarter_turn(points,int(angle/90),512))
            np.testing.assert_array_equal(bank.images[i],np.rot90(image,int(angle/90)))


def test_padded_rotated_pixels_cannot_enter_coarse_or_native_descriptors():
    bank=RotationBank(np.full((512,512),255,np.uint8),EXPANDED_ANGLES)
    yy,xx=np.mgrid[12:116:4,12:116:4];low=np.column_stack([xx.ravel(),yy.ravel()]);masks=bank.coarse_mask(low)
    assert masks.shape==(16,676)and masks[::4].all()
    assert all(not masks[i].all()for i in range(16)if i%4)
    for i in range(16):
        for xy in low[masks[i]]:
            x,y=xy;native=(xy+.5)*4-.5
            source=bank.transform(native[None],i,True)[0]
            assert(source>=0).all()and(source<=511).all()
            assert bank.native_support(native,i)
    assert not bank.native_support([49.5,49.5],2) #45degree corner is padded
    assert not bank.native_support([1.,1.],0)
    assert bank.native_support([255.5,255.5],2)


def test_known_nonquarter_rotation_has_correct_inverse_image_and_coordinate_convention():
    image=np.zeros((512,512),np.uint8);cv2.circle(image,(200,255),3,255,-1)
    bank=RotationBank(image,EXPANDED_ANGLES)
    for i in range(16):
        x,y=bank.transform(np.array([[200.,255.]]),i)[0]
        measured=cv2.remap(bank.images[i],np.float32([[x]]),np.float32([[y]]),cv2.INTER_LINEAR)[0,0]
        assert measured>240
        np.testing.assert_allclose(bank.transform([[x,y]],i,True),[[200.,255.]],atol=1e-12)


def test_expanded_proposal_grid_keeps256anchors_top8_and_valid_native_candidates():
    torch.manual_seed(71);model=CompactDescriptor(False).eval();image=np.random.default_rng(5).integers(0,256,(512,512),dtype=np.uint8)
    p,q,s,r=candidates(model,image,image,EXPANDED_ANGLES)
    assert p.shape==(256,2)and q.shape==(256,8,2)and s.shape==(256,8)and r.shape==(256,8)
    assert np.isfinite(q).all()and np.isfinite(s).all()and(q>=0).all()and(q<=511).all()and(r>=0).all()and(r<16).all()
    np.testing.assert_allclose(q[:,0],p,atol=1e-6)
    bank=RotationBank(image,EXPANDED_ANGLES)
    for i in range(256):
        for c in range(8):assert bank.native_support(bank.transform(q[i,c][None],r[i,c])[0],r[i,c])


def test_undeclared_angle_banks_rejected():
    with pytest.raises(ValueError,match='Predeclared'):proposal_angles([0.,45.])
    assert proposal_angles()==QUARTER_ANGLES and proposal_angles(list(EXPANDED_ANGLES))==EXPANDED_ANGLES
