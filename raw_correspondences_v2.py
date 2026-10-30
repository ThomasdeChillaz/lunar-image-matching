"""Extract correspondences before planar fitting for depth/pose evaluation."""
import cv2
import numpy as np


def raw_points(matcher,method,a,b,ma,mb):
    empty=lambda:(np.empty((0,2),np.float32),np.empty((0,2),np.float32))
    if not ma.any() or not mb.any() or min(a[ma].std(),b[mb].std())<1:return empty()
    if hasattr(matcher,'points'):return matcher.points(a,b,ma,mb)
    if method=='sift':
        detector=cv2.SIFT_create(nfeatures=2000)
        ka,da=detector.detectAndCompute(a,ma.astype(np.uint8)*255)
        kb,db=detector.detectAndCompute(b,mb.astype(np.uint8)*255)
        if da is None or db is None or min(len(da),len(db))<2:return empty()
        pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
        matches=sorted([p[0] for p in pairs if len(p)==2 and p[0].distance<.75*p[1].distance],key=lambda m:m.distance)
        unique={}
        for m in matches:unique.setdefault(m.trainIdx,m)
        aa=np.array([ka[m.queryIdx].pt for m in unique.values()],np.float32).reshape(-1,2)
        bb=np.array([kb[m.trainIdx].pt for m in unique.values()],np.float32).reshape(-1,2)
        return aa,bb
    if method=='original':
        import torch
        from contrastive_terrain import valid_centers,sample_descriptors,RADIUS
        yy,xx=np.mgrid[RADIUS:a.shape[0]-RADIUS:4,RADIUS:a.shape[1]-RADIUS:4]
        grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
        pa,pb=[grid[valid_centers(m)[grid[:,1].astype(int),grid[:,0].astype(int)]] for m in (ma,mb)]
        if min(len(pa),len(pb))<6:return empty()
        inputs=torch.from_numpy(np.stack([np.where(ma,a,0),np.where(mb,b,0)])).to(matcher.device).float()[:,None]/255
        with torch.inference_mode():
            d=matcher.model(inputs)
            da=sample_descriptors(d[:1],torch.from_numpy(pa).to(matcher.device)[None])[0]
            db=sample_descriptors(d[1:],torch.from_numpy(pb).to(matcher.device)[None])[0]
            similarity=da@db.T;score,index=similarity.max(1);reverse=similarity.argmax(0)
            ia=torch.where((reverse[index]==torch.arange(len(pa),device=matcher.device))&(score>=.5))[0]
            ia=ia[torch.argsort(score[ia],descending=True)[:512]]
            return pa[ia.cpu().numpy()],pb[index[ia].cpu().numpy()]
    if method=='matchanything':
        from eloftr_matcher import native_pixel_centers
        from loftr_matcher import valid_correspondences
        batch={}
        for i,(image,mask) in enumerate(((a,ma),(b,mb))):
            values=cv2.resize(np.where(mask,image,0),None,fx=matcher.factor,fy=matcher.factor,interpolation=cv2.INTER_LINEAR)
            batch[f'image{i}']=matcher.torch.from_numpy(np.ascontiguousarray(values)).float()[None,None].to(matcher.device)/255
        with matcher.torch.inference_mode():matcher.model(batch)
        pa,pb=[native_pixel_centers(batch[k].cpu().numpy(),matcher.factor) for k in ('mkpts0_f','mkpts1_f')]
        return valid_correspondences(pa,pb,ma,mb)
    if method=='roma':
        from PIL import Image
        from roma_matcher import cpu_density_precision,normalized_to_native_centers
        from loftr_matcher import valid_correspondences
        inputs=[Image.fromarray(np.repeat(np.where(m,image,0)[...,None],3,axis=2)) for image,m in ((a,ma),(b,mb))]
        matcher.torch.manual_seed(20261029)
        with matcher.torch.inference_mode(),cpu_density_precision():
            warp,certainty=matcher.model.match(*inputs,device='cpu')
            points,_=matcher.model.sample(warp,certainty,num=matcher.sample_count)
        points=points.float().cpu().numpy()
        return valid_correspondences(normalized_to_native_centers(points[:,:2],a.shape),
                                     normalized_to_native_centers(points[:,2:],b.shape),ma,mb)
    raise ValueError('No raw correspondence adapter; never substitute affine-filtered matches')
