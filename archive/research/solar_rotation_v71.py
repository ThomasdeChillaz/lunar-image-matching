"""Matched rotation-bank engineering control, preserving native coordinates.

This is not a learned contribution. All experimental ablations must receive
identical angles. Quarter turns use the historical exact np.rot90 path; added
22.5-degree images use bilinear resampling and explicit full-support FOV masks.
"""
import cv2
import numpy as np
from rotation_bank_pose_v4 import quarter_turn

QUARTER_ANGLES=(0.,90.,180.,270.)
EXPANDED_ANGLES=tuple(float(x)for x in np.arange(0.,360.,22.5))


def proposal_angles(value=None):
    angles=QUARTER_ANGLES if value is None else tuple(float(x)for x in value)
    if angles not in(QUARTER_ANGLES,EXPANDED_ANGLES):raise ValueError('Predeclared historical4 or matched16 rotation angles required')
    return angles


class RotationBank:
    def __init__(self,image,angles=None):
        self.angles=proposal_angles(angles)
        if np.asarray(image).shape!=(512,512)or np.asarray(image).dtype!=np.uint8:raise ValueError('Native512uint8 query image required')
        self.images=[];self.valid=[];self.affines=[];self.inverse_affines=[]
        for angle in self.angles:
            affine=cv2.getRotationMatrix2D((255.5,255.5),angle,1.)
            if angle%90==0:
                result=np.rot90(image,int(angle/90)).copy();valid=np.ones((512,512),bool)
            else:
                result=cv2.warpAffine(image,affine,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                support=cv2.warpAffine(np.ones((512,512),np.float32),affine,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                valid=support>=1.-1e-6
            self.images.append(result);self.valid.append(valid);self.affines.append(affine);self.inverse_affines.append(cv2.invertAffineTransform(affine))

    def transform(self,points,index,inverse=False):
        points=np.asarray(points,float)
        if points.shape[-1]!=2 or not np.isfinite(points).all()or not 0<=int(index)<len(self.angles):raise ValueError('Finite native coordinates and valid rotation-bank index required')
        angle=self.angles[int(index)]
        if angle%90==0:return quarter_turn(points,int(angle/90),512,inverse).reshape(points.shape)
        affine=(self.inverse_affines if inverse else self.affines)[int(index)]
        return points@affine[:,:2].T+affine[:,2]

    def coarse_mask(self,low_xy):
        """Entire25px low-image receptive field and area-resize support valid.

        The frozen descriptor has radius12 low-pixel support. No padded pixel
        may contribute to a candidate descriptor. Native57px support is checked
        separately immediately before sparse refinement.
        """
        points=np.asarray(low_xy)
        if points.ndim!=2 or points.shape[1]!=2 or not np.array_equal(points,np.rint(points)):raise ValueError('Integer128px coarse candidate grid required')
        out=[]
        for mask in self.valid:
            resized=cv2.resize(mask.astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)
            valid=(resized>=1.-1e-6).astype(np.uint8)
            valid=cv2.erode(valid,np.ones((25,25),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            out.append(valid[points[:,1].astype(int),points[:,0].astype(int)].astype(bool))
        return np.asarray(out)

    def native_support(self,proposal,index):
        centre=np.rint(np.asarray(proposal,float)).astype(int);origin=centre-28
        if(origin<0).any()or(origin+57>512).any():return False
        x,y=origin;return bool(self.valid[int(index)][y:y+57,x:x+57].all())
