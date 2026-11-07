"""Generated pattern catalogue and known image shift, not physical sunlight."""
import numpy as np
from catalogue_v138 import Catalogue


def fixture():
    rng=np.random.default_rng(20261105)
    images=np.array([.1+rng.random((32,32)).repeat(16,0).repeat(16,1) for _ in range(3)])
    cat=Catalogue(images,np.eye(3),(0,8,16))
    depth=np.full((512,512),1100.)
    k=np.array([[440.,0.,255.5],[0.,440.,255.5],[0.,0.,1.]])
    poses=np.repeat(np.eye(4)[None],3,axis=0); poses[:,0,3]=[0.,20.,-40.]
    query=np.full((512,512),.1); query[:,:-8]=1.2*images[1,:,8:]+.03
    return cat,query,depth,k,poses
