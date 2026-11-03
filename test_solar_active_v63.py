import numpy as np
import pytest
from solar_active_v63 import disagreement_scores,choose_action,acquire_patch_actions


def test_robust_intervals_prefer_stable_pose_disagreement():
    x=np.zeros((2,3,7,4));x[1,0,:,2]=1
    x[0,1,:,2]=np.linspace(0,1,7);x[1,1,:,2]=np.linspace(.1,1.1,7)
    x[:,2,:,:]=np.nan
    s=disagreement_scores(x,[.5,.5]);assert s[0]>0 and s[1]==0 and s[2]==0
    assert choose_action(x,[.5,.5],np.array([[50,50],[200,200],[400,400]]),np.zeros(3,bool))==0
    assert (disagreement_scores(x,[.5,.5],geometry_only=True)==0).all()


def test_geometry_only_information_and_used_measurements_respected():
    x=np.zeros((2,3,7,4));x[1,1,:,0]=2
    p=np.array([[50,50],[200,200],[400,400]])
    assert choose_action(x,[.5,.5],p,[False,False,False],'geometry')==1
    assert choose_action(x,[.5,.5],p,[False,True,False],'geometry')!=1
    assert choose_action(x,[.5,.5],p,[True,True,True])is None


def test_no_native_measurement_is_requested_before_selection_and_budget_holds():
    calls=[];p=np.column_stack([np.arange(50),np.arange(50)])
    def predict():return np.zeros((2,50,7,4)),np.ones(2),None
    def measure(i):calls.append(i);return dict(found=True)
    records=acquire_patch_actions(predict,measure,lambda *a:None,p,32,'uniform')
    assert len(records)==32 and len(set(calls))==32
    assert choose_action(np.zeros((2,50,7,4)),[1,1],p,np.zeros(50,bool),'appearance',np.arange(50))==49


def test_corrupt_weights_and_undeclared_budgets_rejected():
    with pytest.raises(ValueError):disagreement_scores(np.zeros((2,3,7,4)),[0,0])
    with pytest.raises(ValueError):acquire_patch_actions(None,None,None,np.zeros((3,2)),33)
