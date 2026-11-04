"""Raw diagnostics must not move query truth into primary inference."""
from types import SimpleNamespace
import numpy as np
import pytest
from solar_evaluation_v70 import InferenceInput,evaluate_case


def run_control(bad_pool=False,bad_sun=False,timeout=False):
    events=[]
    p=np.array([[80.,80.],[400.,80.],[80.,400.],[400.,400.],[240.,160.],[160.,240.]])
    q=p+[8.,-4.]
    k=np.array([[400.,0,255.5],[0,400.,255.5],[0,0,1.]])
    geometry=SimpleNamespace(depth=np.full((512,512),10.),intrinsics=k,cam2world=np.eye(4))
    inp=InferenceInput(np.ones((512,512),np.uint8),np.ones((512,512),np.uint8),geometry,k,np.array([0.,0.,1.]))
    class Backend:
        profile={}
        def infer(self,inputs):
            events.append('inference')
            if timeout:raise TimeoutError('constructed per-request resource deadline')
            assert inputs.known_query_sun is None
            assert not hasattr(inputs,'scoring_query_sun_world')
            pool=np.repeat(q[:,None],7 if bad_pool else 8,axis=1)
            return dict(reference_xy=p,query_xy=q,pose=None,sun_enu=np.array([2. if bad_sun else 1.,0.,0.]),
                coarse_reference_xy=p,coarse_query_candidates=pool,
                coarse_rotation_indices=np.zeros(pool.shape[:2],int))
    def labels(case,points):
        assert events[0]=='inference'
        events.append('scoring')
        return dict(positive=True,query_xy=points+[8.,-4.],visible=np.ones(len(points),bool),query_cam2world=np.eye(4)),'procedural_exact'
    def estimator(*args):
        events.append('independent_pose_replay')
        return None,dict(reason='constructed_missing_pose',inliers=0)
    case=dict(positive=True,secondary_known_query_sun_world=[0.,1.,0.])
    row,arrays=evaluate_case(Backend(),inp,case,scoring_loader=labels,estimator=estimator)
    return events,row,arrays,p,q


def test_native_candidates_and_sun_labels_are_saved_only_after_inference():
    events,row,arrays,p,q=run_control()
    assert events==['inference','independent_pose_replay','scoring','scoring']
    np.testing.assert_array_equal(arrays['coarse_reference_xy'],p)
    np.testing.assert_array_equal(arrays['scoring_coarse_query_xy'],q)
    np.testing.assert_array_equal(arrays['estimated_sun_world'],[1.,0.,0.])
    np.testing.assert_array_equal(arrays['scoring_query_sun_world'],[0.,1.,0.])
    assert arrays['coarse_query_candidates'].shape==(6,8,2)
    assert not row['metrics']['pose_returned']


def test_invalid_candidate_export_is_an_experiment_defect():
    with pytest.raises(ValueError,match='Eight finite native coarse candidates'):
        run_control(bad_pool=True)


def test_nonunit_estimated_sun_is_an_experiment_defect():
    with pytest.raises(ValueError,match='Estimated Sun'):
        run_control(bad_sun=True)


def test_per_request_timeout_retains_the_eligible_failed_case():
    events,row,arrays,p,q=run_control(timeout=True)
    assert row['inference_failure']['status']=='timeout'
    assert row['inference_failure']['type']=='TimeoutError'
    assert not row['metrics']['pose_returned']and not row['metrics']['correct']
    assert arrays['reference_xy'].shape==(0,2)and arrays['pose'].shape==(0,4)
    assert events==['inference','scoring']
