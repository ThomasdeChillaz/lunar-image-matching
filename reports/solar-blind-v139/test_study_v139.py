from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from study_v139 import forecast_fits,sealed
from engine_v139 import decode_clean16


def timing():
    samples=[dict(layered=v,all_six_pose_controls_pass=True,cameras=2048,lights=24,
                  extra_searches=32,elapsed_seconds=10.) for v in (False,True)]
    return dict(samples=samples,comparison_forecast_seconds=150.)


def test_full_budget_gate_preserves_original_deadline_and_refuses_smaller_pilot():
    t=timing()
    assert forecast_fits(t,150.) and not forecast_fits(t,149.999)
    for key,value in [('cameras',3),('lights',3),('extra_searches',2),('all_six_pose_controls_pass',False)]:
        wrong=deepcopy(t);wrong['samples'][0][key]=value
        with pytest.raises(ValueError):forecast_fits(wrong,480.)
    wrong=deepcopy(t);wrong['samples'][0]['elapsed_seconds']=-1.
    with pytest.raises(ValueError):forecast_fits(wrong,480.)


def test_rgb16_decode_preserves_sub_rgb8_variation_and_colour_order(tmp_path):
    p=tmp_path/'source.png';image=np.full((512,512,3),10000,np.uint16)
    image[:,:256,2]+=1
    assert cv2.imwrite(str(p),image)
    decoded=decode_clean16(p)
    assert decoded[0,0]>decoded[0,300]
    rgb=image[0,0,::-1]/65535.
    expected=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)@np.array([.2126,.7152,.0722])
    assert decoded[0,0]==pytest.approx(expected,abs=1e-15)
    p8=tmp_path/'downcast.png';cv2.imwrite(str(p8),image.astype(np.uint8))
    with pytest.raises(ValueError):decode_clean16(p8)


def test_unsealed_or_incomplete_predictions_cannot_open_scoring(tmp_path,monkeypatch):
    import study_v139 as study
    monkeypatch.setattr(study,'WORK',tmp_path)
    p=tmp_path/'predictions.json';p.write_text(json.dumps(dict(rows=[dict(id='a')],query_truth_used=False)))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'PREDICTIONS_SEALED.json').write_text(json.dumps(dict(rows=8,sha256=digest)))
    with pytest.raises(ValueError):sealed(dict(case_ids=['a','b']))
    data=json.loads(p.read_text());data['query_truth_used']=True;p.write_text(json.dumps(data))
    with pytest.raises(ValueError):sealed(dict(case_ids=['a']))


def test_six_arm_small_generated_wiring_and_no_extra_identity():
    # Deliberately small software wiring, not the24-light/2048-camera/32-search
    # timing gate. Real predict()/timing() always use the frozen default budget.
    from procedural_fixture_v138 import fixture
    from engine_v139 import models,setup,METHODS
    cat,q,d,k,poses=fixture();cache=setup()
    result=models(cat,q,d,k,np.eye(4),k,np.ones_like(d,bool),cache,0,
        extra_budget=2,control_poses=poses)
    assert tuple(result['methods'])==METHODS and not result['query_truth_used']
    for name,m in result['methods'].items():
        assert m['pose'] is not None
        assert np.linalg.norm(np.asarray(m['pose'])[:3,3]-[20.,0.,0.])<2.
        if name not in ('direct','joint_no_extra'):
            assert m['acquisition']['attempted_extra_patches']==2
    assert result['initialization']['lighting_selected_from_images']
