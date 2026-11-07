import numpy as np
import pytest

from north_pose_prior_diagnostic_v41 import (fit_training_templates,proposals,
    evaluate_metadata)
from north_render_plan_v31 import camera_to_world


def fixture():
    rows=[]
    for component,height in enumerate((1100.,1200.,1300.)):
        a=np.asarray(camera_to_world([0.,0.,height],[0.,0.,0.]))
        for dx,dy,dz,tx,ty in ((250.,150.,0.,0.,0.),(450.,-200.,-200.,0.,0.),(-350.,300.,200.,100.,150.)):
            b=np.asarray(camera_to_world([dx,dy,height+dz],[tx,ty,0.]))
            rows.append(dict(reference=a,query=b,component=str(component),split='training'if component<2 else'model_selection'))
    return rows


def test_fitted_templates_reconstruct_heldout_metadata_without_query_ids():
    rows=fixture();bank=fit_training_templates(rows[:6])
    result=evaluate_metadata(rows[6:],bank)
    assert len(bank)==3 and all(r['oracle_near_exact']for r in result)
    assert sum(r['first_fixed_template_close']for r in result)==1
    assert len(proposals(rows[-1]['reference'],bank))==3


def test_selection_cameras_cannot_enter_template_fitting():
    with pytest.raises(ValueError,match='training metadata only'):
        fit_training_templates(fixture())


def test_changing_query_truth_does_not_change_proposals():
    rows=fixture();bank=fit_training_templates(rows[:6]);ref=rows[-1]['reference']
    old=proposals(ref,bank);rows[-1]['query']=np.eye(4)
    new=proposals(ref,bank)
    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
