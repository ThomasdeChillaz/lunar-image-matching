"""Coordinate association regression; no models, imagery or new inference."""
import numpy as np
import pytest
from correct_pose_free_metrics_v86 import exact_label_indices,counts


def test_permuted_larger_label_inventory_scores_sparse_predictions_by_coordinates():
    predicted=np.array([[10.,10.],[20.,20.]])
    label_reference=np.array([[20.,20.],[99.,99.],[10.,10.]])
    target=np.array([[100.,100.],[77.,77.],[200.,200.]])
    candidates=np.repeat(np.array([[200.,200.],[100.,100.]])[:,None],8,axis=1)
    ids=exact_label_indices(predicted,label_reference)
    np.testing.assert_array_equal(ids,[2,0])
    bank=np.array([[100.,100.],[200.,200.]])
    corrected,_=counts(predicted,candidates,target[ids],np.ones(2,bool))
    wrong,_=counts(predicted,candidates,target[:2],np.ones(2,bool))
    assert corrected['recalled']['1']==2 and wrong['recalled']['16']==0
    permutation=np.array([2,0,1]);new=exact_label_indices(predicted,label_reference[permutation])
    np.testing.assert_array_equal(target[permutation][new],target[ids])


@pytest.mark.parametrize('which',['labels','predictions'])
def test_duplicate_reference_coordinates_fail_closed(which):
    predicted=np.array([[10.,10.],[20.,20.]]);labels=predicted.copy()
    if which=='labels':labels=np.vstack([labels,labels[0]])
    else:predicted=np.vstack([predicted,predicted[0]])
    with pytest.raises(ValueError,match='Duplicate'):exact_label_indices(predicted,labels)


def test_missing_label_and_nonfinite_coordinates_fail_closed():
    with pytest.raises(ValueError,match='Missing'):exact_label_indices(np.array([[10.,10.]]),np.array([[11.,10.]]))
    with pytest.raises(ValueError,match='Finite'):exact_label_indices(np.array([[np.nan,10.]]),np.array([[10.,10.]]))


def test_256_sparse_anchors_join_permuted676_labels_with_known_translation():
    axis=49.5+16*np.arange(26);y,x=np.meshgrid(axis,axis,indexing='ij');labels=np.column_stack([x.ravel(),y.ravel()])
    selected=np.rint(np.linspace(0,25,16)).astype(int);yy,xx=np.meshgrid(selected,selected,indexing='ij')
    prediction=labels[(yy*26+xx).ravel()]
    translation=np.array([3.,-2.]);target=labels+translation
    permutation=np.random.default_rng(20261106).permutation(676)
    ids=exact_label_indices(prediction,labels[permutation])
    truth=target[permutation][ids]
    np.testing.assert_array_equal(truth,prediction+translation)
    candidates=np.repeat(truth[:,None],8,axis=1)
    result,error=counts(prediction,candidates,truth,np.ones(256,bool))
    assert result['all_retained_reference_anchors']==256
    assert result['recalled']=={str(t):256 for t in(1,2,4,8,12,16)}
    np.testing.assert_array_equal(error,np.zeros(256))
