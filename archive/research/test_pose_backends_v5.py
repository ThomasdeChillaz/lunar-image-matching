from combine_pose_backends_v5 import choose


def test_backend_priority_is_independent_of_truth_labels():
    sift={'pose':[[1]],'pose_correct':False};cnn={'pose':[[2]],'pose_correct':True}
    assert choose(sift,cnn)[0] is sift
    assert choose({'pose':None},cnn)[0] is cnn
    assert choose(None,None)[0] is None
