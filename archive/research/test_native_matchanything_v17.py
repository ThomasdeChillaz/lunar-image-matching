import numpy as np
import pytest

from native_matchanything_v17 import NativeMatchAnything,to_native


def test_original_pixel_centers_survive_internal_resize():
    native=np.array([[0.,0.],[255.5,255.5],[511.,511.],[91.375,212.625]])
    internal=(native+.5)*832/512-.5
    np.testing.assert_allclose(to_native(internal,832),native,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(to_native(native,512),native)


def test_empty_support_and_invalid_native_inputs_do_not_invoke_the_network():
    class ForbiddenNetwork:
        def __call__(self,*args,**kwargs):raise AssertionError('Invalid input reached the network')
    matcher=NativeMatchAnything.__new__(NativeMatchAnything);matcher.model=ForbiddenNetwork()
    image=np.indices((512,512)).sum(0).astype(np.uint8);blank=np.zeros_like(image)
    assert len(matcher.points(blank,blank)[0])==0
    assert len(matcher.points(image,image,np.zeros_like(image,bool))[0])==0
    with pytest.raises(ValueError,match='uint8'):matcher.points(image.astype(float),image)
    with pytest.raises(ValueError,match='512'):matcher.points(image[:128,:128],image[:128,:128])
    with pytest.raises(ValueError,match='mask'):matcher.points(image,image,np.ones((128,128),bool))
