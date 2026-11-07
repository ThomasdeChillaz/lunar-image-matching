"""Input/content guards for the prepared MOONSt3R integration, without weights."""
import numpy as np
import pytest

from native_moonst3r_v19 import native_inputs


def test_full_native_content_and_mask_guards_do_not_require_weights():
    image=np.tile(np.arange(512,dtype=np.uint16)%256,(512,1)).astype(np.uint8)
    masks,usable=native_inputs(image,image)
    assert usable and all(m.shape==(512,512) and m.all() for m in masks)
    assert not native_inputs(image,image,np.zeros_like(image,bool),None)[1]
    assert not native_inputs(np.zeros_like(image),image)[1]
    with pytest.raises(ValueError):native_inputs(image[:384],image[:384])
    with pytest.raises(ValueError):native_inputs(image.astype(float),image)
    with pytest.raises(ValueError):native_inputs(image,image,np.ones((128,128),bool),None)
