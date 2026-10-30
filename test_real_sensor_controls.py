import numpy as np
import pytest

from dataset import sha256
from real_sensor_controls import real_image


def test_real_photo_loader_masks_native_nodata_before_area_resize(tmp_path):
    rasterio=pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    path=tmp_path/'real.tif'
    values=np.full((256,256),80,np.uint8)
    values[:16,:]=255
    with rasterio.open(path,'w',driver='GTiff',height=256,width=256,count=1,dtype='uint8',
        nodata=255,transform=from_origin(10,20,1,1)) as dst:
        dst.write(values,1)
    # Real assets normally have repository-relative paths; use a temporary cwd.
    import os
    old=os.getcwd()
    try:
        os.chdir(tmp_path)
        image,mask=real_image(dict(path='real.tif',sha256=sha256(path)))
    finally:
        os.chdir(old)
    assert image.shape==(128,128)
    assert not mask[:8].any() and mask[8:].all()
    assert (image[:8]==0).all() and (image[8:]==80).all()
