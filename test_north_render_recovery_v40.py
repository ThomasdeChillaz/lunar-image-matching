"""Recovery preserves prior bytes and the unchanged image/label kernel."""
import hashlib
import json
from pathlib import Path

import pytest

from north_render_recovery_v40 import copy_component,sha,verify_component


def fixture(tmp_path):
    folder=tmp_path/'prior';folder.mkdir();images=[];geometry={};labels=[];exact=[]
    for view in range(4):
        for light in range(6):
            p=folder/f'view-{view}'/f'light-{light}'/'image.png';p.parent.mkdir(parents=True)
            p.write_bytes(f'image {view} {light}'.encode())
            images.append(dict(view=view,light=light,image_sha256=sha(p)))
        p=folder/f'view-{view}'/'geometry.npz';p.write_bytes(b'geometry')
        geometry[str(p.relative_to(folder))]=sha(p)
    for view in (1,2,3):
        p=folder/f'correspondences-0-{view}.npz';p.write_bytes(b'dense')
        labels.append(dict(query_view=view,sha256=sha(p)))
    for i in range(15):
        p=folder/f'supervision-{i}.npz';p.write_bytes(b'exact')
        exact.append(dict(file=p.name,sha256=sha(p)))
    (folder/'summary.json').write_text(json.dumps(dict(component='fixture',images=images,
        labels=labels,geometry_sha256=geometry,exact_supervision=exact)))
    record=dict(component='fixture',files_sha256={str(p.relative_to(folder)):sha(p)
        for p in folder.rglob('*') if p.is_file()})
    return folder,record


def test_component_copy_is_byte_identical_and_cannot_mutate_original(tmp_path):
    source,record=fixture(tmp_path);dest=tmp_path/'recovery'
    copy_component(source,dest,record)
    p=Path('view-0/light-0/image.png')
    assert (source/p).read_bytes()==(dest/p).read_bytes()
    assert (source/p).stat().st_ino!=(dest/p).stat().st_ino
    (dest/p).write_bytes(b'changed recovery copy')
    verify_component(source,record)
    with pytest.raises(ValueError,match='bytes changed'):verify_component(dest,record)


def test_component_corruption_refuses_reuse_before_copy(tmp_path):
    source,record=fixture(tmp_path);(source/'supervision-4.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='bytes changed'):
        copy_component(source,tmp_path/'recovery',record)
    assert not (tmp_path/'recovery').exists()


def test_existing_recovery_is_never_overwritten(tmp_path):
    source,record=fixture(tmp_path);dest=tmp_path/'recovery';dest.mkdir()
    with pytest.raises(FileExistsError):copy_component(source,dest,record)


def test_actual_render_and_supervision_kernel_is_identical_to_v37():
    start='            with np.load(path, allow_pickle=False) as f:'
    end='            records.append(item)'
    kernel=lambda name:Path(name).read_text().split(start)[1].split(end)[0]
    old=kernel('render_north_development_v37.py');new=kernel('render_north_development_v40.py')
    assert old==new
    contract=json.loads(Path('reports/north_render_recovery_contract_v40.json').read_text())
    assert hashlib.sha256(new.encode()).hexdigest()==contract['identical_render_kernel_sha256']
    assert contract['original_v37_deadline_met'] is False
    assert contract['supplemental_max_calendar_seconds']==1800
