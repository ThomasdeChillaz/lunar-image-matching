import hashlib
import io
import zipfile
import numpy as np
import pytest
from infer_dense_shard import load_inputs


def test_remote_loader_recovers_exact_native_arrays_and_rejects_changes(tmp_path):
    arrays = dict(reference=np.arange(64, dtype=np.uint8).reshape(8,8),
        query=np.arange(64, dtype=np.uint8).reshape(8,8),
        reference_mask=np.ones((8,8), bool), query_mask=np.ones((8,8), bool))
    case = dict(files={name: name+'.npy' for name in arrays},
        input_array_sha256={name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in arrays.items()})
    path = tmp_path/'input.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, a in arrays.items():
            stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
            archive.writestr(case['files'][name], stream.getvalue())
    with zipfile.ZipFile(path) as archive:
        loaded=load_inputs(archive,case)
        assert all(np.array_equal(loaded[name],a) for name,a in arrays.items())
        case['input_array_sha256']['query']='changed'
        with pytest.raises(ValueError,match='Remote native input bytes differ'):
            load_inputs(archive,case)
