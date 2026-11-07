import argparse
import io
import pickle
import zipfile

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from extract_moonst3r_v17 import decode_metadata,tensor_values


def test_tensor_extractor_recovers_values_and_inert_arguments_without_optimizer_loading():
    matrix=torch.arange(24,dtype=torch.float32).reshape(3,8)
    state={'encoder.weight':matrix,'transposed.weight':matrix.T,
        'head.bias':torch.tensor([.25,-.5],dtype=torch.float32)}
    buffer=io.BytesIO();torch.save(dict(model=state,args=argparse.Namespace(model='UnevaluatedModel()',train_dataset='fixture'),optimizer={}),buffer)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as bundle:
        path=next(n for n in bundle.namelist() if n.endswith('/data.pkl'));metadata=decode_metadata(bundle.read(path));prefix=path[:-8]
        assert metadata['args'].model=='UnevaluatedModel()'
        for name,spec in metadata['model'].items():
            np.testing.assert_array_equal(tensor_values(bundle,prefix,spec),state[name].numpy())


def test_unknown_pickle_globals_are_rejected_without_calling_them(tmp_path):
    marker=tmp_path/'executed.txt'
    class Attack:
        def __reduce__(self):return (eval,("__import__('pathlib').Path("+repr(str(marker))+").write_text('bad')",))
    with pytest.raises(ValueError,match='Unsupported checkpoint global'):
        decode_metadata(pickle.dumps(Attack(),protocol=2))
    assert not marker.exists()
