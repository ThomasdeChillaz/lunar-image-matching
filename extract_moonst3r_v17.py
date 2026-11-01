"""Read pinned MOONSt3R tensors without executing checkpoint classes or code.

Only inert metadata stand-ins are unpickled. Four exact global names are
accepted; tensor bytes are separately read from CRC-checked ZIP members.
The full upstream file must first match its published Hugging Face LFS hash.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import sys
import zipfile

import numpy as np

EXPECTED_SHA256='4ecc74a6f8190d2cdc3837c2cc1b101a0feffa84b114a5a27c1aaa9ee91d1725'
EXPECTED_BYTES=5821036807


class FloatStorageTag:
    pass


class ArgumentsTag:
    pass


@dataclass(frozen=True)
class StorageSpec:
    key:str
    elements:int


@dataclass(frozen=True)
class TensorSpec:
    storage:StorageSpec
    offset:int
    shape:tuple
    stride:tuple


def tensor_spec(storage,offset,shape,stride,*unused):
    if not isinstance(storage,StorageSpec) or not isinstance(offset,int) or offset<0:
        raise ValueError('Invalid inert tensor metadata')
    if len(shape)>8 or len(shape)!=len(stride) or any(not isinstance(v,int) or v<0 for v in (*shape,*stride)):
        raise ValueError('Invalid tensor dimensions/strides')
    maximum=offset+sum((n-1)*s for n,s in zip(shape,stride) if n)
    if maximum>=storage.elements and all(shape):raise ValueError('Tensor view exceeds storage')
    return TensorSpec(storage,offset,tuple(shape),tuple(stride))


class MetadataReader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):OrderedDict,('argparse','Namespace'):ArgumentsTag,
            ('torch','FloatStorage'):FloatStorageTag,('torch._utils','_rebuild_tensor_v2'):tensor_spec}
        if (module,name) not in allowed:raise ValueError('Unsupported checkpoint global: '+module+'.'+name)
        return allowed[(module,name)]

    def persistent_load(self,value):
        if (not isinstance(value,tuple) or len(value)!=5 or value[0]!='storage'
                or value[1] is not FloatStorageTag or not re.fullmatch(r'[0-9]+',str(value[2]))
                or not isinstance(value[4],int) or not 0<=value[4]<=1000000000):
            raise ValueError('Unsupported storage metadata')
        return StorageSpec(str(value[2]),value[4])


def decode_metadata(data):
    if len(data)>8*1024*1024:raise ValueError('Checkpoint metadata exceeds the bound')
    value=MetadataReader(io.BytesIO(data)).load()
    if not isinstance(value,dict) or not isinstance(value.get('model'),dict):raise ValueError('Expected model state mapping')
    model=value['model']
    if not 1<=len(model)<=5000 or any(not isinstance(k,str) or not isinstance(v,TensorSpec) for k,v in model.items()):
        raise ValueError('Unexpected tensor-state schema')
    return value


def tensor_values(bundle,prefix,spec):
    name=prefix+'data/'+spec.storage.key;info=bundle.getinfo(name)
    if info.file_size!=4*spec.storage.elements or info.file_size>512*1024*1024:raise ValueError('Unexpected tensor storage length')
    values=np.frombuffer(bundle.read(info),dtype='<f4')
    # The inert parser already bounds the largest reachable storage index.
    # Materialize bounded positive-stride views, including transposed weights
    # and arbitrary singleton strides, without changing their actual values.
    if not spec.shape:return np.array(values[spec.offset],dtype=np.float32)
    elements=1
    for n in spec.shape:elements*=n
    if elements*4>512*1024*1024:raise ValueError('Materialized tensor exceeds the declared bound')
    return np.lib.stride_tricks.as_strided(values[spec.offset:],shape=spec.shape,
        strides=tuple(v*4 for v in spec.stride),writeable=False).copy()


def extract(checkpoint,output,dependency_dir=None):
    checkpoint,output=Path(checkpoint),Path(output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Preserve earlier extracted weights')
    if checkpoint.stat().st_size!=EXPECTED_BYTES:raise ValueError('Upstream checkpoint length differs')
    with checkpoint.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=EXPECTED_SHA256:raise ValueError('Published full-checkpoint SHA256 differs')
    if dependency_dir:sys.path.insert(0,str(Path(dependency_dir).resolve()))
    import torch
    from safetensors.torch import save_file
    tensors={};storage_bytes=0
    with zipfile.ZipFile(checkpoint) as bundle:
        paths=[p for p in bundle.namelist() if p.endswith('/data.pkl') or p=='data.pkl']
        if len(paths)!=1:raise ValueError('Unique checkpoint metadata required')
        prefix=paths[0][:-len('data.pkl')];data=bundle.read(paths[0]);metadata=decode_metadata(data)
        if prefix+'byteorder' in bundle.namelist() and bundle.read(prefix+'byteorder')!=b'little':raise ValueError('Unexpected checkpoint byte order')
        for name,spec in metadata['model'].items():
            values=tensor_values(bundle,prefix,spec);storage_bytes+=values.nbytes
            tensors[name]=torch.from_numpy(values)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(output.suffix+'.part')
    if temporary.exists():raise FileExistsError('Earlier partial conversion retained')
    save_file(tensors,str(temporary));temporary.replace(output)
    arguments=getattr(metadata.get('args'), '__dict__',{})
    proof=dict(passed=True,source_checkpoint_sha256=digest,source_checkpoint_bytes=EXPECTED_BYTES,
        extracted_tensors=len(tensors),tensor_bytes=storage_bytes,output_sha256=sha256_file(output),
        metadata_sha256=hashlib.sha256(data).hexdigest(),model_configuration=arguments.get('model'),
        train_dataset=arguments.get('train_dataset'),test_dataset=arguments.get('test_dataset'),epoch=metadata.get('epoch'),
        source_sha256=sha256_file(__file__),
        method='Strict four-global inert metadata parser; CRC-checked Float32 ZIP tensor storage. No checkpoint eval, author-class execution, optimizer tensor loading or inference.',
        license_note='Upstream model-card weight license unspecified; local research artifact, not authorized for redistribution. Nested MASt3R/DUSt3R code carries CC BY-NC-SA4.0.')
    output.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('train_dataset','test_dataset')},indent=2));return proof


def sha256_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--dependency-dir');args=parser.parse_args()
    extract(args.checkpoint,args.output,args.dependency_dir)
