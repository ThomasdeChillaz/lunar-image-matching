"""A repeated physical point must not become its own contrastive negative."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from solar_training_v63 import TrainingBank


def make_bank(tmp_path, points):
    patches=np.full((points,2,33,33),100,np.uint8)
    teacher=np.full((points,2,24,32),np.nan,np.float32)
    available=np.zeros((points,2,24),bool)
    for i in range(points):
        teacher[i,:,::2]=np.eye(32,dtype=np.float32)[i%32]
        available[i,:,::2]=True
    arrays=dict(patches=patches,conditions=np.zeros((points,2,9),np.float32),
        target_suns=np.broadcast_to(np.array([0.,0.,1.],np.float32),(points,2,24,3)).copy(),
        teacher=teacher,available=available,observed_lights=np.zeros((points,2),np.int64))
    path=tmp_path/'samples.npz';np.savez_compressed(path,**arrays)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    groups=[f'procedural-only-{i}' for i in range(16)]
    records=[dict(component=g,split='training',source_lights=list(range(0,24,2)),path=str(path),sha256=digest)for g in groups]
    manifest=dict(schema_version=63,split='training',lighting_indices=list(range(0,24,2)),heldout_image_reads=0,
        teacher_descriptor_sha256='procedural',expected_frozen_descriptor_sha256='procedural',canonicalization_identical_to_inference=True,
        terrain_groups=groups,geometry_gate_passed=True,records=records)
    file=tmp_path/'manifest.json';file.write_text(json.dumps(manifest))
    return TrainingBank(file)


@pytest.mark.parametrize('points',(1,2,5,8,16))
def test_small_usable_bank_has_no_contradictory_duplicate_negative(tmp_path,points):
    bank=make_bank(tmp_path,points)
    batch,record=bank.sample(np.random.default_rng(20261101),augment=False)
    # Perfect teacher predictions must have nearly zero correspondence loss.
    # Repeated sampled identities create unavoidable log(k) loss instead.
    descriptors=batch['teacher_features'][:,0]
    pairs=batch['positive_pairs']
    a,b=descriptors[pairs[:,0]],descriptors[pairs[:,1]]
    labels=torch.arange(len(a))
    discrimination=F.cross_entropy(a@b.T/.1,labels)
    assert float(discrimination)<.001
    assert len(set(record['sample_indices']))==len(record['sample_indices'])
    assert len(record['sample_indices'])==min(8,points)
    assert batch['patches'].shape[0]==2*len(record['sample_indices'])
    assert torch.isnan(batch['teacher_features'][:,1::2]).all()


def test_small_bank_sensor_augmentation_and_replay_are_exact(tmp_path):
    bank=make_bank(tmp_path,2)
    a,sa=bank.sample(np.random.default_rng(20261102),augment=True)
    b,sb=bank.sample(np.random.default_rng(20261102),augment=True)
    assert sa==sb
    for name in ('patches','condition','target_suns','available'):
        assert torch.equal(a[name],b[name])
    assert torch.allclose(a['teacher_features'],b['teacher_features'],equal_nan=True)
    np.testing.assert_array_equal(a['positive_pairs'],b['positive_pairs'])


@pytest.mark.parametrize('count',(0,-1,True,1.5))
def test_invalid_physical_sample_count_is_rejected(tmp_path,count):
    bank=make_bank(tmp_path,2)
    with pytest.raises(ValueError,match='sample count'):
        bank.sample(np.random.default_rng(1),count=count)
