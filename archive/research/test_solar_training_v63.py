import json
import pytest
from solar_training_v63 import TrainingBank
from solar_experiments_v63 import variants


def test_training_loader_rejects_heldout_or_selection_manifest_before_opening_arrays(tmp_path):
    p=tmp_path/'bad.json'
    p.write_text(json.dumps(dict(schema_version=63,split='model_selection',lighting_indices=list(range(0,24,2)),heldout_image_reads=0)))
    with pytest.raises(ValueError,match='training-only'):TrainingBank(p)
    p.write_text(json.dumps(dict(schema_version=63,split='training',lighting_indices=list(range(24)),heldout_image_reads=0)))
    with pytest.raises(ValueError,match='training-only'):TrainingBank(p)


def test_all_eight_ablations_hold_patch_budget_fixed():
    cases=variants();assert len(cases)==8
    assert len({(x['solar_mode'],x['response'],x['active'])for x in cases})==8
    assert {x['budget']for x in cases}=={128}
    assert not any(x['pooled']for x in cases)
