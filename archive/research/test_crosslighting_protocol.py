from collections import Counter
import json
from pathlib import Path
import numpy as np
import pytest
from evaluate_crosslighting import plan, prepare_pair, WARP
from geometry import transform
from verify_crosslighting import translated_input


def test_complete_sun_pair_plan_includes_non_anchor_pairs_without_duplicates():
    manifest = json.loads(Path('moon_manifest.json').read_text())
    cases = plan(manifest,'test')
    assert len(cases) == 16*36 and len({c['pair_id'] for c in cases}) == 16*36
    assert set(Counter(c['tile_id'] for c in cases).values()) == {36}
    assert sum(c['reference_light_index'] == 0 for c in cases) == 16*8
    assert all(c['reference_light_index'] < c['query_light_index'] for c in cases)


def test_every_sun_pair_gets_same_nonidentity_geometry_and_valid_targets():
    reference = np.zeros((128,128),np.uint8)
    reference[60,40] = 200
    valid = np.ones((128,128),bool)
    query, mask, matrix, points, targets = prepare_pair(reference,reference,valid,valid)
    assert WARP['angle'] == 0 and np.allclose(matrix,[[1,0,6],[0,1,-4]])
    assert query[56,46] == 200 and query[60,40] == 0
    reconstructed,reconstructed_mask = translated_input(reference,valid)
    assert np.array_equal(reconstructed,query) and np.array_equal(reconstructed_mask,mask)
    assert np.allclose(targets-points,[6,-4])
    assert np.allclose(transform(targets,np.array([[1,0,-6],[0,1,4]])),points)
    assert mask[np.rint(targets[:,1]).astype(int),np.rint(targets[:,0]).astype(int)].all()
    with pytest.raises(ValueError,match='No valid known-geometry'):
        prepare_pair(reference,reference,~valid,valid)
