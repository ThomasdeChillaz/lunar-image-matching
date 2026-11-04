"""Procedural canonical-support masking; no actual image/model extraction."""
from copy import deepcopy
import numpy as np
import pytest

import solar_training_bank_v68 as bank
from build_solar_training_bank_v68 import validate_record_arrays
from lunar_geometry_v5 import world_points_z, project_z
from solar_native_v63 import canonical_patch


SEED = 20261101
JITTER = dict(translation_sigma_m=25., rotation_sigma_deg=1.5)


def boundary_points():
    k = np.array([[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]])
    reference = np.diag([1.,-1.,-1.,1.]); reference[:3,3] = [0.,0.,100.]
    query = reference.copy(); query[:3,3] += [-2.,-1.,0.]
    geometry = {role:dict(intrinsics=k.copy(), cam2world=pose.copy(), axial_depth=np.full((512,512),100.))
        for role, pose in (("reference",reference),("query",query))}
    p = np.array([[24.,24.],[480.,24.],[24.,480.],[480.,480.],
        [180.,200.],[210.,200.],[240.,200.],[270.,200.],
        [180.,240.],[210.,240.],[240.,240.],[270.,240.]])
    world = world_points_z(p, np.full(len(p),100.), k, reference)
    q, _, front = project_z(world, k, query); assert front.all()
    zeros = np.zeros((512,512),np.uint8); points = []
    for i, (source, target, xyz) in enumerate(zip(p,q,world)):
        pair, maps = canonical_patch(zeros, zeros, source, geometry["reference"]["axial_depth"], k, reference, k, query, query_center=target)
        assert pair is not None  # Every original exact-geometry sample is supported.
        points.append(dict(label_index=i, reference_xy=source, query_xy=target, world=xyz,
            normal=np.array([0.,0.,1.]), normal_provenance={},
            reference_map=maps["reference_xy"], query_map=maps["query_xy"]))
    return geometry, points


class ProceduralReader:
    def __init__(self): self.decoded = []; self.cache = {}
    def get(self, role, light):
        assert role in bank.ROLES and light in bank.EVEN
        key = (role,light)
        if key not in self.cache:
            y,x = np.mgrid[:512,:512]
            self.cache[key] = ((x+2*y+3*light+17*(role=="query"))%256).astype(np.uint8)
            self.decoded.append(dict(role=role,light=light))
        return self.cache[key]


def analytical_features(patches):
    mean = np.asarray(patches).mean((1,2))/255.
    result = np.zeros((len(mean),32),np.float32)
    result[:,0] = np.cos(mean); result[:,1] = np.sin(mean)
    return result


def test_boundary_jitter_masks_support_without_reroll_reindex_or_relabelling(monkeypatch):
    geometry, points = boundary_points(); before = deepcopy(points); reader = ProceduralReader()
    original_trial = bank.training_trial_pose; draws = []; teacher_inputs = []
    def trial(*args, **kwargs): draws.append(kwargs.copy()); return original_trial(*args, **kwargs)
    def encode(encoder, patches, *args): teacher_inputs.append(patches.copy()); return analytical_features(patches)
    monkeypatch.setattr(bank,"training_trial_pose",trial); monkeypatch.setattr(bank,"encode_teacher",encode)
    arrays, details = bank.build_unit_arrays(geometry,points,reader,86.,object(),SEED,trial_jitter=JITTER)
    kept = [2,4,5,6,7,8,9,10,11]
    assert details["selected_indices"] == kept and validate_record_arrays(arrays) == 9
    assert [r["sample_index"] for r in details["canonical_trial_support_failures"]] == [0,1,3]
    assert details["omitted_trial_support_samples"] == 3 and details["omitted_unusable_teacher_samples"] == 0
    assert details["omitted_samples_total"] == 3 and details["trial_support_is_not_visibility_or_negative_label"]
    assert draws == [JITTER] and len(teacher_inputs) == 1 and len(teacher_inputs[0]) == 9*2*12
    expected_trial = original_trial(geometry["query"]["cam2world"],np.random.default_rng(SEED),**JITTER)
    np.testing.assert_array_equal(details["query_training_trial_pose"],expected_trial)
    original_index = np.arange(len(points)); light_index = (original_index+SEED%12)%12
    observed = np.column_stack([np.asarray(bank.EVEN)[light_index], np.asarray(bank.EVEN)[(light_index+5+original_index//12)%12]])
    np.testing.assert_array_equal(arrays["observed_lights"],observed[kept])
    ref, query = geometry["reference"], geometry["query"]
    for row, i in enumerate(kept):
        pair, _ = canonical_patch(reader.get("reference",int(observed[i,0])),reader.get("query",int(observed[i,1])),
            points[i]["reference_xy"],ref["axial_depth"],ref["intrinsics"],ref["cam2world"],query["intrinsics"],expected_trial,query_center=points[i]["query_xy"])
        np.testing.assert_array_equal(arrays["patches"][row],np.stack(pair))
    assert arrays["available"][:,:,::2].all() and np.isnan(arrays["teacher"][:,:,1::2]).all()
    assert not arrays["available"][:,:,1::2].any() and {r["light"] for r in reader.decoded} == set(bank.EVEN)
    for current, original in zip(points,before):
        for key in ("reference_xy","query_xy","world","normal"):
            np.testing.assert_array_equal(current[key],original[key])


def test_teacher_unavailability_is_counted_separately_from_support(monkeypatch):
    geometry, points = boundary_points(); reader = ProceduralReader()
    def encode(encoder, patches, *args):
        result = analytical_features(patches)
        # The first of nine supported points has unavailable descriptors at
        # both roles/every even light. It remains distinct from missing support.
        for offset in range(0,len(result),18): result[offset:offset+2] = 0.
        return result
    monkeypatch.setattr(bank,"encode_teacher",encode)
    arrays, details = bank.build_unit_arrays(geometry,points,reader,86.,object(),SEED,trial_jitter=JITTER)
    assert details["selected_indices"] == list(range(4,12)) and validate_record_arrays(arrays) == 8
    assert details["omitted_trial_support_samples"] == 3 and details["omitted_unusable_teacher_samples"] == 1
    assert details["omitted_samples_total"] == 4 and len(details["zero_norm_teacher_targets"]) == 24
    assert {key[0] for key in details["zero_norm_teacher_targets"]} == {2}
    assert np.isnan(arrays["teacher"][:,:,1::2]).all() and arrays["available"][:,:,::2].all()


def test_no_trial_support_is_explicit_shortfall_before_images_or_teacher(monkeypatch):
    geometry, points = boundary_points(); reader = ProceduralReader()
    monkeypatch.setattr(bank,"encode_teacher",lambda *a:pytest.fail("No teacher forward without canonical input support"))
    with pytest.raises(ValueError,match="No paired sample with canonical support"):
        bank.build_unit_arrays(geometry,[points[i] for i in (0,1,3)],reader,86.,object(),SEED,trial_jitter=JITTER)
    assert not reader.decoded
