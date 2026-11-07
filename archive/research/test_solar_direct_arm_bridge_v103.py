"""Procedural metadata controls with a recording stand-in for Direct inference."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from solar_direct_arm_bridge_v103 import bind_case, MAX_LOOKUP_METADATA_BYTES
from solar_evaluation_v70 import InferenceInput


class Recorder:
    def __init__(self):
        self.received = None

    def infer(self, inp):
        self.received = inp
        return {"sentinel": True}


class DirectBridgeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.name = "reports/r/lookup.json"
        self.camera = "runs/known/component/motion-0/reference/geometry.npz"
        self.archive = "runs/precomputed/geometry.npz"
        self.mesh = "data/full-map.npz"
        self.audit = "reports/audit.json"
        self.lookup = dict(
            cpu_threads=2, schema_version=70, query_truth_used=False, reference_geometry=self.camera,
            reference_geometry_sha256="a" * 64, mesh=self.mesh, mesh_sha256="b" * 64,
            scene_mesh=self.mesh, scene_mesh_sha256="b" * 64, approved_reference_key="component/motion-0",
            geometry_audit=self.audit, geometry_audit_sha256="c" * 64,
        )
        self.case = dict(split="model_selection", component="component", unit_index=3,
                         reference_geometry=self.archive, solar_lookup_protocol=self.name,
                         scoring_geometry="not_read", secondary_known_query_sun_world="not_read")
        self.meta = self.case["reference_asset_metadata"] = dict(
            component="component", unit_index=3, reference_geometry=self.archive,
            reference_geometry_sha256="d" * 64, source_reference_geometry_path=self.camera,
            source_reference_geometry_sha256="a" * 64, scene_mesh_path=self.mesh,
            scene_mesh_sha256="b" * 64, source_audit_sha256="c" * 64, solar_lookup_protocol=self.name,
        )
        self.recorder = Recorder()
        self.inp = InferenceInput(object(), object(), object(), object(), object())
        self.reseal_fixture()

    def reseal_fixture(self):
        raw = json.dumps(self.lookup).encode()
        self.raw = raw
        pin = sha256(raw).hexdigest()
        self.meta["solar_lookup_protocol_sha256"] = pin
        self.case["input_sha256"] = {self.name: pin, self.camera: "a" * 64, self.mesh: "b" * 64,
                                      self.archive: "d" * 64, self.audit: "c" * 64}
        self.execution = dict(self.case["input_sha256"])
        path = self.root / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def bind(self, proof=None):
        return bind_case(self.recorder, self.case, self.execution, self.root,
                         public_bytes=proof or (lambda root, name: self.raw))

    def test_truth_free_input_is_copied_and_only_three_digests_reach_direct(self):
        bound = self.bind()
        self.assertEqual(bound.infer(self.inp), {"sentinel": True})
        actual = self.recorder.received
        self.assertIsNone(self.inp.solar_lookup)
        self.assertIs(actual.geometry, self.inp.geometry)
        self.assertIs(actual.query, self.inp.query)
        self.assertIsNone(actual.known_query_sun)
        self.assertEqual(set(actual.solar_lookup.protocol),
                         {"reference_geometry_sha256", "mesh_sha256", "scene_mesh_sha256"})
        with self.assertRaises(TypeError):
            actual.solar_lookup.protocol["mesh_sha256"] = "tampered"

    def test_wrong_execution_pin_and_changed_public_bytes_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "original public"):
            self.bind(lambda root, name: b"other")
        self.execution[self.name] = "0" * 64
        with self.assertRaisesRegex(ValueError, "same reference metadata input"):
            self.bind()
        self.assertIsNone(self.recorder.received)

    def test_map_and_camera_substitution_fail_even_with_updated_lookup_hash(self):
        for key, value in (("mesh_sha256", "0" * 64), ("reference_geometry_sha256", "0" * 64),
                           ("approved_reference_key", "other/camera"), ("query_truth_used", True)):
            with self.subTest(key=key):
                old = self.lookup[key]
                self.lookup[key] = value
                self.reseal_fixture()
                with self.assertRaisesRegex(ValueError, "full reference map and camera"):
                    self.bind()
                self.lookup[key] = old

    def test_primary_only_and_no_legacy_online_provider(self):
        bound = self.bind()
        for modified in (replace(self.inp, known_query_sun=object()), replace(self.inp, solar_lookup=object())):
            with self.assertRaisesRegex(ValueError, "truth-free"):
                bound.infer(modified)
        self.case["split"] = "final"
        with self.assertRaisesRegex(ValueError, "development cohort"):
            self.bind()

    def test_oversized_or_changed_lookup_file_fails_before_public_proof(self):
        path = self.root / self.name
        path.write_bytes(b"x" * (MAX_LOOKUP_METADATA_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "bounded"):
            self.bind()
        path.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "original public"):
            self.bind()


if __name__ == "__main__":
    unittest.main()
