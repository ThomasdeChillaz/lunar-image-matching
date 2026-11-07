import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

import solar_distributional_evaluation_v90 as adapter
from solar_frontend_v63 import ReferenceGeometry, candidates, reference_anchors
from solar_inference_cache_v84 import SolarMatcherV84


class ToyCoarse(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, image):
        self.events.append(("coarse", tuple(image.shape[-2:])))
        z = torch.cat((image, torch.ones_like(image), image.expand(-1, 30, -1, -1) * 0.01), 1)
        return torch.nn.functional.normalize(z, dim=1), None


class ToyDistribution(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.coarse = ToyCoarse(events)
        self.parameter_count_stub = torch.nn.Parameter(torch.zeros(95116))
        self.events = events

    def coarse_reference_features(self, field, points):
        self.events.append(("reference_field", tuple(field.shape[-2:])))
        dense, _ = self.coarse(field[:, :1])
        return adapter.sample_descriptors(dense, points)

    def marginal_distribution(self, base, condition, suns, *, diagonal_only):
        assert not diagonal_only
        self.events.append(("marginal", len(suns)))
        mean = base.clone()
        mean[:, 0] += suns[:, 0]
        covariance = torch.eye(32).expand(len(base), -1, -1).clone()
        return dict(mean=mean, covariance=covariance)


def toy_geometry():
    depth = np.full((512, 512), 100.0)
    k = np.array([[400.0, 0.0, 255.5], [0.0, 400.0, 255.5], [0.0, 0.0, 1.0]])
    p = reference_anchors()
    xyz = (np.column_stack((p, np.ones(len(p)))) @ np.linalg.inv(k).T) * 100.0
    world = np.repeat(xyz[:, None], 7, axis=1)
    return ReferenceGeometry(
        depth, k, np.eye(4), np.tile([0.0, 0.0, 1.0], (256, 1)), np.array([[0.0, 0.0, 1.0]]),
        np.zeros((256, 1), np.uint8), np.zeros((256, 1, 7), np.uint8), np.array([0.0, 0.0, 1.0]),
        world,
    )


def toy_image():
    return np.random.default_rng(123).integers(0, 256, (512, 512), dtype=np.uint8)


def test_sun_moment_matching_keeps_between_component_covariance():
    model = ToyDistribution([])
    base = torch.zeros((2, 32))
    condition = torch.zeros((2, 10))
    suns = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mean, covariance = adapter.marginal_sun_moments(model, base, condition, suns)
    assert torch.equal(mean, torch.zeros((2, 32), dtype=torch.float64))
    assert torch.equal(covariance[0], torch.diag(torch.tensor([2.0] + [1.0] * 31, dtype=torch.float64)))
    query = torch.zeros((2, 32))
    query[1, 0] = 1.0
    scores = adapter._learned_similarity(mean, covariance, query)
    assert torch.all(scores[:, 0] == 1.0)
    assert torch.all(scores[:, 1] < 1.0)


def test_zero_fusion_reproduces_protected_v84_candidate_arrays(monkeypatch):
    model = ToyDistribution([])
    geom = toy_geometry()
    a, b = toy_image(), np.roll(toy_image(), 16, axis=1)
    monkeypatch.setattr(adapter, "FUSION_WEIGHT", 0.0)
    frozen = candidates(model.coarse, a, b)
    got = adapter.distributional_candidates(model, a, b, geom, None, geom.sun_vectors)
    assert all(np.array_equal(x, y) for x, y in zip(frozen, got[:4]))
    assert got[4]["learned_supported_anchors"] == 256


def test_frontend_copy_has_only_proposal_swap_and_reporting_changes():
    inherited = inspect.getsource(SolarMatcherV84.localize)
    actual = inspect.getsource(adapter.DistributionalCoarseMatcher.localize)
    first = "            proposal_sun_ids ="
    start = actual.index(first)
    end = actual.index("            world, use =", start)
    restored = actual[:start] + "            p, q, appearance, rotation = candidates(self.coarse, reference, query, self.proposal_angles)\n" + actual[end:]
    restored = restored.replace("                result['coarse_response'] = distributional_cost\n", "")
    assert restored == inherited
    assert hashlib.sha256(Path(inspect.getsourcefile(SolarMatcherV84)).read_bytes()).hexdigest() == adapter.BASELINE_FRONTEND_SHA256


def test_native_callback_stays_deferred_to_v84_solver(monkeypatch):
    events = []

    class Refiner:
        def __init__(self, encoder):
            self.forward_calls = 0

        def refine(self, reference, query, ref_xy, query_xy):
            events.append(("native", 1))
            self.forward_calls += 1
            return query_xy, {}

    def solver(graph, intrinsics, measure, *args, **kwargs):
        assert graph.query_xy.shape == (256, 8, 2)
        assert not any(event[0] == "native" for event in events)
        for anchor in range(256):
            got = measure(anchor, 0, None, 0)
            if got.get("query_xy") is not None:
                break
        assert [event for event in events if event[0] == "native"] == [("native", 1)]
        return {"pose": None}

    monkeypatch.setattr(adapter, "SparseNativeRefiner", Refiner)
    monkeypatch.setattr(adapter, "solve_v84", solver)
    matcher = adapter.DistributionalCoarseMatcher(ToyDistribution(events))
    g = toy_geometry()
    result = matcher.localize(toy_image(), toy_image(), g, g.intrinsics, active=True, budget=32)
    assert result["native_encoder_forwards"] == 1
    assert result["coarse_response"]["native_view_128_teacher"]
    assert [shape for kind, shape in events if kind in {"coarse", "reference_field"}] == [(128, 128)] * 3
