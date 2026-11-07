from types import SimpleNamespace

import numpy as np

import solar_render_bootstrap_v92 as module
from solar_assignment_v63 import CandidateGraph


def test_24_reference_views_remain_coarse_and_budgeted(monkeypatch):
    calls = []
    # Four admitted physical directions. The remaining twenty are still
    # matched on coarse images but have no exact above-horizon catalogue slot.
    angles = np.radians([0., 10., 20., 30.])
    suns = np.column_stack((np.sin(angles), np.zeros(4), np.cos(angles)))
    entries = [dict(light_index=i, sun_world=suns[i] if i < 4 else [0., 0., -1.]) for i in range(24)]
    geometry = SimpleNamespace(
        depth=np.full((512, 512), 100.), intrinsics=np.eye(3), cam2world=np.eye(4), sun_vectors=suns,
    )
    geometry_sha = module.geometry_digest(geometry.depth, geometry.intrinsics, geometry.cam2world)
    library = dict(map_mesh_sha256="mesh", scene_mesh_sha256="mesh",
                   reference_geometry_array_sha256=geometry_sha, entries=entries,
                   original_reference_preparation_seconds=99.)
    image = np.random.default_rng(123).integers(0, 256, (512, 512), dtype=np.uint8)

    class Matcher:
        def query_features(self, query):
            calls.append(("query", query.shape))
            return None

        def match(self, ref, query):
            calls.append(("match", ref.shape))
            p = np.array([[10., 10.], [12., 12.], [20., 10.], [25., 20.], [40., 30.], [80., 20.]])
            return p, p, 0.0

    def pose(world, q, k, config):
        calls.append(("ap3p", config["iterations"]))
        assert np.allclose(q[0], [41.5, 41.5])
        return np.eye(4), dict(inliers=6, reason=None, clipped_squared_reprojection_cost=0.)

    monkeypatch.setattr(module, "estimate", pose)
    bootstrap = module.RenderBootstrap.__new__(module.RenderBootstrap)
    bootstrap.direct = SimpleNamespace(libraries={"geometry": library}, matcher=Matcher(), image_loader=lambda entry: image)
    lookup = SimpleNamespace(protocol=dict(reference_geometry_sha256="geometry", mesh_sha256="mesh", scene_mesh_sha256="mesh"))
    inp = SimpleNamespace(
        reference=image, query=image, geometry=geometry, solar_lookup=lookup, query_intrinsics=np.eye(3),
        known_query_sun=None, reference_sun=suns[0],
    )
    ref, sun, seeds, trace = bootstrap.propose(inp)
    assert ref is image
    np.testing.assert_array_equal(sun, suns[0])
    assert len(seeds) == 4 and trace["requested_render_samples"] == 512
    assert calls == [("query", (128, 128))] + [("match", (128, 128))] * 24 + [("ap3p", 128)] * 4


def test_native_rounds_keep_original_total_and_single_final_estimator():
    y, x = np.mgrid[90:450:80, 90:450:80]
    p = np.column_stack((x.ravel(), y.ravel())).astype(float)
    k = np.array([[400., 0., 255.5], [0., 400., 255.5], [0., 0., 1.]])
    world = np.column_stack((p, np.ones(len(p)))) @ np.linalg.inv(k).T * 100.
    q = np.repeat(p[:, None], 8, axis=1)
    q[:, 1:, 0] += np.arange(1., 8.) * 9
    appearance = np.repeat(np.linspace(1., 0., 8)[None], len(p), axis=0)
    states = np.zeros((len(p), 1), dtype=np.uint8)
    graph = CandidateGraph(
        p, world, q, appearance, np.full(appearance.shape, 0.5), states, np.array([[0., 0., 1.]]),
        np.repeat(world[:, None], 7, axis=1), np.repeat(states[:, :, None], 7, axis=2),
    )
    seeds = [dict(pose=np.eye(4), sun_index=0, requested_samples=128, reason=None)] * 4
    events = []

    def measure(anchor, candidate, pose, sun):
        events.append("measure")
        return dict(query_xy=p[anchor].copy(), observed_shadow=0.5)

    def final(a, b):
        events.append("final")
        np.testing.assert_array_equal(a, b)
        return np.eye(4), dict(reason=None, inliers=len(a), iterations_requested=2000)

    result = module.solve_render_bootstrap(
        graph, k, measure, seeds, active=False, budget=32,
        final_estimator=final, pose_initialization="quantization_aware",
    )
    assert result["proposal_samples_requested"] == 4096
    assert result["final_estimator_iterations"] == 2000
    assert len(result["measurements"]) == 32
    assert events == ["measure"] * 32 + ["final"]


def test_seed_geometry_inserts_a_native_window_and_retains_other_edges():
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, image):
            value = torch.full((len(image), 32, 128, 128), 1 / np.sqrt(32), dtype=torch.float32)
            return value, None

    image = np.zeros((512, 512), np.uint8)
    bank = module.RotationBank(image, module.QUARTER_ANGLES)
    p = np.array([[128., 160.], [240., 340.]])
    k = np.array([[400., 0., 255.5], [0., 400., 255.5], [0., 0., 1.]])
    world = np.column_stack((p, np.ones(len(p)))) @ np.linalg.inv(k).T * 100.
    q = np.repeat(p[:, None], 8, axis=1)
    scores = np.zeros((2, 8), np.float32)
    rotation = np.zeros((2, 8), dtype=int)
    pose = np.eye(4)
    pose[0, 3] = 2.
    geometry = SimpleNamespace(cam2world=np.eye(4))
    q_new, score_new, r_new, count = module.insert_seed_windows(
        Encoder(), image, bank, p, q, scores, rotation, world, k, geometry,
        [dict(pose=pose)] * 4,
    )
    assert count == 2  # Repeated seeds do not take four copies of the same window.
    np.testing.assert_allclose(q_new[:, 4], p - [8., 0.], atol=1e-12)
    np.testing.assert_array_equal(q_new[:, :4], q[:, :4])
    np.testing.assert_array_equal(q_new[:, 5:], q[:, 5:])
    np.testing.assert_allclose(score_new[:, 4], 1.0, rtol=2 * np.finfo(np.float32).eps)
    np.testing.assert_array_equal(r_new, rotation)
