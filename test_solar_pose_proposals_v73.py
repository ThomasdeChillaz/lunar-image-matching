"""Geometry-only controls; no lunar accuracy or learned-model result."""
import numpy as np
import pytest
from diagnose_pose_quantization_v73 import fixture, error
from solar_pose_proposals_v73 import propose, QUANTIZATION_DISC_NATIVE_PX
from localization_robust_pose_v4 import estimate


def test_quantization_aware_raw_initializer_reduces_rounded_grid_bias():
    old, new = [], []
    for seed in (1, 3, 5, 7):
        world, pixels, k, truth = fixture(seed, 16)
        a, _ = estimate(world, pixels, k, dict(solver='ap3p', iterations=128, reclassify=True))
        b, info = propose(world, pixels, k, 128)
        old.append(error(a, truth)['translation_m'])
        new.append(error(b, truth)['translation_m'])
        assert info['minimal_sets_executed'] == 128
        assert info['accepted'] is False
        assert info['final_reprojection_threshold_native_px'] == 3.
    # This is a constructed diagnosis of a coarse-grid estimation bias.
    # It is not a learned/localization benchmark or a generalization test.
    assert np.median(new) < .3*np.median(old)


def test_initial_cell_consensus_handles_outliers_and_refuses_degenerate_data():
    world, pixels, k, truth = fixture(1, 16, .4)
    pose, info = propose(world, pixels, k, 128)
    assert pose is not None and error(pose, truth)['rotation_deg'] < 2
    assert info['quantization_disc_native_px'] == QUANTIZATION_DISC_NATIVE_PX
    none, info = propose(np.zeros((8, 3)), np.ones((8, 2))*100, k, 128)
    assert none is None and info['reason'] == 'degenerate_coarse_geometry'
    assert propose(world[:5], pixels[:5], k)[0] is None
    with pytest.raises(ValueError): propose(world, pixels, k, 4096)


def test_quantized_initialization_keeps_final_native_estimator_and_budget():
    import solar_assignment_v63 as module
    from lunar_geometry_v5 import world_points_z
    from native_baseline_pose_v7 import propose_pose
    from localize_terrain import public_result
    axis = np.linspace(60, 450, 8)
    y, x = np.meshgrid(axis, axis, indexing='ij')
    p = np.column_stack([x.ravel(), y.ravel()])
    k = np.array([[430., 0, 255.5], [0, 430., 255.5], [0, 0, 1.]])
    world = world_points_z(p, np.full(len(p), 100.), k, np.eye(4))
    target = p + [8., -4.]
    coarse = 49.5 + 16*np.rint((target-49.5)/16)
    states = np.ones((len(p), 1), np.uint8)
    graph = module.CandidateGraph(p, world, coarse[:, None], np.ones((len(p), 1)),
        np.zeros((len(p), 1)), states, np.array([[0., 0., 1.]]),
        np.repeat(world[:, None], 7, 1), np.repeat(states[:, :, None], 7, 2))
    calls = []
    def final(a, b):
        pose, info = propose_pose(a, b, np.full((512, 512), 100.), k, np.eye(4), k)
        calls.append((a.copy(), b.copy()))
        return pose, dict(info, iterations_requested=2000, reprojection_threshold_native_px=3.)
    result = module.solve(graph, k, lambda i, c, *unused: dict(query_xy=target[i]),
        solar_mode='off', budget=32, final_estimator=final,
        pose_initialization='quantization_aware')
    assert len(calls) == 1 and result['pose'] is not None
    assert result['proposal_samples_requested'] == 4096
    assert result['final_estimator_iterations'] == 2000
    assert result['initialization']['policy'] == 'quantization_aware'
    assert result['initialization']['proposal_info'][0]['minimal_sets_executed'] == 1024
    assert result['initialization']['final_reprojection_threshold_native_px'] == 3.
    assert public_result(result)['pose'] is None and not public_result(result)['accepted']
