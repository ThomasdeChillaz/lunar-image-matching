import inspect
import numpy as np
import pytest

torch = pytest.importorskip('torch')
import pose_recovery_v14 as recovery


def test_broad_pose_recovers_known_camera_without_query_geometry():
    rng = np.random.default_rng(19)
    world = np.column_stack([rng.uniform(-2, 2, (70, 2)), rng.uniform(8, 12, 70)])
    k = np.array([[300., 0, 255.5], [0, 300., 255.5], [0, 0, 1.]])
    camera = world-[.2, -.1, .3]; projected = camera@k.T
    q = projected[:, :2]/projected[:, 2, None]
    pose, info = recovery.broad_proposal(world, q, k)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(pose[:3, 3], [.2, -.1, .3], atol=1e-5)
    assert info['inliers'] == 70 and info['proposal_threshold_px'] == 12
    assert not {'query_depth', 'query_pose', 'query_cam2world'} & set(inspect.signature(recovery.recover).parameters)


def test_broad_pose_rejects_invalid_and_degenerate_geometry():
    k = np.eye(3)
    assert recovery.broad_proposal(np.zeros((5, 3)), np.zeros((5, 2)), k)[0] is None
    assert recovery.broad_proposal(np.zeros((8, 3)), np.zeros((8, 2)), k)[0] is None
    with pytest.raises(ValueError):
        recovery.broad_proposal(np.zeros((7, 3)), np.zeros((8, 2)), k)


def test_broad_proposal_is_never_returned_when_strict_updates_fail(monkeypatch):
    from types import SimpleNamespace
    p = np.array([[40, 40], [50, 40], [60, 40], [40, 50], [50, 50], [60, 50]], np.float32)
    branch = dict(p=p, q=p, pose=None, info={}, quarter_turn=0)
    coarse = SimpleNamespace(model=torch.nn.Identity())
    monkeypatch.setattr(recovery, 'initialize', lambda *a, **kw: (None, [branch]))
    monkeypatch.setattr(recovery, 'broad_proposal', lambda *a: (np.eye(4), dict(inliers=6, median_proposal_residual=0)))
    def failed(*args):
        return dict(args[-1], geometry_history=[dict(iteration=1, estimated=False)])
    monkeypatch.setattr(recovery, 'refine_surface', failed)
    image = np.zeros((100, 100), np.uint8)
    result, _ = recovery.recover(coarse, image, image, np.ones(image.shape), np.eye(3), np.eye(4), np.eye(3))
    assert result['pose'] is None
    assert not result['recovery']['recovered']
    assert result['info']['reason'] == 'strict_refinement_failed'


def test_successful_strict_update_can_survive_a_later_failure(monkeypatch):
    from types import SimpleNamespace
    p = np.array([[40, 40], [50, 40], [60, 40], [40, 50], [50, 50], [60, 50]], np.float32)
    branch = dict(p=p, q=p, pose=None, info={}, quarter_turn=0)
    coarse = SimpleNamespace(model=torch.nn.Identity())
    monkeypatch.setattr(recovery, 'initialize', lambda *a, **kw: (None, [branch]))
    monkeypatch.setattr(recovery, 'broad_proposal', lambda *a: (np.eye(4), dict(inliers=6, median_proposal_residual=0)))
    strict_pose = np.eye(4); strict_pose[0, 3] = .1
    monkeypatch.setattr(recovery, 'refine_surface', lambda *a: dict(a[-1], pose=strict_pose,
        geometry_history=[dict(iteration=1, estimated=True), dict(iteration=2, estimated=False)]))
    image = np.zeros((100, 100), np.uint8)
    result, _ = recovery.recover(coarse, image, image, np.ones(image.shape), np.eye(3), np.eye(4), np.eye(3))
    np.testing.assert_array_equal(result['pose'], strict_pose)
    assert result['recovery']['recovered'] and result['recovery']['final_threshold_px'] == 3
