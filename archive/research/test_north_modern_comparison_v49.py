from types import SimpleNamespace

import numpy as np
import pytest

import north_modern_comparison_v49 as comparison


def test_backend_receives_only_images_and_pose_estimator_only_reference_geometry():
    inputs = tuple(object() for _ in range(6)); seen = []
    class Backend:
        def points(self, *args):
            assert args == inputs[:2]; seen.append('images')
            return [[20, 40]], [[28, 36]]
        def sync(self): seen.append('sync')
    def estimator(p, q, *geometry):
        assert all(a is b for a, b in zip(geometry, inputs[2:]))
        assert len(geometry) == 4
        np.testing.assert_array_equal(q - p, [[8, -4]])
        seen.append('reference_geometry'); return None, {'reason': 'insufficient_matches'}
    p, q, pose, info = comparison.image_only_prediction(Backend(), inputs, estimator)
    assert pose is None and info['reason'] == 'insufficient_matches'
    assert seen == ['images', 'reference_geometry', 'sync']


def test_extra_query_geometry_argument_is_refused_before_backend():
    with pytest.raises(ValueError, match='reference-only'):
        comparison.image_only_prediction(None, tuple(range(7)), None)


@pytest.mark.parametrize('p,q', [([[1, 2]], []), ([[float('nan'), 2]], [[1, 2]]),
    ([[1, 2]], [[float('inf'), 2]]), ([[-.1, 2]], [[1, 2]]), ([[1, 2]], [[512, 2]])])
def test_invalid_native_output_is_not_clamped_or_silently_accepted(p, q):
    with pytest.raises(ValueError): comparison.checked_points(p, q)


def test_empty_failed_matches_keep_an_explicit_empty_pair():
    p, q = comparison.checked_points([], [])
    assert p.shape == q.shape == (0, 2)


def test_atomic_evidence_is_idempotent_but_different_evidence_is_refused(tmp_path):
    path = tmp_path / 'row.json'; comparison.write_once(path, {'case': 1})
    before = path.read_bytes(); comparison.write_once(path, {'case': 1})
    assert path.read_bytes() == before
    with pytest.raises(ValueError): comparison.write_once(path, {'case': 2})


def test_interrupted_temporary_evidence_is_preserved(tmp_path):
    path = tmp_path / 'row.json'; temporary = tmp_path / 'row.json.temporary'
    temporary.write_text('incomplete')
    with pytest.raises(FileExistsError): comparison.write_once(path, {'case': 1})
    assert temporary.read_text() == 'incomplete' and not path.exists()


def test_incomplete_upstream_state_blocks_before_process_or_model_use(monkeypatch):
    monkeypatch.setattr(comparison, 'read', lambda _: {'state': 'running'})
    with pytest.raises(ValueError, match='unfinished'): comparison.upstream_gate()


def test_terminal_state_alone_cannot_bypass_actual_live_process(monkeypatch):
    monkeypatch.setattr(comparison, 'read', lambda _: {'state': 'complete', 'pid': 42})
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='live process'))
    with pytest.raises(ValueError, match='active or reused'): comparison.upstream_gate()


def test_completed_gates_require_passed_independent_reports(monkeypatch):
    def read(path):
        return {'state': 'complete'} if path.startswith('runs/') else {'passed': False}
    monkeypatch.setattr(comparison, 'read', read)
    with pytest.raises(ValueError, match='independent checks'): comparison.upstream_gate()


def test_procedural_exact_correspondences_recover_common_camera_pose():
    # Analytical software fixture, not a pretrained or lunar accuracy result.
    from native_baseline_pose_v7 import propose_pose
    y, x = np.mgrid[80:440:60, 80:440:60]
    p = np.column_stack([x.ravel(), y.ravel()]).astype(float); q = p + [8, -4]
    image = np.zeros((512, 512), np.uint8); depth = np.full((512, 512), 10.)
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    class Fixture:
        def points(self, a, b): return p, q
        def sync(self): pass
    _, _, pose, info = comparison.image_only_prediction(Fixture(),
        (image, image, depth, k, np.eye(4), k), propose_pose)
    assert info['inliers'] == len(p)
    np.testing.assert_allclose(pose[:3, 3], [-.2, .1, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), rtol=0, atol=1e-6)


def test_procedural_insufficient_matches_return_no_pose():
    from native_baseline_pose_v7 import propose_pose
    image = np.zeros((512, 512), np.uint8); depth = np.full((512, 512), 10.)
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    class Fixture:
        def points(self, a, b): return [], []
        def sync(self): pass
    p, q, pose, info = comparison.image_only_prediction(Fixture(),
        (image, image, depth, k, np.eye(4), k), propose_pose)
    assert pose is None and len(p) == len(q) == 0
    assert info['reason'] == 'insufficient_or_invalid_matches'


def control_fixture():
    from native_baseline_pose_v7 import propose_pose
    y, x = np.mgrid[80:440:60, 80:440:60]
    p = np.column_stack([x.ravel(), y.ravel()]).astype(float); q = p + [8, -4]
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    pose, info = propose_pose(p, q, np.full((512, 512), 10.), k, np.eye(4), k)
    native = dict(passed=True, matches=len(p), inliers=info['inliers'], image_seed=20261101,
        known_translation_native=[8., -4.], reference_points_native=p.tolist(), query_points_native=q.tolist(),
        estimated_cam2world=pose.tolist(), median_native_error_px=0., warm_match_seconds=.1,
        translation_error_m=float(np.linalg.norm(pose[:3, 3] - [-.2, .1, 0.])),
        rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1) / 2, -1, 1)))),
        blank_empty=True, invalid_mask_empty=True)
    compact_pose = np.eye(4); compact_pose[:3, :3] = [[0., -1, 0], [1., 0, 0], [0, 0, 1.]]
    compact = dict(passed=True, estimated_cam2world=compact_pose.tolist(), image_seed=733,
        rotation_matrix_error=0., translation_error_m=0., blank_pose_null=True)
    return dict(pretrained_or_sift=native, compact=compact)


def test_saved_procedural_control_pose_and_errors_replay_without_network():
    from verify_north_modern_v49 import control_replay
    result = control_replay(control_fixture())
    assert result['saved_native_control_pose_and_errors_replayed']
    assert not result['blank_and_invalid_model_forwards_repeated']


@pytest.mark.parametrize('field,value', [('matches', 1), ('median_native_error_px', .5),
    ('blank_empty', False), ('invalid_mask_empty', False), ('warm_match_seconds', float('nan'))])
def test_control_success_flag_does_not_hide_tampered_evidence(field, value):
    from verify_north_modern_v49 import control_replay
    checks = control_fixture(); checks['pretrained_or_sift'][field] = value
    with pytest.raises(ValueError): control_replay(checks)


def test_missing_control_records_are_not_a_vacuous_pass():
    from verify_north_modern_v49 import control_replay
    with pytest.raises(ValueError): control_replay({})


def test_false_compact_rotation_success_is_refused():
    from verify_north_modern_v49 import control_replay
    checks = control_fixture(); checks['compact']['estimated_cam2world'] = np.eye(4).tolist()
    with pytest.raises(ValueError): control_replay(checks)


@pytest.mark.parametrize('kind', ['last_row', 'scaled_rotation'])
def test_camera_pose_requires_a_rigid_transform(kind):
    from verify_north_modern_v49 import rigid_pose
    pose = np.eye(4)
    if kind == 'last_row': pose[3, 0] = .1
    else: pose[:3, :3] *= 1.001
    with pytest.raises(ValueError): rigid_pose(pose)
