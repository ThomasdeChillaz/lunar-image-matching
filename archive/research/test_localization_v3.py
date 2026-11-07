import numpy as np
import pytest

from localization_data_v3 import freshness_gate, require_training_gate
from localization_diagnostics_v3 import native_points, point_metrics, refine_native


def test_conservative_groups_do_not_turn_repeated_views_into_fresh_terrain():
    gate = freshness_gate(['nadir']*1000+['oblique', 'dynamic'], ['nadir', 'oblique', 'dynamic'])
    assert gate['fresh_group_count'] == 0 and not gate['passed']
    with pytest.raises(ValueError, match='training and final-test'):
        require_training_gate({'gate': gate})
    enough_folders = {'gate': freshness_gate([str(i) for i in range(12)], [])}
    with pytest.raises(ValueError):
        require_training_gate(enough_folders)
    require_training_gate(dict(enough_folders, verified_fresh_footprint_count=12,
                               split_audit_passed=True, trajectory_or_batch_grouping_verified=True))


def test_fresh_group_overlap_is_excluded_and_missing_gate_fails_closed():
    gate = freshness_gate([str(i) for i in range(12)], ['0'])
    assert gate['fresh_group_count'] == 11 and not gate['passed']
    with pytest.raises(ValueError):
        require_training_gate({})


def test_native_coordinates_preserve_pixel_centers_and_nonsquare_scaling():
    np.testing.assert_allclose(native_points([[0, 0], [127, 127]], (512, 1024)),
                               [[3.5, 1.5], [1019.5, 509.5]])
    assert native_points([], (512, 512)).shape == (0, 2)


def test_metrics_use_original_pixels_and_keep_unscorable_matches_in_returned_count():
    result = point_metrics([[0, 0], [3, 0], [10, 0]], [[0, 0], [0, 0]], [True, True, False])
    assert result['returned'] == 3 and result['scorable'] == 2
    assert result['correct_at_native_px'] == {'1': 1, '2': 1, '4': 2, '8': 2, '12': 2}


def test_refinement_finds_shift_without_query_depth_pose_or_ground_truth():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(7)
    a = torch.from_numpy(rng.normal(size=(1, 32, 80, 80)).astype(np.float32))
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.roll(a, shifts=(2, 3), dims=(2, 3))
    p = np.array([[30., 30.], [40., 40.]], np.float32)
    rp, rq, keep = refine_native(p, p.copy(), a, b, radius=5)
    np.testing.assert_allclose(rp, p)
    np.testing.assert_allclose(rq, p+[3, 2], atol=.3)
    np.testing.assert_array_equal(keep, [0, 1])


def test_refinement_drops_unsupported_borders_and_handles_no_matches():
    torch = pytest.importorskip('torch')
    a = torch.zeros((1, 32, 80, 80))
    rp, rq, keep = refine_native([[1, 1]], [[1, 1]], a, a)
    assert rp.shape == rq.shape == (0, 2) and len(keep) == 0
    assert refine_native([], [], a, a)[0].shape == (0, 2)
    with pytest.raises(ValueError, match='Paired'):
        refine_native([[20, 20]], [], a, a)


def test_inlier_pose_refit_uses_only_reference_geometry_and_observed_pixels():
    from localization_pose_refit_v3 import estimate_refined_pose
    from lunar_geometry_v2 import project, pose_error
    rng = np.random.default_rng(5)
    world = rng.uniform([-3, -3, 10], [3, 3, 15], (60, 3))+[1e6, 2e6, 3e6]
    camera = np.eye(4); camera[:3, 3] = [1e6, 2e6, 3e6]
    k = np.array([[100., 0, 64], [0, 100, 64], [0, 0, 1]])
    pixels = project(world, k, camera)[0]
    pixels[:10] = rng.uniform(0, 128, (10, 2))
    pose, inliers = estimate_refined_pose(world, pixels, k)
    assert inliers >= 45 and pose_error(pose, camera)['translation_m'] < 1e-3
    assert estimate_refined_pose(world[:5], pixels[:5], k) == (None, 0)


def test_capsule_install_validates_every_path_before_writing(tmp_path):
    import hashlib
    import json
    import zipfile
    from package_localization_v3 import install
    archive = tmp_path/'bad.zip'
    files = {'safe.txt': b'ok', '../escape.txt': b'bad'}
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr('CAPSULE_MANIFEST.json', json.dumps({'files': {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}))
    archive.with_suffix('.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='Unsafe'):
        install(archive, tmp_path/'destination')
    assert not (tmp_path/'destination').exists()


def test_capsule_rejects_missing_checksum(tmp_path):
    from package_localization_v3 import install
    path = tmp_path/'no-checksum.zip'; path.write_bytes(b'not trusted')
    with pytest.raises(ValueError, match='checksum'):
        install(path, tmp_path/'destination')
