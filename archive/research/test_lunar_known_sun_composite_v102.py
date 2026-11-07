from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import direct_render_adapter_v73 as original
from lunar_known_sun_composite_v102 import (
    EVEN_LIBRARY,
    SuppliedSunCompositeAdapter,
    contrast_u8,
    linear_grayscale,
    plan_supplied_sun,
    synthesize_linear_contrast,
)
from solar_evaluation_v70 import common_pose


def sun(azimuth, elevation=30.0):
    azimuth, elevation = np.deg2rad([azimuth, elevation])
    return np.array([np.cos(elevation)*np.cos(azimuth),
                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])


def even_suns():
    return {index: sun(-82.5 + 7.5*index) for index in EVEN_LIBRARY}


def srgb_uint8(linear):
    encoded = np.where(linear <= 0.0031308, 12.92*linear,
                       1.055*linear**(1/2.4)-0.055)
    return np.rint(encoded*255).astype(np.uint8)


def test_horizontal_bracket_and_exact_reference():
    lights = even_suns()
    target = sun(0)
    plan = plan_supplied_sun(lights, target)
    assert plan.component_indices == (10, 12)
    vectors = np.stack([lights[index] for index in plan.component_indices])
    np.testing.assert_allclose(np.asarray(plan.weights) @ vectors[:, :2], target[:2], atol=1e-12)
    assert min(plan.weights) >= 0
    exact = plan_supplied_sun(lights, lights[6])
    assert exact.component_indices == (6,)
    assert exact.weights == (1.0,)
    assert plan_supplied_sun(lights, sun(180)).component_indices == ()
    with pytest.raises(ValueError, match="unit Sun"):
        plan_supplied_sun(lights, np.zeros(3))


def test_analytic_lambertian_offset_and_radiometric_transform():
    lights = even_suns()
    target = sun(0, elevation=40)
    plan = plan_supplied_sun(lights, target)
    phase = np.linspace(0, 2*np.pi, 320).reshape(16, 20)
    tilt = 0.2
    nz = np.sqrt(1-tilt**2)
    normal = np.stack((tilt*np.cos(phase), tilt*np.sin(phase), np.full(phase.shape, nz)), axis=2)
    radiance = {i: normal @ lights[i] for i in plan.component_indices}
    composite = synthesize_linear_contrast(plan, radiance)
    np.testing.assert_allclose(normal @ target-composite, nz*plan.vertical_residual, atol=1e-12)
    assert np.corrcoef(composite.ravel(), (normal @ target).ravel())[0, 1] > 0.999999
    code = np.array([[0, 255], [17, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(srgb_uint8(linear_grayscale(code)), code)
    np.testing.assert_array_equal(contrast_u8(np.full((2, 2), .1)), np.zeros((2, 2), dtype=np.uint8))


def test_native_sift_and_common_pose_on_synthetic_translation(monkeypatch):
    lights, target = even_suns(), sun(0)
    phase = np.random.default_rng(20261101).normal(size=(512, 512)).astype(np.float32)
    phase = cv2.GaussianBlur(phase, (0, 0), 1.3)
    phase = phase.astype(np.float64) * 6.0
    tilt = 0.25
    normal = np.stack((tilt*np.cos(phase), tilt*np.sin(phase),
                       np.full(phase.shape, np.sqrt(1-tilt**2))), axis=2)
    templates = {i: srgb_uint8(normal @ s) for i, s in lights.items()}
    base = srgb_uint8(normal @ target)
    shift = np.array([8.0, -4.0])
    query = cv2.warpAffine(base, np.column_stack([np.eye(2), shift]), (512, 512))
    depth = np.full((512, 512), 10.0)
    camera = np.array([[400, 0, 255.5], [0, 400, 255.5], [0, 0, 1.]])
    geometry = SimpleNamespace(depth=depth, intrinsics=camera, cam2world=np.eye(4))
    library = dict(
        reference_geometry_array_sha256=original.geometry_digest(depth, camera, np.eye(4)),
        map_mesh_sha256="mesh", scene_mesh_sha256="scene",
        entries=[dict(light_index=i, sun_world=s) for i, s in lights.items()],
    )
    physical = dict(reference_geometry_sha256="geometry", mesh_sha256="mesh", scene_mesh_sha256="scene")
    inp = SimpleNamespace(reference=templates[0], query=query, geometry=geometry, query_intrinsics=camera,
                          known_query_sun=target, solar_lookup=SimpleNamespace(protocol=physical))
    monkeypatch.setattr(original, "validate_render_profile", lambda profile: ({}, {"geometry": library}, {}))
    profile = dict(configuration=dict(nfeatures=2048, ratio_threshold=.75))
    for method in ("nearest_even", "two_reference_ic"):
        adapter = SuppliedSunCompositeAdapter(profile, method,
            image_loader=lambda e: templates[e["light_index"]], estimator=common_pose)
        result = adapter.infer(inp)
        assert result["supplied_sun_control"]["final_pose_calls"] == 1
        assert result["pose"] is not None
        p, q = result["reference_xy"], result["query_xy"]
        assert len(p) > 100
        assert np.median(np.linalg.norm(q-p-shift, axis=1)) < .3
        np.testing.assert_allclose(result["pose"][:3, 3], [-.2, .1, 0.], atol=.03)
    inp.known_query_sun = None
    with pytest.raises(ValueError, match="supplied-Sun track"):
        adapter.infer(inp)
