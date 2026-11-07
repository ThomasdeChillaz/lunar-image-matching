"""Two-reference horizontal Lambertian synthesis for a supplied-Sun control.

The public terrain-coherence description in US20260219064A1 motivates
cancelling horizontal light components with two co-registered references.
This module uses a declared angular bracketing rule and returns linear
contrast templates. It neither estimates the Sun nor calls a pose solver.
"""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


EVEN_LIBRARY = tuple(range(0, 24, 2))
MIN_HORIZONTAL_SINE = math.sin(math.radians(5.0))
_VECTOR_ATOL = 1e-10


@dataclass(frozen=True)
class SynthesisPlan:
    nearest_index: int
    nearest_angle_deg: float
    component_indices: tuple[int, ...]
    weights: tuple[float, ...]
    vertical_residual: float
    horizontal_condition: float | None


def _sun_vector(value):
    sun = np.asarray(value, dtype=np.float64)
    if (sun.shape != (3,) or not np.isfinite(sun).all()
            or abs(np.linalg.norm(sun)-1.0) > _VECTOR_ATOL):
        raise ValueError("Require a finite unit Sun direction in reference ENU")
    return sun


def plan_supplied_sun(lights, supplied_sun):
    """Choose two nearby even-index lights whose horizontal cone contains the target.

    Selection uses only public geometry. A missing angular bracket returns
    no composite, while the nearest-template comparison remains available.
    """
    if tuple(sorted(lights)) != EVEN_LIBRARY:
        raise ValueError("Require exactly the frozen even-index reference library")
    suns = np.stack([_sun_vector(lights[index]) for index in EVEN_LIBRARY])
    target = _sun_vector(supplied_sun)
    angles = np.degrees(np.arccos(np.clip(suns @ target, -1.0, 1.0)))
    nearest = int(np.argmin(angles))
    nearest_index = EVEN_LIBRARY[nearest]
    nearest_angle = float(angles[nearest])

    if target[2] <= 0 or np.linalg.norm(target[:2]) < _VECTOR_ATOL:
        return SynthesisPlan(nearest_index, nearest_angle, (), (), 0.0, None)
    exact = np.flatnonzero(np.all(np.abs(suns-target) <= 1e-12, axis=1))
    if len(exact):
        return SynthesisPlan(nearest_index, nearest_angle, (EVEN_LIBRARY[int(exact[0])],), (1.0,), 0.0, 1.0)

    candidates = []
    for i, j in combinations(range(len(suns)), 2):
        if min(suns[i, 2], suns[j, 2]) <= 0:
            continue
        horizontal = suns[[i, j], :2].T
        lengths = np.linalg.norm(horizontal, axis=0)
        if min(lengths) < _VECTOR_ATOL:
            continue
        sine = abs(np.linalg.det(horizontal)) / np.prod(lengths)
        if sine < MIN_HORIZONTAL_SINE:
            continue
        weights = np.linalg.solve(horizontal, target[:2])
        if min(weights) < -1e-12:
            continue
        weights = np.maximum(weights, 0.0)
        residual = float(target[2] - weights @ suns[[i, j], 2])
        condition = float(np.linalg.cond(horizontal))
        order = (max(angles[i], angles[j]), np.linalg.norm(weights), i, j)
        candidates.append((order, (EVEN_LIBRARY[i], EVEN_LIBRARY[j]), weights, residual, condition))

    if not candidates:
        return SynthesisPlan(nearest_index, nearest_angle, (), (), 0.0, None)
    _, indices, weights, residual, condition = min(candidates, key=lambda row: row[0])
    return SynthesisPlan(nearest_index, nearest_angle, indices, tuple(map(float, weights)), residual, condition)


def linear_grayscale(rgb):
    """Decode the declared sRGB render before taking luminance."""
    value = np.asarray(rgb)
    if value.dtype != np.uint8 or value.ndim not in (2, 3):
        raise ValueError("Require original grayscale or RGB uint8 content")
    if value.ndim == 3 and value.shape[2] != 3:
        raise ValueError("Require three sRGB channels")
    encoded = value.astype(np.float64) / 255.0
    linear = np.where(encoded <= 0.04045, encoded/12.92, ((encoded+0.055)/1.055)**2.4)
    if value.ndim == 3:
        linear = linear @ np.array([0.2126, 0.7152, 0.0722])
    return linear


def synthesize_linear_contrast(plan, co_registered):
    """Combine co-registered linear intensities at the reference viewpoint.

    Under unclipped Lambertian response the missing term is
    vertical_residual * n_up, up to a constant albedo/radiometric factor.
    The result is a contrast template, not a recovered radiance image.
    """
    if not plan.component_indices:
        return None
    images = [np.asarray(co_registered[index], dtype=np.float64) for index in plan.component_indices]
    if any(image.ndim != 2 or image.shape != images[0].shape
           or not np.isfinite(image).all() for image in images):
        raise ValueError("Require co-registered finite linear grayscale arrays")
    result = plan.weights[0] * images[0]
    if len(images) == 2:
        result += plan.weights[1] * images[1]
    return result


def contrast_u8(image):
    """Map a nonconstant linear plane to SIFT's native integer input."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Require a finite linear grayscale plane")
    low, high = float(image.min()), float(image.max())
    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.rint(np.clip((image-low)/(high-low), 0.0, 1.0)*255.0).astype(np.uint8)


class SuppliedSunCompositeAdapter:
    """Use equal-budget nearest or synthesized reference in the secondary track."""

    def __init__(self, original_direct_profile, method, *, image_loader=None, estimator=None):
        from direct_render_adapter_v73 import DirectRenderAdapter

        if method not in ("nearest_even", "two_reference_ic"):
            raise ValueError("Declare nearest-even or two-reference synthesis")
        self.direct = DirectRenderAdapter(original_direct_profile, estimator=estimator)
        self.method = method
        self.load_image = image_loader or _load_srgb_reference

    def infer(self, inp):
        from direct_render_adapter_v73 import BaselineUnavailable, geometry_digest, validate_images, validate_points

        validate_images(inp)
        if inp.known_query_sun is None:
            raise ValueError("This adapter is restricted to the declared supplied-Sun track")
        if inp.solar_lookup is None or not hasattr(inp.solar_lookup, "protocol"):
            raise BaselineUnavailable("Require the identical qualified reference map")
        physical = inp.solar_lookup.protocol
        library = self.direct.libraries.get(physical["reference_geometry_sha256"])
        if (library is None or physical["mesh_sha256"] != library["map_mesh_sha256"]
                or physical["scene_mesh_sha256"] != library["scene_mesh_sha256"]):
            raise BaselineUnavailable("Require the original full, unperturbed reference map")
        geometry = inp.geometry
        if geometry_digest(geometry.depth, geometry.intrinsics, geometry.cam2world) != library["reference_geometry_array_sha256"]:
            raise ValueError("Reference camera geometry differs from its render library")
        entries = {e["light_index"]: e for e in library["entries"] if e["light_index"] in EVEN_LIBRARY}
        plan = plan_supplied_sun({i: e["sun_world"] for i, e in entries.items()}, inp.known_query_sun)
        indices = (plan.nearest_index,) if self.method == "nearest_even" else plan.component_indices
        result = dict(
            reference_xy=np.empty((0, 2)), query_xy=np.empty((0, 2)), pose=None,
            query_truth_used=False,
            supplied_sun_control=dict(
                method=self.method, known_sun_metadata_used=True, template_indices=list(indices),
                supplied_sun_enu=np.asarray(inp.known_query_sun, dtype=float).tolist(),
                nearest_index=plan.nearest_index, nearest_angle_deg=plan.nearest_angle_deg,
                horizontal_bracket_available=bool(plan.component_indices),
                synthesis_weights=list(plan.weights), vertical_residual=plan.vertical_residual,
                horizontal_condition=plan.horizontal_condition, final_pose_calls=0,
            ),
        )
        if not indices:
            return dict(result, reason="no_supported_horizontal_bracket")
        radiance = {i: linear_grayscale(self.load_image(entries[i])) for i in indices}
        reference = radiance[indices[0]] if self.method == "nearest_even" else synthesize_linear_contrast(plan, radiance)
        query = contrast_u8(linear_grayscale(inp.query))
        features = self.direct.matcher.query_features(query)
        reference_xy, query_xy, _ = self.direct.matcher.match(contrast_u8(reference), features)
        reference_xy, query_xy = validate_points(reference_xy, query_xy)
        pose, info = self.direct.estimator(reference_xy, query_xy, inp)
        result["supplied_sun_control"]["final_pose_calls"] = 1
        return dict(result, reference_xy=reference_xy, query_xy=query_xy, pose=pose,
                    reason=info.get("reason"), common_estimator=info)


def _load_srgb_reference(entry):
    from PIL import Image
    from direct_render_adapter_v73 import sha

    if sha(entry["image"]) != entry["image_sha256"]:
        raise ValueError("Reference render bytes differ from the frozen source")
    with Image.open(entry["image"]) as image:
        if image.format != "PNG" or image.size != (512, 512) or image.mode not in ("L", "RGB"):
            raise ValueError("Require original native 512-pixel sRGB reference PNG")
        return np.asarray(image).copy()
