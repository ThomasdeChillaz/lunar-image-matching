"""Fixed blind exposure normalization for the original 24-light direct baseline."""

import numpy as np

from direct_render_adapter_v73 import METHOD, DirectRenderAdapter, SiftLibraryMatcher, load_native_render


DIRECT = METHOD
NORMALIZED = "reference_camera_render_match_p995_exposure"
CONTROL_POLICY = "identity_grayscale_with_identical_observation_logging"
POLICY = dict(name="per_image_p995_white_v101", rank="ceil(199*N/200), one-indexed",
              target_white_code=255, black_offset=0, output="uint8 with half-up rounding and clipping",
              uniform_policy="unchanged", zero_or_full_white_policy="unchanged", roles=["reference_render", "query"],
              truth_fields_used=[], lighting_budget=24, candidate_pose_calls=8, final_pose_calls=1)


def normalize_exposure(image, enabled):
    """Map a fixed high percentile to 255 without reading any paired context."""
    if not isinstance(image, np.ndarray) or image.shape != (512, 512) or image.dtype != np.uint8:
        raise ValueError("Original native512 uint8 grayscale required")
    if type(enabled) is not bool:
        raise ValueError("Explicit identity or frozen exposure policy required")
    histogram = np.bincount(image.ravel(), minlength=256)
    rank = (199 * image.size + 199) // 200
    white = int(np.searchsorted(np.cumsum(histogram), rank, side="left"))
    distinct = int(np.count_nonzero(histogram))
    action = "identity_control"
    if enabled:
        if distinct == 1:
            action = "uniform_unchanged"
        elif white == 0:
            action = "zero_white_unchanged"
        elif white == 255:
            action = "full_white_unchanged"
        else:
            action = "scaled"
    scaled = action == "scaled"
    if scaled:
        # A code below white cannot round to 255 because the gain exceeds one.
        result = np.minimum((image.astype(np.uint16) * 255 + white // 2) // white, 255).astype(np.uint8)
    else:
        result = image
    tied = int(histogram[white])
    above = int(histogram[white + 1:].sum())
    stats = dict(action=action, p995_input_code=white, p995_input_rank=rank, percentile_plateau_pixels=tied,
                 percentile_has_duplicate_codes=tied > 1, percentile_plateau_fraction=tied / image.size,
                 input_distinct_codes=distinct, input_black_pixels=int(histogram[0]),
                 input_saturated_pixels=int(histogram[-1]), input_above_white_pixels=above,
                 gain=255. / white if scaled else 1., clipped_pixels=above if scaled else 0,
                 output_saturated_pixels=tied + above if scaled else int(histogram[-1]), pixel_count=image.size)
    return result, stats


def matched_profiles(original):
    """Leave the original v92 profile byte-identical and derive one arm."""
    if (original.get("name") != DIRECT or original.get("method") != DIRECT or original.get("backend") != "direct_render"
            or original.get("native_size") != 512 or original.get("cpu_threads") != 2
            or original["configuration"].get("top_pose_candidates") != 8 or "radiometric_policy" in original):
        raise ValueError("Exact original v92 direct profile with eight candidates required")
    return original, dict(original, name=NORMALIZED, radiometric_policy=dict(POLICY))


class _ObservedSift:
    def __init__(self, config, backend):
        self.sift = SiftLibraryMatcher(config)
        self.backend = backend

    def query_features(self, image):
        prepared, record = normalize_exposure(image, self.backend.enabled)
        features = self.sift.query_features(prepared)
        self.backend.query_stats = dict(record, detected_keypoints=len(features[0]))
        return features

    def match(self, render, query_features):
        result = self.sift.match(render, query_features)
        self.backend.reference_stats[-1]["sift_ratio_unique_matches"] = len(result[0])
        return result


class RadiometricDirectBackend:
    """Fresh control or one exposure arm using the unchanged direct adapter."""

    def __init__(self, profile):
        from run_solar_crosslight_mechanism_v98 import pair_fingerprint

        name = profile.get("name")
        if name == DIRECT:
            matched_profiles(profile)
        elif name == NORMALIZED:
            original = {k: v for k, v in profile.items() if k != "radiometric_policy"}
            _, expected = matched_profiles(dict(original, name=DIRECT))
            if profile != expected:
                raise ValueError("Exact predeclared p99.5 exposure arm required")
        else:
            raise ValueError("Use the original direct control or exact exposure arm")
        self.profile = profile
        self.enabled = name == NORMALIZED
        self.model = DirectRenderAdapter(profile, matcher_factory=lambda config: _ObservedSift(config, self),
                                         image_loader=self.load_reference)
        self.provenance = dict(self.model.provenance, radiometric_policy=POLICY if self.enabled else CONTROL_POLICY,
                               radiometric_scope="Independently applied to each original image before unchanged SIFT.")
        original = self.model.estimator

        def observed(p, q, inp):
            if len(self.pose_calls) >= 9:
                raise ValueError("Inference exceeded the original nine pose calls")
            checksum = pair_fingerprint(p, q)
            pose, info = original(p, q, inp)
            self.pose_calls.append(dict(pairs=len(p), input_sha256=checksum, pose_returned=pose is not None))
            return pose, info

        self.model.estimator = observed

    def load_reference(self, entry):
        image = load_native_render(entry)
        prepared, report = normalize_exposure(image, self.enabled)
        self.reference_stats.append(dict(light_index=entry["light_index"], **report))
        return prepared

    def infer(self, inp):
        if inp.known_query_sun is not None:
            raise ValueError("This blind control uses the original unknown-Sun primary track")
        self.query_stats, self.reference_stats, self.pose_calls = None, [], []
        result = self.model.infer(inp)
        if (len(self.reference_stats) != 24 or {r["light_index"] for r in self.reference_stats} != set(range(24))
                or len(self.pose_calls) != 9 or self.query_stats is None):
            raise ValueError("Original 24 reference lights, one query and nine solver calls required")
        trace = dict(policy=POLICY if self.enabled else CONTROL_POLICY, query=self.query_stats,
                     reference_renders=self.reference_stats, same_native_fov=True)
        render_search = dict(result["render_search"], observed_pose_calls=list(self.pose_calls), radiometry=trace)
        return dict(result, render_search=render_search)
