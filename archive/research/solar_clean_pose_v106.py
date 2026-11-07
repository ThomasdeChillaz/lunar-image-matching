"""Blind, paired native pose controls for dither-free rerenders."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from direct_render_adapter_v73 import DirectRenderAdapter, METHOD
from run_solar_crosslight_mechanism_v98 import pair_fingerprint, verify_saved_final_call
from run_solar_distributional_evaluation_v90 import fixed_v86_cases, read, sha, write
from run_solar_full_crosslight_rescue_v99 import CASE_FIELDS, completion, outcomes, verify_replayed
from run_solar_matched_matchanything_recovery_v91 import _replay_metrics
from solar_aligned_scoring_v91 import load_aligned_scoring
from solar_direct_arm_bridge_v103 import reference_identity
from solar_evaluation_v70 import InferenceInput, common_pose, digest, evaluate_case
from solar_radiometric_control_v101 import NORMALIZED, normalize_exposure


ARCHIVED = "reports/solar-radiometric-pilot-dispatch-v101.json"
ARCHIVED_SHA = "b7e821211828566d7d60d5250fb230d02e1e73d6092a96418ce5435ab3306b28"
ARCHIVED_REPLAY = "reports/solar_radiometric_pilot_replay_v101.json"
ARCHIVED_REPLAY_SHA = "4cb8d85c39cabcf84aff8f805bc9160406a268bf47a130acfe346c4f6da258f9"
TRACK = "estimated_sun"
POLICIES = (
    ("legacy8_exposure", "default_rgb8", False),
    ("clean8_exposure", "clean_rgb8", False),
    ("clean16_exposure", "clean_rgb16", False),
    ("clean16_independent_noise_exposure", "clean_rgb16", True),
)
# One near-black sRGB8 code, converted to scene-linear space, divided by
# sqrt(12) to match the RMS of a unit-width quantization error. This is a
# fixed sensitivity intervention, not a fit to a physical camera.
READ_NOISE_SIGMA = 1.0 / (255.0 * 12.92 * math.sqrt(12.0))
NOISE_NAMESPACE = b"v106-independent-linear-read-noise/"


def profile_specs(original_direct_profile):
    """Freeze the same four full-library, common-pose arms."""
    if (original_direct_profile["method"] != METHOD or original_direct_profile["name"] != METHOD
            or original_direct_profile["configuration"]["top_pose_candidates"] != 8):
        raise ValueError("Require original v101 blind direct profile")
    return [
        dict(name=name, backend="clean_radiance_v106", direct_profile=original_direct_profile,
             configuration=dict(variant=variant, image_count=24, candidate_pose_calls=8, final_pose_calls=1,
                                independent_linear_read_noise_sigma=READ_NOISE_SIGMA if noise else None))
        for name, variant, noise in POLICIES
    ]


def _checked(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen native asset changed: " + str(path))
    return raw


def _asset_bytes(asset, bit_depth):
    raw = _checked(asset["path"], asset["sha256"])
    if (raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or raw[16:24] != b"\x00\x00\x02\x00"*2
            or raw[24:26] != bytes((bit_depth, 2))):
        raise ValueError("Exact 512-square RGB PNG with frozen bit depth required")
    return raw


def _gray8(raw, *, archived=False):
    from PIL import Image

    with Image.open(BytesIO(raw)) as source:
        modes = ("L", "RGB") if archived else ("RGB",)
        if source.format != "PNG" or source.mode not in modes or source.size != (512, 512):
            raise ValueError("Legacy PIL RGB to L path must preserve native pixels")
        return np.asarray(source.convert("L")).copy()


def _gray16(raw):
    import cv2

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (512, 512, 3) or bgr.dtype != np.uint16:
        raise ValueError("RGB16 must decode without truncation or resizing")
    return (0.114*bgr[..., 0] + 0.587*bgr[..., 1] + 0.299*bgr[..., 2]) / 65535.0


def _noise_seed(key):
    return int.from_bytes(sha256(NOISE_NAMESPACE + key.encode("ascii")).digest()[:16], "big")


def high_bit_contrast(gray, exposure_key, noisy):
    """Set a p99.5 white point before the first uint8 quantization."""
    if (gray.shape != (512, 512) or gray.dtype != np.float64 or not np.isfinite(gray).all()
            or (gray < 0).any() or (gray > 1).any() or type(noisy) is not bool):
        raise ValueError("Native high-bit displayed grayscale in [0, 1] required")
    seed, noise_low, noise_high = None, 0, 0
    if noisy:
        seed = _noise_seed(exposure_key)
        linear = np.where(gray <= 0.04045, gray / 12.92, ((gray + 0.055) / 1.055)**2.4)
        rng = np.random.Generator(np.random.PCG64(seed))
        linear += rng.normal(0., READ_NOISE_SIGMA, gray.shape)
        noise_low, noise_high = int((linear < 0).sum()), int((linear > 1).sum())
        linear = np.clip(linear, 0., 1.)
        gray = np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055)
    gray = np.clip(gray, 0., 1.)
    rank = (199 * gray.size + 199) // 200
    white = float(np.partition(gray.ravel(), rank - 1)[rank - 1])
    scale = white if 0 < white < 1 and float(gray.min()) < float(gray.max()) else 1.
    pixels = np.floor(np.minimum(gray / scale, 1.)*255 + 0.5).astype(np.uint8)
    return pixels, dict(p995_input_fraction=white, p995_input_rank=rank, gain=1. / scale,
                        clipped_pixels=int((gray > scale).sum()), output_saturated_pixels=int((pixels == 255).sum()),
                        linear_noise_seed=seed, noise_clipped_low=noise_low, noise_clipped_high=noise_high)


def _load(row, variant, noisy):
    asset = row[variant]
    is_high = variant == "clean_rgb16"
    raw = _asset_bytes(asset, 16 if is_high else 8)
    pixels, trace = (high_bit_contrast(_gray16(raw), row["key"], noisy) if is_high
                     else normalize_exposure(_gray8(raw), True))
    return pixels, dict(exposure_key=row["key"], variant=variant, input_sha256=asset["sha256"], **trace)


def _protocol_inputs(protocol, cohort_path, render_receipt_path):
    from run_solar_radiometric_pilot_v101 import library_inventory

    cohort = read(cohort_path)
    if (sha(cohort_path) != protocol["cohort_sha256"] or protocol["split"] != "model_selection"
            or cohort["split"] != "model_selection" or protocol["calibration_or_final_access"]
            or cohort["calibration_or_final_access"] or cohort["query_truth_in_inference"] or cohort["native_shape"] != [512, 512]
            or protocol["tracks"] != [TRACK] or protocol["cpu_threads"] != 2):
        raise ValueError("Require exact public native-development cohort")
    receipt = read(render_receipt_path)
    render_root = Path(cohort["output_root"])
    if (Path(render_receipt_path) != render_root / "COMPLETE.json" or receipt["complete"] is not True
            or receipt["split"] != "model_selection" or receipt["cohort_sha256"] != protocol["cohort_sha256"]):
        raise ValueError("Require the exact completed physical rerender receipt")
    if sha(ARCHIVED) != ARCHIVED_SHA or sha(ARCHIVED_REPLAY) != ARCHIVED_REPLAY_SHA:
        raise ValueError("Require exact original archive protocol and independent replay")
    archived = read(ARCHIVED)
    if protocol["profiles"] != profile_specs(archived["profiles"][0]):
        raise ValueError("Changed blinded comparison arm list or original Direct source")
    basis, _, all_cases = fixed_v86_cases(full=True)
    if (cohort["case_manifest"] != basis["manifest"] or protocol["manifest"] != basis["manifest"]
            or cohort["case_manifest_sha256"] != basis["input_sha256"][basis["manifest"]]):
        raise ValueError("Require the exact original v101 camera/scoring manifest")
    by_case = {row["case_id"]: row for row in all_cases}
    plans = cohort["cases"]
    ids = [plan["case_id"] for plan in plans]
    if ids != protocol["case_ids"] or len(ids) != len(set(ids)) or not set(ids) <= set(archived["case_ids"]):
        raise ValueError("Require ordered unique v101 development cases")
    plans_by_key = {row["key"]: row for row in cohort["renders"]}
    renders = {row["key"]: row for row in receipt["rows"]}
    if (set(renders) != set(plans_by_key) or len(renders) != len(receipt["rows"])
            or len(renders) != len(cohort["renders"])):
        raise ValueError("Rerender receipt must cover every frozen exposure once")
    fields = ("unit_index", "role", "light_index", "archive_path", "archive_sha256",
              "archive_geometry_path", "archive_geometry_sha256")
    for key, row in renders.items():
        plan = plans_by_key[key]
        if (any(row[k] != plan[k] for k in fields) or row["geometry"]["exact_camera_depth_and_rays"] is not True
                or row["legacy_rgb_pixels_equal"] is not True
                or row["legacy_sha_equal"] is not (row["default_rgb8"]["sha256"] == row["archive_sha256"])):
            raise ValueError("Native render and historical camera/geometry identities must agree: " + key)
        for variant in ("default_rgb8", "clean_rgb8", "clean_rgb16"):
            generated = Path(row[variant]["path"])
            if (str(generated) != plan["output_" + variant] or not generated.is_relative_to(render_root)
                    or generated.resolve() != Path.cwd() / generated):
                raise ValueError("A generated exposure left its exact frozen output path")

    libraries = library_inventory(archived["profiles"][0])
    case_inputs = []
    used_renders = set()
    for plan in plans:
        case = by_case[plan["case_id"]]
        if (case["split"] != "model_selection" or plan["original_case_input_sha256"] != case["input_sha256"]
                or any(protocol["input_sha256"].get(k) != value for k, value in case["input_sha256"].items())):
            raise ValueError("Blind diagnostic may access model-selection cases only")
        physical_key = case["reference_asset_metadata"]["source_reference_geometry_sha256"]
        library = libraries[physical_key]
        reference = [renders[key] for key in plan["reference_keys"]]
        query = renders[plan["query_key"]]
        if (plan["unit_index"] != case["unit_index"] or plan["query_unit_index"] != case["query_unit_index"]
                or [row["light_index"] for row in reference] != list(range(24))
                or any(row["role"] != "reference" or row["unit_index"] != case["unit_index"]
                       or row["archive_geometry_sha256"] != physical_key for row in reference)
                or [(row["archive_path"], row["archive_sha256"]) for row in reference]
                   != [(entry["image"], entry["image_sha256"]) for entry in library["entries"]]
                or (query["role"], query["unit_index"], query["archive_path"], query["archive_sha256"])
                   != ("query", case["query_unit_index"], case["query"], case["input_sha256"][case["query"]])):
            raise ValueError("24 exact reference-role renders and the original query view required")
        nominal = renders[plan["nominal_reference_key"]]
        if nominal not in reference or nominal["archive_path"] != case["reference"]:
            raise ValueError("Nominal reference frame differs from original case")
        case_inputs.append((case, reference, nominal, query))
        used_renders.update([row["key"] for row in reference] + [query["key"]])
    if used_renders != set(renders):
        raise ValueError("Unscheduled rerender images are not allowed")
    return archived, case_inputs


class CleanDirect:
    """Change only radiometry around the unchanged Direct selection and solver."""

    def __init__(self, profile):
        self.profile = profile
        config = profile["configuration"]
        self.variant = config["variant"]
        self.noisy = config["independent_linear_read_noise_sigma"] is not None
        self.model = DirectRenderAdapter(profile["direct_profile"], image_loader=self._reference, estimator=self._pose)
        self.provenance = dict(self.model.provenance, method=profile["name"], radiometric_scope="Frozen native input encoding only.",
                               actual_rerender_cost_reported_separately=True, independent_read_noise_is_sensitivity=True)

    def bind(self, reference, query_trace, identity):
        self.reference = {row["archive_path"]: row for row in reference}
        self.query_trace = query_trace
        self.identity = identity

    def _reference(self, entry):
        row = self.reference[entry["image"]]
        if row["archive_sha256"] != entry["image_sha256"]:
            raise ValueError("Changed original library image identity")
        image, trace = _load(row, self.variant, self.noisy)
        self.seen.append(dict(light_index=entry["light_index"], **trace))
        return image

    def _pose(self, p, q, inp):
        if len(self.calls) >= 9:
            raise ValueError("Original blind AP3P budget exceeded")
        pose, info = common_pose(p, q, inp)
        self.calls.append(dict(pairs=len(p), input_sha256=pair_fingerprint(p, q), pose_returned=pose is not None))
        return pose, info

    def infer(self, inp):
        if inp.known_query_sun is not None or inp.solar_lookup is not None:
            raise ValueError("The frozen matcher cannot receive query truth or live geometry services")
        self.calls, self.seen = [], []
        raw = self.model.infer(replace(inp, solar_lookup=SimpleNamespace(protocol=self.identity)))
        if [row["light_index"] for row in self.seen] != list(range(24)) or len(self.calls) != 9:
            raise ValueError("Exactly 24 ordered image visits and 8+1 common poses required")
        a, b = raw["reference_xy"], raw["query_xy"]
        return dict(raw, render_search=dict(raw["render_search"], observed_pose_calls=list(self.calls),
                    clean_radiometry=dict(query=self.query_trace, reference=self.seen, same_native_fov=True,
                    selected_identity_within_1px=int((np.linalg.norm(a-b, axis=1) <= 1.).sum()))))


def _check_scoring(case):
    for field in ("scoring_labels", "scoring_geometry"):
        path = case.get(field)
        if path is not None and sha(path) != case["input_sha256"][path]:
            raise ValueError("Changed scoring-only asset: " + path)


def _load_scoring(case, points):
    _check_scoring(case)
    return load_aligned_scoring(case, points)


def evaluate(protocol, cohort_path, render_receipt_path, output_dir, check):
    """Seal one cell per ordered arm/case under the caller's live watchdog."""
    from run_solar_evaluation_v70 import InferenceResultCapture, capture_case_report, worker_peak_rss_snapshot
    from solar_frontend_v63 import ReferenceGeometry

    _, case_inputs = _protocol_inputs(protocol, cohort_path, render_receipt_path)
    root = Path(output_dir)
    if root != Path(protocol["output"]) / "pose":
        raise ValueError("Use only the admitted output directory")
    rows, profile_memory = [], []
    for profile in protocol["profiles"]:
        check()
        before = worker_peak_rss_snapshot()
        t = perf_counter()
        backend = InferenceResultCapture(CleanDirect(profile))
        startup = perf_counter() - t
        write(root / (profile["name"] + ".provenance.json"), backend.provenance)
        for i, (case, references, nominal, query) in enumerate(case_inputs):
            check()
            started = perf_counter()
            variant = profile["configuration"]["variant"]
            noise = profile["configuration"]["independent_linear_read_noise_sigma"] is not None
            ref_pixels, _ = _load(nominal, variant, noise)
            qry_pixels, query_trace = _load(query, variant, noise)
            image_seconds = perf_counter() - started
            t = perf_counter()
            if sha(case["reference_geometry"]) != case["input_sha256"][case["reference_geometry"]]:
                raise ValueError("Changed original reference geometry")
            geometry = ReferenceGeometry.load(case["reference_geometry"])
            identity = reference_identity(case, protocol["input_sha256"])
            setup = dict(image_decode_and_exposure_seconds=image_seconds,
                         reference_geometry_and_identity_seconds=perf_counter()-t)
            inp = InferenceInput(ref_pixels, qry_pixels, geometry, np.asarray(case["query_intrinsics"], float),
                                 np.asarray(case["reference_sun_world"], float))
            backend.bind(references, query_trace, identity)
            row, arrays = evaluate_case(backend, inp, case, scoring_loader=_load_scoring)
            row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(backend, arrays)
            key = digest(dict(case=case, profile=profile, track=TRACK, sources=protocol["source_sha256"]))
            path, array_path, seal = [root / (key + suffix) for suffix in (".json", ".npz", ".COMPLETE.json")]
            with array_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row.update({k: case[k] for k in ("case_id", *CASE_FIELDS)})
            row.update(profile=profile["name"], track=TRACK, reference_setup=setup,
                       complete_localization_cold_seconds=row["inference_seconds"]+sum(setup.values()),
                       case_calendar_seconds=perf_counter()-started, profile_load_seconds=startup if i == 0 else 0.,
                       saved_arrays=str(array_path), saved_arrays_sha256=sha(array_path))
            write(path, row)
            write(seal, dict(complete=True, identity=key, row_path=str(path), row_sha256=sha(path),
                             arrays_path=str(array_path), arrays_sha256=row["saved_arrays_sha256"]))
            rows.append(row)
            print(profile["name"], case["case_id"], row["reason"], flush=True)
        profile_memory.append(dict(profile=profile["name"], before=before, after=worker_peak_rss_snapshot()))
    return rows, profile_memory


def _saved_trace(row, profile, references, query):
    search = row["render_search"]
    if (not search["reference_viewpoint_only"] or search["exact_known_sun"] or search["library_entries_considered"] != 24
            or search["common_pnp_candidates"] != 8 or search["final_common_pnp_calls"] != 1
            or row["raw_inference_result"]["query_truth_used"] is not False):
        raise ValueError("Saved native full-library pose trace differs")
    trace = search["clean_radiometry"]
    variant = profile["configuration"]["variant"]
    noisy = profile["configuration"]["independent_linear_read_noise_sigma"] is not None
    expected = lambda asset: (asset["key"], variant, asset[variant]["sha256"])
    observed = lambda record: tuple(record[name] for name in ("exposure_key", "variant", "input_sha256"))
    if (observed(trace["query"]) != expected(query) or not trace["same_native_fov"]
            or [observed(record) for record in trace["reference"]] != [expected(ref) for ref in references]):
        raise ValueError("Saved pose input paths or hashes differ")
    if variant == "clean_rgb16":
        records = [trace["query"], *trace["reference"]]
        seeds = [record["linear_noise_seed"] for record in records]
        if seeds != [_noise_seed(record["exposure_key"]) if noisy else None for record in records]:
            raise ValueError("Saved deterministic independent noise trace differs")
        if noisy and len(set(seeds)) != 25:
            raise ValueError("Reference and query exposures must use different independent noise draws")
    verify_saved_final_call(row)
    with np.load(row["saved_arrays"], allow_pickle=False) as data:
        identity_count = int((np.linalg.norm(data["reference_xy"]-data["query_xy"], axis=1) <= 1.).sum())
    if trace["selected_identity_within_1px"] != identity_count:
        raise ValueError("Saved identity-lock count differs from native arrays")
    return identity_count


def _legacy_pixel_equal(row):
    if row["legacy_sha_equal"]:
        return True
    archived = _gray8(_checked(row["archive_path"], row["archive_sha256"]), archived=True)
    fresh = _gray8(_asset_bytes(row["default_rgb8"], 8))
    return bool(np.array_equal(archived, fresh))


def replay(protocol, cohort_path, render_receipt_path, output_dir, rows):
    """Score saved rows independently and keep archive comparisons separate."""
    from run_solar_full_native_baselines_v93 import seal_rows
    from run_solar_radiometric_pilot_v101 import exact_arrays

    archived, inputs = _protocol_inputs(protocol, cohort_path, render_receipt_path)
    actual, _ = seal_rows(Path(output_dir), protocol, protocol["case_ids"])
    if rows != actual:
        raise ValueError("Supplied v106 rows differ from the sealed grid")
    old_summary, _, old_seal, old_rows, _ = completion(ARCHIVED, archived)
    verify_replayed(ARCHIVED_REPLAY, ARCHIVED, old_summary, old_seal, old_rows, two_arms=True)
    old_index = {(row["profile"], row["case_id"]): row for row in old_rows}
    ids = protocol["case_ids"]
    archived_subset = [old_index[(NORMALIZED, case_id)] for case_id in ids]
    verified, fresh, archive_equality, archive_pixels = [], {}, [], {}
    offset = 0
    for profile in protocol["profiles"]:
        part = rows[offset:offset+len(inputs)]
        offset += len(inputs)
        for (case, references, _, query), row in zip(inputs, part, strict=True):
            if row["case_id"] != case["case_id"] or any(row[k] != case[k] for k in CASE_FIELDS):
                raise ValueError("Paired case identity differs")
            _check_scoring(case)
            checked = _replay_metrics(row, case)
            identity = None if row["inference_failure"] is not None else _saved_trace(row, profile, references, query)
            verified.append(dict(profile=profile["name"], selected_identity_within_1px=identity, **checked))
            if profile["name"] == POLICIES[0][0]:
                archived_row = old_index[(NORMALIZED, row["case_id"])]
                exposures = [*references, query]
                for exposure in exposures:
                    key = exposure["key"]
                    if key not in archive_pixels:
                        archive_pixels[key] = _legacy_pixel_equal(exposure)
                archive_equality.append(dict(case_id=row["case_id"], archive_byte_equal_exposures=sum(r["legacy_sha_equal"] for r in exposures),
                                             archive_gray_pixels_equal_exposures=sum(archive_pixels[r["key"]] for r in exposures),
                                             input_exposures=len(exposures), exact_native_outputs=exact_arrays(row, archived_row)))
        fresh[profile["name"]] = outcomes(part)
    archive_gate = all(row["exact_native_outputs"] and row["archive_gray_pixels_equal_exposures"] == row["input_exposures"]
                       for row in archive_equality)
    return dict(passed=True, replayed_rows=len(verified), rows=verified, fresh_arm_outcomes=fresh,
                archived_v101_raw_outcomes=outcomes([old_index[(METHOD, case_id)] for case_id in ids]),
                archived_v101_exposure_outcomes=outcomes(archived_subset), fresh_legacy_archive_equality=archive_equality,
                exact_archived_exposure_pose_reproduction=archive_gate, historical_causal_link_valid=archive_gate,
                render_receipt_sha256=sha(render_receipt_path), accepted_results=False,
                scope="Retrospective model-selection challenge cases and original matched nonoverlaps; no generalization, sensor or superiority claim.")
