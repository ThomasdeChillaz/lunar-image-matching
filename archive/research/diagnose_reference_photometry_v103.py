"""Predeclared reference-only held-out-light contrast diagnostic."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import lunar_known_sun_composite_v102 as method


PROTOCOL = "reports/solar-direct-render-library-v90.json"
PROTOCOL_SHA = "93cb7b7ddbe76b2c97d89fde830f0d02376df89d36d2a4ba4c907dd360890b11"
CATALOGUE = "runs/direct-reference-render-library-v73/manifest.json"
CATALOGUE_SHA = "c610005ade77b302c7b8fec7ae0d12d2de6b175e3039ac893cdb7873c2329667"
RECEIPT = "runs/direct-reference-render-library-v73/COMPLETE.json"
RECEIPT_SHA = "e6ee922399822edf32297143a48b37b129a7adc08943f68a7fc1c068d8c6562c"
METHOD_SHA = "6c5bd0a8d3716aac3eb085ab77d0a6de69582c9954f28896909d918a6264827f"
ODD = tuple(range(1, 24, 2))
MAIN_STRATUM = (1, 17)
RADIUS = 12
CODE_STEP = 2
EDGE_PERCENT = 5
DARK_P95 = 3
PRIMARY = "informative_patch_derivative_mae_median"
AXIS = np.rint(49.5 + 16 * np.rint(np.linspace(0, 25, 16))).astype(int)
YY, XX = np.meshgrid(AXIS, AXIS, indexing="ij")
PATCH_Y = YY.ravel()[:, None, None] + np.arange(-RADIUS, RADIUS + 1)[None, :, None]
PATCH_X = XX.ravel()[:, None, None] + np.arange(-RADIUS, RADIUS + 1)[None, None, :]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("Frozen input differs: " + str(path))
    return data


def read(path, expected):
    return json.loads(checked(path, expected))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metadata_freeze():
    """Verify catalogue and producer metadata. Open zero image payloads."""
    if digest(method.__file__) != METHOD_SHA:
        raise ValueError("Use the unchanged v102 synthesis algorithm")
    protocol = read(PROTOCOL, PROTOCOL_SHA)
    catalogue = read(CATALOGUE, CATALOGUE_SHA)
    receipt = read(RECEIPT, RECEIPT_SHA)
    if (str(Path(protocol["output"]) / "manifest.json") != CATALOGUE or protocol["prepared_only"]
            or protocol["split"] != "model_selection" or protocol["calibration_or_final_access"]
            or protocol["query_truth_used"] or not catalogue["complete"] or catalogue["native_size"] != 512
            or catalogue["split"] != "model_selection" or catalogue["calibration_or_final_access"]
            or catalogue["query_truth_used"] or catalogue["query_role_images_used"]
            or catalogue["viewpoint_search"] != "reference_camera_only"
            or catalogue["protocol_sha256"] != PROTOCOL_SHA
            or catalogue["producer_source_sha256"] != protocol["source_sha256"]
            or receipt["protocol_sha256"] != PROTOCOL_SHA or receipt["manifest_sha256"] != CATALOGUE_SHA
            or receipt["publication_commit"] != catalogue["publication_commit"]
            or not receipt["complete"] or receipt["libraries"] != 32 or receipt["entries"] != 768):
        raise ValueError("Require the completed original 32-view reference-only development catalogue")
    groups = Counter(library["component"] for library in catalogue["libraries"])
    if len(groups) != 8 or set(groups.values()) != {4}:
        raise ValueError("Exactly eight original terrains and four reference views each required")
    views = []
    images = set()
    for lib in sorted(catalogue["libraries"], key=lambda entry: entry["unit_index"]):
        source_path = Path(lib["source_summary"])
        source = read(source_path, lib["source_summary_sha256"])
        refs = [row for row in source["images"] if row["role"] == "reference"]
        originals = {row["light"]: row for row in refs}
        entries = {row["light_index"]: row for row in lib["entries"]}
        if (lib["source_role"] != "reference" or lib["query_camera_fields_used"]
                or source["original_unit_index"] != lib["unit_index"] or not source["complete"]
                or source["calibration_or_final_access"] or not source["views"]["reference"]["complete_surface"]
                or source["views"]["reference"]["sha256"] != lib["reference_geometry_sha256"]
                or lib["scene_mesh_sha256"] != lib["map_mesh_sha256"]
                or len(refs) != 24 or len(lib["entries"]) != 24 or set(originals) != set(range(24))
                or set(entries) != set(range(24))):
            raise ValueError("Reference view lacks exact role, geometry or catalogue identity")
        inventory = []
        for light in range(24):
            entry, original = entries[light], originals[light]
            source_name = source_path.parent / original["path"]
            if (entry["source_role"] != "reference" or not entry["primary_generic"]
                    or entry["native_shape"] != [512, 512] or entry["image_sha256"] != original["sha256"]
                    or source_name != Path(entry["image"]) or Path(original["path"]).parts != ("reference", f"light-{light}.png")
                    or entry["image"] in images):
                raise ValueError("Image is outside the unique original reference-role PNG inventory")
            images.add(entry["image"])
            inventory.append(dict(light=light, path=entry["image"], sha256=entry["image_sha256"], sun=entry["sun_world"]))
        even = {i: entries[i]["sun_world"] for i in method.EVEN_LIBRARY}
        cases = []
        for light in ODD:
            plan = method.plan_supplied_sun(even, entries[light]["sun_world"])
            if len(plan.component_indices) not in (0, 2) or any(i not in even for i in plan.component_indices):
                raise ValueError("An odd target may use only zero or two even component templates")
            cases.append(dict(target=light, **asdict(plan)))
        views.append(dict(unit=lib["unit_index"], terrain=lib["component"], geometry_sha256=lib["reference_geometry_sha256"],
                          source_summary=str(source_path), source_summary_sha256=lib["source_summary_sha256"],
                          images=inventory, cases=cases))
    if len({v["unit"] for v in views}) != 32 or len(images) != 768:
        raise ValueError("Original unique view and image count required")
    return dict(
        schema="reference_only_photometric_v103", source_sha256=digest(__file__), method_sha256=METHOD_SHA,
        library_protocol_sha256=PROTOCOL_SHA, library_manifest_sha256=CATALOGUE_SHA, library_receipt_sha256=RECEIPT_SHA,
        publication_commit=catalogue["publication_commit"], reference_views=views,
        cases=384, main_stratum=list(MAIN_STRATUM), cpu_thread_ceiling=2,
        calibration_or_final_access=False, query_role_inputs=False, matcher_calls=0, pose_estimator_calls=0,
        protocol=dict(
            primary=PRIMARY, primary_unit="mean absolute error between corresponding signed x/y derivatives in stretched code units",
            primary_reduction="median across informative patches within each view/light condition",
            stretch="unchanged v102 per-image min-max contrast_u8 applied to prediction and odd target",
            informative_target="at least 5% of native intra-patch horizontal/vertical raw grayscale edges differ by >=2 codes",
            raw_grayscale="Pillow L conversion of original sRGB PNG; linear method uses exact v102 sRGB conversion",
            patch_centres=AXIS.tolist(), patch_radius=RADIUS, image_size=[512, 512],
            dark_original_target="raw 95th percentile <=3 codes",
            comparison="positive nearest-minus-composite error difference favors synthesis, common scored support only",
            low_signal="report unsupported target windows and predicted texture in them; record target stretch too",
            aggregation="report all, 1/17, each illumination, each terrain and terrain macro medians without independence claims",
        ),
        scope=("Reference-role same-camera reconstruction of withheld odd renderer states. Odd pixels are read after "
               "the metadata-only plan is fixed. No query-view, matcher, navigation, transfer or visibility conclusion."),
    )


def gradients(image):
    patches = np.asarray(image, dtype=np.int16)[PATCH_Y, PATCH_X]
    return np.diff(patches, axis=2), np.diff(patches, axis=1)


def edge_support(gradient):
    dx, dy = gradient
    count = np.count_nonzero(np.abs(dx) >= CODE_STEP, axis=(1, 2)) + np.count_nonzero(np.abs(dy) >= CODE_STEP, axis=(1, 2))
    total = np.prod(dx.shape[1:]) + np.prod(dy.shape[1:])
    return 100 * count >= EDGE_PERCENT * total


def median(values):
    return None if len(values) == 0 else float(np.median(values))


def target_stats(target_linear, target_raw):
    target_u8 = method.contrast_u8(target_linear)
    target_grad = gradients(target_u8)
    informative = edge_support(gradients(target_raw))
    stretched_informative = edge_support(target_grad)
    histogram = np.bincount(target_raw.ravel(), minlength=256)
    p95 = int(np.searchsorted(histogram.cumsum(), np.ceil(.95 * target_raw.size)))
    return target_u8, target_grad, informative, dict(
        raw_codes=int(np.count_nonzero(histogram)), raw_p95=p95, dark_p95_le3=p95 <= DARK_P95,
        raw_informative_patches=int(informative.sum()), raw_low_signal_patches=int((~informative).sum()),
        stretched_informative_patches=int(stretched_informative.sum()),
        stretched_new_texture_on_raw_low_signal=int((stretched_informative & ~informative).sum()),
        linear_dynamic_range=float(np.ptp(target_linear)),
    )


def prediction_stats(prediction, target_linear, target_u8, target_grad, informative):
    candidate = method.contrast_u8(prediction)
    candidate_grad = gradients(candidate)
    delta_x, delta_y = (np.abs(a - b) for a, b in zip(candidate_grad, target_grad))
    local = .5 * (delta_x.mean(axis=(1, 2)) + delta_y.mean(axis=(1, 2)))
    support = edge_support(candidate_grad)
    low = ~informative
    return dict(
        raw_linear_mae=float(np.abs(prediction - target_linear).mean()),
        contrast_image_mae=float(np.abs(candidate.astype(np.int16) - target_u8).mean()),
        informative_patch_derivative_mae_median=median(local[informative]),
        all_patch_derivative_mae_median=median(local),
        low_signal_patch_derivative_mae_median=median(local[low]),
        stretched_texture_on_raw_low_signal=int((support & low).sum()),
        stretched_texture_on_raw_informative=int((support & informative).sum()),
        prediction_linear_dynamic_range=float(np.ptp(prediction)),
        prediction_outside_unit_fraction=float(((prediction < 0.) | (prediction > 1.)).mean()),
    )


def evaluate_pair(nearest, composite, target_linear, target_raw):
    target_u8, target_grad, informative, signal = target_stats(target_linear, target_raw)
    arguments = (target_linear, target_u8, target_grad, informative)
    return dict(target_signal=signal, nearest=prediction_stats(nearest, *arguments),
                composite=None if composite is None else prediction_stats(composite, *arguments))


def load_image(entry):
    blob = checked(entry["path"], entry["sha256"])
    with Image.open(io.BytesIO(blob)) as image:
        if image.format != "PNG" or image.size != (512, 512) or image.mode not in ("L", "RGB"):
            raise ValueError("Original native uint8 reference PNG required")
        return method.linear_grayscale(np.asarray(image)), np.asarray(image.convert("L"))


def group_summary(rows):
    supported = [r for r in rows if r["composite"] is not None]
    common = [r for r in supported if r["nearest"][PRIMARY] is not None]

    def contrast(metric, subset):
        nearest = [r["nearest"][metric] for r in subset]
        composite = [r["composite"][metric] for r in subset]
        deltas = np.asarray(nearest) - composite
        return dict(paired=len(subset), nearest_median=median(nearest), composite_median=median(composite),
                    nearest_minus_composite_median=median(deltas), wins=int((deltas > 0).sum()),
                    ties=int((deltas == 0).sum()), losses=int((deltas < 0).sum()))

    return dict(
        conditions=len(rows), angular_bracket_available=len(supported),
        angular_bracket_unavailable=len(rows) - len(supported),
        raw_dark_p95_le3=sum(r["target_signal"]["dark_p95_le3"] for r in rows),
        no_raw_informative_patch=sum(r["target_signal"]["raw_informative_patches"] == 0 for r in rows),
        informative_patch_count_median=median([r["target_signal"]["raw_informative_patches"] for r in rows]),
        primary=contrast(PRIMARY, common),
        raw_linear_mae=contrast("raw_linear_mae", supported),
        contrast_image_mae=contrast("contrast_image_mae", supported),
        low_signal_patch_derivative_mae=contrast("low_signal_patch_derivative_mae_median", [
            r for r in supported if r["target_signal"]["raw_low_signal_patches"]]),
        low_signal_patches_total=sum(r["target_signal"]["raw_low_signal_patches"] for r in supported),
        texture_after_stretch_on_same_raw_low_signal_patches=dict(
            target=sum(r["target_signal"]["stretched_new_texture_on_raw_low_signal"] for r in supported),
            nearest=sum(r["nearest"]["stretched_texture_on_raw_low_signal"] for r in supported),
            composite=sum(r["composite"]["stretched_texture_on_raw_low_signal"] for r in supported),
        ),
    )


def summarize(rows):
    by_terrain, by_light, by_unit = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_terrain[row["terrain"]].append(row)
        by_light[row["target"]].append(row)
        by_unit[row["unit"]].append(row)

    def strata(groups, main_only=False):
        return {str(key): group_summary([r for r in part if not main_only or r["target"] in MAIN_STRATUM])
                for key, part in sorted(groups.items())}

    def macro(groups):
        deltas = [part["primary"]["nearest_minus_composite_median"] for part in groups.values()]
        available = [d for d in deltas if d is not None]
        return dict(groups=len(groups), groups_with_primary=len(available),
                    median_group_primary_delta=median(available),
                    positive_group_delta=sum(d > 0 for d in available), negative_group_delta=sum(d < 0 for d in available))

    t, t_main = strata(by_terrain), strata(by_terrain, main_only=True)
    return dict(
        all_odd=group_summary(rows),
        lights_1_17=group_summary([r for r in rows if r["target"] in MAIN_STRATUM]),
        by_terrain=t, by_terrain_lights_1_17=t_main, by_light=strata(by_light),
        macro_terrain=macro(t), macro_terrain_lights_1_17=macro(t_main), macro_view=macro(strata(by_unit)),
    )


def run(freeze_path, freeze_sha, output):
    if Path(output).exists():
        raise FileExistsError("Preserve every previously written diagnostic")
    frozen = read(freeze_path, freeze_sha)
    if frozen != json.loads(json.dumps(metadata_freeze())):
        raise ValueError("The exact reviewed pre-pixel diagnostic plan is required")
    rows = []
    for view in frozen["reference_views"]:
        entries = {entry["light"]: entry for entry in view["images"]}
        even = {}

        def even_image(light):
            if light not in method.EVEN_LIBRARY:
                raise ValueError("Odd target pixels cannot enter template construction")
            if light not in even:
                even[light] = load_image(entries[light])[0]
            return even[light]

        for case in view["cases"]:
            target_light = case["target"]
            plan = method.SynthesisPlan(**{key: value for key, value in case.items() if key != "target"})
            nearest = even_image(plan.nearest_index)
            composite = method.synthesize_linear_contrast(plan, {i: even_image(i) for i in plan.component_indices})
            target, raw = load_image(entries[target_light])
            scores = evaluate_pair(nearest, composite, target, raw)
            rows.append(dict(unit=view["unit"], terrain=view["terrain"], **case, **scores))
    if len(rows) != 384:
        raise ValueError("Every original odd reference condition must be represented")
    write(output, dict(
        protocol_sha256=freeze_sha, source_sha256=digest(__file__), method_sha256=METHOD_SHA,
        library_manifest_sha256=CATALOGUE_SHA, no_query_role_inputs=True, matcher_calls=0, pose_estimator_calls=0,
        calibration_or_final_access=False, summary=summarize(rows), rows=rows,
        limits=("One renderer and fixed viewpoint per camera. Low signal is a measurement from raw stored codes. "
                "Neither lighting visibility, physical shadow, restored radiance, correspondence accuracy nor "
                "pose recovery is evaluated. Patch/lighting conditions share cameras and terrain."),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("freeze-metadata")
    prep.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--freeze-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze-metadata":
        write(args.output, metadata_freeze())
    else:
        run(args.freeze, args.freeze_sha256, args.output)
