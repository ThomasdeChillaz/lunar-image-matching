"""Fresh v92 software smoke with a fixture-only 24-view render inventory."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(output):
    import cv2
    import numpy as np
    import torch

    import solar_render_bootstrap_v92 as experimental
    from control_direct_render_v90 import CONFIGURATION, GEOMETRY, native_geometry, render_library
    from direct_render_adapter_v73 import SiftLibraryMatcher, load_native_render
    from lunar_geometry_v5 import world_points_z
    from solar_evaluation_v70 import InferenceInput
    from solar_frontend_v63 import ReferenceGeometry, reference_anchors
    from solar_native_v63 import state_digest

    if Path(experimental.__file__).resolve() != Path("solar_render_bootstrap_v92.py").resolve():
        raise ValueError("Smoke must import the integrated repository v92 source")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(20261101)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    fixture_root = root / "synthetic-reference-library"
    fixture_root.mkdir()
    plane = native_geometry()
    lookup, library, renders = render_library(fixture_root, plane, lambda: None)
    vectors = np.asarray([entry["sun_world"] for entry in library["entries"]])
    anchor_xy = reference_anchors()
    world = world_points_z(
        anchor_xy, np.full(len(anchor_xy), GEOMETRY["reference_depth_m"]), plane.intrinsics, plane.cam2world,
    )
    # All physical states are explicit UNKNOWN. This fixture exercises exact
    # Sun identity, not a claim about physical shadow predictions.
    states = np.zeros((256, len(vectors)), dtype=np.uint8)
    geometry = ReferenceGeometry(
        plane.depth, plane.intrinsics, plane.cam2world, np.tile([0., 0., -1.], (256, 1)),
        vectors, states, np.repeat(states[:, :, None], 7, axis=2), vectors[0],
        np.repeat(world[:, None], 7, axis=1),
    )
    observed_reference = np.full((512, 512), 7, dtype=np.uint8)
    affine = np.asarray(GEOMETRY["reference_to_query_affine"])
    truth = np.asarray(GEOMETRY["expected_query_cam2world"])
    query = cv2.warpAffine(renders[GEOMETRY["target_light_index"]], affine, (512, 512))
    blank = np.zeros_like(query)
    model_contract = Path("reports/solar_learning_contract_v79.json")
    lock = json.loads(model_contract.read_text())
    directory = Path(lock["initializer"])
    for name, checksum in lock["initializer_sha256"].items():
        if sha(directory / name) != checksum:
            raise ValueError("Original frozen smoke encoder checkpoint changed")
    profile = dict(
        direct_profile={"fixture_only": True}, teacher_directory=str(directory),
        frozen_descriptor_sha256=lock["expected_frozen_descriptor_sha256"],
        coarse_response_checkpoint=None,
        configuration=dict(active=True, policy="active", budget=32, render_feedback="full"),
    )
    image_reads = []

    def load(entry):
        image = load_native_render(entry)
        image_reads.append(entry["light_index"])
        return image

    fixture = SimpleNamespace(
        libraries={lookup.protocol["reference_geometry_sha256"]: library},
        matcher=SiftLibraryMatcher(CONFIGURATION), image_loader=load,
        provenance={"scope": "synthetic fixture-only reference inventory"},
    )
    # Bypass only the production inventory gate for this synthetic fixture.
    # The real v92 constructor still loads and validates the frozen encoder.
    with patch.object(experimental, "DirectRenderAdapter", return_value=fixture):
        backend = experimental.RenderBootstrapBackend(profile)

    shapes = []
    hook = backend.coarse.register_forward_pre_hook(
        lambda model, inputs: shapes.append(tuple(inputs[0].shape[-2:])),
    )
    try:
        inputs = InferenceInput(observed_reference, query, geometry, plane.intrinsics, vectors[0], None, lookup)
        result = backend.infer(inputs)
        translated_shapes = shapes.copy()
        first_reads = image_reads.copy()
        shapes.clear()
        image_reads.clear()
        invalid = backend.infer(InferenceInput(blank, blank, geometry, plane.intrinsics, vectors[0], None, lookup))
        blank_shapes = shapes.copy()
        blank_reads = image_reads.copy()
    finally:
        hook.remove()

    p, q = result["reference_xy"], result["query_xy"]
    raw_pose = result["pose"]
    error = np.linalg.norm(q - (p @ affine[:, :2].T + affine[:, 2]), axis=1)
    shift = None if raw_pose is None else float(np.linalg.norm(raw_pose[:3, 3] - truth[:3, 3]))
    rotation = None if raw_pose is None else float(np.rad2deg(np.arccos(np.clip(
        (np.trace(raw_pose[:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1,
    ))))
    native_count = sum(shape in ((27, 27), (57, 57)) for shape in translated_shapes)
    boot = result["render_bootstrap"]
    gates = dict(
        analytic_native_matches=len(error) >= 6 and float(np.median(error)) <= 1.0,
        analytic_pose=shift is not None and shift <= 0.05 and rotation <= 0.5,
        exact_24_reference_inputs=first_reads == list(range(24)) and blank_reads == list(range(24)),
        selected_render_and_windows=boot["selected_render"] is not None and boot.get("seed_windows_inserted", 0) > 0,
        deferred_native_features=translated_shapes[:2] == [(128, 128)] * 2 and native_count > 0
            and translated_shapes[2:] == [shape for shape in translated_shapes if shape in ((27, 27), (57, 57))]
            and native_count == result.get("native_encoder_forwards"),
        paid_patch_and_pose_limits=len(result.get("measurements", [])) == 32
            and result.get("proposal_samples_requested") == 4096 and result.get("final_estimator_iterations") == 2000,
        blank_rejected=invalid["pose"] is None and not len(invalid["reference_xy"])
            and invalid["reason"] == "blank_or_uninformative_image" and not blank_shapes,
        frozen_encoder_unchanged=state_digest(backend.coarse) == lock["expected_frozen_descriptor_sha256"],
    )
    raw_path = root / "predictions.npz"
    np.savez_compressed(raw_path, reference_xy=p, query_xy=q,
                        estimated_cam2world=np.empty((0, 4)) if raw_pose is None else raw_pose,
                        analytic_cam2world=truth, native_affine=affine)
    for name, image in (("observed-reference.png", observed_reference), ("translated-query.png", query)):
        if not cv2.imwrite(str(root / name), image):
            raise ValueError("Could not preserve the procedural input image")
    sources = [Path(__file__), Path(experimental.__file__), Path("control_direct_render_v90.py"), model_contract]
    summary = dict(
        passed=bool(all(gates.values())), gates={key: bool(value) for key, value in gates.items()},
        scope="Generated plane, 24 analytic reference-only renders, known translation, actual frozen SIFT/compact v92 path. No lunar data, model fitting or science admission.",
        native_size=512, cpu_threads=2, image_seed=GEOMETRY["image_seed"], returned=len(p),
        median_native_error_px=None if not len(error) else float(np.median(error)),
        translation_error_m=shift, rotation_error_deg=rotation, blank_reason=invalid["reason"],
        live_render_seeds=sum(seed["pose"] is not None for seed in boot["seeds"]),
        seed_windows_inserted=boot.get("seed_windows_inserted", 0), selected_render=boot["selected_render"],
        paid_native_searches=len(result.get("measurements", [])), encoder_shapes=translated_shapes,
        raw_predictions_sha256=sha(raw_path), source_sha256={str(path): sha(path) for path in sources},
        fixture_reference_sha256={str(entry["light_index"]): entry["image_sha256"] for entry in library["entries"]},
        input_image_sha256={name: sha(root / name) for name in ("observed-reference.png", "translated-query.png")},
        frozen_descriptor_sha256=lock["expected_frozen_descriptor_sha256"],
        calendar_seconds=perf_counter() - start,
    )
    with (root / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = run(args.output)
    print(json.dumps(evidence, indent=2, allow_nan=False))
    raise SystemExit(0 if evidence["passed"] else 1)
