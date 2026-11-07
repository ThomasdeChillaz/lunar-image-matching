"""Exercise the real native CLI on generated images, with truth used only for scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunar_matching.artifacts import sha256, write_json_exclusive


def check(destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    archive = ROOT / "archive/research"
    # This standalone software-control process may import the frozen generator.
    sys.path.insert(0, str(archive))
    sys.path.insert(0, str(archive / "reports/solar-blind-v139"))
    from engine_v139 import generated
    catalogue, query, depth, k, pose, qk, valid, truth = generated(False)

    def image(path, linear):
        value = linear / 1.5  # Fit the complete generated radiance range without clipping.
        assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        srgb = np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)
        encoded = np.repeat(np.rint(65535 * srgb).astype(np.uint16)[:, :, None], 3, axis=2)
        assert cv2.imwrite(str(path), encoded)
        return {"path": str(path), "sha256": sha256(path), "bit_depth": 16}

    entries = [dict(light_id=light, sun_world=sun.tolist(), image=image(destination / f"reference-{light}.png", raster))
               for light, sun, raster in zip(catalogue.light_ids, catalogue.sun_world, catalogue.images, strict=True)]
    geometry = destination / "reference.npz"
    np.savez(geometry, axial_depth=depth, intrinsics=k, cam2world=pose, valid=valid)
    case = dict(id="generated-native", reference_geometry={"path": str(geometry), "sha256": sha256(geometry)},
                catalogue=entries, query_image=image(destination / "query.png", query), query_intrinsics=qk.tolist())
    manifest = destination / "manifest.json"
    write_json_exclusive(manifest, [case])
    output = destination / "result.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    completed = subprocess.run([sys.executable, "-m", "lunar_matching", "localize", "--manifest", str(manifest),
        "--case-id", case["id"], "--policies", "beam_active", "--research-output", "--output", str(output)],
        cwd=destination, env=env, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    row = result["raw_research"]["methods"]["beam_active"]
    predicted = None if row["pose"] is None else np.asarray(row["pose"])
    translation = None if predicted is None else float(np.linalg.norm(predicted[:3, 3] - truth[:3, 3]))
    passed = (result["accepted"] is False and result["pose"] is None
              and translation is not None and translation < 2 and len(row["reference_xy"]) >= 6)
    receipt = dict(passed=passed, generated_only=True, physical_sunlight_simulation=False,
                   lunar_images_opened=False, truth_used_only_for_scoring=True,
                   translation_error_m=translation, native_matches=len(row["reference_xy"]),
                   actual_native_fields=row["acquisition"]["attempted_native_fields"],
                   public_pose_null=result["pose"] is None, accepted=result["accepted"],
                   source_manifest_sha256=sha256(manifest), result_sha256=sha256(output))
    write_json_exclusive(destination / "verification.json", receipt)
    if not passed:
        raise RuntimeError("The generated native CLI control failed; preserve its output.")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(check(parser.parse_args().output), indent=2))
