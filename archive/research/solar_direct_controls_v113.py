"""Fresh actual Direct forward of the frozen v90 procedural controls."""

from copy import copy
from dataclasses import replace
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from types import MappingProxyType

from control_direct_render_v90 import (
    CASES, CONFIGURATION, GEOMETRY, METHOD, native_geometry, render_library, replay_controls,
)


def controls(adapter, protocol_sha256, check, *, qualification=False, measure=None):
    """Run known/blank/invalid analytic native512 tests with the original SIFT."""
    import cv2
    import numpy as np
    from direct_render_adapter_v73 import load_native_render
    from historical_compact_adapters_v73 import value_digest
    from solar_direct_arm_bridge_v103 import BoundDirectArmCase
    from solar_evaluation_v70 import InferenceInput

    if adapter.profile["configuration"] != CONFIGURATION:
        raise ValueError("Actual Direct configuration differs from the frozen procedural control")
    arm = platform.system() == "Linux" and platform.machine() == "aarch64"
    if qualification and (not arm or measure is None):
        raise ValueError("Qualified measured controls require the actual ARM stratum")
    cv2.setNumThreads(1)
    procedural = copy(adapter)
    output = {
        "method": METHOD, "profile": METHOD, "protocol_sha256": protocol_sha256,
        "native_shape": [512, 512], "cpu_threads": 2,
        "configuration_sha256": value_digest(CONFIGURATION), "artifact_sha256": {},
        "source_sha256": {}, "query_truth_used": False, "lunar_images_loaded": 0,
        "calibration_or_final_access": False, "opencv_threads": cv2.getNumThreads(),
        "fresh_actual_arm": bool(arm and qualification),
    }
    with TemporaryDirectory(prefix="direct-v113-analytic-") as temp:
        geometry = native_geometry()
        lookup, library, views = render_library(Path(temp), geometry, check)
        procedural.libraries = {lookup.protocol["reference_geometry_sha256"]: library}
        bound = BoundDirectArmCase(procedural, MappingProxyType(dict(lookup.protocol)))
        target = views[GEOMETRY["target_light_index"]]
        query = cv2.warpAffine(target, np.asarray(GEOMETRY["reference_to_query_affine"]), (512, 512))
        blank = np.zeros_like(query)
        reads = 0

        def audited_loader(entry):
            nonlocal reads
            check()
            image = load_native_render(entry)
            reads += 1
            return image

        procedural.image_loader = audited_loader
        for kind in CASES:
            check()
            before = reads
            started = None if measure is None else measure()
            inp = InferenceInput(
                reference=blank if kind == "blank" else views[0], query=blank if kind == "blank" else query,
                geometry=native_geometry(False) if kind == "invalid_support" else geometry,
                query_intrinsics=geometry.intrinsics, reference_sun=np.asarray(library["entries"][0]["sun_world"]),
            )
            rejection = None
            try:
                result = bound.infer(inp)
            except ValueError as error:
                if kind != "invalid_support" or str(error) != "Common valid native reference geometry required":
                    raise
                result, rejection = {}, str(error)
            pose = result.get("pose")
            detail = {
                "reference_xy": np.asarray(result.get("reference_xy", [])).reshape(-1, 2).tolist(),
                "query_xy": np.asarray(result.get("query_xy", [])).reshape(-1, 2).tolist(),
                "pose_returned": pose is not None, "estimated_cam2world": None if pose is None else pose.tolist(),
                "render_search": result.get("render_search"), "rejection": rejection, "render_reads": reads - before,
            }
            if started is not None:
                detail["elapsed_seconds"] = measure() - started
            if kind == "known_transform":
                detail["reference_to_query_affine"] = GEOMETRY["reference_to_query_affine"]
            output[kind] = detail
        try:
            bound.infer(replace(inp, known_query_sun=np.asarray([0., 0., 1.])))
        except ValueError as error:
            if str(error) != "ARM direct inference requires the primary truth-free materialized input":
                raise
            output["injected_query_sun_rejected"] = True
        else:
            raise ValueError("Direct bridge leaked a query-Sun prior")
    output["actual_render_forward"] = len(library["entries"]) == 24 and output["known_transform"]["render_reads"] == 24
    output["replay"] = replay_controls(output, {"source_sha256": {}})
    output["passed"] = output["replay"]["passed"]
    check()
    return output
