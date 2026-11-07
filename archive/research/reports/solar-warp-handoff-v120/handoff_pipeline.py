"""Assemble a native handoff from reference-only geometry and scene-linear images."""

import numpy as np

from handoff_localizer import ANCHORS, MAX_SEEDS, NATIVE, Templates, localize
from handoff_patch import search_patch
from reference_patch_warp import ReferencePatchWarper


def seed_candidates(predictions, tag, reference_seed, query_seed):
    """Use the published 32-pixel 3D ranking for all three handoff arms."""
    rows = [r for r in predictions["rows"]
            if (r["tag"], r["reference_seed"], r["query_seed"], r["cell_pixels"])
            == (tag, reference_seed, query_seed, 32)]
    if len(rows) != 1 or predictions["calibration_or_final_access"] is not False:
        raise ValueError("Require one sealed development row from the primary detector scale")
    ranked = rows[0]["orders"]["geometry_3d"][:MAX_SEEDS]
    bank = np.asarray(predictions["banks"][tag], dtype=float)
    if (bank.ndim != 3 or bank.shape[1:] != (4, 4)
            or len(set(ranked)) != len(ranked) or
            any(type(i) is not int or not 0 <= i < len(bank) for i in ranked)):
        raise ValueError("Require the original candidate bank and distinct frozen indices")
    return bank[ranked], ranked


def detector_templates(banks):
    real, plane = banks["geometry_3d"], banks["planar"]
    if (not np.array_equal(real.detector_valid, plane.detector_valid)
            or not np.array_equal(real.anchor_visible, plane.anchor_visible)
            or not np.array_equal(real.predicted_xy, plane.predicted_xy, equal_nan=True)
            or not np.array_equal(real.reference_xy, plane.reference_xy)
            or not np.array_equal(real.reference_world, plane.reference_world, equal_nan=True)):
        raise ValueError("The 3D and fitted-plane control must share anchors, centres and detector support")
    anchors = (real.anchor_visible & np.isfinite(real.reference_world).all(axis=1)
               & np.isfinite(real.predicted_xy).all(axis=1))
    return Templates(real.raster, plane.raster, real.detector_valid, real.reference_xy,
                     real.reference_world, real.predicted_xy, anchors)


def run(reference_luminance, reference_axial_depth, reference_intrinsics, reference_pose,
        query_luminance, query_intrinsics, seeds, *, reference_valid=None, query_valid=None):
    """Raw inference on separate radiance arrays. The caller supplies no query labels."""
    if np.asarray(reference_luminance).shape != (NATIVE, NATIVE):
        raise ValueError("Expected the original native reference radiance")
    warper = ReferencePatchWarper(reference_luminance, reference_axial_depth,
                                  reference_intrinsics, reference_pose, ANCHORS,
                                  reference_valid=reference_valid)

    def warp(seed):
        return detector_templates(warper.warp(query_intrinsics, seed, (NATIVE, NATIVE)))

    return localize(query_luminance, query_intrinsics, seeds, warp, search_patch, query_valid)
