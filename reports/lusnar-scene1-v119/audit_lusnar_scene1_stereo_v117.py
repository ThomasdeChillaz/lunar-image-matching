"""Verify released Scene 1 depth independently with synchronized stereo disparity."""

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_lusnar_scene1_v117 import checked_read, digest, format_header, triplet_names, write_json


BASELINE_M = .310
FOCAL_PX = 610.17784
UNIT_SCALES = (("millimetre", .001), ("centimetre", .01), ("metre", 1.))
PALETTE_SOURCE = (
    "https://github.com/zqyu9/LuSNAR-dataset/blob/"
    "9c4a6073722be7268e65830c796aff67e4dd1576/README.md#-semantic-categories"
)
PALETTE_RGB = {
    "regolith": 0xBB469C,
    "crater": 0x7800C8,
    "rock": 0xE8FA50,
    "mountain": 0xAD451F,
    "sky": 0x22C9F8,
}


class SemanticPaletteError(ValueError):
    """The decoded label is incompatible with the frozen publisher palette."""


@dataclass(frozen=True)
class StereoPolicy:
    grid: int = 32
    radius: int = 5
    horizontal: int = 192
    vertical: int = 5
    min_ncc: float = .85
    min_peak_gap: float = .09
    max_cycle_px: float = .75
    min_tracks: int = 36
    min_off_axis: int = 18
    min_near_axis: int = 8
    min_tracks_per_quadrant: int = 4
    min_disparity_px: float = 2.
    max_vertical_median_px: float = .40
    max_vertical_p90_px: float = 1.
    max_reprojection_median_px: float = .8
    max_reprojection_p90_px: float = 1.8
    min_model_margin_px: float = .25


def decode_stereo_png(raw):
    format_header("left", raw)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (1024, 1024, 4) or image.dtype != np.uint8:
        raise ValueError("Expected frozen native RGBA PNG")
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), image[:, :, 3] == 255


def decode_pfm_storage(raw):
    """Return file-order float values; orientation and metric units stay unresolved."""
    header = format_header("depth", raw)
    file = io.BytesIO(raw)
    for _ in range(3):
        file.readline(64)
    dtype = "<f4" if header["storage_endian"] == "little" else ">f4"
    values = np.frombuffer(raw, dtype=dtype, offset=file.tell()).reshape(1024, 1024)
    return values.astype(np.float64) * header["scale_magnitude"]


def semantic_terrain_mask(image):
    """Interpret exact publisher RGB values after OpenCV palette expansion."""
    if image.ndim != 3 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
        raise SemanticPaletteError("Expected decoded 8-bit BGR or BGRA label")
    opaque = image[:, :, 3] == 255 if image.shape[-1] == 4 else np.ones(image.shape[:2], bool)
    pixels = image[:, :, :3].astype(np.uint32)
    rgb = (pixels[:, :, 2] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 0]
    counts = {}
    known = np.zeros(opaque.shape, bool)
    for name, colour in PALETTE_RGB.items():
        hit = (rgb == colour) & opaque
        known |= hit
        counts[name] = int(hit.sum())
    unknown = opaque & ~known
    if unknown.any() or not known.any():
        examples = np.unique(rgb[unknown])[:8]
        raise SemanticPaletteError(
            f"Unrecognized opaque semantic pixels={int(unknown.sum())}, examples={[f'#{code:06X}' for code in examples]}"
        )
    counts["nonopaque"] = int((~opaque).sum())
    return known & (rgb != PALETTE_RGB["sky"]), counts


def decode_semantic_png(raw):
    format_header("semantic", raw)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[:2] != (1024, 1024):
        raise SemanticPaletteError("Expected frozen native semantic PNG")
    return semantic_terrain_mask(image)


def search_patch(source, target, source_valid, target_valid, centre, policy):
    """Independent native NCC peak with a competing-peak gap and subpixel offset."""
    h, w = source.shape
    x, y = map(float, centre)
    ix, iy = round(x), round(y)
    r = policy.radius
    if (not r + 1 <= ix < w - r - 1 or not r + 1 <= iy < h - r - 1
            or not source_valid[iy-r-1:iy+r+2, ix-r-1:ix+r+2].all()):
        return None
    patch = cv2.getRectSubPix(source, (2*r + 1, 2*r + 1), (x, y))
    if patch.std() < 5.:
        return None
    xmin, xmax = max(r, ix-policy.horizontal), min(w-r-1, ix+policy.horizontal)
    ymin, ymax = max(r, iy-policy.vertical), min(h-r-1, iy+policy.vertical)
    image = target[ymin-r:ymax+r+1, xmin-r:xmax+r+1]
    correlation = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF_NORMED)
    valid = target_valid[ymin-r:ymax+r+1, xmin-r:xmax+r+1]
    support = cv2.boxFilter(valid.astype(np.float32), cv2.CV_32F, (2*r+1, 2*r+1), normalize=False)
    correlation[support[r:-r, r:-r] != (2*r+1)**2] = -np.inf
    py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = float(correlation[py, px])
    if peak < policy.min_ncc or not (0 < px < correlation.shape[1]-1 and 0 < py < correlation.shape[0]-1):
        return None
    adjacent = correlation[py-1:py+2, px-1:px+2]
    if not np.isfinite(adjacent).all():
        return None
    other = correlation.copy()
    other[max(0, py-4):py+5, max(0, px-4):px+5] = -np.inf
    second = max(-1., float(other.max()))
    gap = peak - second
    if gap < policy.min_peak_gap:
        return None
    dx, dy = (parabolic_offset(adjacent[1, :]), parabolic_offset(adjacent[:, 1]))
    return np.array([xmin + px + dx, ymin + py + dy]), peak, gap


def parabolic_offset(three):
    lower, centre, upper = map(float, three)
    curvature = lower - 2.*centre + upper
    if curvature >= -1e-6:
        return 0.
    return float(np.clip(.5 * (lower - upper) / curvature, -.5, .5))


def stereo_tracks(left, right, left_valid, right_valid, policy):
    """Pick a spatially spread corner in each cell and measure forward/reverse tracks."""
    if left.ndim != 2 or left.shape != right.shape or left.shape != left_valid.shape or left.shape != right_valid.shape:
        raise ValueError("Aligned grayscale/validity arrays required")
    corners = cv2.cornerMinEigenVal(left, 5)
    h, w = left.shape
    tracks = []
    for y in range(0, h, policy.grid):
        for x in range(0, w, policy.grid):
            patch = corners[y:y+policy.grid, x:x+policy.grid]
            py, px = np.unravel_index(np.argmax(patch), patch.shape)
            if patch[py, px] < .0001:
                continue
            xy = np.array([x+px, y+py])
            forward = search_patch(left, right, left_valid, right_valid, xy, policy)
            if forward is None:
                continue
            back = search_patch(right, left, right_valid, left_valid, forward[0], policy)
            if back is None:
                continue
            error = float(np.linalg.norm(back[0] - xy))
            if error <= policy.max_cycle_px:
                tracks.append((xy[0], xy[1], *forward[0], min(forward[1], back[1]), min(forward[2], back[2]), error))
    return np.asarray(tracks, dtype=float).reshape(-1, 7)


def smooth_valid(depth, x, y):
    samples = np.stack([depth[y+dy, x+dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    valid = np.isfinite(samples).all(axis=0) & (samples > 0).all(axis=0)
    with np.errstate(invalid="ignore"):
        spread = samples.max(axis=0) - samples.min(axis=0)
    return valid & (spread <= .10 * samples[4])


def report_error(values):
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .90)))


def audit_tracks(tracks, storage, policy, *, focal_px=FOCAL_PX, baseline_m=BASELINE_M, terrain_mask=None):
    """Score all depth hypotheses on the exact same image-selected terrain samples."""
    h, w = storage.shape
    if tracks.ndim != 2 or tracks.shape[1] != 7 or not np.isfinite(tracks).all():
        raise ValueError("Expected finite native stereo tracks")
    if not np.isfinite([focal_px, baseline_m]).all() or min(focal_px, baseline_m) <= 0:
        raise ValueError("Finite documented camera scale required")
    if terrain_mask is not None and (terrain_mask.shape != storage.shape or terrain_mask.dtype != np.bool_):
        raise ValueError("Terrain mask must be Boolean on the native left detector")
    centres = (("pixel_centres", (w-1)/2., (h-1)/2.), ("half_dimensions", w/2., h/2.))
    orientation = {"bottom_to_top_storage": storage[::-1], "top_to_bottom_storage": storage}
    report = dict(
        registered_tracks=len(tracks), common_depth_tracks=0, discriminating_tracks=0,
        near_axis_tracks=0, off_axis_tracks=0, quadrant_tracks=[0, 0, 0, 0],
        semantic_static_mask_used=terrain_mask is not None, models=[], selected_model=None, reason=None,
    )
    if not len(tracks):
        return dict(report, reason="no_confident_stereo_registration")
    x, y = tracks[:, :2].T.astype(int)
    if (not np.array_equal(tracks[:, :2], np.column_stack([x, y]))
            or not ((0 < x) & (x < w-1) & (0 < y) & (y < h-1)).all()):
        raise ValueError("Native left track coordinates must be integer and interior")
    common = np.logical_and.reduce([smooth_valid(depth, x, y) for depth in orientation.values()])
    if terrain_mask is not None:
        common &= np.logical_and.reduce([terrain_mask[y+dy, x+dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    valid = tracks[common]
    report["common_depth_tracks"] = len(valid)
    if not len(valid):
        return dict(report, reason="no_shared_stable_depth_support")
    disparity = valid[:, 0] - valid[:, 2]
    vertical = valid[:, 1] - valid[:, 3]
    report["registration"] = dict(
        ncc=report_error(valid[:, 4]), peak_gap=report_error(valid[:, 5]), cycle_px=report_error(valid[:, 6]),
        signed_horizontal_median_px=float(np.median(disparity)),
        signed_vertical_median_px=float(np.median(vertical)), absolute_vertical_px=report_error(np.abs(vertical)),
        reverse_sign_fraction=float(np.mean(disparity < -.5)),
    )
    select = np.abs(disparity) >= policy.min_disparity_px
    valid, disparity, vertical = valid[select], disparity[select], vertical[select]
    x, y = valid[:, :2].T.astype(int)
    radius = np.hypot(x - centres[0][1], y - centres[0][2]) / focal_px
    off_axis, near_axis = radius >= .42, radius <= .22
    quadrant = (x >= w/2.).astype(int) + 2*(y >= h/2.).astype(int)
    counts = np.bincount(quadrant, minlength=4)
    report.update(discriminating_tracks=len(valid), near_axis_tracks=int(near_axis.sum()),
                  off_axis_tracks=int(off_axis.sum()), quadrant_tracks=counts.tolist())
    if (len(valid) < policy.min_tracks or off_axis.sum() < policy.min_off_axis
            or near_axis.sum() < policy.min_near_axis or (counts >= policy.min_tracks_per_quadrant).sum() < 3):
        return dict(report, reason="insufficient_spatial_discrimination")
    registration = report["registration"]
    if (registration["reverse_sign_fraction"] > .02
            or registration["absolute_vertical_px"]["median"] > policy.max_vertical_median_px
            or registration["absolute_vertical_px"]["p90"] > policy.max_vertical_p90_px):
        return dict(report, reason="documented_rectified_baseline_not_supported")
    report["depth_scoring_evidence"] = {
        "native_track_columns": ["left_x", "left_y", "right_x", "right_y", "min_ncc", "min_peak_gap", "cycle_px"],
        "native_tracks": valid.tolist(),
        "stored_depth": {order: depth[y, x].tolist() for order, depth in orientation.items()},
    }
    for order, depth in orientation.items():
        raw_depth = depth[y, x]
        for kind in ("axial", "ray"):
            for units, scale in UNIT_SCALES:
                centre_reports = []
                for centre, cx, cy in centres:
                    ray_norm = np.sqrt(1. + ((x-cx)**2 + (y-cy)**2) / focal_px**2)
                    axial_per_unit = raw_depth if kind == "axial" else raw_depth / ray_norm
                    ideal = focal_px * baseline_m / axial_per_unit
                    prediction = ideal / scale
                    residual = disparity - prediction
                    reprojection = np.hypot(residual, vertical)
                    if not np.isfinite(reprojection).all():
                        raise ValueError("Depth hypothesis does not produce finite stereo projection")
                    centre_reports.append(dict(centre=centre, reprojection_px=report_error(reprojection),
                                               off_axis_horizontal_px=report_error(np.abs(residual[off_axis])),
                                               near_axis_horizontal_px=report_error(np.abs(residual[near_axis])),
                                               implied_metres_per_stored_unit=float(np.median(ideal / disparity))))
                best = min(centre_reports, key=lambda row: row["off_axis_horizontal_px"]["median"])
                report["models"].append(dict(row_order=order, depth=kind, units=units, scale_m=scale,
                                             best_centre=best, centres=centre_reports))
    report["models"].sort(key=lambda row: row["best_centre"]["off_axis_horizontal_px"]["median"])
    best, second = (row["best_centre"] for row in report["models"][:2])
    report["runner_up_off_axis_margin_px"] = (
        second["off_axis_horizontal_px"]["median"] - best["off_axis_horizontal_px"]["median"]
    )
    if (best["reprojection_px"]["median"] > policy.max_reprojection_median_px
            or best["reprojection_px"]["p90"] > policy.max_reprojection_p90_px):
        return dict(report, reason="no_documented_depth_hypothesis_meets_reprojection")
    if report["runner_up_off_axis_margin_px"] < policy.min_model_margin_px:
        threshold = best["off_axis_horizontal_px"]["median"] + policy.min_model_margin_px
        competitors = [row for row in report["models"] if row["best_centre"]["off_axis_horizontal_px"]["median"] < threshold]
        report["ambiguous_fields"] = [
            field for field in ("row_order", "depth", "units") if len({row[field] for row in competitors}) > 1
        ]
        return dict(report, reason="depth_orientation_units_or_ray_axis_ambiguous")
    chosen = report["models"][0]
    return dict(report, selected_model={key: chosen[key] for key in ("row_order", "depth", "units", "scale_m")})


def semantic_entries(path, expected_sha, freeze, freeze_sha):
    receipt = json.loads(checked_read(path, expected_sha))
    expected = {case["timestamp_ns"]: f"Moon_1/image0/label/{case['timestamp_ns']}.png" for case in freeze["triplets"]}
    rows = receipt["entries"]
    if (receipt["schema"] != "lusnar_scene1_semantic_support_v119"
            or receipt["scene"] != 1 or receipt["complete"] is not True
            or receipt["parent_extraction_freeze_sha256"] != freeze_sha
            or receipt["reserved_scenes_opened"] != [] or receipt["semantic_sample_arrays_decoded"] != 0
            or len(rows) != 5 or {row["timestamp_ns"]: row["path"] for row in rows} != expected
            or any(row["role"] != "semantic" for row in rows)):
        raise ValueError("Require exact sealed left semantic sidecars for the same five Scene 1 timestamps")
    return {row["timestamp_ns"]: row for row in rows}


def run(freeze_path, extracted, receipt_sha, output, *, semantic_root=None, semantic_receipt=None, semantic_receipt_sha=None):
    if output.exists():
        raise FileExistsError("Preserve previous stereo audit")
    receipt = json.loads(checked_read(extracted / "COMPLETE.json", receipt_sha))
    freeze = json.loads(checked_read(freeze_path, receipt["freeze_sha256"]))
    if (freeze["schema"] != "lusnar_scene1_exact_local_extraction_v117"
            or receipt["schema"] != "lusnar_scene1_extraction_receipt_v117"
            or freeze["scene"] != 1 or receipt["scene"] != 1 or freeze["split"] != "exposed_development"
            or freeze["reserve_access_allowed"] is not False or receipt["reserved_scenes_opened"] != []
            or receipt["complete"] is not True or len(freeze["triplets"]) != 5):
        raise ValueError("Require original sealed five-triplet Scene 1 extraction")
    timestamps = [case["timestamp_ns"] for case in freeze["triplets"]]
    if len(set(timestamps)) != 5 or any(
        not isinstance(t, str) or len(t) != 19 or not t.isascii() or not t.isdigit() for t in timestamps
    ):
        raise ValueError("Exact native Scene 1 timestamps required")
    if any(x is None for x in (semantic_root, semantic_receipt, semantic_receipt_sha)) and any(
        x is not None for x in (semantic_root, semantic_receipt, semantic_receipt_sha)
    ):
        raise ValueError("Semantic root, receipt and receipt digest must be provided together")
    semantic = semantic_entries(semantic_receipt, semantic_receipt_sha, freeze, receipt["freeze_sha256"]) if semantic_root else {}
    if semantic:
        semantic_root = semantic_root.resolve(strict=True)
    images = [row for row in receipt["entries"] if row["role"] != "metadata"]
    entries = {row["path"]: row for row in images}
    policy = StereoPolicy()
    results = []
    cv2.setNumThreads(1)
    root = extracted.resolve(strict=True)
    expected_names = {name for case in freeze["triplets"] for name in triplet_names(case["timestamp_ns"]).values()}
    if len(expected_names) != 15 or len(images) != 15 or set(entries) != expected_names:
        raise ValueError("Exact extracted triplet set required")
    for case in freeze["triplets"]:
        timestamp = case["timestamp_ns"]
        data = {}
        for role, name in triplet_names(timestamp).items():
            entry, member = entries[name], case["members"][role]
            path = root / name
            if (entry["role"] != role or entry["timestamp_ns"] != timestamp or member["name"] != name
                    or entry["local_zip_crc32"] != member["crc32"] or path.resolve(strict=True) != path):
                raise ValueError("Extraction receipt does not bind the native Scene 1 member")
            raw = checked_read(path, entry["sha256"])
            if len(raw) != member["bytes"] or len(raw) != entry["bytes"]:
                raise ValueError("Frozen member byte count differs")
            data[role] = raw
        left, left_valid = decode_stereo_png(data["left"])
        right, right_valid = decode_stereo_png(data["right"])
        terrain, palette_counts = None, None
        if semantic:
            row = semantic[timestamp]
            path = semantic_root / row["path"]
            if path.resolve(strict=True) != path:
                raise ValueError("Semantic sidecar resolves outside its scoped path")
            raw = checked_read(path, row["sha256"])
            if len(raw) != row["bytes"] or format_header("semantic", raw) != row["header"]:
                raise ValueError("Frozen semantic member or header differs")
            try:
                terrain, palette_counts = decode_semantic_png(raw)
            except SemanticPaletteError as exc:
                results.append(dict(timestamp_ns=timestamp, selected_model=None, reason="unrecognized_semantic_palette",
                                    semantic_error=str(exc), semantic_static_mask_used=False))
                continue
            left_valid &= terrain
        stereo = stereo_tracks(left, right, left_valid, right_valid, policy)
        depth = decode_pfm_storage(data["depth"])
        audited = audit_tracks(stereo, depth, policy, terrain_mask=terrain)
        results.append(dict(timestamp_ns=timestamp, semantic_palette_counts=palette_counts, **audited))
    decisive = [row["selected_model"] for row in results if row["selected_model"] is not None]
    semantic_failed = any(row["reason"] == "unrecognized_semantic_palette" for row in results)
    if not semantic_failed and len(decisive) >= 3 and all(row == decisive[0] for row in decisive):
        consensus = decisive[0]
    else:
        consensus = None
    write_json(output, dict(
        schema="lusnar_scene1_stereo_geometry_audit_v117", source_sha256=digest(Path(__file__).read_bytes()),
        extraction_freeze_sha256=receipt["freeze_sha256"], extraction_receipt_sha256=receipt_sha,
        scene=1, exposed_development=True, reserved_scenes_opened=[], cases=results, case_count=len(results),
        semantic_receipt_sha256=semantic_receipt_sha, semantic_palette_source=PALETTE_SOURCE if semantic else None,
        documented_focal_px=FOCAL_PX, documented_baseline_m=BASELINE_M, policy=policy.__dict__,
        decisive_cases=len(decisive), conditional_stereo_depth_convention=consensus if semantic else None,
        unmasked_diagnostic_depth_convention=consensus if not semantic else None, principal_point_verified=False,
        camera_trajectory_extrinsics_verified=False,
        static_terrain_evidence="verified_left_mask_and_reciprocal_stereo" if semantic and not semantic_failed
        else "requested_palette_unrecognized" if semantic_failed else "unverified_semantics_diagnostic_only",
        scope="Same-timestamp stereo geometry under the published rig; no camera-pose or held-out transfer claim.",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--semantic-root", type=Path)
    parser.add_argument("--semantic-receipt", type=Path)
    parser.add_argument("--semantic-receipt-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.freeze, args.extracted, args.receipt_sha256, args.output, semantic_root=args.semantic_root,
        semantic_receipt=args.semantic_receipt, semantic_receipt_sha=args.semantic_receipt_sha256)
