"""Measure top-tail shape agreement at independently projected native pixels."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

from audit_clean_signal_v107 import BASE, COHORT, DISPATCH, NOISE_SIGMA, PINS, checked, write

PUBLIC = "dba59e49c974e2feca40c4c7b25e4e2d12e042a8"
V108_PROTOCOL = "reports/solar-clean-covision-protocol-v108.json"
V108_EVIDENCE = {
    V108_PROTOCOL: "ccd1d2d4220f3ef118703b42f57109328df0671d271e0e5ae044d71ba3fc86e0",
    "reports/solar_clean_covision_v108.json": "d92ab62049b149cc9188d1267d4add78bfca892cd1c95b7cf085099d0c57f4c5",
    "reports/solar_clean_covision_replay_v108.json": "3c4262e6eda60233ef49f245ceb0f138d7674eaad1189b5163e8e728a28eb403",
}
REPO_SOURCES = ("audit_clean_signal_v107.py", "lunar_geometry_v5.py", "lunar_geometry_v2.py")
NATIVE = (512, 512)
TILE = 128
OFFSETS = ((0, 0), (-32, 0), (32, 0), (0, -32), (0, 32))
PERCENTILES = (98, 99, 99.5)
ABS_Z_TOLERANCE = .05
REL_Z_TOLERANCE = .001
ABS_Z_SPREAD = .10
REL_Z_SPREAD = .01
ROUND_TRIP_PX = .5


def digest(raw):
    return sha256(raw).hexdigest()


def freeze():
    """Check only source and published metadata, with no image or depth loads."""
    for path, checksum in {**PINS, **V108_EVIDENCE}.items():
        checked(path, checksum)
    for path in (*REPO_SOURCES, COHORT, DISPATCH):
        original = subprocess.check_output(["git", "show", f"{PUBLIC}:{path}"], timeout=5)
        if original != Path(path).read_bytes():
            raise ValueError("Earlier public source differs: " + path)
    dispatch, cohort, render, prior = [
        json.loads(Path(path).read_bytes()) for path in
        (DISPATCH, COHORT, BASE + "/render/COMPLETE.json", V108_PROTOCOL)
    ]
    manifest_path, manifest_checksum = cohort["case_manifest"], cohort["case_manifest_sha256"]
    manifest = json.loads(checked(manifest_path, manifest_checksum))
    if (not manifest["complete"] or not render["complete"] or dispatch["manifest"] != manifest_path
            or dispatch["cohort_sha256"] != PINS[COHORT] or render["cohort_sha256"] != PINS[COHORT]
            or prior["calibration_or_final_access"] or prior["inference"]
            or any(m["split"] != "model_selection" or m["calibration_or_final_access"]
                   for m in (dispatch, cohort, manifest, render))):
        raise ValueError("Only the completed model-selection cohort is eligible")
    cases = {x["case_id"]: x for x in manifest["cases"]}
    images = {x["clean_rgb16"]["path"]: x for x in render["rows"]}
    items = []
    for prior_item in prior["items"]:
        case = cases[prior_item["case_id"]]
        geom = case["scoring_geometry"]
        seal = case["input_sha256"][geom]
        if (not case["positive"] or case["split"] != "model_selection"
                or case["unit_index"] != case["query_unit_index"] or case["unit_index"] != prior_item["unit"]
                or case["light_b"] != prior_item["light_index"] or case["component"] != case["query_component"]
                or case["scoring_inputs"]["geometry"] != geom or dispatch["input_sha256"][geom] != seal):
            raise ValueError("Fixed positive model-selection case and scoring geometry required")
        views = {}
        for role, asset in prior_item["exposures"].items():
            row = images[asset["path"]]
            if (row["role"] != role or row["unit_index"] != prior_item["unit"] or row["clean_rgb16"] != asset
                    or row["light_index"] != prior_item["light_index"] or asset["bit_depth"] != 16
                    or row["clean_dither"] != 0 or not row["geometry"]["exact_camera_depth_and_rays"]
                    or row["geometry"]["native_rays"] != NATIVE[0]*NATIVE[1]):
                raise ValueError("v106 exact geometry and unchanged same-light view required")
            views[role] = dict(image=asset, archive_geometry=dict(
                path=row["archive_geometry_path"], sha256=row["archive_geometry_sha256"]))
        if set(views) != {"reference", "query"}:
            raise ValueError("Both prior view roles required")
        items.append(dict(case_id=prior_item["case_id"], component=case["component"],
                          purpose=prior_item["purpose"], light_index=prior_item["light_index"], views=views,
                          scoring_geometry=dict(path=geom, sha256=seal)))
    if (len(items) != 4 or [item["purpose"] for item in items] !=
            ["sparse_tail_primary", "bright_control", "near_uniform_control", "near_uniform_control"]):
        raise ValueError("Exact v108 audited case order required")
    return dict(
        schema="solar_tail_morphology_protocol_v109",
        public_predecessor_commit=PUBLIC,
        prior_measurement_seals=V108_EVIDENCE,
        render_and_dispatch_seals=PINS,
        case_manifest=dict(path=manifest_path, sha256=manifest_checksum),
        source_sha256=digest(Path(__file__).read_bytes()),
        repo_source_sha256={path: digest(Path(path).read_bytes()) for path in REPO_SOURCES},
        items=items,
        native_shape=list(NATIVE),
        query_offsets_pixels=[list(x) for x in OFFSETS],
        depth_visibility=dict(
            coordinate_rule="All finite positive reference axial depths at integer pixel centres are projected through exact scoring cameras.",
            sample_rule="Require all four finite positive query depths at the bilinear stencil. Use inverse axial depth interpolation.",
            max_bilinear_depth_spread_m=f"{ABS_Z_SPREAD} + {REL_Z_SPREAD} * nearest depth in stencil",
            max_axial_depth_disagreement_m=f"{ABS_Z_TOLERANCE} + {REL_Z_TOLERANCE} * predicted depth",
            reciprocal_max_reference_pixel_error=ROUND_TRIP_PX,
            reciprocal_reference_depth_checked=True,
            label_scope="Conservative depth-based subpixel co-visibility, without a new mesh raycast. Unknown and occluded samples excluded.",
        ),
        common_support=(
            "Use aligned co-visible reference pixels whose nearest query locations and every shifted query location "
            "have an in-frame, all-positive-depth 3x3 neighborhood. Rounding ties go to the larger pixel index. "
            "Keep one reference pixel per aligned query pixel, choosing smallest projected distance to its centre "
            "with reference row-major order as tie-break. No shift wraps. Offsets share identical reference support."
        ),
        photometry=dict(
            grayscale="v107 weighted sRGB BGR code before inverse sRGB transfer.",
            threshold="A pixel is bright if its scene-linear value strictly exceeds both its own view's depth-valid percentile and that view's P01 plus 3 sigma. All ranks are linear empirical quantiles.",
            linear_noise_sigma=NOISE_SIGMA,
            primary_percentile=99,
            percentile_sensitivity=list(PERCENTILES),
            tolerance_sensitivity="Symmetric 3x3 maximum filter on native binary masks for one-pixel Chebyshev tolerance. Same common support.",
        ),
        comparison=dict(
            primary="Case u061, percentile 99, no dilation. Report Jaccard and phi at true projection and each shifted null.",
            descriptive_signal="Both aligned Jaccard and phi must strictly exceed every shifted value. Any undefined comparison produces an indeterminate primary outcome.",
            controls="Same fixed statistics for the one bright and two near-uniform controls. No p-values for dependent raster samples.",
            spatial_counts=f"Primary shared bright intersections in {TILE}-pixel native reference tiles.",
        ),
        query_truth_policy="Independent sealed cameras/depth and physical query light are used for scoring only.",
        scope="Descriptive retrospective morphology audit on four exposed positives. No matching, pose fitting, rendering, training, human labels, calibration or final inputs.",
        calibration_or_final_access=False,
    )


def inverse_depth_sample(depth, xy):
    import numpy as np

    x, y = xy.T
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = x - ix, y - iy
    stencil = np.stack([depth[iy, ix], depth[iy, ix+1], depth[iy+1, ix], depth[iy+1, ix+1]], axis=1)
    complete = (np.isfinite(stencil) & (stencil > 0)).all(axis=1)
    safe = np.where(complete[:, None], stencil, 1.)
    nearest = safe.min(axis=1)
    weights = np.column_stack(((1-a)*(1-b), a*(1-b), (1-a)*b, a*b))
    supported = complete & (np.ptp(safe, axis=1) <= ABS_Z_SPREAD + REL_Z_SPREAD*nearest)
    return 1 / np.sum(weights / safe, axis=1), supported


def valid_neighborhoods(depth):
    import cv2
    import numpy as np

    valid = (np.isfinite(depth) & (depth > 0)).astype(np.uint8)
    return cv2.erode(valid, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=0).astype(bool)


def native_support(reference_depth, query_depth, k_ref, ref_pose, k_query, query_pose):
    import numpy as np
    from lunar_geometry_v5 import project_z, world_points_z

    height, width = query_depth.shape
    y, x = np.nonzero(np.isfinite(reference_depth) & (reference_depth > 0))
    p = np.column_stack((x, y))
    depth = reference_depth[y, x]
    world = world_points_z(p, depth, k_ref, ref_pose)
    q, expected_z, in_front = project_z(world, k_query, query_pose)
    in_frame = in_front & np.isfinite(q).all(axis=1) & (
        (q >= 0) & (q < [width-1, height-1])).all(axis=1)
    counts = dict(reference_valid=len(p), query_bilinear_footprint=int(in_frame.sum()))
    p, q, depth, expected_z = [a[in_frame] for a in (p, q, depth, expected_z)]
    observed_z, complete = inverse_depth_sample(query_depth, q)
    counts["query_supported_smooth_depth_stencil"] = int(complete.sum())
    tolerance = ABS_Z_TOLERANCE + REL_Z_TOLERANCE*expected_z
    counts["inconsistent_nearer_query_surface"] = int((complete & (observed_z < expected_z-tolerance)).sum())
    visible = complete & (np.abs(observed_z-expected_z) <= tolerance)
    counts["query_depth_agreement"] = int(visible.sum())
    p, q, depth, observed_z = [a[visible] for a in (p, q, depth, observed_z)]
    sampled_world = world_points_z(q, observed_z, k_query, query_pose)
    p_back, p_back_z, back_front = project_z(sampled_world, k_ref, ref_pose)
    reciprocal = (back_front & (np.linalg.norm(p_back-p, axis=1) <= ROUND_TRIP_PX)
                  & (np.abs(p_back_z-depth) <= ABS_Z_TOLERANCE + REL_Z_TOLERANCE*depth))
    p, q = p[reciprocal], q[reciprocal]
    counts["reciprocal_cov_visible"] = len(p)
    centers = np.floor(q + .5).astype(int)
    support = valid_neighborhoods(query_depth)
    shared = valid_neighborhoods(reference_depth)[p[:, 1], p[:, 0]]
    for offset in OFFSETS:
        xy = centers + offset
        inside = ((xy >= 0) & (xy < [width, height])).all(axis=1)
        shift_supported = np.zeros(len(xy), bool)
        shift_supported[inside] = support[xy[inside, 1], xy[inside, 0]]
        shared &= shift_supported
    p, q, centers = [a[shared] for a in (p, q, centers)]
    counts["common_support_before_unique"] = len(p)
    # Stable deterministic tie-breaking by original reference raster order.
    key = centers[:, 1]*width + centers[:, 0]
    order = np.lexsort((np.arange(len(p)), ((q-centers)**2).sum(axis=1), key))
    keep = order[np.unique(key[order], return_index=True)[1]]
    counts["common_support_unique"] = len(keep)
    return p[keep], centers[keep], counts


def load_geometry(item):
    import numpy as np

    geometry = item["scoring_geometry"]
    with np.load(BytesIO(checked(geometry["path"], geometry["sha256"])), allow_pickle=False) as scoring:
        result = {key: scoring[key] for key in scoring.files}
    for role, view in item["views"].items():
        archive = view["archive_geometry"]
        with np.load(BytesIO(checked(archive["path"], archive["sha256"])), allow_pickle=False) as native:
            depth = native["axial_depth"]
            if (depth.shape != NATIVE or not np.array_equal(native["valid"], np.isfinite(depth) & (depth > 0))
                    or not np.array_equal(depth, result[role+"_depth"], equal_nan=True)
                    or any(not np.array_equal(native[key], result[role+"_"+key])
                           for key in ("intrinsics", "cam2world"))):
                raise ValueError("Scoring axial depths and cameras differ from the original archive")
    return tuple(result[k] for k in ("reference_depth", "query_depth", "reference_intrinsics",
                                    "reference_cam2world", "query_intrinsics", "query_cam2world"))


def decode(asset):
    import cv2
    import numpy as np

    raw = checked(asset["path"], asset["sha256"])
    if (raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            or raw[16:24] != b"\x00\x00\x02\x00"*2 or raw[24:26] != b"\x10\x02"):
        raise ValueError("Expected native RGB16 PNG")
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (*NATIVE, 3) or bgr.dtype != np.uint16:
        raise ValueError("Expected unchanged native RGB16")
    encoded = (0.114*bgr[..., 0] + 0.587*bgr[..., 1] + 0.299*bgr[..., 2]) / 65535.
    return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055)/1.055)**2.4)


def tail_masks(linear, depth):
    import numpy as np

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("No valid terrain pixels")
    floor, *quantiles = np.percentile(linear[valid], [1, *PERCENTILES], method="linear")
    out = {}
    for percentile, quantile in zip(PERCENTILES, quantiles, strict=True):
        threshold = max(float(quantile), float(floor + 3*NOISE_SIGMA))
        mask = valid & (linear > threshold)
        out[percentile] = (mask, dict(linear_cut=threshold, p01_linear=float(floor),
                                     ranked_cut_linear=float(quantile), bright_valid_pixels=int(mask.sum()),
                                     valid_pixels=int(valid.sum())))
    return out


def overlap(left, right):
    import numpy as np

    n = len(left)
    l, r, joint = int(left.sum()), int(right.sum()), int((left & right).sum())
    union = l + r - joint
    phi_base = l*r*(n-l)*(n-r)
    return dict(common_pixels=n, reference_bright=l, query_bright=r, both_bright=joint,
                jaccard=joint/union if union else None,
                phi=float((joint*n-l*r)/np.sqrt(float(phi_base))) if phi_base else None,
                lift=joint*n/(l*r) if l and r else None)


def assess_primary(scores):
    for metric in ("jaccard", "phi"):
        if any(row[metric] is None for row in scores):
            return "indeterminate"
    if all(scores[0][metric] > max(row[metric] for row in scores[1:]) for metric in ("jaccard", "phi")):
        return "above_every_shift_on_both_metrics"
    return "does_not_exceed_every_shift_on_both_metrics"


def measure(frozen):
    import cv2
    import numpy as np

    cv2.setNumThreads(2)
    rows = []
    for item in frozen["items"]:
        geometry = load_geometry(item)
        p, q, support = native_support(*geometry)
        ref, query = [decode(item["views"][role]["image"]) for role in ("reference", "query")]
        ref_tails, query_tails = tail_masks(ref, geometry[0]), tail_masks(query, geometry[1])
        tails = {}
        for percentile in PERCENTILES:
            left, left_stats = ref_tails[percentile]
            right, right_stats = query_tails[percentile]
            results = {}
            for dilation in (0, 1):
                if dilation:
                    kernel = np.ones((3, 3), np.uint8)
                    left = cv2.dilate(left.astype(np.uint8), kernel).astype(bool)
                    right = cv2.dilate(right.astype(np.uint8), kernel).astype(bool)
                a = left[p[:, 1], p[:, 0]]
                scores = []
                for dx, dy in OFFSETS:
                    b = right[q[:, 1]+dy, q[:, 0]+dx]
                    scores.append(dict(query_offset_xy=[dx, dy], **overlap(a, b)))
                    if percentile == 99 and not dilation and not (dx or dy):
                        tiles_x, tiles_y = NATIVE[1]//TILE, NATIVE[0]//TILE
                        tiles = p[a & b]//TILE
                        scores[-1]["both_bright_reference_tiles_yx"] = np.bincount(
                            tiles[:, 1]*tiles_x + tiles[:, 0], minlength=tiles_x*tiles_y,
                        ).reshape(tiles_y, tiles_x).tolist()
                results[str(dilation)] = scores
            tails[str(percentile)] = dict(reference=left_stats, query=right_stats, dilation_pixels=results)
        rows.append(dict(case_id=item["case_id"], purpose=item["purpose"], geometry=support, tails=tails,
                         primary_shape_result=assess_primary(tails["99"]["dilation_pixels"]["0"])))
    return rows


def self_check():
    import cv2
    import numpy as np

    cv2.setNumThreads(2)
    shape = (80, 80)
    depth = np.full(shape, 10.)
    camera = np.eye(4)
    k = np.array([[70., 0., 39.5], [0., 70., 39.5], [0., 0., 1.]])
    p, q, counts = native_support(depth, depth, k, camera, k, camera)
    if (not np.array_equal(p, q) or p.min() != 33 or p.max() != 46
            or counts["common_support_unique"] != 196):
        raise AssertionError("Identity projection or common shift footprint differs")
    query = depth.copy()
    query[37:43, 37:43] = 5.
    qn = query.copy()
    qn[37:43, 5:11] = np.nan
    p, _, counts = native_support(depth, qn, k, camera, k, camera)
    if (((p >= 37) & (p < 43)).all(axis=1).any() or not counts["inconsistent_nearer_query_surface"]
            or counts["common_support_unique"] >= 196):
        raise AssertionError("Occlusions or a null-shift missing footprint were accepted")
    sample, valid = inverse_depth_sample(np.array([[10., 11.], [11., 12.]]), np.array([[.5, .5]]))
    if valid[0] or not 10 < sample[0] < 12:
        raise AssertionError("Discontinuous stencil must not certify visibility")
    if any(mask.any() for mask, _ in tail_masks(np.zeros(shape), depth).values()):
        raise AssertionError("Constant images must have no bright-tail points")
    n = np.array([True, False, True, False])
    if overlap(n, n)["phi"] != 1 or overlap(n, ~n)["phi"] != -1:
        raise AssertionError("Synthetic morphology metrics differ")
    print("v109 procedural geometry and metric controls passed; no archived inputs opened.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "measure", "self-check"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
        return
    if args.protocol is None:
        parser.error("freeze and measure require --protocol")
    current = freeze()
    if args.action == "freeze":
        write(args.protocol, current)
        print(digest(args.protocol.read_bytes()))
        return
    if args.protocol_sha256 is None or args.output is None:
        parser.error("measure requires --protocol-sha256 and --output")
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads(checked(args.protocol, args.protocol_sha256))
    if current != frozen:
        raise ValueError("Current source and assets differ from the predeclared protocol")
    write(args.output, dict(schema="solar_tail_morphology_v109", protocol_sha256=args.protocol_sha256,
                            rows=measure(frozen), matcher_calls=0, optimizer_updates=0,
                            renderer_calls=0, calibration_or_final_access=False))


if __name__ == "__main__":
    main()
