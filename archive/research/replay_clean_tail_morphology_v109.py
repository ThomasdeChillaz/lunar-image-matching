"""Recompute the frozen tail morphology report without importing the audit."""
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


def read(asset):
    raw = Path(asset["path"]).read_bytes()
    if sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError("Asset checksum differs: " + str(asset["path"]))
    return raw


def data(asset):
    return np.load(BytesIO(read(asset)), allow_pickle=False)


def image(asset):
    pixels = cv2.imdecode(np.frombuffer(read(asset), np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels.shape != (512, 512, 3) or pixels.dtype != np.uint16:
        raise ValueError("Expected unchanged native RGB16")
    code = (pixels @ [.114, .587, .299]) / 65535
    radiance = code / 12.92
    high = code > .04045
    radiance[high] = ((code[high] + .055) / 1.055)**2.4
    return radiance


def support_mask(positive):
    """Return a native image interior with no invalid pixel in a 3x3 stencil."""
    out = np.zeros_like(positive)
    patches = np.lib.stride_tricks.sliding_window_view(positive, (3, 3))
    out[1:-1, 1:-1] = patches.all(axis=(2, 3))
    return out


def footprint_test(mask, xy):
    valid = ((xy >= 0) & (xy < mask.shape[::-1])).all(axis=1)
    ix, iy = xy[valid].T
    valid[valid] = mask[iy, ix]
    return valid


def geometry_replay(ref_depth, qry_depth, k_ref, p_ref, k_qry, p_qry, shifts):
    """Compose relative extrinsics, then evaluate visibility in camera axes."""
    r_qr = p_qry[:3, :3].T @ p_ref[:3, :3]
    t_qr = p_qry[:3, :3].T @ (p_ref[:3, 3] - p_qry[:3, 3])
    ref_valid = np.isfinite(ref_depth) & (ref_depth > 0)
    qry_valid = np.isfinite(qry_depth) & (qry_depth > 0)
    row, col = np.where(ref_valid)
    ref_xy = np.column_stack([col, row])
    axial = ref_depth[row, col]
    rays_ref = np.linalg.solve(k_ref, np.vstack([col, row, np.ones(len(row))]))
    in_query = r_qr @ (rays_ref * axial[None, :]) + t_qr[:, None]
    xy_h = k_qry @ in_query
    with np.errstate(divide="ignore", invalid="ignore"):
        target = (xy_h[:2] / xy_h[2]).T
    projection = ((in_query[2] > 0) & np.isfinite(target).all(axis=1)
                  & ((target >= 0) & (target < np.array(qry_depth.shape[::-1]) - 1)).all(axis=1))
    counts = dict(reference_valid=len(row), query_bilinear_footprint=int(projection.sum()))
    ref_xy, axial, target, pred = (v[projection] for v in (ref_xy, axial, target, in_query[2]))
    ix, iy = target.astype(int).T
    fx, fy = (target - np.column_stack([ix, iy])).T
    z00, z10 = qry_depth[iy, ix], qry_depth[iy, ix+1]
    z01, z11 = qry_depth[iy+1, ix], qry_depth[iy+1, ix+1]
    z_max = np.maximum.reduce([z00, z01, z10, z11])
    z_min = np.minimum.reduce([z00, z01, z10, z11])
    stencil = (np.isfinite(z_max) & (z_min > 0) & ((z_max - z_min) <= .10 + .01*z_min))
    counts["query_supported_smooth_depth_stencil"] = int(stencil.sum())
    ref_xy, axial, target, pred, fx, fy, z00, z01, z10, z11 = (
        v[stencil] for v in (ref_xy, axial, target, pred, fx, fy, z00, z01, z10, z11))
    top = (1-fx)/z00 + fx/z10
    bottom = (1-fx)/z01 + fx/z11
    observed = 1 / ((1-fy)*top + fy*bottom)
    delta = observed - pred
    tolerance = .05 + .001*pred
    counts["inconsistent_nearer_query_surface"] = int((delta < -tolerance).sum())
    consistent = np.abs(delta) <= tolerance
    counts["query_depth_agreement"] = int(consistent.sum())
    ref_xy, axial, target, observed = (v[consistent] for v in (ref_xy, axial, target, observed))
    rays_qry = np.linalg.solve(k_qry, np.vstack([target.T, np.ones(len(target))]))
    back_ref = r_qr.T @ (rays_qry * observed[None, :] - t_qr[:, None])
    back_h = k_ref @ back_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        back_xy = (back_h[:2] / back_h[2]).T
    dx, dy = (back_xy - ref_xy).T
    close = ((back_ref[2] > 0) & (dx*dx + dy*dy <= .25)
             & (np.abs(back_ref[2]-axial) <= .05+.001*axial))
    ref_xy, target = (v[close] for v in (ref_xy, target))
    counts["reciprocal_cov_visible"] = len(ref_xy)
    query_xy = (target + .5).astype(int)
    ref_support, query_support = support_mask(ref_valid), support_mask(qry_valid)
    common = footprint_test(ref_support, ref_xy)
    for shift in shifts:
        common &= footprint_test(query_support, query_xy + shift)
    ref_xy, target, query_xy = (v[common] for v in (ref_xy, target, query_xy))
    counts["common_support_before_unique"] = len(query_xy)
    # A stable distance sort followed by first occurrence independently selects
    # the winning reference sample. Output order does not affect metrics.
    dx, dy = (target - query_xy).T
    by_distance = np.argsort(dx*dx + dy*dy, kind="stable")
    keys = np.ravel_multi_index(query_xy[by_distance].T[::-1], qry_depth.shape)
    _, first = np.unique(keys, return_index=True)
    chosen = by_distance[first]
    counts["common_support_unique"] = len(chosen)
    return ref_xy[chosen], query_xy[chosen], counts


def load_geometry(item):
    with data(item["scoring_geometry"]) as scoring:
        geometry = {key: scoring[key] for key in scoring.files}
    for role, view in item["views"].items():
        with data(view["archive_geometry"]) as archive:
            for old, key in (("axial_depth", "depth"), ("intrinsics", "intrinsics"), ("cam2world", "cam2world")):
                if not np.array_equal(archive[old], geometry[f"{role}_{key}"], equal_nan=True):
                    raise ValueError("Original and scoring geometry differ")
    return tuple(geometry[key] for key in ("reference_depth", "query_depth", "reference_intrinsics",
                                           "reference_cam2world", "query_intrinsics", "query_cam2world"))


def ranked_tail(luminance, depth, percentile):
    valid = np.isfinite(depth) & (depth > 0)
    ordered = np.sort(luminance[valid])

    def at(percent):
        index = (len(ordered)-1)*percent/100
        base = math.floor(index)
        return float(ordered[base] + (index-base)*(ordered[math.ceil(index)] - ordered[base]))

    baseline, top = at(1), at(percentile)
    cut = max(top, baseline + 3/(255*12.92*math.sqrt(12)))
    high = valid & (luminance > cut)
    return high, dict(linear_cut=cut, p01_linear=baseline, ranked_cut_linear=top,
                      bright_valid_pixels=int(high.sum()), valid_pixels=len(ordered))


def dilation(mask):
    padded = np.pad(mask, 1)
    return np.lib.stride_tricks.sliding_window_view(padded, (3, 3)).any(axis=(2, 3))


def metrics(a, b):
    n = len(a)
    code = 2*a.astype(np.uint8) + b
    n00, n01, n10, n11 = map(int, np.bincount(code, minlength=4))
    l, r = n10+n11, n01+n11
    den = math.sqrt(l*r*(n-l)*(n-r))
    return dict(common_pixels=n, reference_bright=l, query_bright=r, both_bright=n11,
                jaccard=n11/(n01+n10+n11) if n01+n10+n11 else None,
                phi=(n11*n00-n10*n01)/den if den else None,
                lift=(n11/l)/(r/n) if l and r else None)


def compare(expected, got, scope):
    if expected.keys() != got.keys():
        raise ValueError("Report keys differ for " + scope)
    for key, value in expected.items():
        observed = got[key]
        if isinstance(value, float) and isinstance(observed, float):
            matches = math.isclose(value, observed, rel_tol=1e-13, abs_tol=1e-14)
        else:
            matches = value == observed
        if not matches:
            raise ValueError(f"{scope} {key}: expected {value}, reproduced {observed}")


def check_case(item, row, shifts):
    if (row["case_id"], row["purpose"]) != (item["case_id"], item["purpose"]):
        raise ValueError("Case identity or order differs")
    geometry = load_geometry(item)
    p, q, counts = geometry_replay(*geometry, shifts)
    compare(row["geometry"], counts, row["case_id"] + " geometry")
    photos = [image(item["views"][role]["image"]) for role in ("reference", "query")]
    for percentile, entry in row["tails"].items():
        a, a_info = ranked_tail(photos[0], geometry[0], float(percentile))
        b, b_info = ranked_tail(photos[1], geometry[1], float(percentile))
        compare(entry["reference"], a_info, row["case_id"] + " reference ranking")
        compare(entry["query"], b_info, row["case_id"] + " query ranking")
        for radius, scores in entry["dilation_pixels"].items():
            if int(radius):
                a, b = dilation(a), dilation(b)
            left = a[p[:, 1], p[:, 0]]
            for (dx, dy), cell in zip(shifts, scores, strict=True):
                right = b[q[:, 1]+dy, q[:, 0]+dx]
                observed = dict(query_offset_xy=[dx, dy], **metrics(left, right))
                if percentile == "99" and radius == "0" and dx == dy == 0:
                    tiles = [[0]*4 for _ in range(4)]
                    for x, y in p[left & right]:
                        tiles[y//128][x//128] += 1
                    observed["both_bright_reference_tiles_yx"] = tiles
                compare(cell, observed, f"{row['case_id']} P{percentile}, radius {radius}, shift {(dx, dy)}")
    primary = row["tails"]["99"]["dilation_pixels"]["0"]
    if any(x["jaccard"] is None or x["phi"] is None for x in primary):
        status = "indeterminate"
    elif all(primary[0]["jaccard"] > x["jaccard"] and primary[0]["phi"] > x["phi"] for x in primary[1:]):
        status = "above_every_shift_on_both_metrics"
    else:
        status = "does_not_exceed_every_shift_on_both_metrics"
    if status != row["primary_shape_result"]:
        raise ValueError("Primary interpretation differs")


def main(protocol_path, result_path, output_path):
    cv2.setNumThreads(2)
    raw = Path(result_path).read_bytes()
    report = json.loads(raw)
    protocol = json.loads(read(dict(path=protocol_path, sha256=report["protocol_sha256"])))
    if report["calibration_or_final_access"] or any(
            report[key] != 0 for key in ("matcher_calls", "optimizer_updates", "renderer_calls")):
        raise ValueError("Replay is limited to fixed scoring inputs")
    shifts = protocol["query_offsets_pixels"]
    for item, row in zip(protocol["items"], report["rows"], strict=True):
        check_case(item, row, shifts)
    cases = len(report["rows"])
    audit = dict(schema="solar_tail_morphology_replay_v109", passed=True,
                 cases_replayed=cases, geometry_tables_replayed=cases,
                 comparison_cells_replayed=cases*3*2*len(shifts),
                 method="Separate relative-camera projection, inverse-depth interpolation, common-support reduction, rank interpolation and contingency arithmetic.",
                 protocol_sha256=report["protocol_sha256"], result_sha256=sha256(raw).hexdigest(),
                 replay_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 calibration_or_final_access=False)
    with Path(output_path).open("x") as file:
        json.dump(audit, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
