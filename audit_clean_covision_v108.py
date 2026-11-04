"""Measure where clean native exposures have contrast at independently projected anchors."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess

from audit_clean_signal_v107 import BASE, COHORT, DISPATCH, NOISE_SIGMA, PINS, checked, write

PUBLIC = "dba59e49c974e2feca40c4c7b25e4e2d12e042a8"
PRIOR = {
    "reports/solar-clean-signal-protocol-v107.json": "6cb071a43be2cc50ee5d3c70c82c111847fe03e55dd5a472c5716b04945aa3cd",
    "reports/solar_clean_signal_v107.json": "7f507f2e68a4912faa5d5679088d6adc68bb646983b3b36a8322c105b27de246",
}
CASES = (
    ("u061-l00-17-positive", 61, 17, "sparse_tail_primary"),
    ("u013-l00-00-positive", 13, 0, "bright_control"),
    ("u025-l00-00-positive", 25, 0, "near_uniform_control"),
    ("u027-l08-01-positive", 27, 1, "near_uniform_control"),
)
REPO_SOURCES = ("audit_clean_signal_v107.py", "north_development_plan_v35.py")
RADIUS = 16
OFFSETS = ((0, 0), (-32, 0), (32, 0), (0, -32), (0, 32))
LOW_PCT = 5
UPPER_PCTS = (95, 99)
MULTIPLES = (1, 3, 6)
SHAPE = (512, 512)
SIZE = SHAPE[0]
TILE_SIZE = 128


def digest(raw):
    return sha256(raw).hexdigest()


def freeze():
    evidence = {**PINS, **PRIOR}
    for path, checksum in evidence.items():
        checked(path, checksum)
    for path in (*PRIOR, *REPO_SOURCES, COHORT, DISPATCH):
        if subprocess.check_output(["git", "show", f"{PUBLIC}:{path}"], timeout=5) != Path(path).read_bytes():
            raise ValueError("Public source differs: " + path)

    def load(path):
        return json.loads(Path(path).read_bytes())

    dispatch, cohort, render = map(load, (DISPATCH, COHORT, BASE + "/render/COMPLETE.json"))
    if (not render["complete"] or render["cohort_sha256"] != PINS[COHORT]
            or dispatch["cohort_sha256"] != PINS[COHORT]
            or dispatch["manifest"] != cohort["case_manifest"]):
        raise ValueError("Sealed render and original manifest must share a cohort")
    for field in ("case_manifest", "geometry_audit"):
        path, checksum = cohort[field], cohort[field + "_sha256"]
        evidence[path] = checksum
        checked(path, checksum)
    manifest = load(cohort["case_manifest"])
    audit = load(cohort["geometry_audit"])
    if (manifest["geometry_audit_sha256"] != cohort["geometry_audit_sha256"]
            or not audit["passed"]
            or any(m["split"] != "model_selection" or m["calibration_or_final_access"]
                   for m in (dispatch, cohort, manifest, render))):
        raise ValueError("Audited and sealed model-selection evidence required")
    by_case = {r["case_id"]: r for r in cohort["cases"]}
    source_cases = {r["case_id"]: r for r in manifest["cases"]}
    by_key = {r["key"]: r for r in render["rows"]}
    planned = {r["key"]: r for r in cohort["renders"]}
    items = []
    for case_id, unit, light, purpose in CASES:
        case, source = by_case[case_id], source_cases[case_id]
        for row in (case, source):
            if not row["positive"] or (row["unit_index"], row["query_unit_index"]) != (unit, unit):
                raise ValueError("Exact same-terrain positive required")
        if (source["split"] != "model_selection" or source["component"] != source["query_component"]
                or source["light_b"] != light):
            raise ValueError("Unexpected source terrain, split or query light")
        labels_path = source["scoring_labels"]
        labels_checksum = source["input_sha256"][labels_path]
        if (source["scoring_inputs"]["labels"] != labels_path
                or dispatch["input_sha256"].get(labels_path) != labels_checksum):
            raise ValueError("Independent scoring label seal mismatch")
        item = dict(case_id=case_id, unit=unit, component=source["component"], light_index=light,
                    purpose=purpose, scoring_labels=dict(path=labels_path, sha256=labels_checksum), exposures={})
        for role, key in (("reference", case["reference_keys"][light]), ("query", case["query_key"])):
            row = by_key[key]
            asset = row["clean_rgb16"]
            if (row["role"] != role or row["unit_index"] != unit or row["light_index"] != light
                    or row["clean_dither"] != 0 or not row["geometry"]["exact_camera_depth_and_rays"]
                    or asset["bit_depth"] != 16 or asset["path"] != planned[key]["output_clean_rgb16"]
                    or not row["legacy_rgb_pixels_equal"]):
                raise ValueError("Unexpected image encoding, geometry, view or lighting")
            item["exposures"][role] = asset
        if by_key[case["query_key"]]["archive_path"] != source["query"]:
            raise ValueError("Query image lineage mismatch")
        items.append(item)
    if len({a["path"] for i in items for a in i["exposures"].values()}) != 8:
        raise ValueError("Exactly eight prior clean RGB16 images required")
    return dict(
        schema="solar_clean_covision_protocol_v108",
        public_predecessor_commit=PUBLIC,
        source_sha256=digest(Path(__file__).read_bytes()),
        repo_source_sha256={path: digest(Path(path).read_bytes()) for path in REPO_SOURCES},
        evidence=evidence,
        items=items,
        native_shape=list(SHAPE),
        anchor_grid="Historical proposal_grid_native, 26x26 with step 16, phase 49.5 native x/y coordinates.",
        geometry="Sealed exact fixed-anchor reference/query projection and independent visibility. Only patch centres have geometric labels.",
        nearest_pixel="Integer pixel centres, ties toward the larger coordinate using floor(x+0.5).",
        patch_radius_pixels=RADIUS,
        thresholds_sigma=list(MULTIPLES),
        linear_noise_sigma=NOISE_SIGMA,
        grayscale="Identical to v107. Weighted sRGB BGR codes before inverse sRGB transfer.",
        primary_contrast_percentiles=[LOW_PCT, UPPER_PCTS[0]],
        sparse_tail_sensitivity_percentiles=[LOW_PCT, UPPER_PCTS[1]],
        quantile_rule="NumPy linear empirical quantiles. No rescaling per view or spatial filtering.",
        primary_threshold_sigma=3,
        spatial_tile_size_pixels=TILE_SIZE,
        query_offsets_pixels=[list(shift) for shift in OFFSETS],
        denominators="Aligned full-window support and paired common support where all shifts fit. Positive visibility applies to the aligned centre only. Null offsets never wrap.",
        scope="Retrospective contrast co-location only on fixed v106 native model-selection cases. No matching, pose fitting, rendering, training, calibration or final inputs.",
        inference=False,
        calibration_or_final_access=False,
    )


def measure(frozen):
    import cv2
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from north_development_plan_v35 import proposal_grid_native

    cv2.setNumThreads(2)

    def decode(asset):
        raw = checked(asset["path"], asset["sha256"])
        if (raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                or raw[16:24] != b"\x00\x00\x02\x00"*2 or raw[24:26] != b"\x10\x02"):
            raise ValueError("Expected a native 16-bit RGB PNG")
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.shape != (*SHAPE, 3) or bgr.dtype != np.uint16:
            raise ValueError("Native unchanged uint16 decoding required")
        encoded = (0.114*bgr[..., 0] + 0.587*bgr[..., 1] + 0.299*bgr[..., 2]) / 65535.
        return np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055)/1.055)**2.4)

    def describe(image, xy):
        width = 2*RADIUS + 1
        windows = sliding_window_view(image, (width, width))
        x, y = xy.T - RADIUS
        pixels = windows[y, x].reshape(len(xy), width*width)
        quantiles = np.percentile(pixels, [LOW_PCT, *UPPER_PCTS], axis=1, method="linear")
        return (quantiles[1:] - quantiles[:1]) / NOISE_SIGMA

    def supported(xy):
        return ((xy >= RADIUS) & (xy < SIZE - RADIUS)).all(axis=1)

    def number(a):
        return int(np.count_nonzero(a))

    def overlap(a, b, a_xy, b_xy, include_spatial):
        report = {}
        for i, pct in enumerate(UPPER_PCTS):
            pairs = {}
            for threshold in MULTIPLES:
                l, r = a[i] >= threshold, b[i] >= threshold
                both = l & r
                union = l | r
                item = dict(reference=number(l), query=number(r), both=number(both),
                            either=number(union), neither=number(~union))
                if include_spatial and pct == UPPER_PCTS[0] and threshold == 3:
                    for role, coords in (("reference", a_xy), ("query", b_xy)):
                        tiles = coords[both] // TILE_SIZE
                        tiles_per_axis = SIZE // TILE_SIZE
                        item[role + "_tiles_yx"] = np.bincount(
                            tiles[:, 1]*tiles_per_axis + tiles[:, 0], minlength=tiles_per_axis**2,
                        ).reshape(tiles_per_axis, tiles_per_axis).tolist()
                pairs[str(threshold)] = item
            report[f"p{pct}_minus_p{LOW_PCT}"] = pairs
        return report

    rows = []
    anchors = proposal_grid_native()
    if anchors.shape != (676, 2) or not np.array_equal(anchors[0], [49.5, 49.5]):
        raise ValueError("Historical grid differs")
    for item in frozen["items"]:
        seal = item["scoring_labels"]
        with np.load(BytesIO(checked(seal["path"], seal["sha256"])), allow_pickle=False) as truth:
            p, q, visible = (truth[k] for k in ("reference_xy", "query_xy", "visible"))
        if (p.shape != anchors.shape or q.shape != p.shape or not np.array_equal(p, anchors)
                or visible.dtype != bool or visible.shape != (len(p),) or not np.isfinite(q[visible]).all()):
            raise ValueError("Exactly aligned independent visibility labels required")
        p = np.floor(p[visible] + .5).astype(np.int32)
        q = np.floor(q[visible] + .5).astype(np.int32)
        aligned = supported(p) & supported(q)
        common = np.logical_and.reduce([aligned, *(supported(q + offset) for offset in OFFSETS)])
        reference = decode(item["exposures"]["reference"])
        query = decode(item["exposures"]["query"])
        a, b = describe(reference, p[aligned]), describe(query, q[aligned])
        common_reference = a[:, common[aligned]]
        rows.append(dict(case_id=item["case_id"], purpose=item["purpose"],
                         grid_anchors=len(anchors), exact_visible_centres=number(visible),
                         aligned_supported=number(aligned), common_supported=number(common),
                         aligned=overlap(a, b, p[aligned], q[aligned], True),
                         common=[
                             dict(query_offset_xy=list(offset),
                                  overlap=overlap(common_reference, describe(query, q[common] + offset),
                                                  p[common], q[common] + offset, False))
                             for offset in OFFSETS],
                         contrast_sigma_quantiles={
                             "quantiles": [5, 50, 95, 99],
                             **{role: np.percentile(values, [5, 50, 95, 99], axis=1).tolist()
                                for role, values in (("reference", a), ("query", b))}}))
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "measure"))
    cli.add_argument("--protocol", type=Path, required=True)
    cli.add_argument("--protocol-sha256")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    current = freeze()
    if args.action == "freeze":
        write(args.protocol, current)
        print(digest(args.protocol.read_bytes()))
        return
    if args.protocol_sha256 is None or args.output is None:
        cli.error("measurement requires the pre-recorded protocol SHA and a new output")
    frozen = json.loads(checked(args.protocol, args.protocol_sha256))
    if frozen != current:
        raise ValueError("Current source and assets do not match the fixed protocol")
    write(args.output, dict(schema="solar_clean_covision_v108", protocol_sha256=args.protocol_sha256,
                            rows=measure(frozen), matcher_calls=0, optimizer_updates=0,
                            renderer_calls=0, calibration_or_final_access=False))


if __name__ == "__main__":
    main()
