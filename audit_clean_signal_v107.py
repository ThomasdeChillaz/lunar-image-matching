"""Retrospective photometry only on fixed completed native v106 exposures."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess


BASE = "runs/solar-clean-radiance-v106"
DISPATCH = "reports/solar-clean-radiance-dispatch-v106.json"
COHORT = "reports/solar-clean-rerender-cohort-v106.json"
PUBLIC = "a435201"
PINS = {
    DISPATCH: "8f65614e740cf8db7d51000deb88c73a147e0b194b12fc35d18b6870634d4e75",
    COHORT: "7510450dfdbaf23f1e5973c5b9adae026834c92349de2d1158f95886acf5b80b",
    BASE + "/COMPLETE.json": "4f2a0f39b4f79f7f4bad848a75548c82ed743580a2cead5a8130a675243cd6da",
    BASE + "/watchdog.json": "ab76b6ca52119c97884d70dc6b966be79d609838eb7573b6f3aed719355094e4",
    BASE + "/summary.json": "fbf53a4d22e3d4cb2962ccc1609fee32ef6a336ab13d523e248dc6451b624030",
    BASE + "/independent_replay.json": "40aab6a459cf733c161dfde5c0769c014f0258c95ecedaeb93d3934ae8a3e518",
    BASE + "/render/COMPLETE.json": "1e3e36c32e552326ba359dca96544e878905086acac9a7c1019daa66b3620beb",
}
# Last field is the pre-existing v101 selected template, kept separate when
# different. No pixel or correspondence inspection selected the v106 inputs.
POSITIVES = (
    ("u013-l00-00-positive", 13, 0, 0),
    ("u025-l00-00-positive", 25, 0, 1),
    ("u027-l08-01-positive", 27, 1, 1),
    ("u061-l00-17-positive", 61, 17, 17),
)
VARIANTS = {"default_rgb8": 8, "clean_rgb8": 8, "clean_rgb16": 16}
REGIONS = {"native": (0, 512), "central_416": (48, 464)}
PCTS = (0, 1, 5, 50, 95, 99.5, 100)
NOISE_SIGMA = 1. / (255. * 12.92 * math.sqrt(12.))


def checked(path, checksum):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError("Sealed asset differs: " + str(path))
    return raw


def exact(path):
    return json.loads(checked(path, PINS[path]))


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def freeze():
    p, c, m, watchdog, summary, replay, render = map(exact, (
        DISPATCH, COHORT, BASE + "/COMPLETE.json", BASE + "/watchdog.json", BASE + "/summary.json",
        BASE + "/independent_replay.json", BASE + "/render/COMPLETE.json"))
    for path in (DISPATCH, COHORT):
        if subprocess.check_output(["git", "show", f"{PUBLIC}:{path}"], timeout=5) != Path(path).read_bytes():
            raise ValueError("Original pre-inference publication differs")
    if (not m["external_watchdog_passed"] or not watchdog["complete"] or watchdog["child_returncode"] != 0
            or m["summary_sha256"] != PINS[BASE + "/summary.json"]
            or m["watchdog_sha256"] != PINS[BASE + "/watchdog.json"]
            or summary["independent_replay_sha256"] != PINS[BASE + "/independent_replay.json"]
            or summary["render_receipt_sha256"] != PINS[BASE + "/render/COMPLETE.json"]
            or replay["passed"] is not True or replay["replayed_rows"] != 32 or not render["complete"]
            or p["cohort_sha256"] != PINS[COHORT] or render["cohort_sha256"] != PINS[COHORT]
            or p["calibration_or_final_access"] or c["calibration_or_final_access"] or render["calibration_or_final_access"]
            or any(x["split"] != "model_selection" for x in (p, c, render))):
        raise ValueError("Exact complete model-selection-only source study required")
    by_case = {r["case_id"]: r for r in c["cases"]}
    by_key = {r["key"]: r for r in render["rows"]}
    plan = {r["key"]: r for r in c["renders"]}
    items = []
    for case_id, unit, physical, historical in POSITIVES:
        case = by_case[case_id]
        if not case["positive"] or case["unit_index"] != unit or case["query_unit_index"] != unit:
            raise ValueError("Fixed same-terrain positive required")
        query = case["query_key"]
        refs = case["reference_keys"]
        choices = [("query", query, physical), ("physical_query_sun_reference", refs[physical], physical)]
        if historical != physical:
            choices.append(("separate_v101_selected_reference", refs[historical], historical))
        for purpose, key, light in choices:
            row, planned = by_key[key], plan[key]
            if (row["light_index"] != light or row["unit_index"] != unit or
                    row["role"] != ("query" if purpose == "query" else "reference")
                    or not row["legacy_rgb_pixels_equal"] or row["clean_dither"] != 0.0
                    or not row["geometry"]["exact_camera_depth_and_rays"]
                    or any(row[v]["path"] != planned["output_" + v] or row[v]["bit_depth"] != b for v, b in VARIANTS.items())):
                raise ValueError("Native matched render differs from fixed role")
            items.append(dict(case_id=case_id, purpose=purpose, key=key, light_index=light,
                              historical_selected_light_index=historical,
                              assets={v: row[v] for v in VARIANTS}))
    if len(items) != 9 or len({r["key"] for r in items}) != 9:
        raise ValueError("Require exactly nine predeclared exposures")
    return dict(
        scope="Retrospective optical pixel statistics. No matching, pose, training, final/calibration imagery, or rendered replacements.",
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), evidence=PINS, items=items,
        crop_bounds=REGIONS, quantiles=PCTS, block_sizes=[8, 32], linear_noise_sigma=NOISE_SIGMA,
        quantile_rule="NumPy method linear, interpolating empirical order statistics",
        grayscale="BGR decoded unchanged. Weighted display codes 0.114 B + 0.587 G + 0.299 R before inverse sRGB.",
        floor="Subtract each crop's own 1st percentile scene-linear luminance and clamp at zero. The floor is diagnostic, not calibrated black.",
        texture="Adjacent raw display differences, robust block-mean span and p95 within-block RMS after per-block constant removal.",
        hidden="Count clean16 intensity variation inside modal clean8 code and clean16 edge differences where clean8 endpoints tie.",
        truth_usage="Same-light reference selected by previously known physical query-light index for analysis only. No correspondence alignment.",
    )


def measure(frozen):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)

    def decode(asset):
        raw = checked(asset["path"], asset["sha256"])
        if (raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                or raw[16:24] != b"\x00\x00\x02\x00"*2 or raw[24:26] != bytes((asset["bit_depth"], 2))):
            raise ValueError("Expected native RGB PNG of the sealed bit depth")
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.shape != (512, 512, 3) or bgr.dtype != (np.uint16 if asset["bit_depth"] == 16 else np.uint8):
            raise ValueError("Native unchanged decoding required")
        image = (0.114*bgr[..., 0] + 0.587*bgr[..., 1] + 0.299*bgr[..., 2]) / (257 if asset["bit_depth"] == 16 else 1)
        return image

    def quantiles(a):
        return dict(zip(map(str, PCTS), map(float, np.percentile(a, PCTS)), strict=True))

    def stats(image):
        encoded = image / 255.
        linear = np.where(encoded <= .04045, encoded / 12.92, ((encoded + .055)/1.055)**2.4)
        floor = float(np.percentile(linear, 1))
        clean = np.maximum(linear - floor, 0.)
        grad = np.concatenate((np.abs(np.diff(image, axis=0)).ravel(), np.abs(np.diff(image, axis=1)).ravel()))
        d = dict(code8_equivalent=quantiles(image), linear=quantiles(linear), black_floor_subtracted_linear=quantiles(clean),
                 black_floor_linear=floor, black_floor_clipped_pixels=int((linear < floor).sum()),
                 adjacent_gradient_code8=quantiles(grad), unique_luminance_values=len(np.unique(image)),
                 floor_subtracted_p995_over_noise_sigma=float(np.percentile(clean, 99.5))/NOISE_SIGMA,
                 scales={})
        for size in (8, 32):
            cells = clean.reshape(clean.shape[0]//size, size, clean.shape[1]//size, size)
            mean = cells.mean(axis=(1, 3))
            rms = cells.std(axis=(1, 3))
            span = float(np.percentile(mean, 95)-np.percentile(mean, 5))
            d["scales"][str(size)] = dict(block_mean_linear=quantiles(mean), block_rms_linear=quantiles(rms),
                                          p95_p05_block_mean_span_over_noise_sigma=span/NOISE_SIGMA)
        return d

    def hidden(low, high):
        values, counts = np.unique(low, return_counts=True)
        modal = values[counts.argmax()]
        tied = np.concatenate(((np.diff(low, axis=0)==0).ravel(), (np.diff(low, axis=1)==0).ravel()))
        high_edges = np.concatenate((np.abs(np.diff(high, axis=0)).ravel(), np.abs(np.diff(high, axis=1)).ravel()))
        return dict(modal_code8=float(modal), modal_pixels=int((low == modal).sum()),
                    clean16_code8_equivalent_within_modal_code8=quantiles(high[low == modal]),
                    clean8_tied_edges=int(tied.sum()), tied_with_nonzero_clean16=int((tied & (high_edges>1e-10)).sum()),
                    tied_with_clean16_above_one_16bit_code=int((tied & (high_edges>1./257. + 1e-10)).sum()))

    rows = []
    for item in frozen["items"]:
        images = {v: decode(asset) for v, asset in item["assets"].items()}
        regions = {}
        for region, (start, end) in REGIONS.items():
            crops = {v: image[start:end,start:end] for v, image in images.items()}
            regions[region] = {v: stats(im) for v, im in crops.items()}
            regions[region]["low_bit_loss"] = hidden(crops["clean_rgb8"], crops["clean_rgb16"])
        rows.append(dict(**item, regions=regions))
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "measure"))
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--protocol", type=Path)
    cli.add_argument("--protocol-sha256")
    args = cli.parse_args()
    derived = freeze()
    if args.action == "freeze":
        write(args.output, derived)
    else:
        p = json.loads(checked(args.protocol, args.protocol_sha256))
        if p != json.loads(json.dumps(derived)):
            raise ValueError("Exact metadata protocol before pixel decoding required")
        rows = measure(p)
        write(args.output, dict(passed=True, protocol_sha256=args.protocol_sha256, image_decodes=27, rows=rows))


if __name__ == "__main__":
    main()
