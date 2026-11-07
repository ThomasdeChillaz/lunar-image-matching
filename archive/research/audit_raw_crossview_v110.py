"""Measure scene-linear cross-view block repeatability on fixed native projections."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess


PUBLIC = "416233c8ce5c11841ed59d545342906c8565a6f2"
STUDY = "reports/solar-raw-acquisition-v109-study.json"
RECEIPT = "runs/solar-raw-acquisition-v109-study/COMPLETE.json"
MORPH = "reports/solar-clean-tail-morphology-protocol-v109.json"
MORPH_RESULT = "reports/solar_tail_morphology_v109.json"
MORPH_REPLAY = "reports/solar_tail_morphology_replay_v109.json"
PINS = {
    STUDY: "d711eb8429a8de80d74b16edc20bd49eae011801d32fcf2cd7410d78db28bb60",
    RECEIPT: "4ce6d2b7364722205f1b067bbb4d6fb4a29c3e3ae224350a92aca0f4c5a31067",
    MORPH: "0388eb9d352afb75c568b564a2fdddb04a24a77ae4ee4eeda0210dc7f0608d91",
    MORPH_RESULT: "91cc4173a5db0f5aa7fbea41c6e86647721a879fb8837344f37e9599fb745569",
    MORPH_REPLAY: "27b13f60066bf86675b29ac7bd87c4e2bb2132c1b98749e3c35c4d1c32da7506",
}
PREDECESSOR_REPLAY_SHA = "51bee959b252e962889346612a7b360422deaac974c2082e5145fae7d21e1426"
PREDECESSOR_REPLAY_SOURCE_SHA = "2670cdb5adc69b6f9e7c224d955ca7be8d2fe705c845a4ab2ce8bd7f17fb4372"
SOURCES = ("audit_clean_tail_morphology_v109.py", "replay_clean_tail_morphology_v109.py",
           "lunar_geometry_v2.py", "lunar_geometry_v5.py")
OFFSETS = ((0, 0), (-32, 0), (32, 0), (0, -32), (0, 32))
SCALES = (32, 16, 1)
NATIVE = 512
EPS = 1e-9


def digest(raw):
    return sha256(raw).hexdigest()


def checked(path, checksum):
    raw = Path(path).read_bytes()
    if digest(raw) != checksum:
        raise ValueError("Changed sealed input: " + str(path))
    return raw


def write(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def freeze(replay_path):
    study, receipt, morph, earlier, geometry_replay = [
        json.loads(checked(path, PINS[path])) for path in (STUDY, RECEIPT, MORPH, MORPH_RESULT, MORPH_REPLAY)
    ]
    replay = json.loads(checked(replay_path, PREDECESSOR_REPLAY_SHA))
    repo_sources = {}
    for path in (STUDY, MORPH, MORPH_RESULT, MORPH_REPLAY, *SOURCES):
        current = Path(path).read_bytes()
        if subprocess.check_output(["git", "show", f"{PUBLIC}:{path}"], timeout=5) != current:
            raise ValueError("The public v109 source differs: " + path)
        if path in SOURCES:
            repo_sources[path] = digest(current)
    checked("replay_raw_acquisition_v109.py", PREDECESSOR_REPLAY_SOURCE_SHA)
    repo_sources["replay_raw_acquisition_v109.py"] = PREDECESSOR_REPLAY_SOURCE_SHA
    if (not receipt["complete"] or not replay["passed"] or not geometry_replay["passed"]
            or study["split"] != "model_selection" or study["phase"] != receipt["phase"]
            or study["phase"] != "study" or study["native_shape"] != [NATIVE, NATIVE]
            or not study["fixed_black_space_spp"] or not study["extended_256_spp"] or study["denoising"]
            or study["exr_float_bits"] != 32 or receipt["protocol_sha256"] != PINS[STUDY]
            or receipt["publication_commit"] != PUBLIC or replay["study"]["protocol_sha256"] != PINS[STUDY]
            or replay["study"]["receipt_sha256"] != PINS[RECEIPT]
            or replay["source_sha256"] != PREDECESSOR_REPLAY_SOURCE_SHA
            or len(replay["study"]["artifacts"]) != len(receipt["rows"])
            or earlier["protocol_sha256"] != PINS[MORPH] or geometry_replay["result_sha256"] != PINS[MORPH_RESULT]
            or morph["query_offsets_pixels"] != [list(offset) for offset in OFFSETS]
            or any(doc["calibration_or_final_access"] for doc in (study, receipt, morph, earlier, geometry_replay, replay))):
        raise ValueError("Need the independently replayed, complete model-selection-only v109 study")
    fields = ("key", "scene", "samples", "replicate", "seed")
    actual = {tuple(r[k] for k in fields): r for r in receipt["rows"]}
    plan = {tuple(r[k] for k in fields) for r in study["render_jobs"]}
    replayed = {tuple(r["job"][k] for k in fields): r["exr_sha256"] for r in replay["study"]["artifacts"]}
    if (len(actual) != len(receipt["rows"]) or plan != actual.keys() or plan != replayed.keys()
            or any({name: a["sha256"] for name, a in r["exr"].items()} != replayed[key]
                   for key, r in actual.items())):
        raise ValueError("Sealed render job identities differ")
    controls = {"u013": 64, "u025": 64, "u027": 64, "u061": 256}
    camera_identity = {entry["key"]: entry for entry in study["camera_mesh_identity"]}
    items = []
    for item, measured in zip(morph["items"], earlier["rows"], strict=True):
        prefix = item["case_id"].split("-")[0]
        if measured["case_id"] != item["case_id"] or prefix not in controls:
            raise ValueError("Unexpected v109 development case")
        spp = controls.pop(prefix)
        views = {}
        for role, view in item["views"].items():
            key = f"{prefix}/{role}/light-{item['light_index']}"
            identity = camera_identity[key]
            expected = view["archive_geometry"]
            if (identity["archive_geometry_path"] != expected["path"]
                    or identity["archive_geometry_sha256"] != expected["sha256"]):
                raise ValueError("Raw view camera/depth provenance differs from scoring geometry")
            choices = [r for r in actual.values() if
                       (r["key"], r["scene"], r["samples"]) == (key, "black_space", spp)]
            choices.sort(key=lambda r: r["replicate"])
            if [r["replicate"] for r in choices] != [0, 1] or any(
                not r["geometry"]["exact_camera_depth_and_rays"] or r["geometry"]["native_rays"] != NATIVE*NATIVE
                or r["adaptive"] or r["exact_legacy_rgb8"] is not None for r in choices
            ):
                raise ValueError("Require two exact black-space raw view replicates")
            assets = []
            for r in choices:
                asset = r["exr"]["combined"]
                path = f"{study['output']}/{key}/black_space/{spp}spp-r{r['replicate']}/combined_0001.exr"
                if asset["path"] != path:
                    raise ValueError("Unexpected combined EXR role or identity")
                assets.append(dict(**asset, seed=r["seed"], replicate=r["replicate"]))
            views[role] = dict(archive_geometry=expected, combined=assets)
        items.append(dict(case_id=item["case_id"], purpose=item["purpose"], light_index=item["light_index"],
                          samples_per_pixel=spp, scoring_geometry=item["scoring_geometry"], views=views,
                          expected_geometry=measured["geometry"]))
    seeds = {a["seed"] for i in items for v in i["views"].values() for a in v["combined"]}
    if controls or len(seeds) != 16:
        raise ValueError("Exactly sixteen disjoint predetermined camera-render seeds required")
    return dict(
        schema="solar_raw_crossview_protocol_v110", public_predecessor_commit=PUBLIC,
        source_sha256=digest(Path(__file__).read_bytes()), repo_source_sha256=repo_sources,
        evidence=PINS, independent_raw_replay_sha256=PREDECESSOR_REPLAY_SHA, items=items,
        native_shape=[NATIVE, NATIVE], query_offsets_pixels=[list(s) for s in OFFSETS], block_sizes=list(SCALES),
        common_support="Reuse v109 depth/reciprocal conservative native support and one reference pixel per aligned query centre. Compare nearest native query pixel at true projection and all four integer offsets on identical support.",
        projection_scope="Camera/depth and physical query Sun are sealed audit inputs only. Dense geometry uses conservative subpixel depth tests, not new mesh rays.",
        block_rule="Fixed reference-native nonoverlapping blocks anchored at x=y=0. Use common pixels only, retain each block with at least half nominal pixels, and weight retained block means equally.",
        photometry="Unclipped black-space Combined EXR float32 RGB, decoded native, linear Rec.709 luminance in float64. No per-view normalization, display transform, EXR albedo or geometry-derived brightness labels.",
        comparison="Report Pearson r, population covariance and spatial RMS for both within-view seed repeats, all four cross-view seed pairs, and the two seed-average maps at each shift.",
        primary="32-pixel blocks on u061 at 256 spp. All four independent cross-view seed pairs must exceed all shifted correlations by more than 1e-9; undefined metrics are indeterminate.",
        controls="u013 bright and u025/u027 near-uniform at 64 spp. Report same fixed 32/16/native-scale tables; sensitivities do not replace the primary.",
        scope="Relative renderer output repeatability on four exposed positives. No pose estimation, matching, render, optimizer, photon model, calibration or final access.",
        calibration_or_final_access=False,
    )


def summaries(a, b, *, replay):
    import numpy as np

    centered_a, centered_b = a-a.mean(), b-b.mean()
    if replay:
        variances = np.cov(np.stack([a, b]), bias=True)
        var_a, var_b, covariance = float(variances[0, 0]), float(variances[1, 1]), float(variances[0, 1])
    else:
        var_a, var_b, covariance = [float(np.dot(left, right) / len(a)) for left, right in
                                    ((centered_a, centered_a), (centered_b, centered_b), (centered_a, centered_b))]
    rms_a, rms_b = math.sqrt(max(0., var_a)), math.sqrt(max(0., var_b))
    r = (float(np.clip(covariance/(rms_a*rms_b), -1., 1.)) if rms_a > 0 and rms_b > 0
         and a.min() != a.max() and b.min() != b.max() else None)
    return dict(pearson_r=r, covariance=covariance, first_spatial_rms=rms_a, second_spatial_rms=rms_b)


def reducer(p, scale, *, replay):
    import numpy as np

    tile_xy = p // scale
    tiles = NATIVE // scale
    keys = tile_xy[:, 1]*tiles + tile_xy[:, 0]
    if replay:
        order = np.argsort(keys, kind="stable")
        _, starts, count = np.unique(keys[order], return_index=True, return_counts=True)
        keep = count >= (scale*scale+1)//2

        def aggregate(v):
            return (np.add.reduceat(v[order], starts)/count)[keep]
    else:
        count = np.bincount(keys, minlength=tiles*tiles)
        keep = count >= (scale*scale+1)//2

        def aggregate(v):
            return np.bincount(keys, weights=v, minlength=tiles*tiles)[keep] / count[keep]
    if keep.sum() < 2:
        raise ValueError("Insufficient common support for spatial correlation")
    return aggregate, int(keep.sum()), int(count[keep].sum())


def compare_case(p, q, left, right, *, replay):
    import numpy as np

    source = [image[p[:, 1], p[:, 0]] for image in left]
    query = [[image[q[:, 1]+dy, q[:, 0]+dx] for image in right] for dx, dy in OFFSETS]
    scales = {}
    for scale in SCALES:
        block_mean, count, pixels = reducer(p, scale, replay=replay)
        a = [block_mean(v) for v in source]
        data = []
        for shift, shifted in zip(OFFSETS, query, strict=True):
            b = [block_mean(v) for v in shifted]
            comparisons = []
            for i in range(2):
                for j in range(2):
                    comparisons.append(dict(reference_replicate=i, query_replicate=j, **summaries(a[i], b[j], replay=replay)))
            data.append(dict(offset_xy=list(shift), seed_pairs=comparisons,
                             seed_average=summaries(np.mean(a, axis=0), np.mean(b, axis=0), replay=replay),
                             query_within_view=summaries(*b, replay=replay)))
        margins = []
        for idx, aligned in enumerate(data[0]["seed_pairs"]):
            null_r = [x["seed_pairs"][idx]["pearson_r"] for x in data[1:]]
            r = aligned["pearson_r"]
            margins.append(None if r is None or None in null_r else r-max(null_r))
        status = ("indeterminate" if None in margins else
                  "all_seed_pairs_above_every_shift" if min(margins) > EPS else "not_all_seed_pairs_above_every_shift")
        scales[str(scale)] = dict(blocks=count, common_pixels_in_retained_blocks=pixels,
                                  reference_within_view=summaries(*a, replay=replay), comparisons=data,
                                  aligned_minus_strongest_shift_by_seed_pair=margins, outcome=status)
    return scales


def run(frozen, *, replay):
    import cv2
    from replay_raw_acquisition_v109 import decode, luminance

    cv2.setNumThreads(2)
    if replay:
        from replay_clean_tail_morphology_v109 import geometry_replay, load_geometry
    else:
        from audit_clean_tail_morphology_v109 import native_support, load_geometry
    rows = []
    for item in frozen["items"]:
        geometry = load_geometry(item)
        p, q, support = (geometry_replay(*geometry, OFFSETS) if replay else native_support(*geometry))
        if support != item["expected_geometry"]:
            raise ValueError("Projected support differs from the frozen v109 morphology replay")
        left, right = [[luminance(decode(checked(a["path"], a["sha256"]), exr_size=NATIVE))
                        for a in item["views"][role]["combined"]] for role in ("reference", "query")]
        rows.append(dict(case_id=item["case_id"], purpose=item["purpose"],
                         samples_per_pixel=item["samples_per_pixel"], geometry=support,
                         scales=compare_case(p, q, left, right, replay=replay)))
    return rows


def check_rows(expected, got, path="rows"):
    if isinstance(expected, list) and isinstance(got, list):
        if len(expected) != len(got):
            raise ValueError("Replay length differs at " + path)
        for index, (a, b) in enumerate(zip(expected, got, strict=True)):
            check_rows(a, b, f"{path}/{index}")
    elif isinstance(expected, dict) and isinstance(got, dict) and expected.keys() == got.keys():
        for key in expected:
            check_rows(expected[key], got[key], f"{path}/{key}")
    elif isinstance(expected, float) and isinstance(got, float):
        if not math.isclose(expected, got, rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError(f"Independent arithmetic differs at {path}: {expected} vs {got}")
    elif expected != got:
        raise ValueError(f"Independent replay differs at {path}: {expected} vs {got}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "measure", "replay"))
    cli.add_argument("--prior-replay", required=True)
    cli.add_argument("--protocol", type=Path, required=True)
    cli.add_argument("--protocol-sha256")
    cli.add_argument("--result", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    fixed = freeze(args.prior_replay)
    if args.action == "freeze":
        write(args.protocol, fixed)
        print(digest(args.protocol.read_bytes()))
        return
    if not args.protocol_sha256 or not args.output:
        cli.error("measurement or replay requires the pre-recorded protocol SHA and new output")
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads(checked(args.protocol, args.protocol_sha256))
    if frozen != fixed:
        raise ValueError("Current source and metadata differ from the protocol")
    if args.action == "measure":
        write(args.output, dict(schema="solar_raw_crossview_v110", protocol_sha256=args.protocol_sha256,
                                rows=run(frozen, replay=False), renderer_calls=0, matcher_calls=0,
                                calibration_or_final_access=False))
        return
    if not args.result:
        cli.error("replay requires the exact prior result path")
    result = args.result.read_bytes()
    measured = json.loads(result)
    if measured["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Measured protocol binding differs")
    check_rows(measured["rows"], run(frozen, replay=True))
    write(args.output, dict(schema="solar_raw_crossview_replay_v110", passed=True, cases_replayed=4,
                            scales_per_case=3, shifts_per_scale=5, cross_view_seed_pair_cells_replayed=240,
                            method="Reproject with the independent v109 camera arithmetic. Sort/reduce spatial blocks and recompute using covariance matrices.",
                            shared_exr_decoder=True, result_sha256=digest(result),
                            protocol_sha256=args.protocol_sha256, calibration_or_final_access=False))


if __name__ == "__main__":
    main()
