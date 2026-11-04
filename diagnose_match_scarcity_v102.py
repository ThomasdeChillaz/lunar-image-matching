"""Saved-only study of camera, lighting and signal in the frozen development pairs."""
import argparse
from collections import defaultdict
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


FULL = Path("reports/solar-full-crosslight-rescue-dispatch-v99.json")
FULL_REPLAY = Path("reports/solar_full_crosslight_rescue_replay_v99.json")
EVEN = Path("reports/solar-offcatalogue-direct-dispatch-v95.json")
EVEN_REPLAY = Path("reports/solar_offcatalogue_direct_replay_v95.json")
DIRECT = "reference_camera_render_match"
NATIVE = Path("reports/solar-full-native-baselines-dispatch-v94.json")
NATIVE_REPLAY = Path("reports/solar_full_native_baselines_replay_v94.json")


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(path, digest):
    payload = Path(path).read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError("Frozen input differs: " + str(path))
    return payload


def run_rows(protocol_path, replay_path, total):
    p, replay = read(protocol_path), read(replay_path)
    root = Path(p["output"])
    seal = read(root / "COMPLETE.json")
    summary = json.loads(checked(root / "summary.json", seal["summary_sha256"]))
    watchdog = json.loads(checked(root / "watchdog.json", seal["watchdog_sha256"]))
    if (watchdog["complete"] is not True or watchdog["child_returncode"] != 0 or replay["passed"] is not True or replay["replayed_rows"] != total or replay["accepted_results"]
            or replay["protocol_sha256"] != sha(protocol_path) or replay["summary_sha256"] != seal["summary_sha256"]
            or replay["watchdog_sha256"] != seal["watchdog_sha256"] or not seal["external_watchdog_passed"]
            or seal["protocol_sha256"] != sha(protocol_path) or not summary["complete"]
            or summary["calibration_or_final_access"] or len(summary["rows"]) != total):
        raise ValueError("Original sealed and independently replayed development results required")
    return p, summary["rows"]


def inspect_image(payload):
    with Image.open(io.BytesIO(payload)) as im:
        if im.size != (512, 512) or im.format != "PNG":
            raise ValueError("Original native 512×512 PNG expected")
        a = np.asarray(im.convert("L")).astype(np.int16)
    histogram = np.bincount(a.ravel(), minlength=256)
    y, x = np.abs(np.diff(a, axis=0)), np.abs(np.diff(a, axis=1))
    # Native mean absolute 1-pixel differences. Local patches below use the
    # same adjacent-code signal, with no filtering or feature extraction.
    gradient = np.zeros((512, 512), float)
    gradient[:-1, :-1] = .5 * (x[:-1] + y[:, :-1])
    integral = np.pad(gradient, [(1, 0), (1, 0)]).cumsum(0).cumsum(1)
    stats = dict(
        codes=int(np.count_nonzero(histogram)), flat=int(np.count_nonzero(histogram)) == 1,
        p95=int(np.searchsorted(histogram.cumsum(), np.ceil(.95 * a.size))),
        mean_gradient=float(gradient[:-1, :-1].mean()),
        gradient_ge2_fraction=float(((x >= 2).sum() + (y >= 2).sum()) / (x.size + y.size)),
    )
    return stats, integral


def patch_mean(integral, p, radius=12):
    p = np.rint(p).astype(int)
    a, b = p - radius, p + radius + 1
    return (integral[b[:, 1], b[:, 0]] - integral[a[:, 1], b[:, 0]]
            - integral[b[:, 1], a[:, 0]] + integral[a[:, 1], a[:, 0]]) / (2 * radius + 1)**2


def projection_labels(case):
    p = case["scoring_inputs"]["labels"]
    with np.load(io.BytesIO(checked(p, case["input_sha256"][p])), allow_pickle=False) as labels:
        r, q, visible = (labels[k] for k in ("reference_xy", "query_xy", "visible"))
    axis = 49.5 + 16 * np.rint(np.linspace(0, 25, 16)).astype(int)
    use = np.isin(r[:, 0], axis) & np.isin(r[:, 1], axis)
    if use.sum() != 256 or not visible.any() or not np.isfinite(q[visible]).all():
        raise ValueError("Original exact anchors and co-visible scoring labels required")
    r, q, visible = r[use], q[use], visible[use]
    # A geometrically available 25-pixel footprint in both views. This
    # bounds image crop only, without asserting patch-interior visibility.
    supported = visible & ((q >= 13) & (q <= 497)).all(1)
    return r[supported], q[supported], int(visible.sum())


def stats(x):
    return None if not len(x) else float(np.median(x))


def ranks(values):
    _, rev, count = np.unique(values, return_inverse=True, return_counts=True)
    centres = np.cumsum(count) - .5 * (count - 1)
    return centres[rev]


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ranked_correlations(rows):
    outcome = "best_saved_candidate"
    variables = ("query_gradient", "exact_template_gradient", "joint_patch_gradient",
                 "visible_anchor_count", "valid_patch_centres", "minimum_bidir_visibility")
    group_sets = ((), ("unit_index",), ("query_light", "terrain"))

    def residual(values, groups):
        if not groups:
            return values
        result = values.copy()
        index = defaultdict(list)
        for i, row in enumerate(rows):
            index[tuple(row[key] for key in groups)].append(i)
        for indices in index.values():
            result[indices] -= values[indices].mean()
        return result

    output = {}
    for group in group_sets:
        b = residual(ranks([r[outcome] for r in rows]), group)
        output["global" if not group else "within_" + "_".join(group)] = {
            key: correlation(residual(ranks([r[key] for r in rows]), group), b) for key in variables
        }
    for key, first, second in (
        ("query_template_intercorrelation", "query_gradient", "exact_template_gradient"),
        ("query_template_joint_intercorrelation", "query_gradient", "joint_patch_gradient"),
    ):
        output[key] = correlation(ranks([r[first] for r in rows]), ranks([r[second] for r in rows]))
    return output


def aggregate(part):
    return dict(
        camera_query_conditions=len(part),
        query_flat=sum(r["query_flat"] for r in part),
        exact_template_flat=sum(r["exact_template_flat"] for r in part),
        median_query_gradient=stats([r["query_gradient"] for r in part]),
        median_exact_template_gradient=stats([r["exact_template_gradient"] for r in part]),
        median_joint_patch_gradient=stats([r["joint_patch_gradient"] for r in part]),
        median_joint_patch_gradient_ge1_fraction=stats([r["joint_patch_gradient_ge1_fraction"] for r in part]),
        median_query_p95=stats([r["query_p95"] for r in part]),
        median_exact_template_p95=stats([r["exact_template_p95"] for r in part]),
        median_valid_patch_centres=stats([r["valid_patch_centres"] for r in part]),
        median_best_full=stats([r["best_saved_candidate"] for r in part]),
        median_selected_full=stats([r["selected_full"] for r in part]),
        median_selected_even=stats([r["selected_even"] for r in part]),
        missing_full=sum(not r["pose_returned"] for r in part),
        full_correct=sum(r["correct"] for r in part),
        even_correct=sum(r["even_correct"] for r in part),
        max_below_six=sum(r["best_saved_candidate"] < 6 for r in part),
        max_at_least_six_but_no_pose=sum(r["best_saved_candidate"] >= 6 and not r["pose_returned"] for r in part),
        even_to_full_correct_gain=sum(r["correct"] and not r["even_correct"] for r in part),
        even_to_full_correct_loss=sum(r["even_correct"] and not r["correct"] for r in part),
    )


def observed_native_contrasts(cases, native_rows, load_image):
    """Existing two native arms let one original image vary while the other stays byte-identical."""
    positives = defaultdict(dict)
    by_case = defaultdict(dict)
    for c in cases:
        if c["positive"]:
            positives[c["unit_index"]][c["light_a"], c["light_b"]] = c
    for row in native_rows:
        if row["positive"]:
            by_case[row["case_id"]][row["profile"]] = row
    specs = (
        ("fixed_query_17_ref_0_to_8", "reference", (0, 17), (8, 17)),
        ("fixed_query_1_ref_16_to_8", "reference", (16, 1), (8, 1)),
        ("fixed_ref_0_query_0_to_17", "query", (0, 0), (0, 17)),
        ("fixed_ref_16_query_16_to_1", "query", (16, 16), (16, 1)),
        ("fixed_ref_8_query_17_to_1", "query", (8, 17), (8, 1)),
    )

    def solar_angle(c):
        return float(np.degrees(np.arccos(np.clip(
            np.dot(c["reference_sun_world"], c["secondary_known_query_sun_world"]), -1., 1.
        ))))

    out = {}
    for name, side, a, b in specs:
        other = "query" if side == "reference" else "reference"
        pairs = []
        for unit, items in positives.items():
            first, second = items[a], items[b]
            if (first[other] != second[other] or first[side] == second[side]
                    or first["reference_geometry"] != second["reference_geometry"]
                    or first["scoring_inputs"] != second["scoring_inputs"]):
                raise ValueError("Require an exactly fixed counterpart and camera pair")
            first_image, second_image = (
                load_image(c[side], c["input_sha256"][c[side]])[0] for c in (first, second)
            )
            before, after = by_case[first["case_id"]], by_case[second["case_id"]]
            if set(before) != set(after) or len(before) != 2:
                raise ValueError("Both original native arms required")
            effects = {}
            for arm, row in before.items():
                prior, later = row, after[arm]
                if prior["inference_failure"] or later["inference_failure"]:
                    raise ValueError("Original native comparison reported a failed scientific cell")
                old, new = (r["metrics"]["correspondence"]["2"] for r in (prior, later))
                effects[arm] = dict(returned_delta=new["returned"] - old["returned"],
                                    correct_2px_delta=new["correct"] - old["correct"],
                                    pose_correct_delta=int(later["metrics"]["correct"]) - int(prior["metrics"]["correct"]))
            pairs.append(dict(unit=unit, first=first["case_id"], second=second["case_id"],
                              changing_side_gradient_delta=second_image["mean_gradient"] - first_image["mean_gradient"],
                              changing_side_p95_delta=second_image["p95"] - first_image["p95"],
                              solar_separation_delta_deg=solar_angle(second) - solar_angle(first), arms=effects))
        by_arm = {}
        for arm in pairs[0]["arms"]:
            counts = [r["arms"][arm]["returned_delta"] for r in pairs]
            gt = [r["arms"][arm]["correct_2px_delta"] for r in pairs]
            by_arm[arm] = dict(median_returned_delta=stats(counts), median_correct_2px_delta=stats(gt),
                               fewer_correct_2px=sum(x < 0 for x in gt), more_correct_2px=sum(x > 0 for x in gt),
                               correct_pose_net_delta=sum(r["arms"][arm]["pose_correct_delta"] for r in pairs),
                               spearman_correct_delta_vs_changing_signal=correlation(
                                   ranks(gt), ranks([r["changing_side_gradient_delta"] for r in pairs])),
                               spearman_correct_delta_vs_solar_separation=correlation(
                                   ranks(gt), ranks([r["solar_separation_delta_deg"] for r in pairs])))
        out[name] = dict(changing_side=side, fixed_side=other, pairs=len(pairs), summary=by_arm, rows=pairs)
    return out


def analyze():
    p, full_rows = run_rows(FULL, FULL_REPLAY, 768)
    even, even_rows = run_rows(EVEN, EVEN_REPLAY, 384)
    native, native_rows = run_rows(NATIVE, NATIVE_REPLAY, 768)
    manifest_path, audit_path = p["manifest"], p["geometry_audit"]
    manifest = json.loads(checked(manifest_path, p["input_sha256"][manifest_path]))
    audit = json.loads(checked(audit_path, p["input_sha256"][audit_path]))
    if (even["manifest"] != manifest_path or native["manifest"] != manifest_path or manifest["split"] != "model_selection"
            or manifest["calibration_or_final_access"] or audit["calibration_or_final_access"]):
        raise ValueError("Only the original development bank and audit allowed")
    camera_units = {u["unit_index"]: u for u in audit["units"] if u["split"] == "model_selection"}
    render = p["profiles"][0]["render_evidence"]
    library = json.loads(checked(render["manifest"], render["manifest_sha256"]))
    views = {lib["reference_geometry_sha256"]: {i["light_index"]: i for i in lib["entries"]}
             for lib in library["libraries"]}
    full = {r["case_id"]: r for r in full_rows if r["profile"] == DIRECT and r["positive"]}
    alternate = {r["case_id"]: r for r in even_rows if r["positive"]}
    # Collapse aliases: the full matcher uses geometry and its own catalogue,
    # not the nominal reference-light PNG. Verify the measured pose-call
    # hashes and scored metrics before counting either alias only once.
    groups = defaultdict(list)
    for case in manifest["cases"]:
        if case["positive"]:
            groups[case["unit_index"], case["query"]].append(case)
    if len(full) != 192 or len(alternate) != 192 or len(groups) != 128:
        raise ValueError("Expected original 192 positive rows and 128 camera/query conditions")
    geometries, cache, rows = {}, {}, []

    def load_image(path, digest, *, need_patches=False):
        if not need_patches and path in cache:
            expected, features = cache[path]
            if expected != digest:
                raise ValueError("One image path has two frozen digests")
            return features, None
        features, integral = inspect_image(checked(path, digest))
        cache[path] = digest, features
        return features, integral

    for (unit, _), cases in groups.items():
        case = cases[0]
        r, e = full[case["case_id"]], alternate[case["case_id"]]
        for alias in cases:
            if (alias["query"] != case["query"] or alias["reference_geometry"] != case["reference_geometry"]
                    or any(full[alias["case_id"]][k] != r[k] for k in ("metrics", "render_search"))
                    or any(alternate[alias["case_id"]][k] != e[k] for k in ("metrics", "render_search"))):
                raise ValueError("Nominal reference-light aliases differ in saved inference")
        search = r["render_search"]
        calls = search["observed_pose_calls"]
        if len(calls) != 9 or search["exact_known_sun"] or search["library_entries_considered"] != 24:
            raise ValueError("Original 24-catalogue trace required")
        best = max(c["pairs"] for c in calls[:8])
        if calls[0]["pairs"] != best:
            raise ValueError("Expected frozen direct proposal count ordering")
        selected = calls[-1]["pairs"]
        if selected != r["metrics"]["correspondence"]["2"]["returned"]:
            raise ValueError("Saved returned points and final call differ")
        light = case["light_b"]
        templates = views[case["reference_asset_metadata"]["source_reference_geometry_sha256"]]
        entry = templates[light]
        chosen = templates[search["selected_light_index"]]
        qstat, iq = load_image(case["query"], case["input_sha256"][case["query"]], need_patches=True)
        tstat, ir = load_image(entry["image"], entry["image_sha256"], need_patches=True)
        sstat, _ = load_image(chosen["image"], chosen["image_sha256"])
        if unit not in geometries:
            geometries[unit] = projection_labels(case)
        anchors, target, visible = geometries[unit]
        qr, rr = patch_mean(iq, target), patch_mean(ir, anchors)
        row = dict(
            aliases=[c["case_id"] for c in cases], unit_index=unit, terrain=case["component"],
            family=camera_units[unit]["family"], query_light=light,
            query=case["query"], exact_template=entry["image"], selected_template=chosen["image"],
            query_gradient=qstat["mean_gradient"], exact_template_gradient=tstat["mean_gradient"],
            query_gradient_ge2_fraction=qstat["gradient_ge2_fraction"],
            exact_template_gradient_ge2_fraction=tstat["gradient_ge2_fraction"],
            selected_template_gradient=sstat["mean_gradient"],
            query_p95=qstat["p95"], exact_template_p95=tstat["p95"],
            query_codes=qstat["codes"], exact_template_codes=tstat["codes"],
            query_flat=qstat["flat"], exact_template_flat=tstat["flat"],
            query_patch_gradient=stats(qr), exact_template_patch_gradient=stats(rr),
            joint_patch_gradient=stats(np.minimum(qr, rr)),
            joint_patch_gradient_ge1_fraction=float(((qr >= 1) & (rr >= 1)).mean()),
            query_patch_gradient_ge1_fraction=float((qr >= 1).mean()),
            exact_template_patch_gradient_ge1_fraction=float((rr >= 1).mean()),
            visible_anchor_count=visible, valid_patch_centres=len(anchors),
            minimum_bidir_visibility=camera_units[unit]["minimum_bidirectional_visible_fraction"],
            best_saved_candidate=best, selected_full=selected, selected_light=search["selected_light_index"],
            selected_even=e["metrics"]["correspondence"]["2"]["returned"], even_selected_light=e["render_search"]["selected_light_index"],
            correct_2px_full=r["metrics"]["correspondence"]["2"]["correct"],
            pose_returned=r["metrics"]["pose_returned"], correct=r["metrics"]["correct"], even_correct=e["metrics"]["correct"],
        )
        rows.append(row)
    if any(r["joint_patch_gradient"] is None for r in rows):
        raise ValueError("Every camera must have some geometrically available patches")
    by_light = {str(light): aggregate([r for r in rows if r["query_light"] == light]) for light in (0, 1, 16, 17)}
    stages = {
        "maximum_below_six": [r for r in rows if r["best_saved_candidate"] < 6],
        "maximum_at_least_six_but_no_pose": [r for r in rows if r["best_saved_candidate"] >= 6 and not r["pose_returned"]],
        "pose_returned": [r for r in rows if r["pose_returned"]],
    }
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["unit_index"]].append(r)
    contrast = {}
    for a, b in ((0, 1), (16, 17), (0, 16), (1, 17)):
        contrasts = []
        for vals in pairs.values():
            index = {r["query_light"]: r for r in vals}
            x, y = index[a], index[b]
            contrasts.append([y[k] - x[k] for k in ("query_gradient", "exact_template_gradient", "best_saved_candidate")])
        diffs = np.array(contrasts)
        contrast[f"{a}_to_{b}"] = dict(cameras=len(diffs), median_query_gradient_delta=float(np.median(diffs[:, 0])),
            median_template_gradient_delta=float(np.median(diffs[:, 1])),
            median_best_saved_count_delta=float(np.median(diffs[:, 2])),
            best_count_lower=int((diffs[:, 2] < 0).sum()), best_count_higher=int((diffs[:, 2] > 0).sum()))
    low = stages["maximum_below_six"]
    adequate = [r for r in rows if r["best_saved_candidate"] >= 6]
    both_bright = [r for r in low if min(r["query_p95"], r["exact_template_p95"]) >= 10]
    gains = [r for r in rows if r["correct"] and not r["even_correct"]]
    losses = [r for r in rows if r["even_correct"] and not r["correct"]]
    strata = {}
    for label, part in (("low_saved_pair_count", low), ("adequate_saved_pair_count", adequate)):
        strata[label] = {
            side: sum(test(r["query_p95"], r["exact_template_p95"]) for r in part)
            for side, test in (("both_p95_le3", lambda q, t: max(q, t) <= 3),
                               ("query_only_p95_le3", lambda q, t: q <= 3 < t),
                               ("exact_template_only_p95_le3", lambda q, t: t <= 3 < q),
                               ("neither_p95_le3", lambda q, t: min(q, t) > 3))
        }
    negative_groups = defaultdict(list)
    full_negative = {r["case_id"]: r for r in full_rows if r["profile"] == DIRECT and not r["positive"]}
    even_negative = {r["case_id"]: r for r in even_rows if not r["positive"]}
    for case in manifest["cases"]:
        if not case["positive"]:
            negative_groups[case["unit_index"], case["query"]].append(case["case_id"])
    negative = []
    for ids in negative_groups.values():
        r, e = full_negative[ids[0]], even_negative[ids[0]]
        for cid in ids:
            if any((grid[cid]["metrics"], grid[cid]["render_search"]) != (row["metrics"], row["render_search"])
                   for grid, row in ((full_negative, r), (even_negative, e))):
                raise ValueError("Nonoverlap aliases differ in saved outcomes")
        negative.append((r, e))
    return dict(
        scope="Frozen eight-terrain development set. Repeated nominal reference light rows collapse to 128 camera/query "
              "conditions with exact saved-output parity. Descriptive PNG/geometry statistics and already saved match outputs "
              "only. Same-light templates and labels are post hoc diagnostic references.",
        evidence={str(path): sha(path) for path in (FULL, FULL_REPLAY, EVEN, EVEN_REPLAY, NATIVE, NATIVE_REPLAY,
                                                     manifest_path, audit_path)},
        original_positive_rows=len(full), unique_camera_query_conditions=len(rows), terrain_groups=len({r["terrain"] for r in rows}),
        by_query_light=by_light,
        by_pose_stage={k: aggregate(v) for k, v in stages.items()},
        by_family={k: aggregate([r for r in rows if r["family"] == k]) for k in sorted({r["family"] for r in rows})},
        by_terrain={k: aggregate([r for r in rows if r["terrain"] == k]) for k in sorted({r["terrain"] for r in rows})},
        within_camera_light_contrasts=contrast, rank_correlations=ranked_correlations(rows),
        signal_strata_p95_le3=strata,
        low_support_geometry=dict(at_least_230_valid_patch_centres=sum(r["valid_patch_centres"] >= 230 for r in low),
                                  both_bright_p95_ge10_cases=[r["aliases"][0] for r in both_bright],
                                  both_bright_by_terrain={k: sum(r["terrain"] == k for r in both_bright)
                                                          for k in sorted({r["terrain"] for r in both_bright})}),
        same_query_catalogue_counterfactual=dict(
            odd_conditions=sum(r["query_light"] in (1, 17) for r in rows), full_only_correct=len(gains),
            even_only_correct=len(losses), gains_selected_matching_light=sum(r["selected_light"] == r["query_light"] for r in gains),
            gain_median_query_p95=stats([r["query_p95"] for r in gains]),
            gain_median_exact_template_p95=stats([r["exact_template_p95"] for r in gains])),
        native_same_camera_side_contrasts=observed_native_contrasts(manifest["cases"], native_rows, load_image),
        nonoverlap=dict(original_rows=len(full_negative), unique_camera_query_conditions=len(negative),
                        full_at_least_six_best_candidate=sum(
                            max(c["pairs"] for c in r["render_search"]["observed_pose_calls"][:8]) >= 6 for r, _ in negative),
                        full_pose_outputs=sum(r["metrics"]["pose_returned"] for r, _ in negative),
                        even_pose_outputs=sum(e["metrics"]["pose_returned"] for _, e in negative)),
        rows=rows,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    report = analyze()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
