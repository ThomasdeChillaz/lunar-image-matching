"""Fixed-coordinate dither fingerprint audit of five sealed v101 development pairs."""

import argparse
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import subprocess


PUBLIC = "bb5e6a4078353d0a143c3df61e08dc991d503485"
DISPATCH = "reports/solar-radiometric-pilot-dispatch-v101.json"
DISPATCH_SHA = "b7e821211828566d7d60d5250fb230d02e1e73d6092a96418ce5435ab3306b28"
REPLAY = "reports/solar_radiometric_pilot_replay_v101.json"
REPLAY_SHA = "4cb8d85c39cabcf84aff8f805bc9160406a268bf47a130acfe346c4f6da258f9"
PILOT_SEAL_SHA = "51573531e76a6b4cac2c0efe30d9ea07a7f55bda3eb4789f217e81f75b3e2e73"
PROCEDURAL = "reports/solar-png-dither-control-v105.json"
PROCEDURAL_SHA = "dfec557dc407893db7c6030a1fb7a2156e1e0e03a521ce65f84e7ea415159841"
PROCEDURAL_SEAL_SHA = "01dbb14138aa71e9778a895a6bd973548a571a240a7ab320bafb6bf65d070a47"
PROCEDURAL_DIR = Path("runs/solar-png-dither-control-v105")
CASES = (
    ("u025-l00-00-positive", 1, "near_black_wrong", False),
    ("u027-l08-01-positive", 1, "near_black_wrong", False),
    ("u061-l00-17-positive", 17, "dim_wrong_comparator", False),
    ("u013-l00-00-positive", 0, "bright_correct_control", True),
    ("u015-l00-17-positive", 17, "bright_correct_control", True),
)
PROFILE = "reference_camera_render_match_p995_exposure"
SHIFTS = ((0,0), (17,29), (31,-13), (-19,23), (43,11))
EDGE, END, THRESHOLD = 48, 464, 2


def read_verified(path, expected):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("Frozen input changed: " + str(path))
    return data


def read_metadata(path, expected):
    return json.loads(read_verified(path, expected))


def public_metadata(path, expected):
    data = read_verified(path, expected)
    if subprocess.check_output(["git", "show", f"{PUBLIC}:{path}"], timeout=5) != data:
        raise ValueError("Metadata does not match the original public Git bytes")
    return json.loads(data)


def metadata():
    pilot = public_metadata(DISPATCH, DISPATCH_SHA)
    replay = public_metadata(REPLAY, REPLAY_SHA)
    procedural = public_metadata(PROCEDURAL, PROCEDURAL_SHA)
    if (pilot["split"] != "model_selection" or pilot["calibration_or_final_access"]
            or pilot["tracks"] != ["estimated_sun"] or not replay["passed"] or replay["replayed_rows"] != 96
            or replay["protocol_sha256"] != DISPATCH_SHA or replay["accepted_results"]
            or procedural["archived_images_access"] or procedural["residual_xy_shifts_px"] != [list(s) for s in SHIFTS]):
        raise ValueError("The exact public development and procedural protocols are required")
    root = Path(pilot["output"])
    seal = read_metadata(root / "COMPLETE.json", PILOT_SEAL_SHA)
    summary = read_metadata(root / "summary.json", seal["summary_sha256"])
    if (not seal["external_watchdog_passed"] or seal["protocol_sha256"] != DISPATCH_SHA
            or seal["summary_sha256"] != replay["summary_sha256"] or seal["watchdog_sha256"] != replay["watchdog_sha256"]
            or not summary["complete"] or summary["calibration_or_final_access"]):
        raise ValueError("The original sealed and independently replayed run is required")
    basis = public_metadata(pilot["basis"], pilot["metadata_sha256"][pilot["basis"]])
    if basis["manifest"] != pilot["manifest"]:
        raise ValueError("The pilot must use the original v86 development case manifest")
    manifest = read_metadata(pilot["manifest"], basis["input_sha256"][pilot["manifest"]])
    ev = pilot["profiles"][0]["render_evidence"]
    render_library = read_metadata(ev["manifest"], ev["manifest_sha256"])
    if (manifest["split"] != "model_selection" or manifest["calibration_or_final_access"]
            or render_library["split"] != "model_selection" or render_library["calibration_or_final_access"]
            or render_library["query_role_images_used"]):
        raise ValueError("Only already observed model-selection image roles may be used")

    observations = []
    for case_id, light, stratum, correct in CASES:
        case = next(c for c in manifest["cases"] if c["case_id"] == case_id)
        row = next(r for r in summary["rows"] if r["case_id"] == case_id and r["profile"] == PROFILE)
        independent = next(r for r in replay["rows"] if r["case_id"] == case_id and r["profile"] == PROFILE)
        search = row["render_search"]
        if (case_id not in pilot["case_ids"] or case["split"] != "model_selection" or not case["positive"]
                or search["selected_light_index"] != light or independent["correct"] != correct
                or independent["wrong_positive"] == correct or row["metrics"]["correct"] != correct):
            raise ValueError("The exact fixed observed positive cases and selected images are required")
        library = next(lib for lib in render_library["libraries"] if lib["unit_index"] == case["unit_index"])
        selected = next(e for e in library["entries"] if e["light_index"] == light)
        ref, query = selected["image"], case["query"]
        if (selected["source_role"] != "reference" or Path(ref).parent.name != "reference"
                or Path(query).parent.name != "query" or Path(ref).suffix != ".png" or Path(query).suffix != ".png"
                or selected["image_sha256"] != pilot["input_sha256"][ref]
                or case["input_sha256"][query] != pilot["input_sha256"][query]):
            raise ValueError("Image is outside the fixed native reference/query hash inventory")
        observations.append(dict(case_id=case_id, stratum=stratum, terrain=case["component"],
            selected_reference_light=light, saved_reference_p995=search["radiometry"]["reference_renders"][light]["p995_input_code"],
            saved_query_p995=search["radiometry"]["query"]["p995_input_code"], saved_correct=correct,
            inputs=[dict(role=role,path=name,sha256=pilot["input_sha256"][name])
                    for role,name in (("reference",ref),("query",query))]))

    proc_seal = read_metadata(PROCEDURAL_DIR / "complete.json", PROCEDURAL_SEAL_SHA)
    if not proc_seal["complete"] or proc_seal["protocol_sha256"] != PROCEDURAL_SHA:
        raise ValueError("Require the bounded complete original procedural fixture")
    proc = read_metadata(PROCEDURAL_DIR / "rendered.json", proc_seal["rendered_sha256"])
    fixtures = {}
    for arm, dither in (("default",1.), ("zero",0.)):
        match = next(row for row in proc["records"] if row["mode"] == arm and row["stimulus"] == "flat_low")
        if (proc["configuration"][arm]["dither_intensity"] != dither or proc["archived_images_access"]
                or match["filename"] != f"{arm}_flat_low.png"):
            raise ValueError("Use only the original sealed constant-radiance fixtures")
        fixtures[arm] = dict(path=str(PROCEDURAL_DIR / match["filename"]), sha256=match["sha256"])
    if len({e["path"] for row in observations for e in row["inputs"]}) != 10:
        raise ValueError("All ten original archived roles must be distinct")
    return dict(
        schema="archived_blender_dither_fingerprint_v105", source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_threads=2, scope="Fixed retrospective pixel fingerprint only. All five positive pairs were already evaluated in v101.",
        pilot_publication_commit=summary["publication_commit"], source_protocol_sha256=DISPATCH_SHA,
        independent_replay_sha256=REPLAY_SHA, source_procedural_protocol_sha256=PROCEDURAL_SHA,
        source_procedural_seal_sha256=PROCEDURAL_SEAL_SHA, fixtures=fixtures, observations=observations,
        metric=dict(primary="phi between archived code>=2 and fixture code>=2 at the same native coordinates",
                    displaced_native_fixture_offsets_xy=SHIFTS, equal_crop=[EDGE, END], secondary="full-native fixed phi and exact 2x2 counts",
                    bright_control="Secondary archived threshold strictly above the lower median grayscale code; fixture threshold stays >=2.",
                    nesting="Report both directional violations; additionally count smaller-mask positives outside the larger mask, with fixture chosen smaller on ties.",
                    constant_mask="Undefined phi is null; keep exact contingency counts."),
        calibration_or_final_access=False, renderer_calls=0, matcher_calls=0, pose_estimator_calls=0,
    )


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def measure(freeze, expected, output):
    import numpy as np
    from PIL import Image

    frozen = read_metadata(freeze, expected)
    if frozen != json.loads(json.dumps(metadata())):
        raise ValueError("Require the exact metadata freeze prepared before archived decoding")

    def load(entry):
        with Image.open(io.BytesIO(read_verified(entry["path"], entry["sha256"]))) as im:
            if im.format != "PNG" or im.size != (512,512) or im.mode not in ("RGB","L"):
                raise ValueError("Require exact original native PNG")
            return np.asarray(im.convert("L")).copy()

    def compare(archived, fixture):
        tp = int((archived & fixture).sum())
        fp = int((archived & ~fixture).sum())
        fn = int((~archived & fixture).sum())
        tn = int((~archived & ~fixture).sum())
        denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        a, f = tp+fp, tp+fn
        smaller = "archive" if a < f else "fixture"
        return dict(both_true=tp, only_archive=fp, only_fixture=fn, both_false=tn,
                    phi=None if denominator == 0 else (tp*tn-fp*fn)/denominator,
                    archive_high=a, fixture_high=f, archive_true_outside_fixture=fp, fixture_true_outside_archive=fn,
                    archive_true_outside_fixture_fraction=fp/a if a else None,
                    fixture_true_outside_archive_fraction=fn/f if f else None,
                    smaller_mask=smaller, smaller_true_outside_larger=fp if smaller == "archive" else fn)

    def compare_grid(mask, fixture):
        crop = mask[EDGE:END, EDGE:END]
        return dict(full_fixed=compare(mask, fixture), equal_crop_shifts=[
            dict(dx=dx,dy=dy,**compare(crop,fixture[EDGE+dy:END+dy,EDGE+dx:END+dx])) for dx,dy in SHIFTS])

    fixtures = {arm: load(entry) >= THRESHOLD for arm,entry in frozen["fixtures"].items()}
    rows = []
    for case in frozen["observations"]:
        for entry in case["inputs"]:
            image = load(entry)
            hist = np.bincount(image.ravel(),minlength=256)
            mid = int(np.searchsorted(hist.cumsum(), image.size//2))
            rows.append(dict(case_id=case["case_id"],stratum=case["stratum"],role=entry["role"],
                code_histogram={int(i):int(hist[i]) for i in np.flatnonzero(hist)},
                code_at_least_two_vs_fixture={arm:compare_grid(image >= THRESHOLD, mask) for arm,mask in fixtures.items()},
                above_lower_median_code=mid,
                above_lower_median_vs_default=compare_grid(image > mid, fixtures["default"])))
    write(output, dict(
        freeze_sha256=expected, source_sha256=frozen["source_sha256"], rows=rows,
        archived_image_count=len(rows), procedural_fixture_count=len(fixtures),
        matcher_calls=0, pose_estimator_calls=0, renderer_calls=0, calibration_or_final_access=False,
        scope="Post hoc analysis of selected observed development failures and bright controls. Counts describe pixels and do not represent independent terrain evidence or pose improvement.",
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    sub.add_parser("freeze").add_argument("--output",required=True,type=Path)
    run = sub.add_parser("measure")
    run.add_argument("--freeze",required=True,type=Path)
    run.add_argument("--freeze-sha256",required=True)
    run.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.command == "freeze":
        write(a.output,metadata())
    else:
        measure(a.freeze,a.freeze_sha256,a.output)
