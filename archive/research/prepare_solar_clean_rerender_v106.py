"""Freeze a fixed, retrospective development cohort without opening optical payloads."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

from solar_physics_v63 import lighting_schedule


PUBLICATION = "fe1d769d69e6b9543da868f60df442f2fdebd253"
DISPATCH = "reports/solar-radiometric-pilot-dispatch-v101.json"
DISPATCH_SHA = "b7e821211828566d7d60d5250fb230d02e1e73d6092a96418ce5435ab3306b28"
REPLAY = "reports/solar_radiometric_pilot_replay_v101.json"
REPLAY_SHA = "4cb8d85c39cabcf84aff8f805bc9160406a268bf47a130acfe346c4f6da258f9"
FINGERPRINT = "reports/solar_dither_fingerprint_result_v105.json"
FINGERPRINT_SHA = "4ebbf0043283c56d2fceb42cd7aca034db5421982c9da11f8d35cf1ec5829232"
PROCEDURAL = "reports/solar_png_dither_control_result_v105.json"
PROCEDURAL_SHA = "54d80e59e0138e0a9ca60af1ced70f0db86929a42457e6946bc896b35f0da7ab"
MASTER = "reports/solar_development_admission_v67.json"
COMPLETION = "reports/solar_completion_contract_v70.json"
COMPLETION_SHA = "6704db3d6e11a62d6dad92bf621eb83d53577875e18cd3543459aaba9656b03c"
OUTPUT = Path("runs/solar-clean-radiance-v106/render")
CAP_SECONDS = 900
EXPOSURE = "reference_camera_render_match_p995_exposure"
PAIRS = (
    ("u013-l00-00-positive", "u013-to-u021-l00-00-nonoverlap", "bright_correct_control"),
    ("u025-l00-00-positive", "u025-to-u057-l00-00-nonoverlap", "near_black_dense_wrong"),
    ("u027-l08-01-positive", "u027-to-u059-l08-01-nonoverlap", "near_black_dense_wrong"),
    ("u061-l00-17-positive", "u061-to-u069-l00-17-nonoverlap", "dim_wrong_comparator"),
)


def unique(rows, field):
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate predecessor metadata identity: " + field)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build():
    bindings = {}
    published = {}

    def read(path, expected, public=False):
        path = Path(path)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".json",
                "The selector may only open repository-relative JSON metadata")
        data = path.read_bytes()
        name = path.as_posix()
        require(sha256(data).hexdigest() == expected, "Predecessor metadata changed: " + name)
        require(bindings.setdefault(name, expected) == expected, "Conflicting metadata identity: " + name)
        if public:
            actual = subprocess.check_output(["git", "show", f"{PUBLICATION}:{name}"], timeout=8)
            require(data == actual, "Public predecessor bytes changed: " + name)
            published[name] = expected
        return json.loads(data)

    pilot = read(DISPATCH, DISPATCH_SHA, public=True)
    replay = read(REPLAY, REPLAY_SHA, public=True)
    read(FINGERPRINT, FINGERPRINT_SHA, public=True)
    read(PROCEDURAL, PROCEDURAL_SHA, public=True)
    require(pilot["split"] == "model_selection" and pilot["tracks"] == ["estimated_sun"]
            and pilot["calibration_or_final_access"] is False and replay["passed"]
            and replay["replayed_rows"] == 96 and replay["protocol_sha256"] == DISPATCH_SHA,
            "An exact independently replayed model-selection pilot is required")
    basis_path = pilot["basis"]
    basis = read(basis_path, pilot["metadata_sha256"][basis_path], public=True)
    manifest_path = pilot["manifest"]
    manifest_hash = basis["input_sha256"][manifest_path]
    manifest = read(manifest_path, manifest_hash)
    audit_path = manifest["geometry_audit"]
    require(audit_path == pilot["geometry_audit"], "Geometry predecessor identity mismatch")
    audit = read(audit_path, manifest["geometry_audit_sha256"], public=True)
    master_hash = manifest["original_master_sha256"]
    master = read(MASTER, master_hash, public=True)
    completion = read(COMPLETION, COMPLETION_SHA, public=True)
    require(manifest["complete"] and manifest["split"] == "model_selection"
            and manifest["calibration_or_final_access"] is False and audit["passed"]
            and audit["complete_dataset"] and audit["calibration_or_final_access"] is False
            and audit["original_master_sha256"] == master_hash and audit["completion_contract_sha256"] == COMPLETION_SHA
            and completion["original_master_sha256"] == master_hash,
            "Require the audited original development acquisition")
    evidence = pilot["profiles"][0]["render_evidence"]
    for key in ("protocol", "receipt", "controls"):
        read(evidence[key], evidence[key + "_sha256"], public=True)
    library = read(evidence["manifest"], evidence["manifest_sha256"], public=True)
    require(library["complete"] and library["native_size"] == 512
            and library["split"] == "model_selection" and library["calibration_or_final_access"] is False
            and library["query_truth_used"] is False and library["query_role_images_used"] is False,
            "Require the original blind full-angle reference-only library")

    audited = unique(audit["units"], "unit_index")
    libraries = unique(library["libraries"], "unit_index")
    cases = unique(manifest["cases"], "case_id")
    exposure_rows = unique([row for row in replay["rows"] if row["profile"] == EXPOSURE], "case_id")
    loaded_units, renders = {}, {}

    def unit(index):
        if index not in loaded_units:
            spec, checked = master["units"][index], audited[index]
            pair = spec["camera_pairs"][0]
            require(index in manifest["eligible_unit_indices"] and checked["eligible"]
                    and checked["split"] == spec["site"]["split"] == pair["split"] == "model_selection"
                    and checked["component"] == spec["site"]["component"] == pair["component"]
                    and spec["calibration_or_final_access"] is False
                    and spec["cpu_threads"] == 2 and spec["lighting_count"] == 24,
                    "Render may only target an eligible original development camera")
            root = Path(checked["source_output"])
            summary_path, seal_path = root / "summary.json", root / "COMPLETE.json"
            summary = read(summary_path, checked["source_summary_sha256"])
            seal = read(seal_path, checked["source_seal_sha256"])["sha256"]
            require(summary["complete"] and summary["calibration_or_final_access"] is False
                    and summary["protocol_sha256"] == master_hash
                    and seal["summary.json"] == checked["source_summary_sha256"]
                    and summary["original_unit_index"] == index
                    and summary["completion_contract_sha256"] == COMPLETION_SHA,
                    "Original camera archive and independent audit disagree")
            images = {(row["role"], row["light"]): row for row in summary["images"]}
            require(len(images) == len(summary["images"]) == 48
                    and set(images) == {(role, n) for role in ("reference", "query") for n in range(24)},
                    "Require the exact original complete image schedule")
            lights = unique(lighting_schedule(spec["site"]["centre_latitude_deg"]), "index")
            loaded_units[index] = (root, spec, checked, summary, seal, images, lights)
        return loaded_units[index]

    def render(index, role, light):
        key = f"u{index:03d}/{role}/light-{light}"
        if key in renders:
            return key
        root, spec, checked, summary, seal, images, lights = unit(index)
        row = images[role, light]
        view = summary["views"][role]
        require(row["path"] == f"{role}/light-{light}.png"
                and view["path"] == f"{role}/geometry.npz" and view["complete_surface"]
                and seal[row["path"]] == row["sha256"] and seal[view["path"]] == view["sha256"],
                "Archived role must match original completed image and geometry pins")
        seconds = row["render_seconds"]
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                "Require original measured render timing")
        output = OUTPUT / key
        renders[key] = dict(
            key=key, unit_index=index, role=role, light_index=light, component=checked["component"],
            archive_path=str(root / row["path"]), archive_sha256=row["sha256"],
            source_summary=str(root / "summary.json"), source_summary_sha256=checked["source_summary_sha256"],
            source_seal=str(root / "COMPLETE.json"), source_seal_sha256=checked["source_seal_sha256"],
            archive_geometry_path=str(root / view["path"]), archive_geometry_sha256=view["sha256"],
            mesh_path=spec["mesh"], mesh_sha256=spec["mesh_sha256"],
            sun_enu=lights[light]["sun_enu"], original_render_seconds=seconds,
            output_default_rgb8=f"{output}-default-rgb8.png", output_clean_rgb8=f"{output}-clean-rgb8.png",
            output_clean_rgb16=f"{output}-clean-rgb16.png",
        )
        return key

    selected = []
    for positive_id, negative_id, stratum in PAIRS:
        positive, negative = cases[positive_id], cases[negative_id]
        require(positive["positive"] is True and negative["positive"] is False
                and negative["matched_positive_case"] == positive_id and positive["unit_index"] == negative["unit_index"],
                "Only the exact original matched negative may accompany a positive")
        for case in (positive, negative):
            case_id, index = case["case_id"], case["unit_index"]
            require(case_id in pilot["case_ids"] and case["split"] == "model_selection",
                    "Require an already evaluated v101 development case")
            observed = exposure_rows[case_id]
            expected_correct = case["positive"] and stratum == "bright_correct_control"
            require(observed["correct"] is expected_correct and observed["raw_nonoverlap"] is False
                    and observed["wrong_positive"] is (case["positive"] and not expected_correct),
                    "Expected original independent pose evidence changed")
            references = [render(index, "reference", light) for light in range(24)]
            query = render(case["query_unit_index"], "query", case["light_b"])
            nominal = references[case["light_a"]]
            for role, key in (("reference", nominal), ("query", query)):
                path = case[role]
                require(path == renders[key]["archive_path"]
                        and case["input_sha256"][path] == pilot["input_sha256"][path] == renders[key]["archive_sha256"],
                        "Archived optical input conflicts with exact case identity")
            lib = libraries[index]
            entries = unique(lib["entries"], "light_index")
            require(list(entries) == list(range(24)) and lib["source_role"] == "reference"
                    and lib["query_camera_fields_used"] is False, "Expected blind original reference library order")
            for light, key in enumerate(references):
                old, new = entries[light], renders[key]
                require(old["image"] == new["archive_path"] and old["image_sha256"] == new["archive_sha256"]
                        and old["sun_world"] == new["sun_enu"]
                        and old["original_render_seconds"] == new["original_render_seconds"]
                        and pilot["input_sha256"][old["image"]] == old["image_sha256"],
                        "Blind reference schedule conflicts with the archived native library")
            selected.append(dict(
                case_id=case_id, unit_index=index, query_unit_index=case["query_unit_index"],
                positive=case["positive"], matched_positive_case=case.get("matched_positive_case"),
                selection_stratum=stratum if case["positive"] else "fixed_matched_nonoverlap",
                reference_keys=references, nominal_reference_key=nominal, query_key=query,
                original_case_input_sha256=case["input_sha256"],
            ))

    query_keys = [case["query_key"] for case in selected]
    require(len(renders) == 104 and len(set(query_keys)) == len(query_keys) == 8
            and len({renders[key]["archive_path"] for key in query_keys}) == 8
            and len({renders[key]["archive_sha256"] for key in query_keys}) == 8,
            "Retain four full reference libraries and eight distinct query exposures")
    views = {(row["unit_index"], row["role"]) for row in renders.values()}
    ray_seconds = math.fsum(unit(i)[3]["views"][role]["centre_ray_seconds"] for i, role in views)
    render_seconds = math.fsum(row["original_render_seconds"] for row in renders.values())
    factor_two = 2 * (render_seconds + ray_seconds)
    ledger_path, ledger_hash = pilot["compute_ledger"], pilot["compute_ledger_sha256"]
    ledger = read(ledger_path, ledger_hash, public=True)
    journal_path = pilot["reservation_journal"]
    journal = json.loads(Path(journal_path).read_bytes())
    require(ledger["accounting_complete"] and journal["ledger_sha256"] == ledger_hash,
            "Original cumulative baseline ledger required")
    left = ledger["remaining_baseline_wall_seconds"] - sum(row["seconds"] for row in journal["reservations"] if row["category"] == "baseline")
    require(len(views) == 12 and factor_two < CAP_SECONDS <= left
            and all(master["units"][i]["max_calendar_seconds"] == CAP_SECONDS for i, _ in views),
            "Fixed cap must cover twice the measured render/geometry work and fit the original budget")
    return dict(
        schema="solar_clean_rerender_cohort_v106", predecessor_publication_commit=PUBLICATION,
        selector_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), split="model_selection", cpu_threads=2,
        calibration_or_final_access=False, query_truth_in_inference=False, native_shape=[512, 512],
        scope="Retrospective mechanism check on four selected previously observed positive cases and their fixed negatives; no generalization claim.",
        original_master=MASTER, original_master_sha256=master_hash, case_manifest=manifest_path,
        case_manifest_sha256=manifest_hash, geometry_audit=audit_path, geometry_audit_sha256=manifest["geometry_audit_sha256"],
        original_library_manifest=evidence["manifest"], original_library_manifest_sha256=evidence["manifest_sha256"],
        original_exposure_dispatch=DISPATCH, original_exposure_dispatch_sha256=DISPATCH_SHA,
        original_exposure_replay=REPLAY, original_exposure_replay_sha256=REPLAY_SHA,
        output_root=str(OUTPUT), cap_seconds=CAP_SECONDS, compute_ledger=ledger_path, compute_ledger_sha256=ledger_hash,
        reservation_journal=journal_path, render_count=104, unique_reference_views=4, unique_query_exposures=8,
        render_cost_basis=dict(original_render_seconds_sum=render_seconds,
                               original_render_seconds_peak=max(row["original_render_seconds"] for row in renders.values()),
                               original_centre_ray_seconds_sum=ray_seconds, factor_two_projection_seconds=factor_two,
                               startup_and_output_margin_seconds=CAP_SECONDS-factor_two,
                               baseline_available_before_dispatch_seconds=left, compute_reserved_seconds=0),
        bindings=bindings, public_predecessor_bindings=published, cases=selected,
        renders=sorted(renders.values(), key=lambda row: (row["unit_index"], row["role"], row["light_index"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build()
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
