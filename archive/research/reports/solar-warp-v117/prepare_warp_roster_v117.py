"""Prepare separate inference and scoring rosters from the sealed v110 archive."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

SOURCE = "reports/solar-raw-crossview-protocol-v110.json"
SOURCE_SHA = "478d90ff7f50ff69d1f030c4875b3a7378b186bf61e4a1471c0bdf7990b5b932"
MANIFEST = "runs/solar-reference-v80-full/case-assets/manifest.json"
MANIFEST_SHA = "750919f111d69cba13b1560409955f7af2c4369ca6cc526b93bc01579993c3e2"
EXPECTED = ("u061-l00-17-positive", "u013-l00-00-positive",
            "u025-l00-00-positive", "u027-l08-01-positive")


def checked(root, name, expected):
    rel = Path(name)
    path = root / rel
    if rel.is_absolute() or ".." in rel.parts or path.resolve() != path:
        raise ValueError(f"Unscoped path {name}")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Input digest differs for {name}")
    return path


def bind(root, record, *, verify=True):
    name, sha = record["path"], record["sha256"]
    path = checked(root, name, sha) if verify else root / name
    return {"path": name, "sha256": sha, "bytes": path.stat().st_size}


def roster(root, *, verify_scoring=True):
    raw = json.loads(checked(root, SOURCE, SOURCE_SHA).read_bytes())
    manifest = json.loads(checked(root, MANIFEST, MANIFEST_SHA).read_bytes())
    if raw["native_shape"] != [512, 512] or raw["calibration_or_final_access"]:
        raise ValueError("The source must be the original 512-pixel development protocol")
    if tuple(i["case_id"] for i in raw["items"]) != EXPECTED:
        raise ValueError("Unexpected development roster")
    cases = {case["case_id"]: case for case in manifest["cases"]}
    predictor, scorer = [], []
    for i, item in enumerate(raw["items"]):
        c = cases[item["case_id"]]
        if c["split"] != "model_selection" or not c["positive"]:
            raise ValueError("All four archival cases are exposed model-selection positives")
        reference, query = (item["views"][name] for name in ("reference", "query"))
        if any([r["replicate"] for r in v["combined"]] != [0, 1] for v in (reference, query)):
            raise ValueError("Require both ordered source render seeds for each camera")
        tag = f"exposure-{i}"
        predictor.append(dict(
            tag=tag, reference_geometry=bind(root, reference["archive_geometry"]),
            reference_radiance=[bind(root, r) for r in reference["combined"]],
            query_radiance=[bind(root, r) for r in query["combined"]],
            query_intrinsics=c["inference_inputs"]["query_intrinsics"],
        ))
        scorer.append(dict(tag=tag, case_id=item["case_id"], purpose=item["purpose"],
                           query_geometry=bind(root, query["archive_geometry"], verify=verify_scoring),
                           joint_scoring_geometry=bind(root, item["scoring_geometry"], verify=verify_scoring),
                           samples_per_pixel=item["samples_per_pixel"]))
    shared = dict(
        source_path=SOURCE, source_sha256=SOURCE_SHA, authority_sha256=MANIFEST_SHA,
        native_image_shape=[512, 512], split="model_selection",
        calibration_or_final_access=False, score_after_prediction_seal=True,
        scope="Archival same-illumination feasibility with preselected matching-light reference.",
    )
    return dict(shared, role="inference", exposures=predictor), dict(shared, role="scoring_only", exposures=scorer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs, scoring = roster(args.repo.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("inference-inputs.json", inputs), ("scoring-inputs.json", scoring)):
        with (args.output / name).open("x") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
