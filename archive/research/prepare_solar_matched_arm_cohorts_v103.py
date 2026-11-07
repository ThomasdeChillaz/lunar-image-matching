"""Predeclare paired modern-comparison development subsets before inference."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from audit_supplied_sun_support_v102 import verified_json
from solar_evaluation_v70 import validate_manifest


BASIS = "reports/solar-preselection-compact-controls-cost-pilot-v86.json"
BASIS_SHA = "9e14f9c1b95eb12a796ec0533e8c9e50af292a86895e763912a91f786f6b0912"
SEED = 20261104
TRIPLES = (((0, 0), (0, 17), (8, 1)), ((16, 16), (16, 1), (8, 17)))
FAMILIES = ("training_motion", "unseen_roll")
LIGHT_COUNTS = {96: (8, 8), 64: (8, 4), 48: (4, 4)}


def prepare():
    basis = verified_json(BASIS, BASIS_SHA)
    manifest = verified_json(basis["manifest"], basis["input_sha256"][basis["manifest"]])
    geometry = verified_json(basis["geometry_audit"], basis["input_sha256"][basis["geometry_audit"]])
    cases = validate_manifest(manifest, geometry)
    if len(cases) != 384 or [row["case_id"] for row in cases] != basis["case_ids"]:
        raise ValueError("Require the complete original eligible development cohort")
    pilot_ids = set(basis["pilot_case_ids"])
    pilot_units = {row["unit_index"] for row in cases if row["positive"] and row["case_id"] in pilot_ids}
    if len(pilot_ids) != 8 or len(pilot_units) != 4:
        raise ValueError("Require the original paired qualification pilot")
    units = [u for u in geometry["units"] if u["split"] == "model_selection"
             and u["eligible"] and u["unit_index"] not in pilot_units]
    terrain_groups = sorted(manifest["terrain_groups"])
    positive_keys = {size: set() for size in LIGHT_COUNTS}
    for i, terrain in enumerate(terrain_groups):
        retained_family = i % 2
        retained_triple = (i // 2) % 2
        for j, family in enumerate(FAMILIES):
            choices = [u["unit_index"] for u in units if u["component"] == terrain and u["family"] == family]
            if not choices:
                raise ValueError("Both camera families must remain eligible outside qualification")
            unit = min(choices, key=lambda n: sha256(f"{SEED}:{terrain}:{family}:{n}".encode()).digest())
            triple = TRIPLES[retained_triple if j == retained_family else 1 - retained_triple]
            keys = [(unit, a, b) for a, b in triple]
            positive_keys[96].update(keys)
            positive_keys[64].update((keys[0], keys[1 + ((i // 4) % 2)]))
            if j == retained_family:
                positive_keys[48].update(keys)

    family_by_unit = {u["unit_index"]: u["family"] for u in units}
    subsets = {}
    for size, keys in positive_keys.items():
        positive_ids = {row["case_id"] for row in cases if row["positive"]
                        and (row["unit_index"], row["light_a"], row["light_b"]) in keys}
        rows = [row for row in cases if row["case_id"] in positive_ids
                or row.get("matched_positive_case") in positive_ids]
        positives = [row for row in rows if row["positive"]]
        negatives = [row for row in rows if not row["positive"]]
        if (len(rows) != size or len(positives) != size//2 or {r["case_id"] for r in rows} & pilot_ids
                or {r["matched_positive_case"] for r in negatives} != positive_ids):
            raise ValueError("Keep paired negatives and exclude qualification cases")
        lights = Counter((r["light_a"], r["light_b"]) for r in positives)
        same, changed = LIGHT_COUNTS[size]
        if (len(lights) != 6 or any(n != (same if a == b else changed) for (a,b), n in lights.items())
                or set(Counter(r["component"] for r in positives).values()) != {size//16}
                or Counter(family_by_unit[r["unit_index"]] for r in positives)
                    != {family: size//4 for family in FAMILIES}):
            raise ValueError("Require balanced lighting, reference terrain and camera-family counts")
        subsets[size] = dict(
            case_ids=[row["case_id"] for row in rows],
            positives=len(positives), nonoverlaps=len(negatives),
            positive_lighting=[dict(reference=a, query=b, cases=n) for (a,b), n in sorted(lights.items())],
            negative_endpoints=[dict(case_id=r["case_id"], reference=r["component"], query=r["query_component"])
                                for r in negatives],
        )

    if not all(set(subsets[size]["case_ids"]) <= set(subsets[96]["case_ids"]) for size in (48,64)):
        raise ValueError("Smaller comparison designs must be subsets of 96 cases")
    return dict(
        scope="Metadata-only candidate subsets for matched modern baseline comparison on observed development terrain.",
        original_full_case_count=len(cases),
        basis=BASIS, basis_sha256=BASIS_SHA,
        manifest=basis["manifest"], manifest_sha256=basis["input_sha256"][basis["manifest"]],
        geometry_audit=basis["geometry_audit"], geometry_audit_sha256=basis["input_sha256"][basis["geometry_audit"]],
        seed=SEED, terrain_groups=terrain_groups, untouched_calibration_and_final=True,
        sizing_rule="Choose the largest of 96, 64 and 48 that fits measured same-worker worst-case cost before inference. If none fits, report the cost gate.",
        pilot_case_ids=sorted(pilot_ids), excluded_qualification_reference_units=sorted(pilot_units),
        subsets=subsets, execution_authorized=False,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = prepare()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
