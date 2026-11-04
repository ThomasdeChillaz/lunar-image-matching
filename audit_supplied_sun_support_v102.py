"""Audit angular support of the known-Sun control using frozen metadata only."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np

from lunar_known_sun_composite_v102 import EVEN_LIBRARY, plan_supplied_sun


PROTOCOL = Path("reports/solar-full-direct-render-v92.json")
PROTOCOL_SHA256 = "9421f06bdaad9d3b8b9da8e424d4d21b6e5ceda90f6a1d228848842d24c0f909"


def verified_json(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen metadata differs")
    return json.loads(raw)


def audit():
    protocol = verified_json(PROTOCOL, PROTOCOL_SHA256)
    manifest = verified_json(protocol["manifest"], protocol["input_sha256"][protocol["manifest"]])
    profile = protocol["profiles"][0]
    library_spec = profile["render_evidence"]
    catalogue = verified_json(library_spec["manifest"], library_spec["manifest_sha256"])
    if (manifest["split"] != "model_selection" or manifest["calibration_or_final_access"]
            or catalogue["calibration_or_final_access"] or catalogue["query_role_images_used"]):
        raise ValueError("Require the frozen reference-only development cohort")
    libraries = {row["unit_index"]: {e["light_index"]:e["sun_world"] for e in row["entries"]
                                     if e["light_index"] in EVEN_LIBRARY}
                 for row in catalogue["libraries"]}
    if len(libraries) != 32 or len(manifest["cases"]) != 384:
        raise ValueError("Unexpected frozen camera or pair count")
    rows = []
    for case in manifest["cases"]:
        target = case["secondary_known_query_sun_world"]
        lights = libraries[case["unit_index"]]
        plan = plan_supplied_sun(lights, target)
        horizontal_error = None
        if plan.component_indices:
            reconstructed = np.asarray(plan.weights) @ np.asarray([lights[i][:2] for i in plan.component_indices])
            horizontal_error = float(np.linalg.norm(reconstructed - np.asarray(target)[:2]))
            if horizontal_error > 1e-10:
                raise ValueError("Supplied-Sun horizontal composition failed")
        rows.append(dict(case_id=case["case_id"], positive=case["positive"],
                         odd_query_light=case["light_b"]%2 == 1, terrain=case["component"],
                         query_terrain=case["query_component"], nearest_index=plan.nearest_index,
                         nearest_angle_deg=plan.nearest_angle_deg, component_indices=list(plan.component_indices),
                         weights=list(plan.weights), vertical_residual=plan.vertical_residual,
                         horizontal_error=horizontal_error, horizontal_condition=plan.horizontal_condition))
    strata = {}
    for positive in (True, False):
        for odd in (True, False):
            group = [row for row in rows if row["positive"] == positive and row["odd_query_light"] == odd]
            nearest_angles = [row["nearest_angle_deg"] for row in group]
            composite = [row for row in group if len(row["component_indices"]) == 2]
            key = f"{'positive' if positive else 'nonoverlap'}_{'odd' if odd else 'even'}"
            strata[key] = dict(
                cases=len(group), exact_single_template=sum(len(r["component_indices"]) == 1 for r in group),
                supported_two_reference=len(composite),
                no_horizontal_bracket=sum(not r["component_indices"] for r in group),
                nearest_angle_deg=[min(nearest_angles),median(nearest_angles),max(nearest_angles)],
                max_supported_horizontal_condition=max((r["horizontal_condition"] for r in composite),default=None),
            )
    return dict(scope="Declared secondary supplied-Sun metadata only. No optical inputs, matching or pose inference.",
                protocol_sha256=PROTOCOL_SHA256, library_sha256=library_spec["manifest_sha256"],
                strata=strata,rows=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    result=audit()
    with args.output.open("x") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result["strata"],indent=2))
