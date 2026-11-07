"""Describe grayscale support in the replayed development queries without matching."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image


AUDIT = Path("reports/solar_crosslight_failure_audit_v100.json")
REPLAY = Path("reports/solar_full_crosslight_rescue_replay_v99.json")
PROTOCOL = Path("reports/solar-full-crosslight-rescue-dispatch-v99.json")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def inspect(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("A query PNG differs from the frozen development input")
    with Image.open(path) as frame:
        if frame.format != "PNG" or frame.size != (512, 512):
            raise ValueError("Require original native 512×512 PNG content")
        image = np.asarray(frame.convert("L"))
    histogram = np.bincount(image.reshape(-1), minlength=256)
    cdf = histogram.cumsum()
    quantiles = np.searchsorted(cdf, np.ceil(image.size * np.array([.05, .50, .95, .99])))
    return dict(unique_codes=int((histogram > 0).sum()), nonzero_pixels=int(image.size-histogram[0]),
                uniform=int((histogram > 0).sum()) == 1, all_black=int(histogram[0]) == image.size,
                grayscale_code_min=int(image.min()), grayscale_code_max=int(image.max()),
                p05=int(quantiles[0]), p50=int(quantiles[1]), p95=int(quantiles[2]), p99=int(quantiles[3]),
                adjacent_code_change_fraction=float(
                    ((image[1:, :] != image[:-1, :]).sum() + (image[:, 1:] != image[:, :-1]).sum())
                    / (2 * 512 * 511)
                ))


def summarize(rows):
    return dict(positive_case_count=len(rows), unique_query_files=len({r["query"] for r in rows}),
                uniform_cases=sum(r["uniform"] for r in rows), all_black_cases=sum(r["all_black"] for r in rows),
                median_grayscale_codes=median(r["unique_codes"] for r in rows),
                median_p95=median(r["p95"] for r in rows),
                median_adjacent_code_change_fraction=median(r["adjacent_code_change_fraction"] for r in rows))


def analyze():
    audit, replay, protocol = (read(path) for path in (AUDIT,REPLAY,PROTOCOL))
    if (replay["passed"] is not True or replay["replayed_rows"] != 768
            or replay["protocol_sha256"] != sha(PROTOCOL)
            or audit["protocol_sha256"] != sha(PROTOCOL)
            or audit["mechanism"]["exact_terminal_replay_sha256"] != sha(REPLAY)):
        raise ValueError("Require the completed v99 terminal replay and exact failure audit")
    path = Path(protocol["manifest"])
    if sha(path) != protocol["input_sha256"][str(path)]:
        raise ValueError("Frozen development case manifest differs")
    manifest = read(path)
    if manifest["split"] != "model_selection" or manifest["calibration_or_final_access"]:
        raise ValueError("Only observed development inputs are allowed")
    cases = {case["case_id"]:case for case in manifest["cases"] if case["positive"]}
    rows, cache = [], {}
    for item in audit["mechanism"]["cases"]:
        if not item["positive"]:
            continue
        case = cases.pop(item["case_id"])
        query = case["query"]
        if query not in cache:
            cache[query] = inspect(Path(query),case["input_sha256"][query])
        pooled = item["pooling"]
        rows.append(dict(case_id=case["case_id"],terrain=case["component"],query=query,
                         stage=item["stage"],changed_light=item["changed_light"],
                         pooled_distinct_pairs=None if pooled is None else pooled["distinct_pairs"],**cache[query]))
    if cases or len(rows)!=192:
        raise ValueError("Retain every positive development pair")
    groups = defaultdict(list)
    for row in rows:
        groups[row["stage"]].append(row)
        if row["pooled_distinct_pairs"] == 0:
            groups["miss_zero_distinct_pairs"].append(row)
    return dict(scope="Descriptive 8-bit optical texture on repeated development queries only. No matcher, "
                      "ground-truth shadow labels, acceptance threshold, or held-out access.",
                protocol_sha256=sha(PROTOCOL),replay_sha256=sha(REPLAY),
                summary={group:summarize(group_rows) for group,group_rows in groups.items()},rows=rows)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    result=analyze()
    with args.output.open("x") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps(result["summary"],indent=2))
