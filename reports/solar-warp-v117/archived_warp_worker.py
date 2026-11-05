"""Known-light archived candidate scoring, separated from camera truth."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from prepare_warp_roster_v117 import checked
from warp_candidates import ARM_NAMES, ProjectionBank, build_reference_cloud, rank_pose_bank, reference_pose_bank

SCALES = (32, 16)
CASE_COUNT = 4
ROWS = CASE_COUNT * 2 * 2 * len(SCALES)
COARSE_ROTATION_DEG = 10
COARSE_RANGE_FRACTION = 0.10
TOP = 8


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def put(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def roster(path, expected, role):
    if digest(path) != expected:
        raise ValueError("Roster SHA-256 differs from the experiment authorization")
    p = json.loads(path.read_bytes())
    if (p["role"] != role or p["native_image_shape"] != [512, 512]
            or p["split"] != "model_selection" or p["calibration_or_final_access"] is not False
            or p["score_after_prediction_seal"] is not True
            or [r["tag"] for r in p["exposures"]] != [f"exposure-{i}" for i in range(CASE_COUNT)]):
        raise ValueError("Unexpected archival roster role or exposure set")
    return p["exposures"]


def asset(root, record):
    return checked(root, record["path"], record["sha256"])


def evidence_name(tag, reference_seed, query_seed, scale):
    return f"cells-{tag}-r{reference_seed}-q{query_seed}-s{scale}.npz"


def evidence_path(root, row):
    entry = row["cell_evidence"]
    expected = evidence_name(row["tag"], row["reference_seed"], row["query_seed"], row["cell_pixels"])
    path = root / expected
    if (entry["path"] != expected or path.resolve() != path.absolute()
            or not path.is_file() or digest(path) != entry["sha256"]):
        raise ValueError("Cell evidence must be bound to the sealed prediction row")
    return path


def infer(root, inputs, output):
    import cv2
    from replay_raw_acquisition_v109 import decode, luminance

    cv2.setNumThreads(1)
    banks, rows = {}, []
    evidence_bytes = 0
    for item in inputs:
        if set(item) != {"tag", "reference_geometry", "reference_radiance", "query_radiance", "query_intrinsics"}:
            raise ValueError("The worker accepts reference geometry and query intrinsics only")
        tag = item["tag"]
        with np.load(asset(root, item["reference_geometry"]), allow_pickle=False) as z:
            reference_depth = np.where(z["valid"], z["axial_depth"], np.nan)
            reference_pose, reference_k = z["cam2world"], z["intrinsics"]
        if reference_depth.shape != (512, 512):
            raise ValueError("Require the native reference depth")
        bank = reference_pose_bank(reference_pose)
        banks[tag] = bank.tolist()

        def read_image(image):
            return luminance(decode(asset(root, image).read_bytes(), exr_size=512))

        queries = [read_image(image) for image in item["query_radiance"]]
        query_k = np.asarray(item["query_intrinsics"])
        shared = None
        for ri, image in enumerate(item["reference_radiance"]):
            cloud = build_reference_cloud(read_image(image), reference_depth, reference_k, reference_pose)
            if shared is None:
                shared = ProjectionBank(cloud, query_k, bank, reference_depth.shape, 4)
            for qi, query in enumerate(queries):
                for scale in SCALES:
                    raw = {}
                    result = rank_pose_bank(cloud, query, query_k, bank, cell_pixels=scale,
                                            evidence=raw, projection_bank=shared)
                    path = output.parent / evidence_name(tag, ri, qi, scale)
                    with path.open("xb") as stream:
                        np.savez(stream, **raw)
                    evidence_bytes += path.stat().st_size
                    if evidence_bytes > 512 * 1024**2:
                        raise ValueError("The original evidence storage cap was exceeded")
                    rows.append(dict(tag=tag, reference_seed=ri, query_seed=qi,
                                     cell_evidence=dict(path=path.name, sha256=digest(path)), **result))
    if len(rows) != ROWS:
        raise ValueError("The complete renderer-seed by detector-scale grid is required")
    put(output, dict(scope="Raw pose candidates, known reference illumination, no local correspondences.",
                     banks=banks, rows=rows, evidence_bytes=evidence_bytes,
                     calibration_or_final_access=False))


def score(root, predictor, labels, predictions, prediction_sha, output):
    if digest(predictions) != prediction_sha:
        raise ValueError("Complete predictions must be sealed before scoring")
    data = json.loads(predictions.read_bytes())
    roster_ids = [x["tag"] for x in predictor]
    expected = [(tag, ri, qi, scale) for tag in roster_ids for ri in (0, 1) for qi in (0, 1)
                for scale in SCALES]
    rows = data["rows"]
    keys = [(r["tag"], r["reference_seed"], r["query_seed"], r["cell_pixels"]) for r in rows]
    if keys != expected or set(data["banks"]) != set(roster_ids):
        raise ValueError("Unsealed or incomplete comparison grid")
    sources = {r["tag"]: r["reference_geometry"] for r in predictor}
    hits = {}
    for label in labels:
        tag = label["tag"]
        bank = np.asarray(data["banks"][tag])
        if bank.ndim != 3 or bank.shape[1:] != (4, 4) or not np.isfinite(bank).all():
            raise ValueError("Unexpected camera bank")
        with np.load(asset(root, label["query_geometry"]), allow_pickle=False) as truth:
            target = truth["cam2world"]
        with np.load(asset(root, sources[tag]), allow_pickle=False) as ref:
            native_rays = ref["position"][ref["valid"]] - ref["cam2world"][:3, 3]
        median_range = np.median(np.linalg.norm(native_rays, axis=1))
        if not np.isfinite(median_range) or median_range <= 0:
            raise ValueError("Missing valid reference range")
        traces = np.einsum("nij,ij->n", bank[:, :3, :3], target[:3, :3])
        angles = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))
        shifts = np.linalg.norm(bank[:, :3, 3] - target[:3, 3], axis=1)
        close = (angles <= COARSE_ROTATION_DEG) & (shifts <= COARSE_RANGE_FRACTION * median_range)
        hits[tag] = (label["case_id"], close)
    result = []
    for row in rows:
        case_id, close = hits[row["tag"]]
        with np.load(evidence_path(predictions.parent, row), allow_pickle=False) as raw:
            supported = raw["indices"]
        if (supported.ndim != 1 or not np.issubdtype(supported.dtype, np.integer)
                or len(supported) != row["common_supported_candidates"]
                or (supported < 0).any() or (supported >= len(close)).any()
                or (np.diff(supported) <= 0).any()):
            raise ValueError("Invalid eligible candidate indices")
        prior_hit = next((rank for rank, i in enumerate(supported[:TOP], 1) if close[i]), None)
        top_ranks = {}
        for arm in ARM_NAMES:
            ordered = row["orders"][arm]
            if len(set(ordered)) != len(ordered) or not all(type(i) is int and 0 <= i < len(close) for i in ordered):
                raise ValueError("Invalid proposal indices")
            top_ranks[arm] = next((rank for rank, i in enumerate(ordered[:TOP], 1) if close[i]), None)
        result.append(dict(case_id=case_id, reference_seed=row["reference_seed"], query_seed=row["query_seed"],
                           cell_pixels=row["cell_pixels"], bank_contains_coarse_basin=bool(np.any(close)),
                           bank_coarse_basin_count=int(close.sum()),
                           supported_coarse_basin_count=int(close[supported].sum()),
                           prior_first_top_eight_hit=prior_hit, top_eight_first_hit=top_ranks,
                           common_supported_candidates=row["common_supported_candidates"],
                           reason=row["reason"]))
    put(output, dict(scope="Candidate-basin recall on exposed known-light archival pairs.",
                     threshold_rotation_deg=COARSE_ROTATION_DEG, threshold_range_fraction=COARSE_RANGE_FRACTION,
                     prediction_sha256=prediction_sha, rows=result, accepted_results=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("predict", "score"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--labels-sha256")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--predictions-sha256")
    parser.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    from control_warp_archive_v117 import worker_ticket

    worker_ticket(args)
    inputs = roster(args.inputs, args.inputs_sha256, "inference")
    if args.mode == "predict":
        if any(x is not None for x in (args.labels, args.labels_sha256, args.predictions, args.predictions_sha256)):
            parser.error("The inference process cannot receive scoring paths")
        infer(args.repo.resolve(), inputs, args.output)
    else:
        if any(x is None for x in (args.labels, args.labels_sha256, args.predictions, args.predictions_sha256)):
            parser.error("The scoring process requires a separately sealed prediction file and labels")
        labels = roster(args.labels, args.labels_sha256, "scoring_only")
        score(args.repo.resolve(), inputs, labels, args.predictions, args.predictions_sha256, args.output)
