"""Whitelisted saved-output capsule; current private execution authority excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/solar-blind-v142"
RUN = ROOT / "runs/solar-blind-v140"


def build(destination):
    derived = {"capsule.json", "capsule-replay.json", "publication-verification.json"}
    files = {p for p in REPORT.iterdir() if p.is_file() and p.name not in derived
             and p.suffix in (".py", ".json", ".md", ".png")}
    files.update(RUN/name for name in ("predictions.json", "PREDICTIONS_SEALED.json", "scores.json", "replay.json",
        "timing.json", "timing-runtime.json", "predict-runtime.json", "score-runtime.json", "replay-runtime.json",
        "timing.log", "predict.log", "score.log", "replay.log"))
    files.update(ROOT/name for name in ("requirements-lock.txt", "LICENSE", "reports/solar-blind-v140/protocol.json",
        "reports/research_requirements_status_v142.json"))
    records = {}
    for path in sorted(files):
        name = str(path.relative_to(ROOT))
        if not path.is_file() or path.is_symlink() or any(word in name.lower() for word in
            ("-allowance/", "ledger", "execution", "ticket", "outreach", "authorization")):
            raise ValueError("Missing or forbidden payload: " + name)
        records[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(schema="blind_comparison_replay_capsule_v142", files=records,
        prerequisite_release="https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-blind-v139-cost-gate",
        prerequisite_assets=["solar-blind-v139-clean-evidence-part1.zip", "solar-blind-v139-clean-evidence-part2.zip"],
        inference_inputs_and_frozen_sources_from_parent=True, no_private_authority=True,
        new_training_or_final_data=False, saved_output_replay_only=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(records):
            info = zipfile.ZipInfo(name, (2026, 11, 6, 11, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT/name).read_bytes())
        archive.writestr("BLIND_REPLAY_INVENTORY_V142.json", json.dumps(manifest, indent=2)+"\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+".sha256").write_text(digest+"  "+destination.name+"\n")
    return dict(archive=destination.name, bytes=destination.stat().st_size, sha256=digest,
        payload_files=len(records), no_private_authority=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
