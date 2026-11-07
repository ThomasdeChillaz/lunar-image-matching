"""Make a whitelisted saved-output capsule, excluding current private authority."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/solar-beam-v144"
FROZEN = ROOT / "reports/solar-beam-v143"
RUN = ROOT / "runs/solar-beam-v143"


def build(destination):
    derived = {"capsule.json", "capsule-replay.json", "publication-verification.json"}
    files = {p for p in REPORT.iterdir() if p.is_file() and p.name not in derived
             and p.suffix in (".py", ".json", ".md", ".png")}
    # v143 source is precisely the public pre-inference freeze; no local test logs.
    files.update(p for p in FROZEN.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md"))
    files.update(RUN / name for name in ("predictions.json", "PREDICTIONS_SEALED.json", "scores.json", "replay.json",
        "timing.json", "timing-runtime.json", "predict-runtime.json", "score-runtime.json", "replay-runtime.json",
        "timing.log", "predict.log", "score.log", "replay.log"))
    files.update(ROOT / name for name in ("requirements-lock.txt", "LICENSE",
        "reports/research_requirements_status_v144.json", "reports/solar-blind-v142/analyze_saved.py"))
    records = {}
    for path in sorted(files):
        name = str(path.relative_to(ROOT))
        if not path.is_file() or path.is_symlink() or any(word in name.lower() for word in
            ("-allowance/", "ledger", "execution", "ticket", "outreach", "authorization")):
            raise ValueError("Missing or forbidden payload: " + name)
        records[name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(schema="persistent_beam_saved_replay_capsule_v144", files=records,
        prerequisite_releases=[
            "https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-blind-v139-cost-gate",
            "https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-blind-v142-development"],
        prerequisite_assets=["solar-blind-v139-clean-evidence-part1.zip", "solar-blind-v139-clean-evidence-part2.zip",
                             "solar-blind-v142-results.zip"],
        prerequisite_sha256={
            "solar-blind-v139-clean-evidence-part1.zip": "cdf3f8c482dfbfeaf99f49886fe1cc8ea74c42ac8753c20b1e7cc765491d7359",
            "solar-blind-v139-clean-evidence-part2.zip": "8f7bdc78393b90ba03e67ef523caa9198a97717eedaf8dc910158bb92fa8aef3",
            "solar-blind-v142-results.zip": "8dc54f86fd8fae0e0bb27492a975605db5bebe1331355515234e085514a77273"},
        no_private_authority=True, saved_output_replay_only=True,
        new_training_or_final_data=False)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(records):
            info = zipfile.ZipInfo(name, (2026, 11, 7, 18, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / name).read_bytes())
        archive.writestr("BEAM_REPLAY_INVENTORY_V144.json", json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "  " + destination.name + "\n")
    return dict(archive=destination.name, bytes=destination.stat().st_size, sha256=digest,
                payload_files=len(records), no_private_authority=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
