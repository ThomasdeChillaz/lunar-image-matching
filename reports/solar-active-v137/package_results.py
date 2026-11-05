"""Create a whitelisted public replay capsule; never include current authority."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HOME=Path("reports/solar-active-v137")
RUN=Path("runs/solar-active-v137")
INVENTORY="ACTIVE_REPLAY_INVENTORY_V137.json"
HISTORICAL_RECEIPTS={
    "runs/solar-local-v129/timing/execution.json",
    "runs/solar-local-v132/timing/execution.json",
    "runs/solar-local-v133/comparison/execution.json",
}


def build(destination):
    protocol=json.loads((HOME/"protocol.json").read_text())
    files=set(protocol["sources"]) | set(protocol["input_sha256"])
    files.update(str(p) for p in HOME.iterdir() if p.is_file() and p.suffix in {".py",".json",".md",".png"})
    files.update(["LICENSE","THIRD_PARTY.md","requirements-lock.txt","reports/research_requirements_status_v137.json"])
    files.update(str(p) for p in Path("output/pdf").glob("lunar-active-acquisition-v137-*.pdf"))
    for stage, names in {
        "comparison":["COMPLETE.json","PREDICTIONS_SEALED.json","predictions.json","scores.json","replay.json",
                      "predict-runtime.json","score-runtime.json","replay-runtime.json","predict.log"],
        "timing":["COMPLETE.json","timing.json","timing-runtime.json","timing.log"],
    }.items():
        files.update(str(RUN/stage/name) for name in names)
    for name,digest in {**protocol["sources"],**protocol["input_sha256"]}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Changed frozen input/source: "+name)
    records={}
    for name in sorted(files):
        p=Path(name)
        if p.is_absolute() or ".." in p.parts or p.is_symlink() or not p.is_file():
            raise ValueError("Invalid or missing payload: "+name)
        if any(word in name.lower() for word in ("-allowance/","ledger","ticket","authorization","outreach")):
            raise ValueError("Private execution or correspondence path excluded: "+name)
        if "execution" in name.lower() and name not in HISTORICAL_RECEIPTS:
            raise ValueError("Only the three frozen historical receipts may be distributed")
        records[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    manifest=dict(schema="active_replay_inventory_v137",files=records,
        inference_source="48c6e9ac4f15a1787a3bfdce6a5b84de54090763",
        protocol_sha256=hashlib.sha256((HOME/"protocol.json").read_bytes()).hexdigest(),
        private_current_authority_included=False,historical_expired_receipts=sorted(HISTORICAL_RECEIPTS),
        no_calibration_or_final_data=True,saved_output_replay_only=True)
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(records):
            info=zipfile.ZipInfo(name,(2026,11,5,22,30,0)); info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            archive.writestr(info,Path(name).read_bytes())
        archive.writestr(INVENTORY,json.dumps(manifest,indent=2)+"\n")
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+".sha256").write_text(digest+"  "+destination.name+"\n")
    return dict(archive=destination.name,bytes=destination.stat().st_size,sha256=digest,payload_files=len(records),
                current_authority_distributed=False,historical_receipts=3)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),indent=2))
