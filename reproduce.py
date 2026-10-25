"""Fetch pinned assets/weights and reproduce the frozen evaluation in a new directory."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from dataset import fetch,validate_manifest
from fetch_models import fetch as fetch_weights,verify_weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--offline',action='store_true');p.add_argument('--split',choices=['train','dev','test','all'],default='all');p.add_argument('--test',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent;os.chdir(root)
    manifest=json.loads(Path('lunar_manifest.json').read_text());lock=json.loads(Path('model_lock.json').read_text())
    if args.offline:
        validate_manifest(manifest,root);verify_weights('data/models',lock)
    else:
        fetch(manifest,root);fetch_weights('data/models',lock)
    if args.test:
        subprocess.run([sys.executable,'-m','pytest','-q'],check=True,env=dict(os.environ,RUN_LEARNED='1'))
    subprocess.run([sys.executable,'evaluate.py','--split',args.split,'--output',args.output],check=True)


if __name__=='__main__':main()
