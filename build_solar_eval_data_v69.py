"""Build only published, fully geometry-qualified development case assets."""
import argparse
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
import json
from solar_eval_data_v69 import build

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True)
    a=p.parse_args();r=build(a.protocol,a.publication_commit)
    print(json.dumps({k:v for k,v in r.items()if k!='cases'},indent=2))
