"""Inventory completed reference-only tables after the full96-unit geometry gate."""
import argparse
import json
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['VECLIB_MAXIMUM_THREADS']='2'
from solar_eval_data_v69 import build_reference_index

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True)
    a=p.parse_args();r=build_reference_index(a.protocol,a.publication_commit)
    print(json.dumps({k:v for k,v in r.items()if k!='units'},indent=2))
