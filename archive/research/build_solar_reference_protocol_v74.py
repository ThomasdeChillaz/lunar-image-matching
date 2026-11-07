"""Prepare immutable reference protocols; never publish or launch their jobs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['VECLIB_MAXIMUM_THREADS']='2'
import argparse,json
from solar_reference_pipeline_v74 import initialize,prepare_next
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--settings');p.add_argument('--contract',required=True);p.add_argument('--next',action='store_true');a=p.parse_args()
    result=prepare_next(a.contract)if a.next else initialize(a.settings,a.contract)
    print(json.dumps(result,indent=2))
