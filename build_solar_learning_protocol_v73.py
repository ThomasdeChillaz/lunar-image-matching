"""Prepare the next concrete learning protocol; never publish or run it."""
import argparse,json
from solar_learning_pipeline_v73 import initialize,prepare_next

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    init=sub.add_parser('initialize');init.add_argument('--settings',required=True);init.add_argument('--contract',required=True)
    nxt=sub.add_parser('prepare-next');nxt.add_argument('--contract',required=True)
    a=p.parse_args();result=initialize(a.settings,a.contract)if a.command=='initialize'else prepare_next(a.contract)
    print(json.dumps(result,indent=2,allow_nan=False))
