"""Prepare artifact-bound evaluation protocols; never publish or execute them."""
import argparse,json
from solar_evaluation_protocol_v74 import prepare,verify_publication

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('prepare');build.add_argument('--settings',required=True);build.add_argument('--phase',choices=['cost_pilot','full'],required=True);build.add_argument('--output',required=True)
    check=sub.add_parser('verify-publication');check.add_argument('--protocol',required=True);check.add_argument('--commit',required=True)
    a=parser.parse_args();result=prepare(a.settings,a.phase,a.output)if a.command=='prepare'else verify_publication(a.protocol,a.commit)
    print(json.dumps(result,indent=2,allow_nan=False))
