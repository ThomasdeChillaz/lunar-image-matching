"""Bounded shared-camera optimization study on the frozen native development cases."""

import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=spec_from_file_location("native_v130_control",HERE.parent/"solar-local-v130/experiment.py")
previous=module_from_spec(spec);spec.loader.exec_module(previous)
base=previous.base
HOME=Path("reports/solar-local-v131")
WORK=Path("runs/solar-local-v131")
CAP=dict(timing=60,comparison=1500)


def build():
    p=deepcopy(base.read("reports/solar-local-v130/protocol.json"))
    p["schema"]="shared_camera_photometric_consensus_v131"
    p["caps"]=CAP.copy()
    for name in ("experiment.py","pose_consensus.py","test_pose_consensus.py","replay_saved.py"):
        file=HOME/name;p["sources"][str(file)]=base.sha(file)
    for name in ("reports/solar-local-v130/protocol.json","reports/solar-local-v130/results.json",
                 "runs/solar-local-v130/comparison/predictions.json","runs/solar-local-v130/comparison/PREDICTIONS_SEALED.json",
                 "runs/solar-local-v130/comparison/COMPLETE.json","runs/solar-local-v129/timing/timing.json"):
        p["input_sha256"][name]=base.sha(name)
    completed=base.read("runs/solar-local-v130/comparison/COMPLETE.json")
    base.check(completed["complete"] and completed["protocol_sha256"]==base.sha("reports/solar-local-v130/protocol.json"),"Preserve completed 0/16 versus 4/16 control")
    p["technique"]="One camera explains multiscale normalized image intensities at fixed reference 3D points. Eight hypotheses, four blur scales and bounded robust six-variable optimization. Final camera still requires independently measured reciprocal native matches and common AP3P."
    p["new_methods"]=["shared_camera_pooled_uniform","shared_camera_pooled_parallax"]
    p["parameters_added"]=0
    p["timing_rule"]="Completed v129 five-arm conservative forecast plus 1.5*32*(worst normalized full-iteration eight-camera optimizer time + two native patch policies), plus20s. Same original cumulative allowance."
    p["historical_control_predictions_must_replay_exactly"]=True
    p["admission"]="New timing and source publication required before full comparison."
    return p


def timing(output):
    import cv2
    import numpy as np
    from pose_consensus import MAX_STEPS,MAX_SEEDS,SIGMAS,measure,refine,pipeline
    from smoke_v126 import generated
    cv2.setNumThreads(1)
    rows=[]
    for variant in (False,True):
        args,valid,truth=generated(variant)
        ref,depth,k,old_pose,query,qk,_=args
        pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2000.
        expected=pose@np.linalg.inv(old_pose)@truth
        seeds=pipeline.coarse.reference_pose_bank(pose,count=8)
        tick=time.perf_counter()
        hypotheses=refine(ref,query,depth,k,pose,qk,seeds,reference_valid=np.ones_like(valid))
        optimizer=time.perf_counter()-tick
        base.check(hypotheses and np.linalg.norm(hypotheses[0]["pose"][:3,3]-expected[:3,3])<2.
                   and np.linalg.norm(hypotheses[0]["pose"][:3,:3]-expected[:3,:3])<.01,"Top-ranked generated camera must be accurate")
        tick=time.perf_counter()
        matched=measure(ref,query,depth,k,pose,qk,hypotheses[0]["pose"][None],reference_valid=np.ones_like(valid))
        measurement=time.perf_counter()-tick
        for row in matched.values():
            base.check(row["pose"] is not None and np.linalg.norm(np.asarray(row["pose"])[:3,3]-expected[:3,3])<2.,"Common estimator must recover generated motion")
        evaluations=sum(sum(h["evaluations"]) for h in hypotheses)
        maximum=MAX_SEEDS*len(SIGMAS)*(1+7*MAX_STEPS)
        rows.append(dict(multidepth=variant,optimizer_seconds=optimizer,measurement_seconds=measurement,
                         evaluations=evaluations,maximum_evaluations=maximum,
                         conservative_optimizer_seconds=optimizer*maximum/evaluations,passed=True))
        print(rows[-1],flush=True)
    old=base.read("runs/solar-local-v129/timing/timing.json")["projected_seconds"]
    projected=old+1.5*32*(max(r["conservative_optimizer_seconds"] for r in rows)+max(r["measurement_seconds"] for r in rows))+20.
    base.put(output,dict(procedural_only=True,archived_images_opened=False,samples=rows,
                         projected_seconds=projected,admitted=projected<=CAP["comparison"]))
    base.check(projected<=CAP["comparison"],"Complete seven-arm comparison exceeds prospective cap")


base.HOME,base.WORK=HOME,WORK
base.CAP=CAP
base.build,base.timing=build,timing
base.__file__=str(Path(__file__).resolve())

if __name__=="__main__":
    base.check(Path.cwd()==ROOT,"Run from repository root")
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("freeze")
    run=sub.add_parser("run");run.add_argument("stage",choices=tuple(CAP));run.add_argument("--commit",required=True)
    worker=sub.add_parser("worker");worker.add_argument("stage",choices=("timing","predict","score","replay"));worker.add_argument("--ticket",type=Path,required=True)
    args=parser.parse_args()
    if args.action=="freeze":base.put(HOME/"protocol.json",build())
    elif args.action=="run":base.execute(args.stage,args.commit)
    else:
        if args.stage=="predict":
            from reference_cache import install
            install()
            import pose_consensus
            pose_consensus.pipeline.run=pose_consensus.run
        base.worker(args.stage,args.ticket)
