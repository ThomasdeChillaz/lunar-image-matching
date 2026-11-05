"""Experimental active acquisition without supplied query lighting.

No lunar loader or compute admission is provided by this library. Any new lunar
execution needs a published, bounded protocol. Public uncalibrated poses are null.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from catalogue_v138 import Catalogue, ProjectionCache, rank

_path = HERE.parent/"solar-active-v136/matcher.py"
sys.path.insert(0,str(_path.parent))
_spec = spec_from_file_location("blind_native_acquisition_v138", _path)
active = module_from_spec(_spec); _spec.loader.exec_module(active)
photometric = active.joint.native.control


def initialize(catalogue,query,depth,k,reference_pose,query_k,*,reference_valid,poses=None,cache=None):
    result = rank(catalogue,query,depth,k,reference_pose,query_k,
                  reference_valid=reference_valid,poses=poses,cache=cache)
    hypotheses=[]
    for slot in sorted(set(s["light_slot"] for s in result["seeds"])):
        selected=[s for s in result["seeds"] if s["light_slot"]==slot]
        seeds=result["cameras"][[s["pose_index"] for s in selected]]
        refined=photometric.refine(catalogue.images[slot],query,depth,k,reference_pose,query_k,
                                    seeds,reference_valid=reference_valid)
        for h in refined:
            hypotheses.append(dict(h,light_slot=slot,light_id=catalogue.light_ids[slot],
                catalogue_sun_world=np.asarray(catalogue.sun_world[slot]).tolist(),
                coarse_pose_index=selected[h["seed_index"]]["pose_index"]))
    hypotheses.sort(key=lambda h:(h["score"],h["light_id"],h["seed_index"]))
    return hypotheses,dict(coarse=result["diagnostics"],seeds=result["seeds"],
        scores=result["scores"],photometric_hypotheses=[dict(h,pose=h["pose"].tolist()) for h in hypotheses],
        query_truth_used=False,lighting_selected_from_images=True,
        matched_query_light_supplied=False,solar_visibility_certified=False)


def run(catalogue,query,depth,k,reference_pose,query_k,*,reference_valid,height_direction_reference,
        policies=("uniform","geometry","active"),budget=32,poses=None,cache=None,research_output=False):
    started=perf_counter()
    catalogue.validate()
    if catalogue.images.shape[1:] != (512,512):
        raise ValueError("The experimental matching stage preserves native512 input")
    if (not policies or len(set(policies)) != len(policies) or any(p not in active.POLICIES for p in policies)
            or type(budget) is not int or not 0 <= budget <= active.MAX_EXTRA):
        raise ValueError("Distinct declared policies and a bounded extra-search budget required")
    up=np.asarray(height_direction_reference,float)
    if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up),1.):
        raise ValueError("Explicit unit reference-frame height direction required")
    hypotheses,initialization=initialize(catalogue,query,depth,k,reference_pose,query_k,
        reference_valid=reference_valid,poses=poses,cache=cache)
    # Preserve the existing JSON-safe diagnostic handling, never repair or alter
    # an accepted measurement, matching decision or final pose gate.
    active.joint.native.direction=active.joint.safe_direction
    if hypotheses:
        best=hypotheses[0]
        result=active.run(catalogue.images[best["light_slot"]],query,depth,k,reference_pose,query_k,
            best["pose"],reference_valid=reference_valid,height_direction_reference=height_direction_reference,
            policies=policies,budget=budget)
        estimated=best["catalogue_sun_world"]
    else:
        null=active.joint.frame.solve([],[],depth,k,reference_pose,query_k)
        result=dict(methods={p:dict(null,reason="no_joint_light_camera_initializer") for p in ("joint_no_extra",*policies)})
        estimated=None
    result.update(initialization=initialization,estimated_catalogue_sun_proposal=estimated,
        calibrated_acceptance=False,query_truth_used=False,known_query_sun_input=False,
        shared_solar_visibility_certified=False,online_seconds=perf_counter()-started,
        scope="Discrete reference-catalogue lighting selected from images; no lunar performance established")
    return active.public_result(result,research_output=research_output)
