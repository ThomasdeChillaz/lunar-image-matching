"""Uncalibrated experimental native lunar localization; no query truth inputs.

match_terrain.py is unchanged. Public pose is null until a separately frozen,
validated acceptance policy exists. --research-output exposes raw hypotheses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from compact_lunar import CompactDescriptor
from solar_response_v63 import LightingResponseModel
from solar_frontend_v63 import ReferenceGeometry,SolarMatcher


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v)for v in value]
    return value


def public_result(result,research=False):
    output=dict(model_version='solar-experimental-v63',inference_revision=result.get('inference_revision','legacy-v63'),accepted=False,pose=None,
        rejection_reason=result.get('reason')or'uncalibrated_experimental_method',
        correspondence_convention='reference-to-query native512pixel centres',
        reference_xy=jsonable(result.get('reference_xy',np.empty((0,2)))),query_xy=jsonable(result.get('query_xy',np.empty((0,2)))),
        estimated_sun_world=jsonable(result.get('sun_enu'))if result.get('solar_supported',False)else None,
        solar_supported=bool(result.get('solar_supported',False)),elapsed_seconds=result.get('elapsed_seconds'),
        total_parameters=result.get('total_parameters'),acceptance_policy=None)
    if research:output['raw_research']=jsonable(result)
    return output


def load_matcher(checkpoint,configuration,allow_untrained=False):
    if configuration.get('inference_revision','solar-v68')!='solar-v68':raise ValueError('Configuration requires a different inference revision')
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if configuration.get('model_version')!='solar-experimental-v63'or configuration.get('checkpoint_sha256')!=digest:raise ValueError('Versioned configuration must pin the exact checkpoint bytes')
    z=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not z.get('training_complete',False)and not allow_untrained:raise ValueError('Untrained models require explicit software-control opt-in')
    coarse=CompactDescriptor(False);fine=CompactDescriptor(False)
    if z.get('format')=='solar-response-v63':
        response=LightingResponseModel(coarse);response.load_state_dict(z['model'],strict=True)
        return SolarMatcher(response.coarse,response.coarse,response,configuration.get('pooled',False),configuration.get('proposal_angles'))
    if z.get('format')!='compact-pair-v63'or configuration.get('response',False):raise ValueError('Declared complete coarse/fine or learned-response checkpoint required')
    coarse.load_state_dict(z['coarse'],strict=True);fine.load_state_dict(z['fine'],strict=True)
    return SolarMatcher(coarse,fine,proposal_angles=configuration.get('proposal_angles'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in('reference','query','reference-geometry','query-intrinsics','config','checkpoint','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--solar-lookup-protocol',help='Published reference-only lazy solar-ray lookup protocol')
    p.add_argument('--publication-commit',help='Commit pinning lazy lookup protocol and complete physical source closure')
    p.add_argument('--blender-executable',help='Official Blender Python/BVH executable for bounded physical lookups')
    p.add_argument('--reference-sun',help='Optional JSON unit world-frame Sun metadata for the REFERENCE image')
    p.add_argument('--known-sun',help='Optional JSON world-frame three-vector; secondary metadata-supplied track only')
    p.add_argument('--research-output',action='store_true');p.add_argument('--allow-untrained-software-control',action='store_true')
    a=p.parse_args(argv);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(1)
    config=json.loads(Path(a.config).read_text());allowed={'model_version','checkpoint_sha256','solar_mode','response','active','budget','policy','pooled','inference_revision','proposal_angles'}
    if set(config)-allowed:raise ValueError('Unknown configuration keys are forbidden')
    matcher=load_matcher(a.checkpoint,config,a.allow_untrained_software_control)
    geometry=ReferenceGeometry.load(a.reference_geometry)
    images=[]
    for path in(a.reference,a.query):
        with Image.open(path)as im:images.append(np.asarray(im.convert('L')).copy())
    k=np.asarray(json.loads(Path(a.query_intrinsics).read_text()),float)
    known=None if a.known_sun is None else json.loads(Path(a.known_sun).read_text())
    reference_sun=None if a.reference_sun is None else json.loads(Path(a.reference_sun).read_text())
    lookup_args=(a.solar_lookup_protocol,a.publication_commit,a.blender_executable)
    if any(lookup_args)and not all(lookup_args):raise ValueError('Lazy lookup requires protocol, publication commit and Blender executable together')
    provider=None
    if all(lookup_args):
        from solar_lookup_v70 import ReferenceSunLookup
        provider=ReferenceSunLookup.from_protocol(*lookup_args)
    try:
        result=matcher.localize(*images,geometry,k,solar_mode=config.get('solar_mode','shared'),response=config.get('response',False),active=config.get('active',False),budget=config.get('budget',128),policy=config.get('policy','active'),known_sun=known,reference_sun=reference_sun,solar_lookup=provider)
    finally:
        if provider is not None:provider.close()
    output=public_result(result,a.research_output);output['configuration_sha256']=hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    with Path(a.output).open('x')as stream:json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    return output


if __name__=='__main__':main()
