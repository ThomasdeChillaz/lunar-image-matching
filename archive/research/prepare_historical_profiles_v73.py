"""Read metadata/checkpoint hashes only; no model loading or image decoding."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import historical_compact_adapters_v73 as h
ROOT=Path.cwd()
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
coarse='runs/compact-v2-expanded-20261030';original='runs/contrastive-terrain-train-v1';context='runs/north-v61-context-study/cross_adaptation-20261101'
configs={h.METHODS[0]:dict(model_directory=original,checkpoint='selected',input_profile='raw_native512'),
 h.METHODS[1]:dict(model_directory=coarse,input_profile='coarse128_native512',surface_iterations=2),
 h.METHODS[2]:dict(coarse_directory=coarse,selected_checkpoint=context+'/selected.pt',variant='cross_adaptation',input_profile='coarse128_native512',surface_iterations=2)}
legacy_surface=read(ROOT/'runs/north-v39-full-baselines/protocol.json')['source_sha256']
legacy_context=read(ROOT/'runs/north-v61-context-study/protocol.json')['source_sha256']
critical=h.SURFACE_SOURCES|h.CONTEXT_SOURCES|{'contrastive_terrain.py','lunar_geometry_v5.py','lunar_geometry_v2.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py'}
profiles=[];catalog=[]
for method,c in configs.items():
 directory=Path(c['coarse_directory']if method==h.METHODS[2]else c['model_directory'])
 artifacts={str(directory/n):sha(ROOT/directory/n)for n in('selected.pt','training.json','COMPLETE.json')}
 if method==h.METHODS[2]:
  for name in('selected.pt','summary.json','COMPLETE.json','selected_controls.json'):
   path=str(Path(context)/name);artifacts[path]=sha(ROOT/path)
  for path in('reports/north_context_completed_results_v61.json','reports/north_context_study_audit_v61.json','runs/north-v61-context-study/protocol.json'):artifacts[path]=sha(ROOT/path)
 legacy=legacy_context if method==h.METHODS[2]else legacy_surface
 algorithm={n:v for n,v in legacy.items()if n in critical}if method!=h.METHODS[0]else{'contrastive_terrain.py':read(ROOT/directory/'training.json')['source_sha256']['contrastive_terrain.py']}
 source={n:sha((OUT if(OUT/n).exists()else ROOT)/n)for n in sorted(h.required_sources(method))}
 profile=dict(name=method,backend='historical_compact',method=method,prepared_only=True,available=False,
  unavailable_reason='Named historical weights exist; fresh source-pinned native512 adapter control outputs are not produced by this preparation task.',
  publication_commit='REQUIRES_PUBLIC_ADAPTER_FREEZE',cpu_threads=2,native_size=512,calibration_or_final_access=False,
  configuration=c,artifact_sha256=artifacts,source_sha256=source,historical_algorithm_source_sha256=algorithm,
  native_control={k:f'reports/{method}-native-control-{k}-v73.json'for k in('protocol','outputs','receipt')})
 for k in('protocol','outputs','receipt'):profile['native_control'][k+'_sha256']='REQUIRES_ACTUAL_NATIVE_CONTROL'
 profiles.append(profile)
 selected=ROOT/(c['selected_checkpoint']if method==h.METHODS[2]else str(directory/'selected.pt'))
 catalog.append(dict(method=method,checkpoint=str(selected.relative_to(ROOT)),checkpoint_sha256=sha(selected),base_artifacts_present=True,
  historical_algorithm_source_currently_matches=all(sha(ROOT/n)==v for n,v in algorithm.items()),
  native_adapter_control_complete=False,scientifically_available_now=False,
  historical_scope='Historical development/control only; new native camera/light evaluation not run.'))
(OUT/'historical_profiles_prepared_v73.json').write_text(json.dumps(dict(prepared_only=True,profiles=profiles),indent=2)+'\n')
(OUT/'historical_artifact_catalog_v73.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),methods=catalog,
 raw_historical_context_result={'correct_raw_poses':15,'positive_cases':96,'calibrated_acceptance_evaluated':False},
 new_forward_or_render_performed=False),indent=2)+'\n')
print(json.dumps(catalog,indent=2))
