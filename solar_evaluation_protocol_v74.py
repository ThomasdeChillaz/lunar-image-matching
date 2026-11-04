"""Artifact-driven development evaluation freezes. Never runs scientific work.

Uses the existing qualified manifest, completed-learning, profile, ranking and
measured-cost helpers. Reads metadata/checkpoint identities, never image pixels,
models, rays or final/calibration cases. Every concrete output still requires a
separate byte-identical source publication and bounded controller invocation.
"""
import ast,copy,hashlib,json,math,subprocess
from datetime import datetime,timedelta,timezone
from pathlib import Path
import solar_learning_pipeline_v73 as learning
import solar_evaluation_v70 as evaluation
import run_solar_evaluation_v70 as runner

MODES={'checkpoint_selection_sh':('checkpoint_selection','sh'),
       'checkpoint_selection_pooled':('checkpoint_selection','pooled'),
       'principal_ablations':('principal_ablations',None),
       'additional_controls_sh':('additional_controls','sh'),
       'additional_controls_pooled':('additional_controls','pooled')}
OWN_SOURCES={'solar_evaluation_protocol_v74.py','build_solar_evaluation_protocol_v74.py','solar_learning_pipeline_v73.py','verify_solar_development_v68.py','solar_reference_pipeline_v74.py'}
DEFAULT_CONFIG=dict(proposal_angles=[0.,90.,180.,270.],pose_initialization='strict3',inference_revision='solar-v68')
sha=evaluation.sha
read=evaluation.read
utc=learning.utc


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def integer(value,maximum):
    if type(value)is not int or not 0<value<=maximum:raise ValueError('Positive finite predeclared integer cap required')
    return value


def source_closure(repository_root):
    root=Path(repository_root).resolve();todo=list(runner.REQUIRED_SOURCES|OWN_SOURCES);seen=set()
    def imports(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return
        if isinstance(node,ast.Import):
            for name in node.names:yield name.name.split('.')[0]+'.py'
        if isinstance(node,ast.ImportFrom)and node.module:yield node.module.split('.')[0]+'.py'
        for child in ast.iter_child_nodes(node):yield from imports(child)
    while todo:
        name=todo.pop()
        if name in seen:continue
        path=root/name
        if not path.is_file():raise ValueError('Evaluation implementation must be integrated before a concrete freeze:'+name)
        seen.add(name);todo.extend(n for n in imports(ast.parse(path.read_text()))if(root/n).is_file()and n not in seen)
    return {name:sha(root/name)for name in sorted(seen)}


def selection_inventory(audit):
    units=sorted((copy.deepcopy(unit)for unit in audit['units']if unit['split']=='model_selection'),key=lambda unit:unit['unit_index'])
    if len(units)!=32 or len({unit['unit_index']for unit in units})!=32 or any(type(unit['eligible'])is not bool for unit in units):raise ValueError('Original32selection-camera inventory with explicit independent eligibility required')
    groups=sorted({unit['component']for unit in units});eligible=[unit for unit in units if unit['eligible']]
    if len(groups)!=8 or any(sum(unit['component']==group for unit in units)!=4 for group in groups):raise ValueError('Original eight generated terrain groups/four cameras each required')
    eligible_groups=sorted({unit['component']for unit in eligible})
    return dict(terrain_groups=groups,generated_selection_units=units,generated_selection_unit_count=32,eligible_selection_unit_count=len(eligible),excluded_selection_units=[unit for unit in units if not unit['eligible']],eligible_terrain_groups=eligible_groups,zero_eligible_terrain_groups=sorted(set(groups)-set(eligible_groups)))


def qualified_cases(manifest_path,audit_path,auditor_source=None):
    manifest=read(manifest_path);audit=read(audit_path)
    if manifest.get('complete')is not True or manifest.get('geometry_audit_sha256')!=sha(audit_path):raise ValueError('Actual complete manifest must bind the exact full geometry audit')
    auditor_source=Path(auditor_source)if auditor_source is not None else Path.cwd()/'verify_solar_development_v68.py'
    if audit.get('audit_source_sha256')!=sha(auditor_source):raise ValueError('Complete geometry audit must use the exact independently frozen auditor source')
    cases=evaluation.validate_manifest(manifest,audit)
    inventory=selection_inventory(audit);eligible=sorted(unit['unit_index']for unit in inventory['generated_selection_units']if unit['eligible'])
    if sorted(manifest.get('eligible_unit_indices',[]))!=eligible or len(cases)!=12*len(eligible):raise ValueError('All independently eligible cameras and exactly12*N cases required; preserve exclusions rather than shrink/substitute cases')
    for name,value in inventory.items():
        if name in manifest and manifest[name]!=value:raise ValueError('Manifest raw eligibility inventory differs from the independent audit:'+name)
    if set(manifest['reference_precompute_receipts'])!={case['reference_geometry']for case in cases}:raise ValueError('Reference receipt set must match exactly the eligible case archives')
    for geometry,record in manifest['reference_precompute_receipts'].items():
        path=record['receipt_path']
        if sha(path)!=record['sha256']:raise ValueError('Measured reference preparation receipt changed')
        receipt=read(path)
        if receipt.get('complete')is not True or not isinstance(receipt.get('calendar_seconds'),(int,float))or isinstance(receipt.get('calendar_seconds'),bool)or not math.isfinite(receipt['calendar_seconds'])or receipt['calendar_seconds']<=0:raise ValueError('Measured positive complete reference preparation cost required')
    return manifest,audit,cases


def _study_profiles(contract,mode):
    stage=learning.completion(contract,mode+'_study',True)
    profiles=evaluation.checkpoint_profiles(Path(stage['summary']).parent,mode=='pooled',training_protocol=stage['protocol'],watchdog_receipt=stage['watchdog_receipt'])
    if any(profile['training_seed']!=20261101 for profile in profiles):raise ValueError('Primary SH/pooled checkpoint selection requires predefined seed20261101')
    return profiles,stage


def _configuration(settings):
    value=copy.deepcopy(settings['inference_configuration'])
    if set(value)!=set(DEFAULT_CONFIG):raise ValueError('Explicit proposal_angles/pose_initialization/inference_revision only; A/B/C settings come from existing matched helpers')
    from solar_rotation_v71 import proposal_angles
    proposal_angles(value['proposal_angles'])
    if value['pose_initialization']not in('strict3','quantization_aware'):raise ValueError('Unsupported explicit initialization policy')
    expected='solar-v68'if value['pose_initialization']=='strict3'else'solar-v73-quantized-init'
    cached='solar-v84-exact-cache'if value['pose_initialization']=='strict3'else'solar-v84-quantized-init-exact-cache'
    if value['inference_revision']not in(expected,cached):raise ValueError('Declared inference revision disagrees with pose policy')
    return value


def profiles_from_artifacts(settings,contract):
    mode=settings['mode']
    if mode not in MODES:raise ValueError('Explicit supported evaluation stage required')
    purpose,response_mode=MODES[mode];config=_configuration(settings);inputs={};selection=None
    def remember(stage):
        for key in('protocol','summary','watchdog_receipt'):inputs[stage[key]]=sha(stage[key])
    if purpose=='checkpoint_selection':
        profiles,stage=_study_profiles(contract,response_mode);remember(stage)
    else:
        # Existing helper requires BOTH completed studies and independently
        # replayed complete all-eligible selections; it never selects by training loss.
        selection=learning.selection(contract)
        selected={row['mode']:row for row in selection['outcomes']};chosen={}
        for model_mode in('sh','pooled'):
            candidates,stage=_study_profiles(contract,model_mode);remember(stage)
            chosen[model_mode]=next(p for p in candidates if p['training_step']==selected[model_mode]['step'])
            for item in selection['evidence'][model_mode].values():inputs[item['path']]=item['sha256']
        if purpose=='principal_ablations':
            a,b=chosen['sh'],chosen['pooled']
            profiles=evaluation.principal_profiles(a['checkpoint'],b['checkpoint'],a['configuration']['checkpoint_sha256'],b['configuration']['checkpoint_sha256'])
            for profile in profiles:
                owner=b if profile['configuration']['pooled']else a
                profile.update(training_evidence=copy.deepcopy(owner['training_evidence']),training_seed=owner['training_seed'],training_step=owner['training_step'])
        else:
            owner=chosen[response_mode];profiles=evaluation.additional_profiles(owner['checkpoint'],owner['configuration']['checkpoint_sha256'])
            for profile in profiles:
                profile['configuration']['pooled']=response_mode=='pooled'
                profile.update(training_evidence=copy.deepcopy(owner['training_evidence']),training_seed=owner['training_seed'],training_step=owner['training_step'])
    for profile in profiles:
        profile['configuration'].update(copy.deepcopy(config))
        evaluation.validate_profile_training(profile,purpose)
        inputs[profile['checkpoint']]=profile['configuration']['checkpoint_sha256']
        for key in('protocol','summary','watchdog_receipt'):
            path=profile['training_evidence'][key];inputs[path]=profile['training_evidence'][key+'_sha256']
    evaluation.validate_profiles(profiles,purpose)
    return profiles,inputs,selection


def _pilot_ids(cases,explicit):
    if len(explicit)<4 or len(explicit)!=len(set(explicit)):raise ValueError('Four or more explicit distinct pilot cases required')
    by={c['case_id']:c for c in cases}
    if not set(explicit)<=set(by):raise ValueError('Pilot contains a case outside frozen development data')
    chosen=[by[key]for key in explicit]
    if len({c['component']for c in chosen})<2 or not any(c['positive']for c in chosen)or not any(not c['positive']for c in chosen):raise ValueError('Pilot must include two terrains, positives and matched nonoverlap')
    return list(explicit)


def _input_pins(settings_path,settings,contract,manifest,cases,training_inputs):
    pins=dict(training_inputs)
    for path in(settings_path,settings['learning_contract'],settings['manifest'],settings['geometry_audit'],contract['compute_ledger']):pins[str(path)]=sha(path)
    # Consume the producer's already measured seals. No image pixels or scoring
    # NPZs are opened by this metadata builder; runtime rechecks their hashes.
    for case in cases:
        required={case[name]for name in('reference','query','reference_geometry')}
        if case.get('solar_lookup_protocol'):required.add(case['solar_lookup_protocol'])
        if case['positive']:required.add(case['scoring_labels'])
        if not required<=set(case['input_sha256']):raise ValueError('Every frozen image/reference/scoring artifact needs a producer hash')
        for path,h in case['input_sha256'].items():
            if path in pins and pins[path]!=h:raise ValueError('Conflicting case/checkpoint artifact identity')
            pins[path]=h
    for record in manifest['reference_precompute_receipts'].values():pins[record['receipt_path']]=record['sha256']
    return pins


def reference_preparation_from_artifacts(settings,manifest,inventory):
    # Reuse the physical orchestration validator; do not parse geometry NPZs or
    # reconstruct successful ray/watchdog evidence in this evaluation builder.
    from solar_reference_pipeline_v74 import completed_reference_preparation
    evidence=settings['reference_preparation_evidence']
    completed=completed_reference_preparation(evidence['protocol'],evidence['summary'])
    for name,value in inventory.items():
        if completed.get(name)!=value:raise ValueError('Physical preparation must preserve the same raw inventory/exclusions:'+name)
    wanted=set(manifest['reference_precompute_receipts'])
    if set(completed['reference_archives'])!=wanted or sorted(completed['reference_unit_indices'])!=sorted(manifest['eligible_unit_indices']):raise ValueError('Full startup-inclusive physical preparation must cover the exact eligible reference archives')
    outer=completed['outer_calendar_seconds_by_archive'];internal=completed['internal_calendar_seconds_by_archive']
    if set(outer)!=wanted or set(internal)!=wanted:raise ValueError('Every reference needs internal and startup-inclusive preparation cost')
    for path in wanted:
        values=(outer[path],internal[path])
        if any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for v in values)or outer[path]<internal[path]or not math.isclose(internal[path],manifest['reference_precompute_receipts'][path]['calendar_seconds'],rel_tol=0,abs_tol=1e-6):raise ValueError('Physical preparation costs differ from manifest or omit cold startup')
    pins=dict(completed['pinned_inputs'])
    for path in(evidence['protocol'],evidence['summary']):
        if pins.get(path)!=sha(path):raise ValueError('Completed physical preparation protocol/summary must be pinned')
    costs=dict(outer_calendar_seconds_by_archive=outer,internal_calendar_seconds_by_archive=internal,baseline_reserved_seconds=completed['baseline_reserved_seconds'],phase_calendar_seconds=completed['phase_calendar_seconds'],scope=completed['scope'])
    return pins,costs


def lease_fields(cap,lease,created=None):
    cap=integer(cap,learning.MAX_BASELINE);lease=integer(lease,3600);created=created or datetime.now(timezone.utc)
    if created.tzinfo is None:raise ValueError('Timezone-aware creation time required')
    latest=created+timedelta(seconds=lease)
    return dict(created_utc=created.isoformat(),publication_dispatch_lease_seconds=lease,latest_start_utc=latest.isoformat(),absolute_deadline_utc=(latest+timedelta(seconds=cap)).isoformat())


def validate_lease(protocol):
    cap=integer(protocol['max_calendar_seconds'],learning.MAX_BASELINE);lease=integer(protocol['publication_dispatch_lease_seconds'],3600)
    created=utc(protocol['created_utc']);latest=utc(protocol['latest_start_utc'])
    if latest!=created+timedelta(seconds=lease)or utc(protocol['absolute_deadline_utc'])!=latest+timedelta(seconds=cap):raise ValueError('Original evaluation dispatch lease/deadline changed')
    return cap,latest,utc(protocol['absolute_deadline_utc'])


def validate_runtime(protocol,protocol_path,at=None):
    """Future runner admission hook; metadata only and no reservation writes."""
    if protocol.get('builder_schema_version')!=74:raise ValueError('Artifact-driven evaluation freeze schema74required')
    cap,latest,absolute=validate_lease(protocol)
    contract=learning.load_contract(protocol['learning_contract'])
    if sha(protocol['learning_contract'])!=protocol['learning_contract_sha256']or protocol['reservation_journal']!=learning.CANONICAL_JOURNAL or protocol['compute_ledger_sha256']!=contract['compute_ledger_sha256']:raise ValueError('Canonical learning/baseline budget identities changed')
    ledger,journal,charged=learning.journal_state(protocol['compute_ledger'],protocol['compute_ledger_sha256'],protocol['reservation_journal'])
    initial=protocol['baseline_admission']['initial_reservations']
    if journal['reservations'][:len(initial)]!=initial:raise ValueError('Baseline reservation history was deleted or rewritten')
    job=sha(protocol_path)+'/evaluation';own=[r for r in journal['reservations']if r['job_id']==job]
    if len(own)>1 or own and(own[0]['category']!='baseline'or own[0]['seconds']!=cap):raise ValueError('Existing evaluation charge differs from its original cap')
    if not own and cap>learning.remaining(contract)['baseline']:raise ValueError('Evaluation cap exceeds remaining canonical baseline allowance')
    current=at or datetime.now(timezone.utc);admission=Path(protocol['output'])/'watchdog_admission.json'
    if current<utc(protocol['created_utc']):raise ValueError('Evaluation cannot start before its declared creation time')
    if own and admission.exists():
        receipt=read(admission);started=utc(receipt['started_utc']);deadline=utc(receipt['absolute_deadline_utc'])
        if receipt['protocol_sha256']!=sha(protocol_path)or started<utc(protocol['created_utc'])or started>latest or deadline!=min(started+timedelta(seconds=cap),absolute):raise ValueError('Original evaluation watchdog/deadline differs from freeze')
        if current>=deadline:raise ValueError('Original evaluation deadline has expired; preserve the run')
    elif current>latest:raise ValueError('Published evaluation dispatch lease expired; no late start or automatic reset')
    if current>=absolute:raise ValueError('Absolute evaluation deadline expired')
    return dict(cap_seconds=cap,absolute_deadline_utc=absolute.isoformat(),already_reserved=bool(own),canonical_journal=learning.CANONICAL_JOURNAL)


def _published_bytes(protocol_path,protocol,commit):
    if len(commit or'')!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact source publication commit required')
    for name in(str(protocol_path),*protocol['source_sha256']):
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Protocol/source publication barrier not satisfied:'+name)


def _complete_pilot(protocol,pilot_spec,manifest,contract):
    """Consume an ACTUAL sealed, independently replayed measured evaluation."""
    paths={key:pilot_spec[key]for key in('protocol','summary','metric_replay')}
    q,s,replay=[read(paths[key])for key in('protocol','summary','metric_replay')]
    root=Path(paths['summary']).parent;seal=read(root/'COMPLETE.json')
    if q.get('builder_schema_version')!=74 or q['purpose']!='cost_pilot'or q['target_purpose']!=protocol['purpose']or q['profiles']!=protocol['profiles']or q['case_ids']!=protocol['case_ids']or q['tracks']!=protocol['tracks']:raise ValueError('Measured pilot must match the exact all-eligible cases/profiles/track policy')
    if q['reference_preparation_evidence']!=protocol['reference_preparation_evidence']or q['reference_preparation_costs']!=protocol['reference_preparation_costs']:raise ValueError('Measured pilot physical setup evidence/costs differ from full study')
    if q['baseline_admission']['maximum_full_calendar_seconds']!=protocol['baseline_admission']['maximum_full_calendar_seconds']:raise ValueError('Maximum full-study allowance changed after the pilot freeze')
    if q['source_sha256']!=protocol['source_sha256']or s.get('software')!=runner.software_identity()or s.get('comparison_identity')!=runner.comparison_identity(protocol,manifest):raise ValueError('Measured pilot software/source/configuration differs from study')
    if s.get('complete')is not True or s.get('purpose')!='cost_pilot'or s['protocol_sha256']!=sha(paths['protocol'])or s['manifest_sha256']!=sha(protocol['manifest']):raise ValueError('Actual complete source-bound cost pilot required')
    if any((root/name).exists()for name in('FAILED.json','WORKER_FAILED.json'))or seal.get('complete')is not True or seal.get('external_watchdog_passed')is not True or seal['protocol_sha256']!=sha(paths['protocol'])or seal['summary_sha256']!=sha(paths['summary']):raise ValueError('Actual external watchdog completion seal required')
    if replay.get('passed')is not True or replay['summary_sha256']!=sha(paths['summary'])or replay['replayed_rows']!=len(s['rows'])or replay['replay_source_sha256']!=q['source_sha256']['verify_solar_evaluation_v70.py']:raise ValueError('Independent actual pilot metric replay required')
    _pilot_ids(manifest['cases'],q['pilot_case_ids'])
    expected={(p['name'],key,track)for p in q['profiles']for key in q['pilot_case_ids']for track in q['tracks']}
    actual=[(r['profile'],r['case_id'],r['track'])for r in s['rows']]
    if s.get('profiles')!=q['profiles']or s.get('case_ids')!=q['pilot_case_ids']or s.get('cached_cases')!=0 or s.get('fresh_inference_cases')!=len(s['rows']):raise ValueError('Cost pilot must contain the exact fresh model/case invocation set')
    bycase={case['case_id']:case for case in manifest['cases']}
    for row in s['rows']:
        case=bycase.get(row['case_id'])
        if case is None or any(row.get(key)!=case[key]for key in('component','query_component','positive','unit_index','query_unit_index','light_a','light_b')):raise ValueError('Measured pilot case/negative endpoint metadata differs')
        if row.get('reused_case_cache')is not False:raise ValueError('Cost pilot cannot replay cached inference latency')
        if any(isinstance(row.get(key),bool)or not isinstance(row.get(key),(int,float))or not math.isfinite(row[key])or row[key]<0 for key in('case_calendar_seconds','profile_load_seconds'))or row['case_calendar_seconds']<=0:raise ValueError('Finite positive measured case costs required')
    if len(actual)!=len(expected)or set(actual)!=expected:raise ValueError('Actual measured pilot grid is incomplete, duplicated or contains substituted cases')
    execution=read(root/'execution.json')
    if execution['protocol_sha256']!=sha(paths['protocol'])or execution['publication_commit']!=s['source_publication_commit']or execution['software']!=s['software']or execution['final_reserve_access']is not False:raise ValueError('Actual pilot execution/publication identity differs')
    _published_bytes(paths['protocol'],q,s['source_publication_commit'])
    charged=read(root/'reservation.json');watchdog=read(root/'watchdog_admission.json');cap,latest,absolute=validate_lease(q)
    job=sha(paths['protocol'])+'/evaluation';entry=charged['reservation']
    if entry['job_id']!=job or entry['category']!='baseline'or entry['seconds']!=cap or entry.get('refund_on_failure')is not False:raise ValueError('Actual pilot must be fully charged to canonical baseline budget')
    ledger,journal,totals=learning.journal_state(contract['compute_ledger'],contract['compute_ledger_sha256'],contract['reservation_journal'])
    if [r for r in journal['reservations']if r['job_id']==job]!=[entry]:raise ValueError('Actual pilot charge absent from canonical journal')
    started=utc(watchdog['started_utc']);deadline=utc(watchdog['absolute_deadline_utc'])
    if seal.get('original_allowance_seconds')!=cap:raise ValueError('Actual pilot seal changed its original allowance')
    elapsed=seal['observed_calendar_seconds'];science=s['calendar_seconds']
    if watchdog['protocol_sha256']!=sha(paths['protocol'])or watchdog['reservation_sha256']!=sha(root/'reservation.json')or started<utc(q['created_utc'])or started>latest or deadline!=min(started+timedelta(seconds=cap),absolute)or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)for v in(elapsed,science))or not 0<science<=elapsed<=min(cap,(deadline-started).total_seconds()):raise ValueError('Actual pilot violated its original controller deadline')
    cost=runner.measured_cost(s['rows'],len(manifest['cases']),protocol['tracks'],protocol['profiles'])
    pins={str(path):sha(path)for path in [*paths.values(),root/'COMPLETE.json',root/'reservation.json',root/'watchdog_admission.json',root/'execution.json']}
    return cost,pins,paths


def prepare(settings_path,phase,output_path,created=None):
    """Prepare one concrete source-freeze artifact; never publish or execute."""
    if Path(output_path).is_absolute()or'..'in Path(output_path).parts:raise ValueError('Published protocol output must be a repository-relative path')
    s=read(settings_path)
    if s.get('prepared_only',False):raise ValueError('Prepared template needs actual qualified/completed artifacts; no generated success evidence')
    if phase not in('cost_pilot','full')or s['calibration_or_final_access']is not False:raise ValueError('Explicit development-only evaluation phase required')
    contract=learning.load_contract(s['learning_contract'])
    manifest,audit,cases=qualified_cases(s['manifest'],s['geometry_audit'],Path(s['repository_root'])/'verify_solar_development_v68.py')
    if sha(s['geometry_audit'])!=contract['geometry_audit_sha256']:raise ValueError('Learning and evaluation must use the same complete96-unit geometry audit')
    physical_inputs,physical_costs=reference_preparation_from_artifacts(s,manifest,selection_inventory(audit))
    profiles,training_inputs,selection=profiles_from_artifacts(s,contract);purpose=MODES[s['mode']][0]
    if purpose=='checkpoint_selection'and s['tracks']!=['estimated_sun']:raise ValueError('Checkpoint selection is primary-only')
    if s['tracks']not in(['estimated_sun'],list(evaluation.TRACKS)):raise ValueError('Primary track required; known-Sun metadata is a separate optional track')
    pilot_ids=_pilot_ids(cases,s['pilot_case_ids']);sources=source_closure(s['repository_root'])
    if selection is not None:
        for mode,evidence in selection['evidence'].items():
            selected_protocol=read(evidence['protocol']['path']);selected_summary=read(evidence['summary']['path'])
            if selected_protocol['case_ids']!=[case['case_id']for case in cases]or selected_summary['manifest_sha256']!=sha(s['manifest'])or selected_protocol['source_sha256']!=sources:raise ValueError('Selected checkpoints must come from the same complete all-eligible manifest and frozen inference sources')
            selected_config=selected_protocol['profiles'][0]['configuration']
            if any(selected_config.get(key,DEFAULT_CONFIG[key])!=value for key,value in _configuration(s).items()):raise ValueError('Cannot change proposal/pose settings after checkpoint selection')
    inputs=_input_pins(settings_path,s,contract,manifest,cases,training_inputs)
    for path,h in physical_inputs.items():
        if path in inputs and inputs[path]!=h:raise ValueError('Physical preparation and dataset pins disagree')
        inputs[path]=h
    available=learning.remaining(contract)['baseline'];pilot_cap=integer(s['pilot_cap_seconds'],learning.MAX_BASELINE);maximum=integer(s['maximum_full_calendar_seconds'],learning.MAX_BASELINE)
    _,journal,charges=learning.journal_state(contract['compute_ledger'],contract['compute_ledger_sha256'],contract['reservation_journal'])
    p=dict(schema_version=70,builder_schema_version=74,prepared_only=False,purpose='cost_pilot'if phase=='cost_pilot'else purpose,target_purpose=purpose,
        stage_mode=s['mode'],settings_path=str(settings_path),settings_sha256=sha(settings_path),learning_contract=s['learning_contract'],learning_contract_sha256=sha(s['learning_contract']),
        split='model_selection',calibration_or_final_access=False,cpu_threads=2,tracks=list(s['tracks']),profiles=profiles,manifest=s['manifest'],geometry_audit=s['geometry_audit'],
        case_ids=[c['case_id']for c in cases],pilot_case_ids=pilot_ids,output=str(Path(s['output_directory'])/phase),
        compute_ledger=contract['compute_ledger'],compute_ledger_sha256=contract['compute_ledger_sha256'],reservation_journal=learning.CANONICAL_JOURNAL,
        baseline_admission=dict(category='baseline',initial_reservations=copy.deepcopy(journal['reservations']),remaining_at_preparation_seconds=available,maximum_full_calendar_seconds=maximum,refund_on_failure=False),
        reference_preparation_evidence=copy.deepcopy(s['reference_preparation_evidence']),reference_preparation_costs=physical_costs,
        blender_executable=s.get('blender_executable'),input_sha256=inputs,source_sha256=sources,case_cache_seals={},
        source_publication_required=True,automatic_execution=False,automatic_model_promotion=False,accepted_results=False,
        scope='All12*N eligible raw development cases from the original32camera/eight-terrain inventory; repeated camera/light conditions are not independent geography.')
    p.update(selection_inventory(audit))
    if phase=='cost_pilot':
        if pilot_cap+maximum>available:raise ValueError('Predeclared pilot plus maximum full study exceeds remaining canonical baseline budget')
        cap=pilot_cap
    else:
        cost,pins,links=_complete_pilot(p,s['cost_pilot_evidence'],manifest,contract);cap=cost['conservative_total_seconds']
        if cap>maximum or cap>available:raise ValueError('Measured complete study does not fit predeclared or remaining baseline allowance; do not drop cases/profiles')
        p.update(cost_pilot_summary=links['summary'],cost_pilot_summary_sha256=sha(links['summary']),measured_cost_admission=cost,cost_pilot_evidence=links);p['input_sha256'].update(pins)
    p['max_calendar_seconds']=cap;p.update(lease_fields(cap,s['publication_dispatch_lease_seconds'],created))
    if Path(output_path).exists()or Path(p['output']).exists():raise FileExistsError('Preserve every prepared/completed/partial stage; no overwrite or restart')
    validate_lease(p);write_new(output_path,p);return p


def verify_publication(protocol_path,commit):
    """Read-only future barrier; does not launch the bounded controller."""
    if len(commit or'')!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact source publication commit required')
    p=read(protocol_path);validate_runtime(p,protocol_path)
    _published_bytes(protocol_path,p,commit)
    return dict(protocol_sha256=sha(protocol_path),publication_commit=commit,ready_for_separate_controller=True,scientific_job_started=False)
