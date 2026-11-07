"""Portable replay of COMPLETED physical reference evidence, never job admission.

Exports run locally only after the strict full96 gate and completed M74 checks.
Consumers verify unchanged historical bytes, public Git object proofs and
recorded charges without opening original Mac executables or live journals.
"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timedelta
from collections import Counter
import base64,hashlib,json,math,os,re,subprocess

SCHEMA=75
PHASES=('reference_one','reference_four','reference_full')
CANONICAL_JOURNAL='runs/solar-compute-reservations-v67.json'
INVENTORY=('terrain_groups','generated_selection_units','generated_selection_unit_count','eligible_selection_unit_count','excluded_selection_units','eligible_terrain_groups','zero_eligible_terrain_groups')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):
    if Path(path).stat().st_size>64*1024**2:raise ValueError('Metadata JSON exceeds declared64MiB limit')
    return json.loads(Path(path).read_text())
def new_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def safe(path):
    if not isinstance(path,str)or not path or '\\'in path or path.startswith('~')or re.match(r'^[A-Za-z]:',path):raise ValueError('Nonportable asset path: '+str(path))
    p=PurePosixPath(path)
    if p.is_absolute()or '..'in p.parts or str(p)!=path or path in('.',''):raise ValueError('Repository-relative normalized asset path required: '+str(path))
    return path

def file(root,path):
    root=Path(root).resolve();p=root/safe(path)
    if not p.resolve().is_relative_to(root):raise ValueError('Asset escaped bundle root')
    current=root
    for part in PurePosixPath(path).parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink inputs are not portable frozen assets: '+path)
    return p

def pin(root,path,digest):
    p=file(root,path)
    if len(digest)!=64 or sha(p)!=digest:raise ValueError('Portable asset hash differs: '+path)
    return p

def time_utc(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('Original timezone-aware timestamp required')
    return result

def positive(value,cap,label):
    if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not 0<value<=cap:raise ValueError('Invalid original '+label)
    return value


def git_oid(kind,data):return hashlib.sha1(kind.encode()+b' '+str(len(data)).encode()+b'\0'+data).hexdigest()

def tree_entries(raw):
    result={};pos=0
    while pos<len(raw):
        end=raw.index(b'\0',pos);mode,name=raw[pos:end].split(b' ',1);oid=raw[end+1:end+21].hex();pos=end+21
        label=name.decode('utf-8');safe(label)
        if '/'in label or label in result or len(oid)!=40:raise ValueError('Malformed public tree proof')
        result[label]=(mode.decode(),oid)
    if pos!=len(raw):raise ValueError('Truncated public tree proof')
    return result


def publication_proof(repo,publications):
    objects={};claims=[]
    def obtain(kind,oid):
        if oid not in objects:
            raw=subprocess.check_output(['git','-C',str(repo),'cat-file',kind,oid])
            if git_oid(kind,raw)!=oid:raise ValueError('Public Git object identity differs')
            objects[oid]=dict(kind=kind,data_base64=base64.b64encode(raw).decode())
        return base64.b64decode(objects[oid]['data_base64'])
    for claim in publications:
        commit=claim['commit'];raw=obtain('commit',commit);root=raw.split(b'\n',1)[0].decode().removeprefix('tree ')
        for path in claim['paths']:
            safe(path);tree=root
            for j,name in enumerate(PurePosixPath(path).parts):
                mode,oid=tree_entries(obtain('tree',tree))[name]
                if j+1<len(PurePosixPath(path).parts):
                    if mode!='40000':raise ValueError('Published ancestor is not a directory')
                    tree=oid
                elif mode not in('100644','100755')or git_oid('blob',file(repo,path).read_bytes())!=oid:raise ValueError('Frozen source/protocol differs from publication')
        claims.append(claim)
    return dict(objects=objects,publications=claims)


def verify_publication_proof(root,proof):
    objects={}
    for oid,row in proof['objects'].items():
        if row['kind']not in('commit','tree'):raise ValueError('Only public commit/tree proof objects are allowed')
        raw=base64.b64decode(row['data_base64'],validate=True)
        if git_oid(row['kind'],raw)!=oid:raise ValueError('Public proof object changed')
        objects[oid]=(row['kind'],raw)
    for claim in proof['publications']:
        commit=claim['commit'];kind,raw=objects[commit]
        if kind!='commit'or not raw.startswith(b'tree '):raise ValueError('Exact public source commit required')
        tree=raw.split(b'\n',1)[0][5:].decode()
        for path,h in claim['paths'].items():
            target=pin(root,path,h);current=tree;parts=PurePosixPath(path).parts
            for j,name in enumerate(parts):
                kind,raw=objects[current]
                if kind!='tree':raise ValueError('Public source ancestor is not a tree')
                mode,oid=tree_entries(raw)[name]
                if j+1<len(parts):
                    if mode!='40000':raise ValueError('Non-directory public ancestor')
                    current=oid
                elif mode not in('100644','100755')or git_oid('blob',target.read_bytes())!=oid:raise ValueError('Frozen bytes are not in the recorded public source commit')
    return True


def selection_inventory(audit):
    rows=sorted((u for u in audit['units']if u['split']=='model_selection'),key=lambda u:u['unit_index'])
    groups=sorted({u['component']for u in rows});good=sorted({u['component']for u in rows if u['eligible']})
    if len(rows)!=32 or len(groups)!=8 or any(sum(r['component']==g for r in rows)!=4 for g in groups):raise ValueError('Complete generated32/eight-group inventory required')
    return dict(terrain_groups=groups,generated_selection_units=rows,generated_selection_unit_count=32,eligible_selection_unit_count=sum(u['eligible']for u in rows),excluded_selection_units=[u for u in rows if not u['eligible']],eligible_terrain_groups=good,zero_eligible_terrain_groups=sorted(set(groups)-set(good)))


def verify_full_gate_metadata(root,e):
    inputs=e['pinned_inputs'];master,data,audit=[read(pin(root,e[k],inputs[e[k]]))for k in('original_master','dataset_summary','geometry_audit')]
    if not data['complete']or len(data['units'])!=96 or not audit['passed']or not audit['complete_dataset']or audit['expected_units']!=96 or audit['completed_units']!=96:raise ValueError('Complete96 independent geometry receipt required')
    if data['original_master_sha256']!=inputs[e['original_master']]or audit['original_master_sha256']!=inputs[e['original_master']]or audit['completion_summary_sha256']!=inputs[e['dataset_summary']]:raise ValueError('Original full-data geometry lineage differs')
    if audit['audit_source_sha256']!=inputs['verify_solar_development_v68.py']:raise ValueError('Published original audit source differs')
    ds={u['unit_index']:u for u in data['units']};ars={u['unit_index']:u for u in audit['units']}
    if set(ds)!=set(range(96))or len(ds)!=96 or set(ars)!=set(ds)or len(master['units'])!=96:raise ValueError('Generated camera inventory was dropped or duplicated')
    groups={};correct=0;directions=0
    for i,spec in enumerate(master['units']):
        row=ars[i];site=spec['site'];groups[site['component']]=site['split']
        if row['component']!=site['component']or row['split']!=site['split']or type(row['eligible'])is not bool:raise ValueError('Original independent split/eligibility differs')
        if site['split']not in('training','model_selection'):raise ValueError('Reserve data forbidden')
        rec=ds[i];folder=safe(rec['output']);summary=folder+'/summary.json';seal=folder+'/COMPLETE.json'
        pin(root,summary,rec['summary_sha256']);pin(root,seal,rec['seal_sha256'])
        if row['source_summary_sha256']!=rec['summary_sha256']or row['source_seal_sha256']!=rec['seal_sha256']:raise ValueError('Independent audit unit source differs')
        s=read(file(root,summary))
        if not s['complete']or s['rendered_images']!=48 or s['matcher_forwards']or s['optimizer_updates']:raise ValueError('Data-only sealed unit required')
        if row['eligible']:
            if len(row['oracle_outcomes'])!=2:raise ValueError('Both eligible oracle directions, including failures, required')
            directions+=2;correct+=sum(v['correct']is True for v in row['oracle_outcomes'])
    if Counter(groups.values())!={'training':16,'model_selection':8}or audit['terrain_counts']!={'training':16,'model_selection':8}:raise ValueError('Original16/8 groups required')
    if directions!=audit['oracle_pose_directions']or correct!=audit['correct_oracle_poses']or not directions or not math.isclose(correct/directions,audit['oracle_success'],rel_tol=0,abs_tol=1e-12)or not .95<=audit['oracle_success']<=1:raise ValueError('Original all-eligible >=95% oracle denominator required')
    return master,data,audit


def full_physics(root,archive,latitude):
    import numpy as np
    from solar_eval_data_v69 import validate_reference_archive,validate_generic_catalogue
    z=validate_reference_archive(file(root,archive));validate_generic_catalogue(z,latitude)
    if z['solar_materialized_mask'].dtype!=bool or not z['solar_materialized_mask'].all():raise ValueError('Portable primary needs EVERY generic physical Sun column; no Mac worker fallback')
    return True


def export_reference_evidence(protocol_path,summary_path,output_path,publication_commit):
    """LOCAL-only export. Strict launch-host evidence/gate checks are never bypassed."""
    from solar_reference_pipeline_v74 import completed_reference_preparation,load_contract,stage_paths,unit_paths
    from solar_learning_pipeline_v73 import full_geometry_gate
    root=Path.cwd();safe(output_path);p=read(protocol_path);c=load_contract(p['contract'],allow_expired=True)
    full_geometry_gate(c['original_master'],c['dataset_summary'],c['geometry_audit'])
    if c.get('reference_precompute')!='full':raise ValueError('Portable primary export requires its own source-frozen FULL pilots')
    result=completed_reference_preparation(protocol_path,summary_path)
    inputs=dict(result['pinned_inputs']);publications=[]
    for path in(c['original_master'],c['dataset_summary'],c['geometry_audit'],c['compute_ledger']):inputs[safe(path)]=sha(path)
    for n,h in c['source_sha256'].items():inputs[safe(n)]=h
    data=read(c['dataset_summary'])
    for row in data['units']:
        for name,h in(('summary.json',row['summary_sha256']),('COMPLETE.json',row['seal_sha256'])):inputs[safe(row['output']+'/'+name)]=h
    for stage in PHASES:
        phasepath,folder=stage_paths(c,stage);q=read(phasepath);summary=read(folder/'summary.json')
        paths={n:sha(n)for n in(q['contract'],phasepath,*c['source_sha256'])}
        for index in c['selection_units']:
            u=unit_paths(c,index)
            for name in('precompute','lookup'):paths[u[name]]=sha(u[name])
        publications.append(dict(commit=summary['publication_commit'],paths=paths))
    own='solar_portable_evidence_v75.py';publications.append(dict(commit=publication_commit,paths={own:sha(own)}));inputs[own]=sha(own)
    for index in c['selection_units']:
        u=unit_paths(c,index);pre=read(u['precompute'])
        for key in('reference_geometry','mesh','scene_mesh','geometry_audit'):inputs[safe(pre[key])]=pre[key+'_sha256']
        full_physics(root,u['archive'],c['reference_units'][str(index)]['latitude_deg'])
    out=Path(output_path);ledger=str(out.parent/'ledger-snapshot.json');journal=str(out.parent/'journal-snapshot.json')
    if out.exists()or Path(ledger).exists()or Path(journal).exists():raise FileExistsError('Preserve prior portable evidence; no overwrite')
    proof=publication_proof(root,publications)
    out.parent.mkdir(parents=True,exist_ok=True)
    for dest,source in((ledger,c['compute_ledger']),(journal,c['reservation_journal'])):
        with Path(dest).open('xb')as target,Path(source).open('rb')as original:
            while chunk:=original.read(1024*1024):target.write(chunk)
        inputs[dest]=sha(dest)
    payload=dict(schema_version=75,kind='completed_reference_physics',complete=True,read_only=True,query_truth_used=False,reference_precompute='full',protocol=protocol_path,summary=summary_path,contract=p['contract'],original_master=c['original_master'],dataset_summary=c['dataset_summary'],geometry_audit=c['geometry_audit'],ledger_snapshot=ledger,journal_snapshot=journal,
        historical_executable_identities={n:dict(path=c[n],sha256=c[n+'_sha256'])for n in('python_executable','blender_executable')},original_canonical_journal=c['reservation_journal'],original_canonical_absolute_identity=c['canonical_journal_absolute'],pinned_inputs=inputs,publication_proof=proof,result=result,
        scope='Replay of source-published COMPLETED full generic reference preparation. Original executable paths are opaque historical identities, not executable inputs or launch authority.')
    new_json(out,payload);return payload


def portable_completed_reference_preparation(bundle_root,evidence_path):
    """Consumer replay: no original executables, live journal, Git or launch checks."""
    e=read(file(bundle_root,evidence_path));inputs=e['pinned_inputs']
    if e['schema_version']!=75 or e['kind']!='completed_reference_physics'or e['complete']is not True or e['read_only']is not True or e['query_truth_used']is not False or e['reference_precompute']!='full':raise ValueError('Completed full reference evidence required')
    for path,h in inputs.items():pin(bundle_root,path,h)
    verify_publication_proof(bundle_root,e['publication_proof'])
    master,data,audit=verify_full_gate_metadata(bundle_root,e);c=read(file(bundle_root,e['contract']));result=e['result'];inventory=selection_inventory(audit)
    if c.get('reference_precompute')!='full'or c['query_truth_in_primary_inference']is not False or c['calibration_or_final_access']is not False:raise ValueError('Original full generic reference-only contract required')
    if any(result[k]!=v or c[k]!=v for k,v in inventory.items()):raise ValueError('Original generated inventory/exclusions changed')
    for name,row in e['historical_executable_identities'].items():
        if c[name]!=row['path']or c[name+'_sha256']!=row['sha256']or not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):raise ValueError('Historical executable identity changed')
    if e['original_canonical_journal']!=CANONICAL_JOURNAL or c['reservation_journal']!=CANONICAL_JOURNAL or e['original_canonical_absolute_identity']!=c['canonical_journal_absolute']:raise ValueError('Historical canonical journal identity changed')
    ledger=read(file(bundle_root,e['ledger_snapshot']));journal=read(file(bundle_root,e['journal_snapshot']))
    if not ledger['accounting_complete']or sha(file(bundle_root,e['ledger_snapshot']))!=c['compute_ledger_sha256']or journal['ledger_sha256']!=c['compute_ledger_sha256']:raise ValueError('Original compute ledger snapshot changed')
    reservations=journal['reservations'];initial=c['initial_reservations']
    if reservations[:len(initial)]!=initial or len({r['job_id']for r in reservations})!=len(reservations):raise ValueError('Original reservation prefix changed or charge duplicated')
    for row in reservations:
        if row['category']not in('training','baseline')or type(row['seconds'])is not int or row['seconds']<=0 or row['refund_on_failure']is not False:raise ValueError('Original full-cap nonrefunded charges required')
    for kind in('training','baseline'):
        if sum(r['seconds']for r in reservations if r['category']==kind)>ledger['remaining_'+kind+'_wall_seconds']:raise ValueError('Historical snapshot exceeds original cumulative budget')
    units=c['selection_units'];rows_by={};phase_times={};reserved=0
    expected=[u['unit_index']for u in audit['units']if u['split']=='model_selection'and u['eligible']]
    if sorted(units)!=sorted(expected)or len(result['reference_archives'])!=len(expected)or sorted(result['reference_unit_indices'])!=sorted(expected):raise ValueError('Every independently eligible reference must remain')
    for stage in PHASES:
        pp=c['protocol_directory']+'/'+stage+'.json';p=read(file(bundle_root,pp));folder=p['output'];s=read(file(bundle_root,folder+'/summary.json'));seal=read(file(bundle_root,folder+'/COMPLETE.json'))
        if p['stage']!=stage or p['contract_sha256']!=inputs[e['contract']]or p['source_sha256']!=c['source_sha256']or s['source_sha256']!=c['source_sha256']or s['complete']is not True or seal['complete']is not True or seal['summary_sha256']!=inputs[folder+'/summary.json']or seal['protocol_sha256']!=inputs[pp]or s['protocol_sha256']!=inputs[pp]:raise ValueError('Original completion source/seal changed')
        started,ended=time_utc(s['started_utc']),time_utc(s['ended_utc']);created,latest,deadline=map(time_utc,(p['created_utc'],p['latest_start_utc'],p['absolute_deadline_utc']))
        positive(s['calendar_seconds'],p['max_calendar_seconds'],'completed phase time')
        if not created<=started<=latest or not started<ended<=deadline or abs((ended-started).total_seconds()-s['calendar_seconds'])>1 or deadline>time_utc(c['preparation_absolute_deadline_utc'])or deadline-latest!=timedelta(seconds=p['max_calendar_seconds']):raise ValueError('Original completed phase time/cap changed')
        charge=[r for r in reservations if r['job_id']==inputs[pp]+'/reference_physics']
        if len(charge)!=1 or charge[0]['category']!='baseline'or charge[0]['seconds']!=p['max_calendar_seconds']:raise ValueError('Completed physical phase lacks exact original canonical charge')
        required_public={name:inputs[name]for name in(e['contract'],pp,*c['source_sha256'])}
        for i in c['selection_units']:
            for label in('precompute','lookup'):
                name=c['protocol_directory']+'/references/'+f'u{i:03d}-'+label+'.json';required_public[name]=inputs[name]
        claims=e['publication_proof']['publications']
        if not any(claim['commit']==s['publication_commit']and all(claim['paths'].get(name)==h for name,h in required_public.items())for claim in claims):raise ValueError('Original phase publication proof is incomplete')
        reserved+=p['max_calendar_seconds'];phase_times[stage]=s['calendar_seconds']
        if stage=='reference_full':rows_by={r['unit_index']:r for r in s['reference_rows']}
    if set(rows_by)!=set(expected)or result['baseline_reserved_seconds']!=reserved or result['phase_calendar_seconds']!=phase_times:raise ValueError('Original completed phase costs or coverage changed')
    archives=[];outer={};inner={}
    for index in expected:
        row=rows_by[index];archive=row['archive'];prepath=c['protocol_directory']+'/references/'+f'u{index:03d}-precompute.json';pre=read(file(bundle_root,prepath));receipt=read(file(bundle_root,row['program_receipt']));w=read(file(bundle_root,row['watchdog_receipt']))
        if pre['precompute']!='full'or pre['query_truth_used']is not False or pre['output']!=archive or receipt['complete']is not True or receipt['precompute']!='full'or receipt['query_truth_used']is not False or receipt['neural_forwards']!=0 or receipt['output_sha256']!=inputs[archive]or receipt['protocol_sha256']!=inputs[prepath]or receipt['source_sha256']!=pre['source_sha256']:raise ValueError('Reference physical recipe/receipt differs')
        if w['complete']is not True or w['child_returncode']!=0 or w['failure']is not None or w['program']!='prepare_solar_reference_v63.py'or w['child_protocol_sha256']!=inputs[prepath]or w['watchdog_source_sha256']!=c['source_sha256']['solar_reference_watchdog_v74.py']:raise ValueError('Original reference external watchdog failed')
        owner=read(file(bundle_root,w['phase_protocol']));owner_summary=read(file(bundle_root,owner['output']+'/summary.json'))
        if w['phase_protocol_sha256']!=inputs[w['phase_protocol']]or w['publication_commit']!=owner_summary['publication_commit']or w['work_cap_seconds']!=c['per_reference_cap_seconds']:raise ValueError('Original physical watchdog phase/publication identity changed')
        start,end,work=map(time_utc,(w['started_utc'],w['ended_utc'],w['work_deadline_utc']))
        if not time_utc(owner['created_utc'])<=start<end<=work or work>start+timedelta(seconds=w['work_cap_seconds'])or work>time_utc(owner['absolute_deadline_utc'])-timedelta(seconds=5)or abs((end-start).total_seconds()-w['calendar_seconds'])>1:raise ValueError('Original completed watchdog time/deadline changed')
        positive(w['calendar_seconds'],owner['max_calendar_seconds'],'startup-inclusive physical time');positive(receipt['calendar_seconds'],w['calendar_seconds'],'internal physical time')
        if row['outer_calendar_seconds']!=w['calendar_seconds']or row['internal_calendar_seconds']!=receipt['calendar_seconds']or row['archive_sha256']!=inputs[archive]or row['program_receipt_sha256']!=inputs[row['program_receipt']]or row['watchdog_receipt_sha256']!=inputs[row['watchdog_receipt']]:raise ValueError('Immutable startup-inclusive reference cost changed')
        approved=c['reference_units'][str(index)]['approved']
        if pre['reference_geometry_sha256']!=approved['reference_geometry_sha256']or pre['scene_mesh_sha256']!=approved['scene_mesh_sha256']or receipt['original_scene_mesh_sha256']!=approved['scene_mesh_sha256']or receipt['original_scene_ceiling_m']!=approved['scene_maximum_height_m']or receipt['map_mesh_sha256']!=pre['mesh_sha256']:raise ValueError('Original map/scene ceiling identity changed')
        for key in('reference_geometry','mesh','scene_mesh','geometry_audit'):
            if inputs.get(pre[key])!=pre[key+'_sha256']:raise ValueError('Required physical input omitted from portable evidence')
        full_physics(bundle_root,archive,c['reference_units'][str(index)]['latitude_deg'])
        archives.append(archive);outer[archive]=w['calendar_seconds'];inner[archive]=receipt['calendar_seconds']
    if sorted(archives)!=result['reference_archives']or outer!=result['outer_calendar_seconds_by_archive']or inner!=result['internal_calendar_seconds_by_archive']:raise ValueError('Portable physical identities/costs differ')
    return dict(result,pinned_inputs=dict(inputs,**{safe(evidence_path):sha(file(bundle_root,evidence_path))}))
