"""Allowlisted, byte-preserving model-selection capsules; no model execution.

The published evaluation protocol and all recursive runtime asset paths remain
unchanged. Packing is local, full96-gated and PRIMARY/full-physics only. A
consumer installs hashes safely; it never obtains reserve or launch authority.
"""
from pathlib import Path,PurePosixPath
import hashlib,json,os,re,shutil,time,zipfile
from solar_portable_evidence_v75 import (sha,read,new_json,safe,file,pin,portable_completed_reference_preparation)

SCHEMA=75
MAX_TOTAL_BYTES=4*1024**3
MAX_MEMBER_BYTES=1900*1024**2
RELEASE_ASSET_MAX_BYTES=1900*1024**2
MAX_FILES=30000
MAX_MANIFEST_BYTES=16*1024**2
MAX_JSON_BYTES=64*1024**2
OWN_SOURCES={'solar_capsule_v75.py','solar_portable_evidence_v75.py'}
OPAQUE_FIELDS={'python_executable','blender_executable','canonical_journal_absolute','original_canonical_absolute_identity'}
INPUT_ROOTS={'reports','runs','data','artifacts','models','weights','vendor','third_party'}
SOURCE_SUFFIXES={'.py','.pyi','.json','.yaml','.yml','.txt','.toml','.cfg','.ini','.sh','.md','.csv'}
PRIVATE_NAMES={'.env','credentials.json','credentials','id_rsa','id_ed25519','.netrc','.npmrc','.pypirc'}


class NonportableInput(ValueError):
    def __init__(self,findings):
        super().__init__('Nonportable frozen runtime inputs; fix BEFORE publication/reservation, never rewrite: '+json.dumps(findings))
        self.findings=findings


def allowed(path,role):
    safe(path);p=PurePosixPath(path);parts={v.lower()for v in p.parts}
    if parts&PRIVATE_NAMES or any(v.startswith('.')and not(role=='source'and j==0 and v=='.github'and len(p.parts)>2 and p.parts[1]=='workflows')for j,v in enumerate(p.parts)):raise ValueError('Private/VCS/hidden files are forbidden capsule inputs:'+path)
    if role=='source':
        if p.suffix not in SOURCE_SUFFIXES:raise ValueError('Unreviewed executable/source asset type:'+path)
    elif p.parts[0]not in INPUT_ROOTS and not(len(p.parts)==1 and p.suffix=='.json'):
        raise ValueError('Asset is outside the explicit scientific input roots:'+path)
    if role!='source'and any(re.search(r'(^|[-_])(final|reserve|calibration)([-_.]|$)',v,re.I)for v in p.parts):raise ValueError('Calibration/final/reserve artifacts forbidden:'+path)
    return path


def absolute_string(value):return value.startswith(('/','~','\\'))or bool(re.match(r'^[A-Za-z]:[\\/]',value))

def portability_findings(root,paths,historical_contracts=()):
    """Opaque old executable fields are reported; they are NEVER executable inputs."""
    failures=[];opaque=[];contracts=set(historical_contracts)
    def visit(value,owner,pointer=()):
        if isinstance(value,dict):
            for k,v in value.items():
                if absolute_string(str(k)):failures.append(dict(file=owner,pointer='/'.join(pointer),value=k,reason='absolute path used as a mapping key'))
                visit(v,owner,pointer+(k,))
        elif isinstance(value,list):
            for i,v in enumerate(value):visit(v,owner,pointer+(str(i),))
        elif isinstance(value,str)and absolute_string(value):
            field=pointer[-1]if pointer else''
            historical=(owner in contracts and field in OPAQUE_FIELDS)or(field=='path'and 'historical_executable_identities'in pointer)
            entry=dict(file=owner,pointer='/'.join(pointer),value=value)
            (opaque if historical else failures).append(entry)
    for path in paths:
        safe(path)
        if path.endswith('.json'):visit(read(file(root,path)),path)
    return dict(failures=failures,opaque_completed_provenance=opaque)


def required_profile_inputs(root,p):
    """Actual checkpoint/history/control prerequisites; missing receipts never become seals."""
    needed={};missing=[]
    def take(path,digest=None):
        safe(path);target=file(root,path)
        if not target.is_file():missing.append(path);return
        actual=sha(target)
        if digest is not None and actual!=digest:raise ValueError('Required profile receipt/checkpoint changed:'+path)
        needed[path]=actual
    if p.get('learning_contract'):take(p['learning_contract'],p.get('learning_contract_sha256'))
    for profile in p['profiles']:
        for path,h in profile.get('artifact_sha256',{}).items():take(path,h)
        if profile.get('native_control'):
            for key in('protocol','outputs','receipt'):take(profile['native_control'][key],profile['native_control'][key+'_sha256'])
        if profile.get('checkpoint'):take(profile['checkpoint'],profile.get('configuration',{}).get('checkpoint_sha256'))
        if profile['backend']=='solar'and not profile.get('initialization_only_cost_control',False):
            evidence=profile['training_evidence']
            for name in('protocol','summary','watchdog_receipt'):take(evidence[name],evidence[name+'_sha256'])
            summary=read(file(root,evidence['summary']));training=read(file(root,evidence['protocol']));folder=str(PurePosixPath(evidence['summary']).parent)
            take(folder+'/history_2000.json',summary['all_history_sha256'])
            for name,h in summary['checkpoint_sha256'].items():take(folder+'/'+safe(name),h)
            # Existing training studies use their successful outer watchdog,
            # not a manufactured COMPLETE. Preserve a real seal if produced.
            if file(root,folder+'/COMPLETE.json').exists():take(folder+'/COMPLETE.json')
            for name,h in training['source_sha256'].items():take(name,h)
        if profile['backend']=='native_pretrained':
            ev=profile['pretrained_evidence']
            for name in('protocol','summary','controls','provenance'):take(ev[name],ev[name+'_sha256'])
        for key in('pretrained_control','pretrained_provenance'):
            if profile.get(key):take(profile[key])
    for evidence in p.get('selection_evidence',{}).values():
        if isinstance(evidence,dict):
            for value in evidence.values():
                if isinstance(value,dict)and 'path'in value:take(value['path'],value.get('sha256'))
    for path,h in p.get('selected_weight_inputs',{}).items():take(path,h)
    for profile in p['profiles']:
        for path in profile.get('weight_paths',[]):take(path,p['input_sha256'].get(path))
    if missing:raise ValueError('Unavailable required actual profile artifacts: '+json.dumps(sorted(set(missing))))
    return needed


def validate_frozen_assets(root,protocol_path,evidence_path):
    p=read(file(root,protocol_path));safe(protocol_path)
    if p.get('prepared_only',False)or p.get('split')!='model_selection'or p.get('calibration_or_final_access')is not False or p.get('tracks')!=['estimated_sun']or p.get('cpu_threads')!=2:raise ValueError('Frozen PRIMARY-only model-selection protocol required')
    if p.get('blender_executable')not in(None,''):raise NonportableInput([dict(pointer='blender_executable',reason='Full-table portable primary has no worker; configure before protocol freeze, never rewrite')])
    pins=dict(p['input_sha256']);sources=dict(p['source_sha256'])
    if any(path in pins and pins[path]!=h for path,h in sources.items()):raise ValueError('Conflicting input/source identities')
    if not OWN_SOURCES<=set(sources)or not pins:raise ValueError('Exact capsule/evidence source and input allowlists required')
    import solar_portable_evidence_v75 as evidence_module
    for name,executed in(('solar_capsule_v75.py',__file__),('solar_portable_evidence_v75.py',evidence_module.__file__)):
        if sha(executed)!=sources[name]:raise ValueError('Actually executed capsule/evidence source differs from frozen source:'+name)
    for path,h in {**sources,**pins}.items():pin(root,path,h)
    if evidence_path not in pins:raise ValueError('Portable reference evidence must be part of the original frozen protocol input allowlist')
    complete=portable_completed_reference_preparation(root,evidence_path)
    required=dict(complete['pinned_inputs']);required.update(required_profile_inputs(root,p))
    if p['manifest']not in pins or p['geometry_audit']not in pins:raise ValueError('Frozen manifest and full geometry audit required')
    manifest=read(file(root,p['manifest']));audit=read(file(root,p['geometry_audit']))
    # Reuses existing complete case/negative endpoint/oracle denominator rules,
    # without a network or any inference/model construction.
    from solar_evaluation_v70 import validate_manifest
    cases=validate_manifest(manifest,audit)
    if p['case_ids']!=[c['case_id']for c in cases]:raise ValueError('All12*N model-selection cases must remain in the original frozen order')
    for c in cases:
        for path,h in c['input_sha256'].items():required[path]=h
    for path,h in required.items():
        if (pins|sources).get(path)!=h:raise ValueError('Required recursive receipt/input is absent from frozen allowlist; refreeze BEFORE reservation:'+path)
    contracts={read(file(root,evidence_path))['contract']}
    # The portable record contains historical paths. Only these old completed
    # identities may remain absolute; live evaluation fields never may.
    historical=portability_findings(root,set(pins)|set(sources)|{protocol_path},historical_contracts=contracts|{evidence_path})
    if historical['failures']:raise NonportableInput(historical['failures'])
    return p,complete,historical


def pack_capsule(protocol_path,output_directory,publication_commit,pre_reservation_authorization,portable_reference_evidence,*,repo_root='.',max_total_bytes=MAX_TOTAL_BYTES,max_calendar_seconds=1800):
    """No rewriting, automatic publication or reservation. Full96 gate is FIRST."""
    from solar_learning_pipeline_v73 import full_geometry_gate
    root=Path(repo_root).resolve()
    if root!=Path.cwd().resolve():raise ValueError('Local gated packing must run from its repository root')
    p=read(file(root,protocol_path));ev=read(file(root,portable_reference_evidence))
    full_geometry_gate(ev['original_master'],ev['dataset_summary'],ev['geometry_audit'])
    tick=time.monotonic()
    if type(max_total_bytes)is not int or not 0<max_total_bytes<=MAX_TOTAL_BYTES or type(max_calendar_seconds)is not int or not 0<max_calendar_seconds<=1800:raise ValueError('Explicit bounded capsule I/O allowance required')
    p,complete,portability=validate_frozen_assets(root,protocol_path,portable_reference_evidence)
    if len(publication_commit)!=40 or any(c not in'0123456789abcdef'for c in publication_commit):raise ValueError('Exact published source commit required')
    import subprocess
    for path in(protocol_path,*p['source_sha256']):
        if subprocess.check_output(['git','-C',str(root),'show',publication_commit+':'+path])!=file(root,path).read_bytes():raise ValueError('Frozen protocol/source bytes are not source-published:'+path)
    auth=pre_reservation_authorization
    if set(auth)!={'path','sha256'}:raise ValueError('One explicit hash-pinned pre-reservation authorization role required')
    pin(root,auth['path'],auth['sha256'])
    authorization=read(file(root,auth['path']))
    if authorization.get('protocol_sha256')!=sha(file(root,protocol_path))or authorization.get('source_publication_commit')!=publication_commit:raise ValueError('Pre-reservation authorization belongs to another immutable protocol/source')
    files={**p['input_sha256'],**p['source_sha256'],protocol_path:sha(file(root,protocol_path)),auth['path']:auth['sha256']}
    if len(files)>MAX_FILES:raise ValueError('Capsule exceeds explicit file-count cap')
    total=0;oversized=[]
    selected=set(required_profile_inputs(root,p))
    for path,h in files.items():
        allowed(path,'source'if path in p['source_sha256']else'input');target=pin(root,path,h);size=target.stat().st_size;total+=size
        if size>MAX_MEMBER_BYTES or path.endswith('.json')and size>MAX_JSON_BYTES:oversized.append(dict(path=path,bytes=size))
        if PurePosixPath(path).suffix.lower()in('.pt','.pth','.ckpt','.safetensors')and path not in selected:raise ValueError('Bulk/unselected model asset forbidden; freeze selected_weight_inputs before reservation:'+path)
    destination=Path(output_directory)
    if total>max_total_bytes or oversized:
        if destination.exists():raise FileExistsError('Preserve prior capsule/shortfall')
        result=dict(complete=False,status='SHORTFALL',reason='Selected-profile capsule exceeds predeclared transport bounds',total_bytes=total,max_total_bytes=max_total_bytes,max_member_bytes=MAX_MEMBER_BYTES,oversized_members=oversized,files_copied=0,refund=False)
        new_json(destination/'SHORTFALL.json',result);return result
    if destination.exists():raise FileExistsError('Preserve prior capsule/partial outputs; no overwrite')
    destination.mkdir(parents=True)
    try:
        for path,h in sorted(files.items()):
            if time.monotonic()-tick>=max_calendar_seconds:raise TimeoutError('Original capsule I/O allowance exhausted')
            source=file(root,path);target=destination/'payload'/path;target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb')as src,target.open('xb')as dst:shutil.copyfileobj(src,dst,1024*1024)
            if sha(target)!=h:raise ValueError('Byte copy changed during packing:'+path)
        result=dict(schema_version=75,capsule_kind='model_selection_primary_full_sun',complete=True,files=files,protocol_path=protocol_path,protocol_sha256=files[protocol_path],publication_commit=publication_commit,pre_reservation_authorization=auth,portable_reference_evidence=dict(path=portable_reference_evidence,sha256=files[portable_reference_evidence]),total_bytes=total,max_total_bytes=max_total_bytes,max_member_bytes=MAX_MEMBER_BYTES,max_files=MAX_FILES,release_asset_max_bytes=RELEASE_ASSET_MAX_BYTES,max_calendar_seconds=max_calendar_seconds,calendar_seconds=time.monotonic()-tick,opaque_completed_provenance=portability['opaque_completed_provenance'],query_truth_policy='Query depth/pose labels remain separate scoring assets; primary inference receives reference geometry/full generic physics only.',scope='Byte-preserving allowlisted transport only; no model, worker, reservation, final data, email or publication.')
        new_json(destination/'manifest.json',result);return result
    except BaseException as exc:
        new_json(destination/'FAILED.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc)));raise


def validate_capsule(capsule_directory):
    root=Path(capsule_directory)
    if (root/'manifest.json').stat().st_size>MAX_MANIFEST_BYTES:raise ValueError('Capsule manifest exceeds declared16MiB limit')
    m=read(root/'manifest.json')
    if (root/'FAILED.json').exists()or m['schema_version']!=75 or m['capsule_kind']!='model_selection_primary_full_sun'or m['complete']is not True or len(m['files'])>MAX_FILES:raise ValueError('Complete allowlisted PRIMARY capsule required')
    if m['protocol_sha256']!=m['files'].get(m['protocol_path']):raise ValueError('Capsule protocol identity changed')
    actual={str(p.relative_to(root/'payload'))for p in(root/'payload').rglob('*')if p.is_file()}
    if actual!=set(m['files']):raise ValueError('Capsule has missing or extra payloads')
    total=0
    for path,h in m['files'].items():
        target=pin(root/'payload',path,h);size=target.stat().st_size;total+=size
        if size>m['max_member_bytes']or size>MAX_MEMBER_BYTES or path.endswith('.json')and size>MAX_JSON_BYTES:raise ValueError('Capsule member exceeds original bound')
    if total!=m['total_bytes']or total>m['max_total_bytes']or total>MAX_TOTAL_BYTES:raise ValueError('Capsule size identity differs')
    p,complete,findings=validate_frozen_assets(root/'payload',m['protocol_path'],m['portable_reference_evidence']['path'])
    for path in m['files']:allowed(path,'source'if path in p['source_sha256']else'input')
    auth=m['pre_reservation_authorization']
    if m['files'].get(auth['path'])!=auth['sha256']:raise ValueError('Pre-reservation authorization is not pinned by capsule')
    if m['portable_reference_evidence']['sha256']!=m['files'].get(m['portable_reference_evidence']['path']):raise ValueError('Portable evidence identity changed')
    expected=set(p['input_sha256'])|set(p['source_sha256'])|{m['protocol_path'],auth['path']}
    if set(m['files'])!=expected:raise ValueError('Capsule includes assets outside the frozen allowlist and explicit authorization role')
    return m


def install_capsule(capsule_directory,destination):
    """Validate all files before installation. Existing different bytes never overwritten."""
    m=validate_capsule(capsule_directory)
    if Path(destination).is_symlink():raise ValueError('Destination symlink is forbidden')
    root=Path(destination).resolve()
    if root.is_symlink():raise ValueError('Destination symlink is forbidden')
    pending=[]
    for path,h in m['files'].items():
        target=file(root,path)
        if target.exists():
            if not target.is_file()or sha(target)!=h:raise FileExistsError('Preserve existing different bytes:'+path)
        else:pending.append((path,h))
    root.mkdir(parents=True,exist_ok=True)
    for path,h in pending:
        source=file(Path(capsule_directory)/'payload',path);target=file(root,path);target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb')as src,target.open('xb')as dst:shutil.copyfileobj(src,dst,1024*1024)
        if sha(target)!=h:raise ValueError('Installed bytes differ:'+path)
    return dict(complete=True,installed_files=len(pending),identical_existing_files=len(m['files'])-len(pending),protocol_sha256=m['protocol_sha256'],manifest_sha256=sha(Path(capsule_directory)/'manifest.json'),scientific_execution=False)


def make_zip(capsule_directory,output_path):
    """Streaming ZIP64; splitting release transport belongs to the remote relay."""
    m=validate_capsule(capsule_directory);root=Path(capsule_directory);out=Path(output_path)
    if out.exists():raise FileExistsError('Preserve prior archive')
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_STORED,allowZip64=True)as z:
        z.write(root/'manifest.json','manifest.json')
        for path in sorted(m['files']):z.write(root/'payload'/path,'payload/'+path)
    return dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,manifest_sha256=sha(root/'manifest.json'))


def unpack_zip(zip_path,destination,*,max_total_bytes=MAX_TOTAL_BYTES):
    """Reject traversal, symlinks, duplicates, zip bombs and unknown files before writing."""
    root=Path(destination)
    if root.exists():raise FileExistsError('Preserve prior/partial extracted capsule')
    with zipfile.ZipFile(zip_path)as z:
        names=[i.filename for i in z.infolist()]
        if len(names)!=len(set(names))or len(names)>MAX_FILES+1:raise ValueError('Duplicate or excessive archive entries')
        total=0
        for item in z.infolist():
            safe(item.filename)
            if item.file_size>MAX_MEMBER_BYTES or item.filename=='manifest.json'and item.file_size>MAX_MANIFEST_BYTES or item.filename.endswith('.json')and item.file_size>MAX_JSON_BYTES or item.is_dir()or (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Only regular explicit payload files allowed')
            if item.filename!='manifest.json'and not item.filename.startswith('payload/'):raise ValueError('Unknown capsule archive entry')
            total+=item.file_size
        if total>max_total_bytes+16*1024**2:raise ValueError('Archive exceeds original decompressed size cap')
        root.mkdir(parents=True)
        for item in z.infolist():
            target=file(root,item.filename);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item)as src,target.open('xb')as dst:shutil.copyfileobj(src,dst,1024*1024)
    validate_capsule(root);return str(root)
