"""Source-pinned, lazy reference-only solar rays via a bounded Blender worker.

Construction validates inputs but launches no worker. Call time includes cold
startup and all ray work. UNKNOWN is a measured coverage state, never a missing
computation. Query truth is neither a protocol field nor a cache dependency.
"""
import hashlib,json,os,selectors,subprocess,time
from pathlib import Path
import numpy as np

SOURCES=('solar_lookup_v70.py','solar_lookup_worker_v70.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py')
REQUIRED={'schema_version','cpu_threads','query_truth_used','max_lookup_calendar_seconds','max_request_seconds','mesh','mesh_sha256','scene_mesh','scene_mesh_sha256','reference_geometry','reference_geometry_sha256','geometry_audit','geometry_audit_sha256','approved_reference_key','source_sha256'}
OPTIONAL=set()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol(path,publication_commit,verify_publication=True):
    path=Path(path).resolve();p=json.loads(path.read_text())
    if not REQUIRED<=set(p)or set(p)-REQUIRED-OPTIONAL:raise ValueError('Strict reference-only lookup protocol required; extra/query fields forbidden')
    if p['schema_version']!=70 or p['cpu_threads']!=2 or p['query_truth_used']is not False:raise ValueError('Source-pinned reference-only schema70/two-thread protocol required')
    if not 0<p['max_lookup_calendar_seconds']<=900 or not 0<p['max_request_seconds']<=300:raise ValueError('Bounded lookup and request deadlines required')
    root=Path(__file__).resolve().parent
    for name in('mesh','scene_mesh','reference_geometry','geometry_audit'):
        if not Path(p[name]).is_absolute():p[name]=str((root/p[name]).resolve())
        if sha(p[name])!=p[name+'_sha256']:raise ValueError('Pinned reference lookup input changed:'+name)
    audit=json.loads(Path(p['geometry_audit']).read_text())
    if not audit['passed']or audit['oracle_success']<.95:raise ValueError('Global95percent exact-correspondence gate required')
    approved=audit['approved_reference_inputs'][p['approved_reference_key']]
    if approved['reference_geometry_sha256']!=p['reference_geometry_sha256']or approved['scene_mesh_sha256']!=p['scene_mesh_sha256']:raise ValueError('Approved reference/original scene identities differ')
    with np.load(p['scene_mesh'],allow_pickle=False)as z:
        v=z['vertices']
        if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Original full scene vertices required')
        ceiling=float(v[:,2].max())
    if abs(ceiling-approved['scene_maximum_height_m'])>1e-6:raise ValueError('Original scene ceiling differs from independent audit')
    root=Path(__file__).resolve().parent
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Exact physical lookup source closure required')
    for name in SOURCES:
        if sha(root/name)!=p['source_sha256'][name]:raise ValueError('Physical lookup source changed:'+name)
    if verify_publication:
        repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip())
        for source in[*(root/name for name in SOURCES),path]:
            rel=source.relative_to(repo).as_posix()
            if subprocess.check_output(['git','show',publication_commit+':'+rel],cwd=repo)!=source.read_bytes():raise ValueError('Lookup source/protocol not published:'+rel)
    # Every physical dependency is in the key; model checkpoints are excluded.
    identity={k:p[k]for k in REQUIRED if k not in('max_lookup_calendar_seconds','max_request_seconds')}
    identity.update(original_scene_ceiling_m=ceiling,anchor_grid='49.5+16*round(linspace(0,25,16))',height_shifts_m=[0,-5,5,-10,10,-20,20],self_hit_epsilon_m=.001,world_frame='reference_scene_ENU',ray_direction='toward_Sun')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return p,key


class ReferenceSunLookup:
    @classmethod
    def from_protocol(cls,protocol_path,publication_commit,blender_executable):
        tick=time.monotonic();p,key=validate_protocol(protocol_path,publication_commit)
        result=cls(protocol_path,publication_commit,blender_executable,p,key);result.stats['validation_seconds']=time.monotonic()-tick
        return result

    def __init__(self,path,publication_commit,blender_executable,protocol,physical_key):
        self.path=str(Path(path).resolve());self.commit=publication_commit;self.blender=str(blender_executable)
        self.protocol=protocol;self.physical_key=physical_key;self.process=None;self.started=None;self.buffer=b'';self.cache={};self.closed=False
        self.stats=dict(physical_key=physical_key,requests=0,requested_columns=0,cache_hits=0,computed_columns=0,solar_rays=0,reference_rays=0,cold_start_seconds=0.,worker_seconds=0.,calendar_seconds=0.,ray_direction='toward_Sun',world_frame='reference_scene_ENU',cache_policy='per-reference provider; caller must declare cold versus warm profile timing')

    def _read(self,deadline):
        selector=selectors.DefaultSelector();selector.register(self.process.stdout,selectors.EVENT_READ)
        try:
            while time.monotonic()<deadline:
                while b'\n'in self.buffer:
                    line,self.buffer=self.buffer.split(b'\n',1)
                    if line.startswith(b'SOLAR_REPLY '):
                        reply=json.loads(line[len(b'SOLAR_REPLY '):])
                        if 'error'in reply:raise RuntimeError('Solar worker failed: '+reply['error'])
                        return reply
                if self.process.poll()is not None:raise RuntimeError('Solar worker exited before reply: '+str(self.process.returncode))
                if not selector.select(max(0.,deadline-time.monotonic())):break
                data=os.read(self.process.stdout.fileno(),65536)
                if not data:raise RuntimeError('Solar worker closed output before reply')
                self.buffer+=data
            raise TimeoutError('Fixed solar lookup/request deadline exhausted')
        finally:selector.close()

    def _deadline(self):
        return min(self.started+self.protocol['max_lookup_calendar_seconds'],time.monotonic()+self.protocol['max_request_seconds'])

    def _start(self):
        self.started=time.monotonic()
        worker=Path(__file__).with_name('solar_lookup_worker_v70.py')
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='2')
        self.process=subprocess.Popen([self.blender,'--background','--threads','2','--python-exit-code','91','--python',str(worker),'--','--protocol',self.path,'--publication-commit',self.commit,'--owner-pid',str(os.getpid())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd=worker.parent,env=env)
        ready=self._read(self._deadline())
        if ready.get('kind')!='ready'or ready.get('physical_key')!=self.physical_key:raise RuntimeError('Physical worker identity/ready mismatch')
        self.stats['cold_start_seconds']=time.monotonic()-self.started;self.stats['reference_rays']+=ready['reference_rays']

    def __call__(self,sun_vectors):
        tick=time.monotonic()
        if self.closed:raise RuntimeError('Reference solar lookup is closed')
        suns=np.asarray(sun_vectors,float)
        if suns.ndim!=2 or suns.shape[1]!=3 or not len(suns)or not np.isfinite(suns).all()or not np.allclose(np.linalg.norm(suns,axis=1),1,atol=1e-6,rtol=0):raise ValueError('Exact unit reference-world Sun vectors required; no angle snapping')
        keys=[np.asarray(s,dtype='<f8').tobytes()for s in suns]
        self.stats['requests']+=1;self.stats['requested_columns']+=len(keys)
        missing=[];seen=set()
        for key,sun in zip(keys,suns):
            if key in self.cache:self.stats['cache_hits']+=1
            elif key not in seen:missing.append((key,sun));seen.add(key)
        try:
            if missing:
                if self.process is None:self._start()
                if time.monotonic()>=self.started+self.protocol['max_lookup_calendar_seconds']:raise TimeoutError('Original solar lookup calendar deadline exhausted')
                request=dict(sun_vectors=[sun.tolist()for _,sun in missing])
                self.process.stdin.write((json.dumps(request,separators=(',',':'))+'\n').encode());self.process.stdin.flush()
                reply=self._read(self._deadline());states=np.asarray(reply['solar_states'],np.uint8);variants=np.asarray(reply['solar_variants'],np.uint8)
                if states.shape!=(256,len(missing))or variants.shape!=(256,len(missing),7)or not np.isin(states,[0,1,2]).all()or not np.isin(variants,[0,1,2]).all():raise RuntimeError('Invalid physical lookup reply')
                for j,(key,_)in enumerate(missing):self.cache[key]=(states[:,j].copy(),variants[:,j].copy())
                for name in('solar_rays','computed_columns','worker_seconds'):self.stats[name]+=reply[name]
            return dict(solar_states=np.stack([self.cache[k][0]for k in keys],axis=1),solar_variants=np.stack([self.cache[k][1]for k in keys],axis=1))
        except BaseException:
            self.close();raise
        finally:self.stats['calendar_seconds']+=time.monotonic()-tick

    def close(self):
        self.closed=True
        if self.process is not None:
            if self.process.poll()is None:
                self.process.stdin.close()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdout.close();self.process=None

    def __del__(self):
        if getattr(self,'process',None)is not None:
            try:self.close()
            except Exception:pass

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
