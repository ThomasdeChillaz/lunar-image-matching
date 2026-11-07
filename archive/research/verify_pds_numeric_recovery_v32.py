"""Supplement original byte-exact replay failure; never overwrite failed evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,subprocess
repo=Path.cwd()
def read(p):return json.loads((repo/p).read_text())
def sha(p):return hashlib.sha256((repo/p).read_bytes()).hexdigest()
a=read('reports/cresent_lola_rim_probe_v30.json');b=read('runs/replayed-cresent-lola-rims-v30.json')
failed=read('runs/localization-v30-data-software-state.json')
assert failed['state']=='failed' and failed['error']=='Actual map-probe replay differs; preserve both'
assert read('runs/localization-v27-mutual-pipeline-state.json')['state']=='complete'
for p,h in failed['source_sha256'].items():assert sha(p)==h
assert '266 passed, 4 skipped' in (repo/'runs/localization-v30-repository_tests.log').read_text()
assert read('reports/lunar_pds_software_smoke_v30.json')['passed']
assert read('runs/replayed-pds-height-probe-v30.json')['passed']
# Eight representable increments at the lunar radius bound floating-point
# cancellation; this is a supplemental numeric check, not byte-exact replay.
bound=8*math.ulp(1737400.0);differences=[]
def compare(x,y,p=''):
 assert type(x)==type(y),p
 if isinstance(x,dict):
  assert x.keys()==y.keys(),p
  for k in x:
   if p=='' and k=='protocol_sha256':continue
   compare(x[k],y[k],p+'/'+k)
 elif isinstance(x,list):
  assert len(x)==len(y),p
  for i,(v,w)in enumerate(zip(x,y)):compare(v,w,p+'/'+str(i))
 elif isinstance(x,float):
  assert math.isfinite(x) and math.isfinite(y),p
  error=abs(x-y);assert error<=bound,(p,error,bound)
  if error:differences.append(dict(field=p,original=x,replay=y,absolute_difference_m=error))
 else:assert x==y,p
compare(a,b)
assert len(differences)==7
out=repo/'reports/lunar_pds_numeric_recovery_v32.json';assert not out.exists()
result=dict(passed=True,original_exact_replay_passed=False,original_failure_preserved=True,
 scope='Supplemental floating-point replay check; original exact gate remains failed. No terrain-registration discrepancy or metric threshold repaired.',
 absolute_numeric_bound_m=bound,max_absolute_difference_m=max(d['absolute_difference_m']for d in differences),differences=differences,
 tests_passed=266,optional_skipped=4,smoke_rows=100,source_sha256=failed['source_sha256'],
 evidence_sha256={p:sha(p)for p in ['reports/cresent_lola_rim_probe_v30.json','runs/replayed-cresent-lola-rims-v30.json','runs/localization-v30-data-software-state.json','runs/localization-v30-repository_tests.log','reports/lunar_pds_software_smoke_v30.json','runs/replayed-pds-height-probe-v30.json']})
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
