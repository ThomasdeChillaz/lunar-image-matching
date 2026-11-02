"""Post-run matched-sampling audit; never executes a neural model or tunes one."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import torch
from compare_north_fine_v58 import read_rows,paired_difference

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def physical(rows):return [{k:v for k,v in r.items()if k not in ('variant','effective_query_center')}for r in rows]


def audit(repo,output):
 repo=Path(repo);out=Path(output)
 if out.exists():raise FileExistsError('Preserve previous matched replay')
 state=read(repo/'runs/north-fine-v58-state.json')
 if state['state']!='complete':raise ValueError('Both original bounded studies must complete first')
 for field in ('pid','child_pid'):
  if state.get(field)and subprocess.run(['ps','-p',str(state[field]),'-o','command='],text=True,capture_output=True).stdout.strip():raise ValueError('Scientific queue still live; inspect before another experiment')
 roots={v:repo/('runs/north-fine-v58-study-'+v)for v in ('aligned','fractional')};summaries={v:read(r/'summary.json')for v,r in roots.items()}
 for v in roots:
  p=repo/('reports/north_fine_'+v+'_study_audit_v58.json');a=read(p)
  if not a['passed']or a['replayed_updates']!=2000 or a['replayed_rows']!=576 or a['summary_sha256']!=sha(roots[v]/'summary.json'):raise ValueError('Both independently replayed studies required')
 a,b=summaries['aligned'],summaries['fractional'];different_queries=different_targets=0
 for step,(x,y)in enumerate(zip(a['history'],b['history']),1):
  if x['step']!=step or y['step']!=step or physical(x['samples'])!=physical(y['samples'])or any(x['input_sha256'][i]!=y['input_sha256'][i]for i in (0,2)):raise ValueError('Matched physical draws/reference inputs differ')
  different_queries+=x['input_sha256'][1]!=y['input_sha256'][1];different_targets+=x['input_sha256'][3]!=y['input_sha256'][3]
 if len(a['history'])!=2000 or len(b['history'])!=2000:raise ValueError('Both full histories required')
 load=lambda r,n:torch.load(r/n,map_location='cpu',weights_only=True)
 ia,ib=[load(r,'initial.pt')for r in roots.values()]
 if ia.keys()!=ib.keys()or any(not torch.equal(v,ib[k])for k,v in ia.items()):raise ValueError('Matched initialization differs')
 la,lb=[load(r,'latest.pt')for r in roots.values()]
 if la['numpy_rng']!=lb['numpy_rng']or not torch.equal(la['torch_rng'],lb['torch_rng']):raise ValueError('Final data/framework RNG differs')
 rows={v:read_rows(r/f'selection_step_{summaries[v]["selected_step"]}.json.gz')for v,r in roots.items()}
 paired=paired_difference(rows['aligned'],rows['fractional'])
 result=dict(passed=True,matched_updates=2000,matched_physical_point_samples=16000,initialization_exact=True,reference_inputs_exact=True,final_numpy_and_torch_rng_exact=True,different_query_batches=different_queries,different_target_batches=different_targets,selected_pose_difference_fractional_minus_aligned=paired,
  selected={v:dict(step=s['selected_step'],selection=s['selection'],negative_counts=s['selected_negative_counts'])for v,s in summaries.items()},summary_sha256={v:sha(r/'summary.json')for v,r in roots.items()},audit_sha256={v:sha(repo/('reports/north_fine_'+v+'_study_audit_v58.json'))for v in roots},source_sha256=sha(__file__),comparison_source_sha256=sha(Path(__file__).with_name('compare_north_fine_v58.py')),model_promoted=False,
  scope='Matched sampling-control replay on exposed northern development data. Only query resampling and its labels differ by design. Pose intervals are exploratory, conditional on development selection/shared renderer. No calibrated acceptance/final/SOTA result.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.repo,a.output)
