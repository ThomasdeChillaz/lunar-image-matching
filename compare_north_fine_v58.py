"""Post-hoc descriptive comparison; never select checkpoints or calibrate risk."""
from collections import defaultdict
import argparse,gzip,hashlib,json,random
from pathlib import Path


def read_rows(path):
 with gzip.open(path,'rt')as f:return json.load(f)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_rows(base,candidate):
 if not base or len(base)!=len(candidate):raise ValueError('Same nonempty full denominator required')
 if len({r['pair_id']for r in base})!=len(base):raise ValueError('Duplicate case')
 keys=('pair_id','component','tile_id','light_a','light_b','query_view','positive')
 for a,b in zip(base,candidate):
  if not a['positive']or any(a[k]!=b[k]for k in keys):raise ValueError('Identical positive cases/order required')
 return list(zip(base,candidate))


def summary(rows):
 if not rows:raise ValueError('Empty denominator')
 estimated=sum(r['estimated_cam2world']is not None for r in rows);correct=sum(bool(r['pose_correct'])for r in rows)
 returned=sum(r['returned']for r in rows)
 return dict(eligible_pairs=len(rows),correct_raw_poses=correct,wrong_raw_poses=estimated-correct,
  no_pose=len(rows)-estimated,raw_pose_success=correct/len(rows),returned_matches=returned,
  precision_all_returned={str(t):None if not returned else sum(r['correct_at_native_px'][str(t)]for r in rows)/returned for t in (1,2,4,8,12)},
  calibrated_acceptance_evaluated=False)


def paired_difference(base,candidate,repetitions=10000):
 pairs=pair_rows(base,candidate);groups=defaultdict(list)
 for a,b in pairs:groups[a['component']].append(int(bool(b['pose_correct']))-int(bool(a['pose_correct'])))
 names=sorted(groups);rng=random.Random(20261102);boot=[]
 for _ in range(repetitions):
  chosen=[groups[rng.choice(names)]for _ in names]
  boot.append(sum(map(sum,chosen))/sum(map(len,chosen)))
 boot.sort();mean=sum(sum(v)for v in groups.values())/len(pairs)
 gains=sum(not a['pose_correct']and b['pose_correct']for a,b in pairs)
 losses=sum(a['pose_correct']and not b['pose_correct']for a,b in pairs)
 return dict(difference_pp=100*mean,gained_cases=gains,lost_cases=losses,
  terrain_groups=len(names),terrain_bootstrap_95_interval_pp=[100*boot[int(.025*(repetitions-1))],100*boot[int(.975*(repetitions-1))]],
  bootstrap_repetitions=repetitions,bootstrap_seed=20261102,
  scope='Exploratory paired component resampling on exposed development data. Shared source terrain, renderer and camera-template dependence remain. Not a selected-model confidence guarantee.')


def compare(base,candidate):
 pairs=pair_rows(base,candidate)
 if len(pairs)!=96 or len({a['component']for a,b in pairs})!=8:raise ValueError('All96 cases/eight development components required')
 out=dict(baseline=summary(base),candidate=summary(candidate),paired=paired_difference(base,candidate))
 for name,key in [('lighting',lambda r:f'{r["light_a"]}-{r["light_b"]}'),('query_view',lambda r:str(r['query_view'])),('terrain',lambda r:r['component'])]:
  groups=defaultdict(list)
  for a,b in pairs:groups[key(a)].append((a,b))
  out[name]={k:dict(baseline=summary([a for a,b in v]),candidate=summary([b for a,b in v]))for k,v in sorted(groups.items())}
 return out


def run(repo,output):
 repo=Path(repo);out=Path(output)
 if out.exists():raise FileExistsError('Preserve prior snapshot')
 base_path=repo/'runs/north-v39-full-baselines/results.json.gz';base=[r for r in read_rows(base_path)if r['positive']and r['method']=='frozen_surface']
 records=[]
 for variant in ('aligned','fractional'):
  root=repo/('runs/north-fine-v58-study-'+variant)
  for step in range(0,2001,500):
   p=root/f'selection_step_{step}.json.gz'
   if not p.exists():continue
   rows=read_rows(p)
   records.append(dict(variant=variant,step=step,path=str(p),sha256=sha(p),comparison=compare(base,rows),
    complete_study=(root/'COMPLETE.json').exists(),independent_run_audit_present=(repo/('reports/north_fine_'+variant+'_study_audit_v58.json')).exists()))
 if not records:raise ValueError('No complete checkpoint to compare')
 result=dict(scope='Interim complete-checkpoint descriptive development comparison; never uses partial case counts as full accuracy, never alters model selection. No calibrated acceptance/final/SOTA result.',baseline_sha256=sha(base_path),analysis_source_sha256=sha(__file__),records=records,model_promoted=False)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([dict(variant=r['variant'],step=r['step'],correct=r['comparison']['candidate']['correct_raw_poses'],paired=r['comparison']['paired'])for r in records],indent=2))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.repo,a.output)
