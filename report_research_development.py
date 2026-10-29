"""Regenerate iteration-2 figures and summaries from complete saved development runs."""
import json, csv, zipfile, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports'
def read(run, name='results.json'):
 path=ROOT/'runs'/run
 if name=='results.json':
  complete=json.loads((path/'COMPLETE.json').read_text())
  if hashlib.sha256((path/name).read_bytes()).hexdigest()!=complete['results_sha256']:
   raise ValueError('Completed measurements changed')
 return json.loads((path/name).read_text())
geometry=read('geometry-dev-v1'); adapter=read('adapter-dev-v1'); blend=read('blend-dev-v1'); cascade=read('cascade-dev-v1'); probe=read('eloftr-dev-probe-v1')
rows=geometry+adapter+blend
families=['translation','rotation','scale','combined']
methods=['disk_lightglue','loftr_outdoor','research_adapter','self_similarity_search','blend_global']
labels=['DISK + LightGlue','Pretrained LoFTR','Expanded adapter','Structural search','Image-selected hybrid']
colors=['#95a4b1','#526d9b','#b68b57','#329f8a','#263c62']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(11.5,6.2))
x=np.arange(4); width=.155
for i,(m,label,color) in enumerate(zip(methods,labels,colors)):
 counts=[sum(r['correct'] for r in rows if r['method']==m and r['geometry_family']==f) for f in families]
 y=np.array(counts)/288*100
 bars=ax.bar(x+(i-2)*width,y,width,color=color,label=label)
 if m=='blend_global':ax.bar_label(bars,labels=[f'{v:.1f}%' for v in y],padding=4,fontsize=10,fontweight='bold')
ax.set_xticks(x,['Translation','Rotation','Scale','Combined'])
ax.set_ylim(0,105);ax.set_ylabel('Correct alignments / all positive cases (%)')
ax.set_title('Structural and learned predictions complement each other',loc='left',fontweight='bold',pad=38)
ax.text(0,1.02,'DEVELOPMENT ONLY  ·  288 cases per family  ·  median endpoint error ≤ 3 native pixels',transform=ax.transAxes,fontsize=10,color='#52606d')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
fig.text(.065,.015,'Eight Tycho patches; only two acquisition/footprint components. No confidence interval or fresh-test claim.',fontsize=9,color='#52606d')
fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.82)
fig.savefig(OUT/'development_geometry_v2.png',dpi=180);plt.close(fig)
quality=read('predictor-quality-dev-v1','quality_gates.json')
blendq=read('blend-dev-v1','quality_gates.json')['policies']['blend_global']['empirical_accepted_error_budgets']
cp=read('cascade-dev-v1','policies.json')
def retained(method,budget):
 counts=quality['policies'][method]['empirical_accepted_error_budgets'][budget]['development_counts']
 return counts['accepted_correct'],counts['accepted_incorrect']
def hybrid_retained(budget):
 counts=blendq[budget]['development_counts']
 return counts['accepted_correct'],counts['accepted_incorrect']
entries=[('LoFTR: strict',*retained('loftr_outdoor','0.0')),
 ('Structural search: strict',*retained('self_similarity_search','0.0')),
 ('Hybrid: strict',*hybrid_retained('0.0')),
 ('Hybrid + fallback: strict',cp['cascade_0.0']['accepted_correct'],cp['cascade_0.0']['accepted_incorrect']),
 ('Hybrid: ≤1% empirical budget',*hybrid_retained('0.01')),
 ('Hybrid + fallback: ≤1% budget',cp['cascade_0.01']['accepted_correct'],cp['cascade_0.01']['accepted_incorrect'])]

fig,ax=plt.subplots(figsize=(11.5,5.8))
y=np.arange(len(entries))
ax.barh(y,[e[1] for e in entries],color=['#95a4b1','#329f8a','#526d9b','#263c62','#526d9b','#263c62'])
for i,(label,good,wrong) in enumerate(entries):ax.text(good+12,i,f'{good} correct · {wrong} wrong',va='center',fontsize=10)
ax.set_yticks(y,[e[0] for e in entries]);ax.invert_yaxis();ax.set_xlim(0,1140)
ax.set_xlabel('Retained correct development alignments / 1,152 positive cases')
ax.set_title('Confidence fallback recovers useful alignments',loc='left',fontweight='bold',pad=35)
ax.text(0,1.025,'DEVELOPMENT SELECTED  ·  all displayed rules reject all 72 non-overlap negatives',transform=ax.transAxes,fontsize=10,color='#52606d')
ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
fig.text(.02,.025,'Development budgets are not held-out bounds. Single predictors/hybrid share a gate grid; the cascade adds a conditional fallback.',fontsize=9,color='#52606d')
fig.subplots_adjust(left=.30,right=.98,bottom=.15,top=.80)
fig.savefig(OUT/'development_rejection_v2.png',dpi=180);plt.close(fig)
# Compact public machine-readable measurements from completed runs only.
ids={r['pair_id'] for r in probe}
common=[]
for rr in [geometry,blend,probe]:
 for m in sorted({r['method'] for r in rr}):
  subset=[r for r in rr if r['method']==m and r['pair_id'] in ids]
  common.append(dict(method=m,positive_cases=288,correct=sum(r['correct'] for r in subset),negative_cases=18,negative_fits=sum(r['estimated'] and r['label_source']=='known_nonoverlap' for r in subset)))
summary=dict(schema_version=1,scope='Completed development experiments; no geographic reserve evaluated.',geometry_summary=read('geometry-dev-v1','summary.json'),adapter_summary=read('adapter-dev-v1','summary.json'),blend_summary=read('blend-dev-v1','summary.json'),cascade_summary=read('cascade-dev-v1','summary.json'),same_two_patch_comparison=common,cascade_policies=cp,predictor_quality_policies=quality,blend_quality_policies=read('blend-dev-v1','quality_gates.json'),test_validation='95 passed with actual DISK/LoFTR/EfficientLoFTR/MatchAnything inference and cascade/refinement tests.',software_smoke_summaries=len(read('research-software-smoke-05','summary.json')))
(OUT/'research_development_v2.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'development_geometry_v2.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['method','geometry_family','cases','returned','correct']);writer.writeheader()
 for run in ['geometry-dev-v1','adapter-dev-v1','blend-dev-v1','cascade-dev-v1']:
  for row in read(run,'summary.json'):writer.writerow({k:row[k] for k in writer.fieldnames})
print('Created two measured figures, completed-results JSON and CSV')
