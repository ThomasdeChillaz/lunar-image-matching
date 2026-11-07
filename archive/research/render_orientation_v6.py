"""Presentation-only orientation plot from immutable measured values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import sha256

source=Path('reports/orientation_training_comparison_v6.json')
result=json.loads(source.read_text());narrow=next(r for r in result['runs'] if r['variant']=='canonical_3d');history=narrow['history']
x=[r['step'] for r in history];fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
axes[0].plot(x,[100*r['selection']['correct_at_two_px']/r['selection']['scorable'] for r in history],marker='o',color='#16867d')
axes[0].set(xlabel='Optimizer step',ylabel='2px precision (%)',ylim=(0,100),title='Narrow-orientation point precision')
axes[1].plot(x,[r['selection']['correct_poses'] for r in history],marker='o',color='#ab6847')
axes[1].set(xlabel='Optimizer step',ylabel='Correct poses / 16',ylim=(0,16),title='Pose recovery does not improve')
fig.suptitle('Orientation training — exposed development selection bank')
fig.savefig('reports/orientation_training_outcomes_v6.png',dpi=180);plt.close(fig)
Path('reports/orientation_plot_render_v6.json').write_text(json.dumps(dict(scope='Presentation only; measured values unchanged.',input_sha256=sha256(source),render_source_sha256=sha256(__file__)),indent=2)+'\n')
