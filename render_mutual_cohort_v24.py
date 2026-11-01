"""Presentation-only plot of the complete audited three-seed result."""
import argparse
import hashlib
import json
from pathlib import Path


def render(report_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads(Path(report_path).read_text());out=Path(output)
    if out.exists():raise FileExistsError('Preserve earlier figures')
    if report['completed_optimizer_steps']!=6000 or len(report['runs'])!=3 or report['independent_saved_rows_replayed']!=1080:
        raise ValueError('Complete audited three-seed report required')
    fig,ax=plt.subplots(figsize=(10,5.5))
    for record in report['runs']:
        values=record['validation'];ax.plot([v['step']for v in values],
            [v['selection']['correct_poses']for v in values],marker='o',label=str(record['seed']))
    ax.axhline(report['frozen_surface']['correct'],color='grey',linestyle='--',label='Frozen compact + surface (54/72)')
    ax.axhline(report['stronger_simple_sift_surface_correct'],color='black',linestyle=':',label='Stronger SIFT-first pipeline (57/72)')
    ax.set(xlabel='Optimizer updates per seed',ylabel='Correct raw camera poses /72',
        ylim=(0,72),xticks=[0,500,1000,1500,2000],
        title='Coupled-assignment study: no learned gain\n6,000 updates; four exposed terrain components')
    ax.legend(loc='lower left',fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)
    return dict(figure=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),scope='Presentation-only saved-results figure; no new inference.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',default='reports/mutual_cohort_complete_v24.json');p.add_argument('--output',default='reports/mutual_cohort_complete_v24.png');a=p.parse_args();print(json.dumps(render(a.report,a.output)))
