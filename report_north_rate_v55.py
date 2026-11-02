"""Present audited saved training-fixture metrics; no inference or rescoring."""
import json
from pathlib import Path

from north_optimization_probe_v54 import read, sha


def report():
    out=Path('reports/north_rate_results_v55.json')
    figure=Path('reports/north_rate_training_curves_v55.png')
    if out.exists() or figure.exists():
        raise FileExistsError('Preserve each report rendering')
    variants=[]
    for rate,root,audit_path in [
        (1e-5,'runs/north-v54-fixed-pair-optimization','reports/north_optimization_audit_v54.json'),
        (1e-4,'runs/north-v55-lr-1e-4','reports/north_rate_1e-4_audit_v55.json'),
        (1e-3,'runs/north-v55-lr-1e-3','reports/north_rate_1e-3_audit_v55.json')]:
        audit=read(audit_path);folder=Path(root);summary=read(folder/'summary.json')
        if (not audit['passed'] or audit['neural_snapshot_forwards_replayed']!=20
                or audit['summary_sha256']!=sha(folder/'summary.json')):
            raise ValueError('Complete audited training fixture results required')
        variants.append(dict(coarse_learning_rate=rate,root=root,audit_sha256=sha(audit_path),
            summary_sha256=sha(folder/'summary.json'),logical_updates=400,
            diagnostic_calendar_seconds=summary['calendar_seconds'],
            fixtures=[dict(name=x['fixture']['name'],
                evaluation=[dict(step=e['step'],objective=e['objective'],metrics=e['metrics'])for e in x['evaluation']],
                parameter_group_weight_change_l2=x['parameter_group_weight_change_l2'])for x in summary['fixtures']]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['identity','relighting','viewpoint','relighting_and_viewpoint']
    titles=['Identity','Physical relighting, same camera','Viewpoint change, same light','Relighting and viewpoint change']
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),sharex=True,sharey=True)
    for ax,name,title in zip(axes.ravel(),names,titles):
        for variant,color in zip(variants,('#777777','#0072B2','#D55E00')):
            item=next(x for x in variant['fixtures']if x['name']==name)
            ev=item['evaluation']
            ax.plot([x['step']for x in ev],
                [100*x['metrics']['positive_row_top1_reachable_fraction']for x in ev],
                marker='o',markersize=4,color=color,label=f"coarse LR {variant['coarse_learning_rate']:.0e}")
        ax.set_title(title,fontsize=11);ax.set_ylim(0,103);ax.set_xlim(0,100)
        ax.set_xticks([0,25,50,75,100]);ax.grid(alpha=.2)
    fig.suptitle('TRAINING examples only: effect of coarse learning rate',fontsize=15)
    fig.supxlabel('Updates on the same fixed example',y=.065)
    fig.supylabel('Highest-scoring coarse location within local search (%)',x=.015)
    axes[0,0].legend(loc='lower right',fontsize=9)
    fig.text(.5,.012,'One training example per condition; same initialization and objective. Not held-out camera-pose accuracy.',ha='center',fontsize=9)
    fig.tight_layout(rect=(.035,.09,1,.95));fig.savefig(figure,dpi=160);plt.close(fig)
    result=dict(complete=True,variants=variants,new_logical_updates=800,
        original_control_updates=400,total_actual_snapshot_forwards_replayed=60,
        model_promoted=False,calibration_or_final_pixels_loaded=0,
        figure_sha256=sha(figure),presentation_source_sha256=sha(__file__),
        scope='One fixed TRAINING example per condition. Reachability is within the existing native33px search window; not precise correspondence, camera pose, generalization or SOTA. All three rates and all four conditions are shown.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(report=str(out),figure=str(figure),new_logical_updates=800),indent=2))


if __name__=='__main__': report()
