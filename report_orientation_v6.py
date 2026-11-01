"""Report the completed orientation ablation, including exact control repeats."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import sha256
from lunargeo_dataset import write_new


def report():
    root=Path('runs/localization-v6-orientation-development');summary=json.loads((root/'summary.json').read_text())
    audit=json.loads(Path('reports/orientation_training_audit_v6.json').read_text())
    if not audit['passed'] or audit['summary_sha256']!=sha256(root/'summary.json'):raise ValueError('Completed orientation audit required')
    runs=[];replications=[]
    for entry in summary['results']:
        folder=root/f"{entry['variant']}-{entry['seed']}";meta=json.loads((folder/'training.json').read_text())
        if sum(meta['training_orientation_counts'].values())!=meta['completed_steps']:raise ValueError('Training orientation count differs')
        if entry['variant']=='true_3d':
            old=Path('runs/localization-v5-coarse-development')/folder.name
            for checkpoint in ('selected.pt','final.pt'):
                a=torch.load(folder/checkpoint,map_location='cpu',weights_only=True);b=torch.load(old/checkpoint,map_location='cpu',weights_only=True)
                if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError('Unchanged wide control did not replicate')
            replications.append(entry['seed'])
        runs.append(dict(variant=entry['variant'],seed=entry['seed'],selected_step=entry['selected_step'],correct=entry['evaluation']['correct_poses'],
            strict=entry['evaluation']['strict_poses'],orientation_counts=meta['training_orientation_counts'],history=meta['history']))
    result=dict(scope='Completed exposed-development orientation ablation; no learned advantage, final test or acceptance claim.',
        runs=runs,wide_controls_exactly_replicated_seeds=replications,completed_optimizer_steps=16000,
        source_sha256=sha256(__file__),summary_sha256=sha256(root/'summary.json'),audit_sha256=sha256('reports/orientation_training_audit_v6.json'))
    write_new('reports/orientation_training_comparison_v6.json',result)
    narrow=next(r for r in runs if r['variant']=='canonical_3d');h=narrow['history'];fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    x=[r['step'] for r in h]
    axes[0].plot(x,[100*r['selection']['correct_at_two_px']/r['selection']['scorable'] for r in h],marker='o',color='#16867d')
    axes[0].set(xlabel='Optimizer step',ylabel='2px precision among co-visible returned points (%)',ylim=(0,100),title='Narrow-orientation point precision')
    axes[1].plot(x,[r['selection']['correct_poses'] for r in h],marker='o',color='#ab6847')
    axes[1].set(xlabel='Optimizer step',ylabel='Correct raw poses / 16 selection pairs',ylim=(0,16),title='The point gain does not improve poses')
    fig.suptitle('Orientation training ablation — exposed development cases only')
    fig.savefig('reports/orientation_training_outcomes_v6.png',dpi=180);plt.close(fig)
    lines='\n'.join(f"| {r['variant']} | {r['seed']} | {r['selected_step']} | {r['correct']}/72 | {r['strict']}/72 |" for r in runs)
    text=f'''# Orientation training ablation: no pose improvement

All four4000-step runs completed (16000 optimizer steps). Training-only geometry canonicalizes the query to its nearest quarter-turn and adds small residual rotation; the control retains the preceding wide-rotation distribution. Model, optimizer, exact3D labels and four-orientation inference are unchanged. No query geometry enters inference.

| Training regime | Seed | Selected step | Correct raw poses | Stricter poses |
|---|---:|---:|---:|---:|
{lines}

The narrow regime retains initial weights and45/72. Its two-pixel point precision improves on the16-case bank during training, but pose recovery does not. The bank selects the wide control; all three wide seeds reproduce the v5 selected and final model tensors exactly. This is a reproducibility check, not three new independent confirmations or a learned gain. Every failed checkpoint remains preserved.

![Point precision and pose recovery](orientation_training_outcomes_v6.png)

The stronger simple SIFT-first fallback remains53/72 with six wrong raw poses. Neither its coverage nor these raw pose counts establish the70% correctly accepted-pose target or1% error requirement. The four selection components/eight geometries produce72 dependent development pairs. No calibration/test pixels, fresh-final validation, modern ranking or verified orbital accuracy is claimed.

The native-resolution study now tests precision and a pair-conditioned no-match outcome. Its100-step pilot gives47/72 versus45/72 frozen, but that small development result is not a consistent gain. An independent corrected exact-in-window diagnostic gives67/72 with oracle projected correspondences; using all visible returned reference points gives72/72. These are query-truth controls, not methods or formal performance bounds. A separate training-source audit finds only3 and23 correct local coarse proposals in two difficult components; uniform projected positives are being tested separately. All work remains development research.

Independent replay validates288 outputs, optimizer completion, actual weight changes, source/cache/checkpoint hashes, selection and all predefined seeds. Own code/weights remain MIT; source data are MoonAnything LunarGeo (Grethen et al., doi:10.5281/zenodo.18415119), CC BY4.0. No human review, paid compute or automatic outreach. The original research goal remains unmet and active.
'''
    Path('reports/ORIENTATION_TRAINING_V6.md').write_text(text);print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':report()
