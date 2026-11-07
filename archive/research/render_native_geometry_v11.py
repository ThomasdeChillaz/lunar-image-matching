"""Presentation-only rendering of preserved, audited native measurements."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def render():
    specs = [
        ('Frozen', 'runs/localization-v7-native-initial-control/initial_results.json.gz',
         'reports/native_initial_equivalence_v7.json', 'new_results_sha256'),
        ('Learned\nseed01', 'runs/localization-v7-native-development/pair_refinement-20261101/development_results.json.gz',
         'runs/localization-v7-native-development/pair_refinement-20261101/COMPLETE.json', 'results_sha256'),
        ('Affine', 'runs/localization-v10-affine-development/results.json.gz',
         'reports/native_affine_audit_v10.json', 'results_sha256'),
        ('Surface', 'runs/localization-v11-surface-warp-development/results.json.gz',
         'reports/native_surface_audit_v11.json', 'results_sha256')]
    data = []
    for name, path, audit_path, key in specs:
        audit = json.loads(Path(audit_path).read_text())
        if audit[key] != sha256(path):
            raise ValueError('Preserved native measurement changed')
        with gzip.open(path, 'rt') as f:
            rows = json.load(f)
        data.append(dict(name=name, correct=sum(r['pose_correct'] for r in rows), strict=sum(r['strict_pose_correct'] for r in rows),
            wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows), cases=len(rows),
            returned=sum(r['returned'] for r in rows), scorable=sum(r['scorable'] for r in rows),
            correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows), input_path=path, input_sha256=sha256(path)))
    if any(r['cases'] != 72 for r in data):
        raise ValueError('Native figure requires all72 cases')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    colors = ['#597b99', '#a57652', '#8979ad', '#218c7f']
    for i, (row, color) in enumerate(zip(data, colors)):
        precision = 100*row['correct_at_two_px']/row['scorable']
        coverage = 100*row['correct']/72
        axes[0].scatter(precision, coverage, c=color, s=65)
        axes[0].annotate(row['name'].replace('\n', ' '), (precision, coverage),
            xytext=(8, -17 if i == 1 else 8), textcoords='offset points', fontsize=9)
    axes[0].set(xlim=(44, 76), ylim=(55, 83), xlabel='2px precision among co-visible returned points (%)',
        ylabel='Correct raw camera poses / all72 pairs (%)', title='Geometry improves the exposed-development result')
    x = np.arange(len(data)); width = .36
    axes[1].bar(x-width/2, [r['correct'] for r in data], width, color='#218c7f', label='Primary pose criterion')
    axes[1].bar(x+width/2, [r['strict'] for r in data], width, color='#597b99', label='Stricter criterion')
    axes[1].set(xticks=x, xticklabels=[r['name'] for r in data], ylim=(0, 72), ylabel='Correct raw poses / all72 pairs',
        title='The gain is not uniform precision dominance')
    axes[1].legend(fontsize=8)
    fig.suptitle('Native controls — 8 geometries / 4 exposed terrain components', fontsize=12)
    fig.supxlabel('Raw proposals, not calibrated acceptance or fresh-test accuracy. Returned-point support differs.', fontsize=9)
    output = Path('reports/native_geometry_controls_v11.png'); fig.savefig(output, dpi=170); plt.close(fig)
    write_new('reports/native_geometry_figure_v11.json', dict(scope='Presentation-only: measurements unchanged.',
        rows=data, source_sha256=sha256(__file__), image_sha256=sha256(output)))


if __name__ == '__main__':
    render()
