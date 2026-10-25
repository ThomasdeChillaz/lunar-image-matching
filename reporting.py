"""Static figures preserve failed registrations and label controlled data."""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(summary, gallery, output):
    methods = sorted({r['method'] for r in summary})
    cases = sorted({r['case'] for r in summary})
    for field, label, filename in [
        ('success_rate', 'Fraction within configured pixel threshold', 'success.png'),
        ('median_error_registered_px', 'Median error among registered pairs (px)', 'error.png'),
        ('median_runtime_ms', 'Median CPU wall time (ms)', 'runtime.png')]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for method in methods:
            values = [next(r[field] for r in summary if r['method']==method and r['case']==c) for c in cases]
            ax.plot(cases, [np.nan if v is None else v for v in values], marker='o', label=method)
        ax.set(title='Controlled perturbations — not real lunar validation', ylabel=label)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        if field == 'success_rate': ax.set_ylim(-.05, 1.05)
        fig.tight_layout(); fig.savefig(output/filename, dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(gallery), 3, figsize=(9, 3*len(gallery)), squeeze=False)
    for row, (name, reference, query, matrix, error) in zip(axes, gallery):
        aligned = np.zeros_like(reference) if matrix is None else cv2.warpAffine(query, cv2.invertAffineTransform(matrix), (reference.shape[1], reference.shape[0]))
        for ax, image, title in zip(row, [reference, query, aligned], ['Reference', name, 'Query warped to reference']):
            ax.imshow(image, cmap='gray', vmin=0, vmax=255); ax.set_title(title, fontsize=9); ax.axis('off')
        row[2].set_title('FAILED' if error is None else f'Aligned query — error {error:.3f}px', fontsize=9)
    fig.suptitle('SIFT examples — controlled transformations, not lunar validation')
    fig.tight_layout(); fig.savefig(output/'gallery.png', dpi=120); plt.close(fig)
