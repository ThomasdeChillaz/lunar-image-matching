"""Render measured extension results without rerunning or changing inference."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataset import sha256
from moon_dataset import load_image

LABELS = {'ncc':'NCC', 'sift':'SIFT', 'disk_lightglue':'DISK + LightGlue',
    'disk_lightglue_clahe':'DISK + LightGlue + CLAHE', 'disk_lightglue_lunar':'Lunar descriptor adapter', 'cnsfm':'Original CNSFM / common grid'}


def write_csv(rows, path, exclude=()):
    fields = sorted(set().union(*(r.keys() for r in rows))-set(exclude))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items() if key in fields})


def sun_separation(row):
    az, a, b = np.deg2rad([row['solar_azimuth_difference_deg'], row['reference_solar_elevation_deg'], row['query_solar_elevation_deg']])
    return float(np.rad2deg(np.arccos(np.clip(np.sin(a)*np.sin(b)+np.cos(a)*np.cos(b)*np.cos(az), -1, 1))))


def paired_interval(rows, manifest, repetitions=5000):
    tiles = {t['tile_id']:t for t in manifest['tiles']}
    grouped = defaultdict(lambda:[0, 0, 0])
    for row in rows:
        if row['split'] != 'test' or row['dataset_kind'] != 'physical_render' or row['method'] not in ('disk_lightglue', 'disk_lightglue_lunar'):
            continue
        group = grouped[tiles[row['tile_id']]['observation_group']]
        group[0 if row['method'] == 'disk_lightglue' else 1] += int(row['correct'])
        if row['method'] == 'disk_lightglue':
            group[2] += 1
    values = np.asarray(list(grouped.values()))
    rng = np.random.default_rng(20261029)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].sum(axis=1)
    deltas = (samples[:, 1]-samples[:, 0])/samples[:, 2]
    return dict(groups=len(values), grouping='Source acquisition; eight illumination pairs per tile',
        delta=float((values[:, 1].sum()-values[:, 0].sum())/values[:, 2].sum()),
        percentile_interval_95=np.percentile(deltas, [2.5, 97.5]).tolist(), bootstrap_repetitions=repetitions,
        warning='Exploratory small-sample interval within one Tycho geographic holdout; not evidence of global or real-image generalization.')


def render(extended, physical, output):
    extended, physical, output = Path(extended), Path(physical), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    real = json.loads((extended/'results.json').read_text())
    rows = json.loads((physical/'results.json').read_text())
    summary = json.loads((physical/'summary.json').read_text())
    manifest = json.loads((physical/'frozen_manifest.json').read_text())
    adapter = json.loads((physical/'frozen_adapter.json').read_text())
    policies = json.loads((physical/'frozen_policies.json').read_text())
    for row in real:
        row['solar_separation_deg'] = sun_separation(row)
    interval = paired_interval(rows, manifest)
    (output/'paired_uncertainty.json').write_text(json.dumps(interval, indent=2))
    write_csv(real, output/'expanded_results.csv', exclude=['reference_points', 'query_points'])
    write_csv(rows, output/'physical_results.csv')
    shutil.copyfile(physical/'frozen_adapter.json', output/'lunar_adapter.json')
    shutil.copyfile(physical/'frozen_policies.json', output/'frozen_policies.json')
    shutil.copyfile(physical/'summary.json', output/'physical_summary.json')
    test = [r for r in summary['registration'] if r['split'] == 'test']
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = [r['method'] for r in test]
    ax.bar(range(len(test)), [r['success_rate']*100 for r in test], color=['#2d83b8', '#91b8c7', '#34a698', '#bc9475'])
    for index, row in enumerate(test):
        ax.text(index, row['success_rate']*100+2, f"{row['correct']}/{row['positive_pairs']}", ha='center')
    ax.set_xticks(range(len(test)), [LABELS[m].replace(' + ', '\n+ ') for m in names], fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel('Correct alignments within 3 native pixels (%)')
    ax.set_title('Held-out physical lighting: 128 pairs from 16 Tycho patches')
    fig.tight_layout(); fig.savefig(output/'physical_success.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for index, method in enumerate(names):
        subset = [r for r in rows if r['split'] == 'test' and r['method'] == method]
        raw_good = sum(r['estimated'] and r['correct'] for r in subset)
        raw_bad = sum(r['estimated'] and not r['correct'] for r in subset)
        good = sum(r.get('accepted', False) and r['correct'] for r in subset)
        bad = sum(r.get('accepted', False) and not r['correct'] for r in subset)
        ax.bar(index-.18, raw_good, width=.32, color='#2d83b8', label='Correct' if index == 0 else None)
        ax.bar(index-.18, raw_bad, width=.32, bottom=raw_good, color='#d27954', label='Incorrect' if index == 0 else None)
        ax.bar(index+.18, good, width=.32, color='#2d83b8', hatch='//')
        ax.bar(index+.18, bad, width=.32, bottom=good, color='#d27954', hatch='//')
        ax.text(index-.18, raw_good+raw_bad+3, 'Raw', ha='center', fontsize=8)
        ax.text(index+.18, good+bad+3, 'Gated', ha='center', fontsize=8)
    ax.set_xticks(range(len(names)), [LABELS[m].replace(' + ', '\n+ ') for m in names], fontsize=9)
    ax.set_ylim(0, 150); ax.set_ylabel('Returned alignments, out of 144 test cases')
    ax.set_title('Frozen development-selected rejection: retention versus incorrect acceptance'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'rejection_tradeoff.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    bins = np.arange(0, 181, 30)
    for ax, region in zip(axes, ['S1', 'S2', 'S3']):
        for method in ['ncc', 'sift', 'disk_lightglue']:
            subset = [r for r in real if r['region_id'] == region and r['method'] == method and r['preprocessing'] == 'none']
            x, rates = [], []
            for low, high in zip(bins[:-1], bins[1:]):
                selected = [r for r in subset if low <= r['solar_separation_deg'] < high+(1e-9 if high == 180 else 0)]
                if selected:
                    x.append((low+high)/2); rates.append(sum(r['estimated'] for r in selected)/len(selected))
            ax.plot(x, rates, marker='o', label=LABELS[method])
        ax.set_title(region); ax.set_xlabel('3D Sun-direction separation (degrees)'); ax.set_ylim(-.03, 1.03)
    axes[0].set_ylabel('Fraction returning an alignment'); axes[0].legend(fontsize=7)
    fig.suptitle('Real repeat observations: returned alignments are not verified correct alignments', fontsize=10)
    fig.tight_layout(); fig.savefig(output/'real_illumination.png', dpi=170); plt.close(fig)
    tile = next(t for t in manifest['tiles'] if t['split'] == 'train')
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, index in zip(axes, [0, 3, 6, 8]):
        byte, valid, values = load_image(tile['images'][index])
        ax.imshow(byte, cmap='gray', vmin=0, vmax=255); ax.axis('off')
        image = tile['images'][index]
        ax.set_title(f"Sun azimuth {image['azimuth_deg']:.1f}°\nelevation {image['elevation_deg']:.1f}°", fontsize=9)
    fig.suptitle('Same terrain and camera, different physical illumination; fixed display mapping', fontsize=10)
    fig.tight_layout(); fig.savefig(output/'physical_lighting_gallery.png', dpi=170); plt.close(fig)
    lines = ['# Lunar research extension: measured results', '',
        'The expanded benchmark and physical-render adaptation experiment have executed. Independent real-image reference validation and real-image adaptation transfer remain open. The original pilot is preserved.', '',
        '## Physical lighting and adaptation', '',
        f"MoonAnything Hapke renders: 48 training patches/acquisitions, 16 development patches/acquisitions, 16 held-out patches/acquisitions. Geographic longitude bands have wide gaps and source acquisitions do not cross splits. This is within-Tycho geographic transfer. Each evaluation patch has eight lighting/known-warp pairs and one disjoint-terrain negative. Native resolution is 128px. All-zero renders stay in evaluation; uninformative descriptors are excluded from training.", '',
        f"A 128×128 linear descriptor projection ({128*128:,} trainable parameters) was fitted on {adapter['training_pairs']} physical training pairs. DISK detection/backbone and LightGlue stayed frozen. Checkpoint {adapter['selected_step']} was selected using {adapter['development_pairs']} development descriptor pairs; test outcomes were not used for selection.", '',
        '| Method | Correct / 128 | Median error among returned (px) | P95 among returned (px) | Non-overlap cases returning a fit / 16 |',
        '|---|---:|---:|---:|---:|']
    for row in test:
        lines.append(f"| {LABELS[row['method']]} | {row['correct']} | {row['median_error_estimated_px']:.2f} | {row['p95_error_estimated_px']:.2f} | {row['negative_pairs_returning_alignment']} |")
    lo, hi = interval['percentile_interval_95']
    lines += ['', f"The adapter gains {interval['delta']*100:.2f} percentage points over the frozen learned matcher. An exploratory source-acquisition-clustered bootstrap interval is [{lo*100:.2f}, {hi*100:.2f}] percentage points ({interval['groups']} groups). This small synthetic experiment does not establish statistical superiority across lunar regions or transfer to real imagery. SIFT uses its original default preset on low-contrast 128px inputs; no SIFT tuning is claimed.", '',
        '![Physical held-out success](physical_success.png)', '', '## Rejection trade-off', '',
        'Each method’s policy was selected from development outcomes with zero accepted incorrect development outcomes. Policies were saved before test evaluation. That development constraint did not generalize perfectly:', '',
        '| Method | Correct fits retained | Incorrect fits accepted | Disjoint-terrain fits accepted / 16 |', '|---|---:|---:|---:|']
    for method, kinds in summary['heldout_rejection'].items():
        p, n = kinds['physical_render'], kinds['physical_nonoverlap']
        lines.append(f"| {LABELS[method]} | {p['accepted_correct']}/{p['correct']} | {p['accepted_incorrect']} | {n['accepted_incorrect']}/16 |")
    lines += ['', 'Both rejection errors and discarded correct fits matter. Zero accepted disjoint-terrain fits in 16 cases is not a safety guarantee. Incorrect physical outcomes include failed registrations; the plot separates returned incorrect alignments from failures to return any fit.', '',
        '![Rejection trade-off](rejection_tradeoff.png)', '', '## Real repeat observations and Sun direction', '',
        'All 42 CNSFM source observations and 321 image pairs were evaluated with three methods and none/CLAHE preprocessing: 1,926 bidirectional evaluations. Sun direction is derived at each crop center from archived LROC subsolar metadata, retaining source-page hashes and 0.01° metadata resolution. It is a spherical geometric direction, not terrain illumination or a fresh SPICE solution. The plot uses 3D separation, so an azimuth flip near solar zenith is not mistaken for opposite illumination.', '',
        'Source-map co-registration uncertainty is unknown. The following primary, no-CLAHE results describe provisional map disagreement, not true registration accuracy:', '',
        '| Scene | Method | Returned / pairs | Median map disagreement among returned (px) |', '|---|---|---:|---:|']
    for region in ['S1', 'S2', 'S3']:
        for method in ['ncc', 'sift', 'disk_lightglue']:
            selected = [r for r in real if r['region_id'] == region and r['method'] == method and r['preprocessing'] == 'none']
            errors = [r['error_px'] for r in selected if r['error_px'] is not None]
            median = f'{np.median(errors):.2f}' if errors else 'unscorable'
            lines.append(f"| {region} | {LABELS[method]} | {sum(r['estimated'] for r in selected)}/{len(selected)} | {median} |")
    lines += ['', 'All pairs within a scene share observations and terrain, so pair-wise confidence intervals would exaggerate independent evidence. S3 was inspected in the pilot; new results there are exploratory. Real bright-overlap measurements are explicitly intensity-threshold proxies on the provisional map grid. They will be recomputed using independently reviewed alignment when available.', '',
        '![Real illumination diagnostics](real_illumination.png)', '', '## Original CNSFM and open acceptance', '',
        'The authors’ pinned Windows executable completed a native-input probe in all three regions. Output coordinates were audited against the published detector files, including the executable’s window-local ID convention. The released YOLOv9 crater checkpoint was separately run on all 42 identical masked 512px inputs. Common-grid CNSFM execution, shared-geometry scoring and the final comparison remain open until the Windows run is verified. Whole-image redetection at the common resolution differs from the authors’ native tiled detector pipeline; no paper-percentage reproduction is claimed.', '',
        'The blank, source-bound [landmark review packet](landmark_review.html) requires human input. It supports 10–20 distributed landmarks, per-image uncertainty, explicit unscorable cases and optional independent second review. No generated or matcher-derived coordinates have been imported as independent references. Real-data rejection calibration, verified registration accuracy and real-image adaptation transfer remain open.', '',
        '## Reproducibility and provenance', '',
        'See [measurement audit](extension_audit.json), [per-evaluation real diagnostics](expanded_results.csv), [physical measurements](physical_results.csv), [adapter](lunar_adapter.json), [frozen policies](frozen_policies.json) and [paired uncertainty](paired_uncertainty.json). Original complete run JSON remains under the fresh run directories; published CSV omits bulky feature-coordinate arrays. Inference sources, input hashes, model locks and metric recomputation were audited. Cached learned extraction timings are descriptive and are not a fair cross-hardware speed comparison.', '',
        'Physical data: Grethen et al., [MoonAnything](https://arxiv.org/html/2604.00682v1), [Zenodo 18415119](https://zenodo.org/records/18415119), CC BY 4.0. Real imagery: [CNSFM research release](https://github.com/Bin501/CNSFM), [Li et al. 2025](https://doi.org/10.3390/rs17132302). No NASA/JPL affiliation or navigation certification is implied.', '',
        f"Generated from saved measurements at {datetime.now(timezone.utc).isoformat()}."]
    (output/'EXTENSION_REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'report_provenance.json').write_text(json.dumps(dict(renderer_sha256=sha256(__file__),
        extended_result_sha256=sha256(extended/'results.json'), physical_result_sha256=sha256(physical/'results.json'),
        report_sha256=sha256(output/'EXTENSION_REPORT.md')), indent=2))
    print(output/'EXTENSION_REPORT.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extended-run', default='runs/expanded-confidence-v1')
    p.add_argument('--physical-run', default='runs/physical-adaptation-v1')
    p.add_argument('--output', default='reports')
    args = p.parse_args()
    render(args.extended_run, args.physical_run, args.output)
