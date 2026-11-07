"""Render measured extension results without rerunning or changing inference."""
import argparse
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
from physical_lighting_analysis import render as render_lighting
from uncertainty_groups import paired_registration_interval
from report_crosslighting import render as render_crosslighting

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
    return paired_registration_interval(rows,manifest,repetitions)


def render(extended, physical, output, cnsfm=None, tycho=None, transfer=None, crosslighting=None):
    extended, physical, output = Path(extended), Path(physical), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    real = json.loads((extended/'results.json').read_text())
    rows = json.loads((physical/'results.json').read_text())
    summary = json.loads((physical/'summary.json').read_text())
    manifest = json.loads((physical/'frozen_manifest.json').read_text())
    adapter = json.loads((physical/'frozen_adapter.json').read_text())
    policies = json.loads((physical/'frozen_policies.json').read_text())
    lighting = render_lighting(physical, output)
    crossed = render_crosslighting(crosslighting,output/'crosslighting_audit.json',output) if crosslighting else None
    for row in real:
        row['solar_separation_deg'] = sun_separation(row)
    interval = paired_interval(rows, manifest)
    (output/'paired_uncertainty.json').write_text(json.dumps(interval, indent=2))
    write_csv(real, output/'expanded_results.csv', exclude=['reference_points', 'query_points'])
    write_csv(rows, output/'physical_results.csv')
    additional, extra_sources = {}, {}
    for name, folder in [('cnsfm', cnsfm), ('tycho', tycho), ('real_adapter', transfer)]:
        if folder is None:
            continue
        folder = Path(folder)
        complete = json.loads((folder/'COMPLETE.json').read_text())
        if sha256(folder/'results.json') != complete['results_sha256']:
            raise ValueError('Additional experiment completion hash mismatch')
        measurements = json.loads((folder/'results.json').read_text())
        additional[name] = measurements
        extra_sources[name] = sha256(folder/'results.json')
        write_csv(measurements, output/(name+'_results.csv'), exclude=['reference_points', 'query_points'])
        if name != 'tycho':
            # Copy only illumination metadata from the identical base pair.
            # Adapter measurements retain their original saved geometry.
            lookup = {r['pair_id']:r for r in real}
            for row in measurements:
                for key in ['solar_azimuth_difference_deg', 'reference_solar_elevation_deg', 'query_solar_elevation_deg']:
                    row[key] = lookup[row['pair_id']][key]
                row['solar_separation_deg'] = sun_separation(row)
            real += measurements
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
        for method in ['ncc', 'sift', 'disk_lightglue', *(['cnsfm'] if cnsfm else []), *(['disk_lightglue_lunar'] if transfer else [])]:
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
        'The expanded benchmark, physical-render adaptation and the additional experiments listed below have executed. Independent real-image reference validation and verified real-image adaptation gains remain open. The original pilot is preserved.', '',
        '## Physical lighting and adaptation', '',
        f"MoonAnything Hapke renders: 48 training patches/acquisitions, 16 development patches/acquisitions, 16 held-out patches/acquisitions. Geographic longitude bands have wide gaps and source acquisitions do not cross splits. This is within-Tycho geographic transfer. Each evaluation patch has eight lighting/known-warp pairs and one disjoint-terrain negative. Native resolution is 128px. All-zero renders stay in evaluation; uninformative descriptors are excluded from training.", '',
        f"A 128×128 linear descriptor projection ({128*128:,} trainable parameters) was fitted on {adapter['training_pairs']} physical training pairs. DISK detection/backbone and LightGlue stayed frozen. Checkpoint {adapter['selected_step']} was selected using {adapter['development_pairs']} development descriptor pairs; test outcomes were not used for selection.", '',
        '| Method | Correct / 128 | Median error among returned (px) | P95 among returned (px) | Non-overlap cases returning a fit / 16 |',
        '|---|---:|---:|---:|---:|']
    for row in test:
        lines.append(f"| {LABELS[row['method']]} | {row['correct']} | {row['median_error_estimated_px']:.2f} | {row['p95_error_estimated_px']:.2f} | {row['negative_pairs_returning_alignment']} |")
    lo, hi = interval['percentile_interval_95']
    lines += ['', f"The adapter gains {interval['delta']*100:.2f} percentage points over the frozen learned matcher. An exploratory bootstrap interval is [{lo*100:.2f}, {hi*100:.2f}] percentage points ({interval['groups']} components across {interval['source_acquisitions']} source acquisitions). Overlapping declared terrain footprints and shared acquisitions are joined before resampling; an earlier source-only grouping did not account for one overlapping pair. This small synthetic experiment does not establish statistical superiority across lunar regions or transfer to real imagery. SIFT uses its original default preset on low-contrast 128px inputs; no SIFT tuning is claimed.", '',
        '![Physical held-out success](physical_success.png)', '', '## Physical illumination and shared terrain support', '',
        'The following analysis recomputes 3D Sun-direction separation from the physical render metadata and joint non-dark support from the exact raw render bytes. Every failed fit stays in its bin denominator. Bin edges are fixed round intervals for this descriptive analysis after test inspection; no model or rejection threshold is retuned. The same source acquisition contributes multiple lighting cases, so pair counts are not independent sample sizes. The frozen experiment also couples light index to rotation; these strata do not isolate a causal illumination effect.', '',
        '| Held-out condition | Pairs | Acquisitions | Frozen correct | Adapter correct |', '|---|---:|---:|---:|---:|']
    primary = [g for g in lighting['grouped'] if g['split'] == 'test' and
        (g['metric'] == 'solar_separation_deg' or g['intensity_threshold'] == .01)]
    for group in [g for g in primary if g['method'] == 'disk_lightglue' and g['pairs']]:
        adapted_group = next(g for g in primary if g['method'] == 'disk_lightglue_lunar' and g['metric'] == group['metric'] and g['bin_index'] == group['bin_index'])
        scale = 1 if group['metric'] == 'solar_separation_deg' else 100
        label = 'Sun separation' if scale == 1 else 'Non-dark support'
        unit = '°' if scale == 1 else '%'
        lines.append(f"| {label} {group['lower']*scale:g}–{group['upper']*scale:g}{unit} | {group['pairs']} | {group['source_acquisitions']} | {group['correct']}/{group['pairs']} | {adapted_group['correct']}/{group['pairs']} |")
    coverage = lighting['design_coverage']['test']
    lines += ['', f"The held-out physical Sun separations span only {coverage['anchor_min_deg']:.2f}–{coverage['anchor_max_deg']:.2f}°. No cases test 90° or larger Sun separations. The training anchor design likewise stays below {lighting['design_coverage']['train']['anchor_max_deg']:.2f}° before co-visible feature filtering. This is a coverage limitation, not evidence that the model generalizes to opposite lighting.", '',
        'Shared non-dark support means both raw renders exceed I/F=.01 at the same valid terrain pixel before the known warp. It is an intensity proxy, not a geometric Sun-visibility or shadow mask. A case without jointly valid terrain would remain unknown rather than be labelled dark. Most cases have high shared support; sparse low-support bins cannot establish robust shadow invariance.', '',
        '![Physical illumination strata](physical_lighting_strata.png)', '',
        '| Raw I/F threshold | Cases with <25% shared support | Acquisitions | Frozen correct | Adapter correct |', '|---|---:|---:|---:|---:|']
    for row in lighting['sensitivity']:
        lines.append(f"| {row['intensity_threshold']:g} | {row['pairs']} | {row['source_acquisitions']} | {row['correct_by_method']['disk_lightglue']}/{row['pairs']} | {row['correct_by_method']['disk_lightglue_lunar']}/{row['pairs']} |")
    lines += ['', 'The support-threshold sensitivity changes membership of a descriptive subset, not any saved matching outcome. Known non-overlap negatives remain in the rejection analysis below; they have no shared terrain or physical Sun-pair reference and are excluded from these positive-case strata. See [all strata and denominators](physical_lighting_strata.csv) and [input hashes, pair metadata and design coverage](physical_lighting_analysis.json).', '']
    if crossed:
        lines += ['## Every Sun pair with fixed geometry', '',
            f"A completed follow-up evaluates every one of the {crossed['pairs']} physical Sun pairs with the same known translation (6,-4) and zero rotation throughout ({crossed['evaluations']:,} method evaluations). This removes the earlier coupling between light index and rotation. It includes {crossed['above_90_pairs']} pairs at Sun separations of 90° or more. The existing models and original development policies stay unchanged. NCC is included within its translation model, alongside SIFT and the frozen learned variants.", '',
            'The design was frozen before the new outcomes, but the terrain was inspected previously: this is an exploratory controlled follow-up, not a fresh geographic holdout. No model or threshold is selected from the new results.', '',
            '| Method | Correct / pairs | Returned / pairs |', '|---|---:|---:|']
        for row in crossed['methods']:
            lines.append(f"| {LABELS.get(row['method'],row['method'])} | {row['correct']}/{row['pairs']} | {row['returned']}/{row['pairs']} |")
        learned_gate = next(r['transported_rejection'] for r in crossed['methods'] if r['method'] == 'disk_lightglue')
        lines += ['', f"The unchanged frozen learned rejection policy retains only {learned_gate['accepted_correct']} of {learned_gate['correct']} correct fits and accepts {learned_gate['accepted_incorrect']} incorrect fits in this follow-up. Its selected translation bound is smaller than the exact known displacement. The old policy is therefore not a reliable quality filter for this design; it remains experimental. No replacement is selected from these test outcomes.", '']
        lines += ['', '![Every physical Sun pair at fixed geometry](crosslighting_strata.png)', '',
            'See the [fixed-geometry report](CROSS_LIGHTING_REPORT.md) for per-Sun-bin counts, transported-policy errors, grouped uncertainty and an audit that reconstructs the exact translation by slicing the source pixels. These physical results do not establish independent real-image accuracy.', '']
    lines += ['## Rejection trade-off', '',
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
        for method in ['ncc', 'sift', 'disk_lightglue', *(['cnsfm'] if cnsfm else []), *(['disk_lightglue_lunar'] if transfer else [])]:
            selected = [r for r in real if r['region_id'] == region and r['method'] == method and r['preprocessing'] == 'none']
            errors = [r['error_px'] for r in selected if r['error_px'] is not None]
            median = f'{np.median(errors):.2f}' if errors else 'unscorable'
            lines.append(f"| {region} | {LABELS[method]} | {sum(r['estimated'] for r in selected)}/{len(selected)} | {median} |")
    lines += ['', 'All pairs within a scene share observations and terrain, so pair-wise confidence intervals would exaggerate independent evidence. S3 was inspected in the pilot; new results there are exploratory. Real bright-overlap measurements are explicitly intensity-threshold proxies on the provisional map grid. They will be recomputed using independently reviewed alignment when available.', '',
        '![Real illumination diagnostics](real_illumination.png)', '', '## Original CNSFM and open acceptance', '',
        ('The authors’ pinned Windows executable completed all 642 executions: 321 pairs independently matched forward and reverse. All ten shards, ordered source inputs, exact pixel/mask hashes, crater IDs and original output hashes were audited before applying the same one-to-one filtering and partial-affine RANSAC (3px, six inliers) as the other feature methods. Execution failures cannot disappear from denominators. See [Windows artifact hashes](cnsfm_windows_artifacts.json) and [common-protocol audit](cnsfm_common_audit.json).' if cnsfm else 'The authors’ pinned Windows executable completed a native-input probe in all three regions. Common-grid execution and scoring remain open.'), '',
        'The released YOLOv9 crater checkpoint produced 2,213 predictions on all 42 identical masked 512px inputs. Whole-image redetection differs from the authors’ native tiled detector pipeline. The paper reports native-input success rates of 100%, 100% and 72.3% for S1/S2/S3, using more than three correct matches as success. Our shared six-inlier geometry criterion, resolution and provisional map scoring differ; neither the published percentages nor our returned-fit rates establish a directly comparable accuracy ranking. No paper-percentage reproduction is claimed.', '',
        'The blank, source-bound [landmark review packet](landmark_review.html) requires human input. It supports 10–20 distributed landmarks, per-image uncertainty, explicit unscorable cases and optional independent second review. No generated or matcher-derived coordinates have been imported as independent references. Real-data rejection calibration, verified registration accuracy and verified real-image adaptation improvement remain open.', '']
    if tycho:
        values = additional['tycho']
        lines += ['## Additional real regions and adapter transfer', '',
            'A checksummed 4.68 GB appearance archive supplied real orthorectified LRO crops, distinct from the Hapke renders. Metadata-only selection froze 54 images from 32 source acquisitions and 27 repeat pairs in five separated Tycho boxes (11 dev, 16 test). No acquisition occurs in the physical train/dev/test sample; no selected real footprint overlaps physical training or development terrain. Real dev/test source groups and footprints are separated. The selection used overlap and Sun-angle strata before matcher outcomes, with available stratum counts retained rather than filled with favorable matches.', '',
            'Each 128px common grid covers the metadata intersection of two original 128px appearance crops. This resampling and geolocation are provisional. These five areas add real terrain beyond the original three CNSFM scenes, but they do not represent five globally independent lunar regions. Physical test terrain may overlap; this transfer experiment is exploratory. The adapter and physical rejection policies were frozen before real outcomes. Transporting a policy is not real-data calibration.', '',
            '| Real split | Method | Returned / pairs | Median map disagreement among returned (128-grid px) | Accepted by transported physical policy |',
            '|---|---|---:|---:|---:|']
        for split in ['dev', 'test']:
            for method in ['sift', 'disk_lightglue', 'disk_lightglue_clahe', 'disk_lightglue_lunar']:
                selected = [r for r in values if r['split'] == split and r['method'] == method]
                errors = [r['error_px'] for r in selected if r['error_px'] is not None]
                median = f'{np.median(errors):.2f}' if errors else '—'
                lines.append(f"| {split} | {LABELS[method]} | {sum(r['estimated'] for r in selected)}/{len(selected)} | {median} | {sum(r.get('accepted_by_transported_physical_policy', False) for r in selected)} |")
        lines += ['', 'Results vary by area and do not establish consistent real-image improvement. A lower median among returned fits can also hide failures. Neither acceptance nor map agreement proves a correct registration. Independent landmarks are required to resolve this. See [real Tycho measurements](tycho_results.csv) and [input/geometry audit](tycho_transfer_audit.json).', '']
    if transfer:
        lines += ['The same frozen adapter was also evaluated on all 321 original CNSFM pairs at the common 512px resolution. Its measurements appear in the scene table above and [real adapter measurements](real_adapter_results.csv); see the [real adapter audit](cnsfm_transfer_audit.json). No physical 128px policy was silently rescaled to 512px. Human annotations from the pilot can be converted through known map-grid affines to score the corresponding pair subset; matcher-estimated transforms never supply that conversion.', '',
            '**The current adapter is not supported as a real-image improvement.** On polar S3 it returns only 99/231 fits versus 158/231 for the frozen matcher, a loss of 59 returned alignments. A lower map-disagreement median among its surviving fits cannot offset that missing coverage or establish better accuracy. Keep the frozen matcher as the default; retain the adapter as an experimental result. Future adaptation should be tested across image scales and constrained against descriptor collapse using development data, with a fresh geographic holdout for any subsequent claim.', '']
    lines += [
        '## Reproducibility and provenance', '',
        'See [measurement audit](extension_audit.json), [per-evaluation real diagnostics](expanded_results.csv), [physical measurements](physical_results.csv), [adapter](lunar_adapter.json), [frozen policies](frozen_policies.json) and [paired uncertainty](paired_uncertainty.json). Original complete run JSON remains under the fresh run directories; published CSV omits bulky feature-coordinate arrays. Inference sources, input hashes, model locks and metric recomputation were audited. Cached learned extraction timings are descriptive and are not a fair cross-hardware speed comparison.', '',
        'Physical data: Grethen et al., [MoonAnything](https://arxiv.org/html/2604.00682v1), [Zenodo 18415119](https://zenodo.org/records/18415119), CC BY 4.0. Real imagery: [CNSFM research release](https://github.com/Bin501/CNSFM), [Li et al. 2025](https://doi.org/10.3390/rs17132302). No NASA/JPL affiliation or navigation certification is implied.', '',
        f"Generated from saved measurements at {datetime.now(timezone.utc).isoformat()}."]
    (output/'EXTENSION_REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'report_provenance.json').write_text(json.dumps(dict(renderer_sha256=sha256(__file__),
        extended_result_sha256=sha256(extended/'results.json'), physical_result_sha256=sha256(physical/'results.json'),
        additional_result_sha256=extra_sources,
        physical_lighting_analysis_sha256=sha256(output/'physical_lighting_analysis.json'),
        crosslighting_results_sha256=sha256(Path(crosslighting)/'results.json') if crosslighting else None,
        uncertainty_group_source_sha256=sha256(Path(__file__).with_name('uncertainty_groups.py')),
        report_sha256=sha256(output/'EXTENSION_REPORT.md')), indent=2))
    print(output/'EXTENSION_REPORT.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extended-run', default='runs/expanded-confidence-v1')
    p.add_argument('--physical-run', default='runs/physical-adaptation-v1')
    p.add_argument('--output', default='reports')
    p.add_argument('--cnsfm-run')
    p.add_argument('--tycho-run')
    p.add_argument('--transfer-run')
    p.add_argument('--crosslighting-run')
    args = p.parse_args()
    render(args.extended_run, args.physical_run, args.output, args.cnsfm_run, args.tycho_run, args.transfer_run, args.crosslighting_run)
