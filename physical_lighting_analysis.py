"""Analyze frozen known-geometry fits against physical lighting and support.

This is a descriptive analysis after test inspection, not model or policy tuning.
Positive I/F support is an intensity proxy, never a geometric shadow annotation.
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from dataset import sha256
from moon_dataset import load_image, validate

SUN_EDGES = [0., 30., 60., 90., 120., 150., 180.]
SUPPORT_EDGES = [0., .05, .25, .5, .75, 1.]
SUPPORT_THRESHOLDS = [0., .005, .01, .02]
LABELS = {'sift':'SIFT', 'disk_lightglue':'DISK + LightGlue',
    'disk_lightglue_clahe':'DISK + LightGlue + CLAHE', 'disk_lightglue_lunar':'Lunar adapter'}


def solar_separation(first, second):
    angles = [first['azimuth_deg'], first['elevation_deg'], second['azimuth_deg'], second['elevation_deg']]
    if not np.isfinite(angles).all() or any(abs(i['elevation_deg']) > 90 for i in [first, second]):
        raise ValueError('Invalid local solar angles')
    az0, el0, az1, el1 = np.deg2rad(angles)
    dot = np.sin(el0)*np.sin(el1)+np.cos(el0)*np.cos(el1)*np.cos(az0-az1)
    return float(np.rad2deg(np.arccos(np.clip(dot, -1, 1))))


def shared_support(first, second, valid_first, valid_second, threshold):
    if first.shape != second.shape or first.shape != valid_first.shape or second.shape != valid_second.shape or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid shared-support inputs')
    valid = valid_first & valid_second & np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return None
    return float(((first > threshold) & (second > threshold) & valid).sum()/valid.sum())


def bin_index(value, edges):
    if value is None:
        return None
    if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
        raise ValueError('Lighting value outside fixed bins')
    return min(int(np.searchsorted(edges, value, side='right')-1), len(edges)-2)


def aggregate(rows, metadata, methods):
    positive = [r for r in rows if r['dataset_kind'] == 'physical_render']
    expected = {(pair, method) for pair in metadata for method in methods}
    identities = [(r['pair_id'], r['method']) for r in positive]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Missing or repeated physical method/case outcomes')
    for row in positive:
        if row.get('label_source') != 'exact_geometry' or type(row.get('correct')) is not bool:
            raise ValueError('Known geometry required; unknown real labels cannot become accuracy')
        if row['split'] != metadata[row['pair_id']]['split']:
            raise ValueError('Physical split drift')
    specs = [('solar_separation_deg', None, SUN_EDGES)] + [
        ('shared_nondark_fraction', threshold, SUPPORT_EDGES) for threshold in SUPPORT_THRESHOLDS]
    grouped = []
    for split in ['dev', 'test']:
        for metric, threshold, edges in specs:
            for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
                selected = {p for p,m in metadata.items() if m['split'] == split and bin_index(
                    m['solar_separation_deg'] if threshold is None else m['shared_support'][str(threshold)], edges) == index}
                for method in methods:
                    subset = [r for r in positive if r['method'] == method and r['pair_id'] in selected]
                    grouped.append(dict(split=split, metric=metric, intensity_threshold=threshold,
                        bin_index=index, lower=low, upper=high, upper_inclusive=index == len(edges)-2,
                        method=method, pairs=len(subset), source_acquisitions=len({metadata[p]['observation_group'] for p in selected}),
                        returned=sum(r['estimated'] for r in subset), correct=sum(r['correct'] for r in subset),
                        no_fit=sum(not r['estimated'] for r in subset),
                        success_rate=sum(r['correct'] for r in subset)/len(subset) if subset else None,
                        dark_queries=sum(metadata[p]['query_zero_render'] for p in selected)))
    return grouped


def analyze(folder, root='.'):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    if sha256(folder/'results.json') != complete['result_sha256']:
        raise ValueError('Physical result hash mismatch')
    config = json.loads((folder/'frozen_config.json').read_text())
    rows = json.loads((folder/'results.json').read_text())
    if complete['rows'] != len(rows):
        raise ValueError('Physical completion count mismatch')
    for name, digest in config['source_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Physical inference source changed: '+name)
    if config['threshold_px'] != 3 or config['image_size'] != 128:
        raise ValueError('Unexpected frozen physical scoring protocol')
    manifest = json.loads((folder/'frozen_manifest.json').read_text())
    validate(manifest, root)
    metadata, design_coverage = {}, {}
    for split in ['train', 'dev', 'test']:
        selected = [t for t in manifest['tiles'] if t['split'] == split]
        anchor = [solar_separation(t['images'][0], image) for t in selected for image in t['images'][1:]]
        possible = [solar_separation(a,b) for t in selected for a,b in itertools.combinations(t['images'], 2)]
        design_coverage[split] = dict(available_anchor_pairs=len(anchor), anchor_min_deg=min(anchor), anchor_max_deg=max(anchor),
            possible_all_light_pairs=len(possible), possible_max_deg=max(possible),
            warning='Metadata design coverage before co-visible feature filtering. All-light combinations are potential inputs, not extra executed experiments.')
    for tile in manifest['tiles']:
        if tile['split'] not in ('dev', 'test'):
            continue
        _, ma, first = load_image(tile['images'][0], root)
        for index, image in enumerate(tile['images'][1:], 1):
            _, mb, second = load_image(image, root)
            pair = tile['tile_id']+f'_light_{index}'
            metadata[pair] = dict(pair_id=pair, tile_id=tile['tile_id'], observation_group=tile['observation_group'],
                split=tile['split'], solar_separation_deg=solar_separation(tile['images'][0], image),
                reference_solar_elevation_deg=tile['images'][0]['elevation_deg'], query_solar_elevation_deg=image['elevation_deg'],
                query_zero_render=image['constant_zero_render'], shared_support={str(t):shared_support(first, second, ma, mb, t) for t in SUPPORT_THRESHOLDS})
    for row in rows:
        if row['dataset_kind'] != 'physical_render':
            continue
        expected = row['error_px'] is not None and row['error_px'] <= config['threshold_px']
        if row['correct'] != expected or row['estimated'] != (row['error_px'] is not None):
            raise ValueError('Known-geometry outcome disagrees with frozen scoring')
        support = metadata[row['pair_id']]['shared_support']['0.01']
        if support is None or not np.isclose(support, row['shared_nondark_fraction'], atol=1e-12, rtol=0):
            raise ValueError('Recomputed non-dark terrain support differs from saved measurement')
    grouped = aggregate(rows, metadata, config['methods'])
    sensitivity = []
    for threshold in SUPPORT_THRESHOLDS:
        selected = {p for p,m in metadata.items() if m['split'] == 'test' and m['shared_support'][str(threshold)] is not None and m['shared_support'][str(threshold)] < .25}
        sensitivity.append(dict(intensity_threshold=threshold, condition='shared support <25% of jointly valid terrain',
            pairs=len(selected), source_acquisitions=len({metadata[p]['observation_group'] for p in selected}),
            correct_by_method={method:sum(r['correct'] for r in rows if r['pair_id'] in selected and r['method'] == method) for method in config['methods']}))
    return dict(protocol=dict(sun_bin_edges_deg=SUN_EDGES, support_bin_edges=SUPPORT_EDGES,
        support_thresholds=SUPPORT_THRESHOLDS, primary_support_threshold=.01,
        geometry='Same co-rendered terrain before the known 2D warp; fixed native 128px evaluation and 3px correctness criterion.',
        interpretation='Exploratory descriptive association after test inspection. No retraining or policy selection. Cases share source acquisitions; counts are not independent sample sizes.',
        geometry_confound='The frozen design couples light index to -8/0/+8 degree rotation plus fixed translation. Strata may reflect geometry as well as illumination; these are not isolated causal lighting effects.',
        support='Intensity proxy from raw physical I/F values, not binary Sun visibility, shadow truth, or a causal lighting estimate. Missing valid area remains unknown.'),
        source=dict(result_sha256=complete['result_sha256'], manifest_sha256=sha256(folder/'frozen_manifest.json'),
            config_sha256=sha256(folder/'frozen_config.json'), analysis_sha256=sha256(__file__)),
        unique_physical_cases=len(metadata), methods=config['methods'], grouped=grouped, sensitivity=sensitivity,
        design_coverage=design_coverage,
        excluded_nonoverlap_cases=len({r['pair_id'] for r in rows if r['dataset_kind'] == 'physical_nonoverlap'}),
        pair_metadata=list(metadata.values()))


def render(folder, output, root='.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = analyze(folder, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'physical_lighting_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    groups = result['grouped']
    with (output/'physical_lighting_strata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader(); writer.writerows(groups)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    colors = {'sift':'#bc9475', 'disk_lightglue':'#2d83b8', 'disk_lightglue_clahe':'#91b8c7', 'disk_lightglue_lunar':'#34a698'}
    for ax, metric, threshold, label in zip(axes, ['solar_separation_deg', 'shared_nondark_fraction'], [None, .01],
            ['3D Sun-direction separation (degrees)', 'Shared non-dark terrain (%) at I/F > .01; intensity proxy']):
        labels = []
        for method in result['methods']:
            selected = [g for g in groups if g['split'] == 'test' and g['metric'] == metric and g['intensity_threshold'] == threshold and g['method'] == method]
            ax.plot(range(len(selected)), [g['success_rate']*100 if g['success_rate'] is not None else np.nan for g in selected], marker='o', label=LABELS[method], color=colors[method])
            if not labels:
                multiplier = 1 if threshold is None else 100
                labels = [f"{g['lower']*multiplier:g}–{g['upper']*multiplier:g}\nn={g['pairs']}\nacq={g['source_acquisitions']}" for g in selected]
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel(label, fontsize=9); ax.set_ylim(-3, 103); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Correct alignments within 3 native pixels (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.93), ncol=4, fontsize=8)
    fig.suptitle('Frozen physical-lighting experiment: failures stay in every bin denominator', fontsize=12)
    fig.text(.5,.02, 'Exploratory after test inspection; 128 held-out cases from 16 Tycho acquisitions. Empty bins have no rate.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.87]); fig.savefig(output/'physical_lighting_strata.png', dpi=170); plt.close(fig)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='runs/physical-adaptation-v1')
    p.add_argument('--output', default='reports')
    p.add_argument('--root', default='.')
    args = p.parse_args()
    result = render(args.run, args.output, args.root)
    print('Analyzed', result['unique_physical_cases'], 'known-geometry cases; no matcher was rerun or tuned')
