"""Report the completed fixed36+36 native comparisons, without a SOTA claim."""
import argparse
import gzip
import json
from pathlib import Path

from combine_native_baselines_v8 import select_assigned, aggregates
from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval


def report(artifact_root):
    artifact_root = Path(artifact_root)
    manifest = json.loads(Path('reports/localization_training_data_v5.json').read_text())
    ids = [c['pair_id'] for c in manifest['cases'] if c['split'] == 'model_selection']
    surface_root = Path('runs/localization-v11-surface-warp-development')
    surface_audit = json.loads(Path('reports/native_surface_audit_v11.json').read_text())
    if not surface_audit['passed'] or surface_audit['results_sha256'] != sha256(surface_root/'results.json.gz'):
        raise ValueError('Completed surface control audit required')
    with gzip.open(surface_root/'results.json.gz', 'rt') as f:
        surface = {r['pair_id']: r for r in json.load(f)}
    tables = []
    for method in ('loma_r', 'romav2'):
        path = Path('reports')/f"native_{'loma' if method == 'loma_r' else 'romav2'}_comparison_v10.json"
        completed = json.loads(path.read_text())
        if not completed['passed'] or completed['eligible_pairs'] != 72 or completed['replayed_records'] != 144:
            raise ValueError('All assigned native pairs must pass replay')
        roots = [artifact_root/f'native-v7-{method}-common'/'native-comparison',
                 artifact_root/f'native-v8-{method}-common-36-36'/'native-comparison']
        collections = []
        for root, evidence in zip(roots, completed['source_runs']):
            files = sorted((root/'cases').glob('*.json'))
            if {p.name: sha256(p) for p in files} != evidence['case_sha256']:
                raise ValueError('Preserved modern case bytes differ')
            collections.append([json.loads(p.read_text()) for p in files])
        rows = select_assigned(*collections, ids, method)
        if aggregates(rows, method) != completed['results']:
            raise ValueError('Complete native aggregate differs')
        modern = {r['pair_id']: r for r in rows if r['method'] == method}
        compact = {r['pair_id']: r for r in rows if r['method'] == 'compact_orientation'}
        m, c = completed['results'][method], completed['results']['compact_orientation']
        tables.append(dict(method=method, cases=72, correct=m['correct'], strict=m['strict'], wrong=m['estimated']-m['correct'],
            median_full_inference_ms=m['median_full_inference_ms'], paired_compact_median_ms=c['median_full_inference_ms'],
            median_paired_compact_speedup=completed['median_paired_compact_speedup'],
            compact_minus_modern_descriptive_component_interval_pp=paired_interval(compact, modern),
            surface_minus_modern_descriptive_component_interval_pp=paired_interval(surface, modern),
            comparison_sha256=sha256(path)))
    result = dict(scope='Complete exposed-development comparison of these exact native common profiles. No calibrated acceptance, fresh test or global SOTA claim.',
        modern=tables, surface_correct=54, surface_wrong=3,
        surface_speed_note='The new surface-control latency has not yet been measured against modern models in the same worker. Original compact speedups do not apply to it.',
        surface_vs_modern_hardware_note='Modern/paired original compact inference ran on supported ARM CPU workers; surface inference ran locally. All native metrics were independently replayed; direct surface CPU validation is a separate experiment.',
        source_sha256=sha256(__file__), calibration_or_test_pixels_loaded=0)
    write_new('reports/native_modern_summary_v10.json', result)
    lines = '\n'.join(f"| {r['method']} common | {r['correct']} | {r['strict']} | {r['wrong']} | {r['median_full_inference_ms']/1000:.2f}s |" for r in tables)
    Path('reports/NATIVE_MODERN_V10.md').write_text(f'''# Native modern comparisons: completed, scoped development evidence

All72 assigned pairs for LoMa-R and RoMa-v2 are complete and independently replayed, including their same-job original compact controls. The fixed positional split uses first36 from the original run and last36 from the declared range worker; all duplicate and partial attempts remain preserved. Both original jobs timed out at64/72, so a successful workflow alone was never treated as complete coverage.

| Method | Correct raw /72 | Strict /72 | Wrong raw | Median full inference |
|---|---:|---:|---:|---:|
{lines}
| Frozen orientation CNN |45|17|12|2.64s /3.17s in paired workers|
| Frozen CNN + reference-surface rectification |54|16|3|Not measured in same worker yet|
| Same SIFT-first + surface pipeline |57|25|2|Not measured in same worker yet|

The original compact control is20.6× faster than LoMa-R and16.8× faster than RoMa-v2 by median per-case paired ratios. **Those speedups do not describe the new surface pipeline.** Its direct same-worker timing/full-score workflow is prepared separately. Model initialization, downloads and reference scoring are outside inference timing. Joint memory includes both loaded models and cannot rank isolated model memory.

These are exact settings, not every author's best configuration: LoMa-R receives the full512pixel field of view internally resized to504; RoMa-v2 uses the declared fast/common512pixel, single-direction profile. Original compact matching uses128pixel coarse proposals and512pixel frozen features. Author-recommended precise RoMa-v2 native inference is not completed, and native EfficientLoFTR/MOONSt3R comparisons remain outstanding. Source revisions, weight hashes and settings are pinned in each artifact. All models passed actual512pixel known-translation pose/point, blank and invalid-mask controls before lunar inference.

All methods receive the same reference geometry and final3native-pixel AP3P estimator. Surface matching additionally uses its image-estimated pose for explicit reference-surface normalization; query depth/true pose are excluded from inference. Its accuracy count exceeds the tested RoMa-v2 profile by one pair, which does **not** establish superiority. The per-method descriptive terrain intervals are in native_modern_summary_v10.json. Only four exposed terrain components/eight geometries generate these72 dependent pairs. No fresh-test confidence or1% risk guarantee follows. The compact CNN has lunar training while the compared large checkpoints remain generic/frozen; this does not isolate architectural advantage.

Raw outputs are not calibrated acceptance. The surface+SIFT pipeline still has two wrong proposals; native nonoverlap and confidence calibration are incomplete. Stricter surface+SIFT success25/72 is below the earlier fallback27/72. No global SOTA or independently verified orbital accuracy is claimed. Every claim is confined to these cases, settings and saved native coordinates.

Replay each method with combine_native_baselines_v8.py using the original and range artifact directories. Then run this reporter with --artifacts pointing to their four extracted ZIP folders. Own code is MIT; MoonAnything LunarGeo data are CC BY4.0 (Grethen et al., doi:10.5281/zenodo.18415119). Upstream model licenses/weights remain governed by their pinned author notices; no third-party weights are redistributed here.
''')
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--artifacts', required=True)
    a = p.parse_args(); report(a.artifacts)
