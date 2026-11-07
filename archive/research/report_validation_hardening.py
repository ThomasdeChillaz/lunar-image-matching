"""Report completed frozen evaluation without fitting or changing measurements."""
import argparse
import html
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256


def load(path):
    return json.loads(Path(path).read_text())


def polish(text,units=True):
    text=re.sub(r'\b(all|in|of|at|on|least|provide|retains|accepts|rejects|suite|core|fresh|full|original|with|across|published|within|iteration|bootstrap|sub|treating|aligns|has|covers|and|The)(?=\d)',r'\1 ',text)
    text=text.replace('30October2026','30 October 2026').replace('May2017','May 2017')
    text=text.replace('Chang’e3','Chang’e 3').replace('Apollo11','Apollo 11')
    if units:
        text=re.sub(r'(\d)(px|mm|UTC)\b',r'\1 \2',text)
    return text


def generate(output,logs):
    output,logs=Path(output),Path(logs)
    reserve=Path('runs/reserve-frozen-v1')
    real=Path('runs/real-sensor-controls-v1')
    polar=Path('runs/polar-reference-v1')
    for root,audit in [(reserve,'audit.json'),(real,'audit_full.json'),(polar,'audit.json')]:
        if not (root/'COMPLETE.json').exists() or not (root/audit).exists():
            raise RuntimeError(f'Completed outputs and full audit required: {root}')
    rs,real_summary,ps=load(reserve/'summary.json'),load(real/'summary.json'),load(polar/'summary.json')
    doc=load(reserve/'frozen_protocol.json')
    terminal=load(reserve/'COMPLETE.json')
    if sha256(reserve/'results.json')!=terminal['results_sha256'] or sha256(reserve/'frozen_protocol.json')!=terminal['protocol_sha256']:
        raise ValueError('Completed reserve bytes changed')
    test_records={}
    for key,name in [('research','validation-hardening-tests.log'),('minimal_core','validation-hardening-core-tests.log')]:
        text=(logs/name).read_text()
        match=re.search(r'(\d+) passed, (\d+) skipped',text)
        if match is None or re.search(r'\d+ failed|ERRORS|FAILURES',text):
            raise ValueError('Successful complete test log required')
        test_records[key]=dict(passed=int(match[1]),skipped=int(match[2]),log_sha256=sha256(logs/name))
    smoke=Path('runs/validation-hardening-smoke-v1')
    if len(load(smoke/'summary.json'))!=20:
        raise ValueError('Full fresh procedural smoke required')
    validation=dict(test_suites=test_records,fresh_smoke_summaries=20,
        smoke_summary_sha256=sha256(smoke/'summary.json'),
        cli_actual_model_control=load('runs/deployment-cli-controls-v1/audit.json'),
        reserve_audit=load(reserve/'audit.json'),real_sensor_audit=load(real/'audit_full.json'),
        polar_reference_audit=load(polar/'audit.json'))
    output.mkdir(parents=True,exist_ok=True)
    (output/'validation_hardening_checks.json').write_text(json.dumps(validation,indent=2)+'\n')
    for root,name in [(reserve,'reserve'),(real,'real_sensor'),(polar,'polar_reference')]:
        (output/f'{name}_validation_summary.json').write_bytes((root/'summary.json').read_bytes())
    methods=['terrain_raw','terrain_deployed','sift']
    labels=['CNN raw proposal','CNN with frozen rejection','SIFT']
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),layout='constrained')
    colors=['#6389a5','#237c67','#8c8c96']
    values=[rs['overall'][m]['correct_rate']*100 for m in methods]
    axes[0].bar(range(3),values,color=colors)
    for i,m in enumerate(methods):
        v=rs['overall'][m]
        axes[0].text(i,values[i]+1.5,f'{v["correct"]:,}/{v["positive_cases"]:,}',ha='center',fontsize=9)
    axes[0].set(ylim=(0,110),ylabel='Correct / all positive cases (%)',title='Fresh physical-render geometry')
    axes[0].set_xticks(range(3),['Raw CNN','Gated CNN','SIFT'])
    for method,label,color in zip(methods[:2],['Raw CNN','Gated CNN'],colors):
        yy=[rs['by_group'][f'{method}:{i}']['correct_rate']*100 for i in range(rs['components'])]
        axes[1].plot(range(1,rs['components']+1),yy,'o-',label=label,color=color)
    axes[1].set(ylim=(0,105),xlabel='Footprint/acquisition group',ylabel='Correct / positives (%)',title='Twelve groups; one crater region')
    axes[1].set_xticks(range(1,rs['components']+1))
    axes[1].legend(frameon=False)
    negatives=[rs['overall'][m]['negative_accepted'] for m in methods]
    axes[2].bar(range(3),negatives,color=colors)
    for i,m in enumerate(methods):
        v=rs['overall'][m]
        axes[2].text(i,negatives[i]+35,f'{negatives[i]:,}/{v["negative_cases"]:,}',ha='center',fontsize=9)
    axes[2].set(ylim=(0,rs['overall']['terrain_raw']['negative_cases']*1.12),ylabel='Nonoverlap proposals/accepted outputs',title='Disjoint terrain: rejection matters')
    axes[2].set_xticks(range(3),['Raw CNN','Gated CNN','SIFT'])
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen standalone evaluation — known geometry, within Tycho',fontsize=15)
    fig.savefig(output/'validation_hardening.png',dpi=170)
    plt.close(fig)
    lines=['# Research iteration5: frozen evaluation and real-sensor checks','',
        'The standalone CNN and strict rejection rule were frozen before reserve access. The original13-predictor ensemble remains a development result; this report does not claim its fresh-test validation or current state-of-the-art superiority.','',
        '## What changed','',
        '1. The primary experiment uses all64 reserve patches in12 footprint/acquisition groups, rather than treating1,152 repeated development pairs as independent terrain tests. The geography remains within Tycho.',
        '2. The fixed checkpoint, source/model hashes, threshold, case plan and evaluation targets were published before opening reserves. No reserve fitting or checkpoint selection occurred.',
        '3. Rejection is enforced in public outputs. Raw transforms remain research proposals and are never returned as accepted alignments when the gate rejects.',
        '4. Real orbital photo controls and independently laser-referenced laboratory diagnostics were executed. Real orbital navigation accuracy remains unverified; neither warped photos nor nominal calibration agreement can establish it.','',
        '## Public preregistration','',
        '[Preregistered commit](https://github.com/ThomasdeChillaz/lunar-image-matching/commit/b7f2e0fe24be08dac5c0e1201e9451b6a625d59f), published30October2026 at08:24:27UTC. The full9,216-positive/1,728-negative plan covers all36 Sun pairs and four geometry families per reserve patch. Negatives use three separately grouped, buffered-disjoint partners and nine lights.','',
        f'Protocol SHA256: `{terminal["protocol_sha256"]}`. Model SHA256: `{doc["training_files_sha256"][doc["training"]+"/selected.pt"]}`. Both reserves are now consumed and must not be reused as fresh data.','',
        '## Untouched-terrain results','',
        '| Fixed method | Correct /9,216 positives | Wrong positive fits | Rejected/failed positives | Accepted nonoverlap /1,728 |',
        '|---|---:|---:|---:|---:|']
    for method,label in zip(methods,labels):
        v=rs['overall'][method]
        lines.append(f'| {label} | {v["correct"]:,} ({100*v["correct_rate"]:.2f}%) | {v["positive_wrong"]:,} | {v["rejected_or_failed_positive"]:,} | {v["negative_accepted"]:,} |')
    deployed=rs['overall']['terrain_deployed']
    interval=deployed['positive_rate_cluster_percentile_95']
    lines+=['',f'The gated positive success rate has an exploratory component-bootstrap95% interval of{100*interval[0]:.2f}–{100*interval[1]:.2f}%. Resampling joins shared acquisitions and overlapping footprints; disconnected components can still be correlated within one crater. Negative pairs reuse both component endpoints, so no independent-pair confidence guarantee is inferred.','',
        f'**Predeclared empirical targets met: {rs["predeclared_targets_met"]}.** Targets: at least40% of all positives correctly accepted, accepted-error fraction≤1%, nonoverlap acceptance≤1%, at least12 components. These are empirical evaluation targets, not statistical deployment guarantees. Failed targets are preserved; no threshold is revised here.','',
        '![Frozen standalone validation](validation_hardening.png)','',
        '## Separate geometric families','',
        '| Family | Raw CNN correct /2,304 | Gated CNN correct /2,304 | Gated wrong positive fits |',
        '|---|---:|---:|---:|']
    for family in ['translation','rotation','scale','combined']:
        raw=rs['by_geometry'][f'terrain_raw:{family}']
        gate=rs['by_geometry'][f'terrain_deployed:{family}']
        lines.append(f'| {family} | {raw["correct"]:,} | {gate["correct"]:,} | {gate["positive_wrong"]:,} |')
    lines+=['','## Real orbital pixels: exact controls and unknown repeat accuracy','',
        'The unchanged Tycho-trained model was evaluated on42 acquisitions across Chang’e3, Apollo11 and near Shackleton. These are three previously inspected source scenes, not42 independent new terrains. Two known warps per geometry family provide336 exact controls;84 pairs come from different scenes. No real-data training or threshold calibration occurs.','',
        '| Method | Correct known-warps /336 | Different-scene accepted /84 |','|---|---:|---:|']
    for method,label in zip(methods,labels):
        v=real_summary['by_method'][method]
        lines.append(f'| {label} | {v["correct"]} | {v["negative_accepted"]} |')
    lines+=['','The321 genuine repeat-observation pairs remain unverified diagnostics. The gate accepts13/45 S1,19/45 S2 and45/231 S3 pairs; these are acceptance counts, not verified-correct counts. Published map coordinates have unknown reference uncertainty.','',
        '## Real sensors and physical relighting: an important failure','',
        'Four NASA Ames POLAR laboratory terrains provide24 pairs: eight same-camera relighting cases and16 stereo cases with published laser XYZ and camera calibration. Original member CRC32, SHA256 and the nominal projection/scoring replay passed. The model never receives laser coordinates.','',
        f'Raw CNN nominally aligns{ps["terrain_raw"]["fixed_camera_nominal_success"]}/8 same-camera relighting pairs within3px; the strict gate retains{ps["terrain_deployed"]["fixed_camera_nominal_success"]}/8. This exposes poor transfer of the rejection rule to real changing-light sensor data. The successful orbital-photo warps do not resolve it.','',
        'Returned inlier matches are scored against external laser projections, with a0.35px nearest-reference tolerance and explicit visibility filtering. These are nominal correspondence-reference diagnostics, not global registration success or a reproduced dense-stereo benchmark.','',
        f'Across all24 mixed cases, raw CNN has{ps["terrain_raw"]["nominal_inlier_matches_within3px"]}/{ps["terrain_raw"]["laser_scorable_inlier_matches"]} laser-scorable returned inlier matches within3px; gated CNN{ps["terrain_deployed"]["nominal_inlier_matches_within3px"]}/{ps["terrain_deployed"]["laser_scorable_inlier_matches"]}; SIFT{ps["sift"]["nominal_inlier_matches_within3px"]}/{ps["sift"]["laser_scorable_inlier_matches"]}. Different returned coverage prevents interpreting these ratios as a method ranking. Full outcomes, missing-reference counts and failures remain in the capsule.','',
        'NASA reports sub2mm laser accuracy, but calibration covariance and cross-light rig drift are not supplied as pixel-accuracy bounds. The reference projection is external; its complete uncertainty is unknown. These are laboratory analogues, not photographs acquired on the Moon. [NASA dataset](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/), [documentation](https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_Readme.pdf).','',
        '## What remains unresolved','',
        '- Cross-crater/polar transfer under genuine changing sunlight is not established by the within-Tycho reserves or by single-photo warps.',
        '- The13-predictor ensemble’s95.75% remains development-only. Its frozen development export retains1,069 outputs and returns null for rejected proposals; that is not held-out reliability evidence.',
        '- Real orbital navigation accuracy requires independent reference geometry with characterized uncertainty. No automatic matcher agreement, laser instrument accuracy alone, or unknown-uncertainty map coordinate is promoted to that claim.',
        '- A subsequent model/gate change must use development data and a new untouched evaluation. Both original reserves are consumed. The NASA relighting failure is a new development lead, not a result to conceal by refitting this report.','',
        '## Reproduce','',
        'Use the locked research environment and existing source archives/training capsule. From the repository, after restoring the completed-measurement capsule:','',
        '```sh','python verify_reserve.py runs/reserve-frozen-v1 --output runs/reserve-audit-replay.json',
        'python verify_real_sensor.py runs/real-sensor-controls-v1 --output runs/real-sensor-audit-replay.json',
        'python fetch_polar_reference.py','python verify_polar_reference.py runs/polar-reference-v1 --output runs/polar-audit-replay.json','```','',
        'Fresh execution uses the committed protocols and new output directories. Do not rerun a freeze command over a published protocol. Source and checkpoint mismatches are errors.','',
        '```sh','python reserve_evaluation.py run --freeze reports/reserve_protocol_v1.json --output runs/reserve-reproduction',
        'python real_sensor_controls.py run --protocol reports/real_sensor_protocol_v1.json --output runs/real-sensor-reproduction',
        'python polar_reference.py run --protocol reports/polar_reference_protocol_v1.json --output runs/polar-reference-reproduction','```','',
        f'Validation: research suite{test_records["research"]["passed"]} passed/{test_records["research"]["skipped"]} skipped; minimal core{test_records["minimal_core"]["passed"]} passed/{test_records["minimal_core"]["skipped"]} skipped; fresh20-summary procedural smoke. The selected CNN ran actual software, CLI and real-data experiments; optional unexecuted matcher tests are not counted as pretrained validation.','']
    lines+=['[Execution environment](validation_environment_v1.json), [ensemble development capsule](ensemble_development_v1.zip), and [public ensemble export audit](ensemble_public_export_audit.json) supplement the complete standalone-validation capsule. The ensemble capsule preserves original development outputs; it is not a fresh-test result.','']
    (output/'RESEARCH_ITERATION_5.md').write_text(polish('\n'.join(lines)))
    payload=dict(reserve=rs,real_sensor=real_summary,polar=ps,checks=validation)
    (output/'validation_hardening.json').write_text(json.dumps(payload,indent=2)+'\n')
    table=''.join(f'<tr><td>{html.escape(label)}</td><td>{rs["overall"][m]["correct"]:,}/{rs["overall"][m]["positive_cases"]:,}</td><td>{rs["overall"][m]["positive_wrong"]:,}</td><td>{rs["overall"][m]["negative_accepted"]:,}/{rs["overall"][m]["negative_cases"]:,}</td></tr>' for m,label in zip(methods,labels))
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lunar matching — validation update</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;color:#183036;background:#f2f5f4;margin:0}}main{{max-width:1050px;margin:40px auto;padding:36px;background:white;border-radius:16px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px}}.muted{{color:#52656b}}.note{{padding:18px 22px;background:#eef5f3;border-left:4px solid #237c67}}.failure{{padding:18px 22px;background:#fff4e6;border-left:4px solid #bd7524}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{padding:12px;border-bottom:1px solid #d9e3df;text-align:left}}img{{width:100%;height:auto}}a{{color:#176d61}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{main{{margin:0;padding:20px}}h1{{font-size:29px}}table{{font-size:13px}}td,th{{padding:7px}}}}</style>
<main><p class="muted">Measured research update · 30 October 2026</p><h1>Frozen terrain evaluation and real-sensor checks</h1>
<p class="note"><b>Model and rejection rule fixed before reserve access.</b> The fresh experiment covers64 patches in12 footprint/acquisition groups. All remain within Tycho. The13-predictor ensemble remains development-only.</p>
<h2>Fresh physical-render results</h2><p>9,216 changing-light/known-motion positives and1,728 disjoint-terrain negatives. Rejections and failed fits remain in the denominators.</p>
<table><tr><th>Method</th><th>Correct positives</th><th>Wrong positive outputs</th><th>Nonoverlap outputs</th></tr>{table}</table>
<p>Predeclared empirical targets met: <b>{rs['predeclared_targets_met']}</b>. An observed rate is not a deployment guarantee.</p><img src="validation_hardening.png" alt="Correctness by method and geographic group, and false nonoverlap outputs">
<h2>Actual lunar photo controls</h2><p>The fixed gated model retains334/336 correct known-warps and rejects84/84 different-scene pairs across Chang’e3, Apollo11 and near Shackleton. This verifies controlled motion on real image pixels. It does not verify changing-light repeat-observation accuracy.</p>
<h2>Real physical relighting exposes a weakness</h2><p class="failure">On four NASA laboratory terrains, raw CNN nominally aligns4/8 fixed-camera relighting pairs within3px, but the strict gate retains0/8. This transfer failure remains visible. External laser-reference diagnostics are included with unknown calibration uncertainty.</p>
<h2>What remains unverified</h2><p>Real orbital navigation accuracy, broad polar/cross-crater illumination transfer, and fresh performance of the full ensemble remain unverified. Both original reserves are consumed; future changes require new untouched evaluation data.</p>
<p><a href="RESEARCH_ITERATION_5.md">Full measured report and reproduction commands</a> · <a href="validation_hardening.json">Measurements and checks</a> · <a href="reserve_protocol_v1.json">Published frozen protocol</a> · <a href="validation_hardening_v1.zip">Complete measurement capsule</a></p></main></html>'''
    (output/'validation_hardening.html').write_text(polish(body,units=False))
    sources=set(doc['source_sha256'])|set(load(real/'frozen_protocol.json')['source_sha256'])|set(load(polar/'frozen_protocol.json')['source_sha256'])
    sources|={p.name for p in Path('.').glob('test_*') if p.is_file() and any(s in p.name for s in ('reserve','real_sensor','polar','deploy_gate'))}
    sources|={'verify_reserve.py','verify_real_sensor.py','verify_polar_reference.py','fetch_polar_reference.py','match_terrain.py','deploy_gate.py','report_validation_hardening.py'}
    snapshot=dict(sources={p:dict(sha256=sha256(p),text=Path(p).read_text()) for p in sorted(sources)})
    (output/'validation_source_snapshot_v1.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    members=[p for root in [reserve,real,polar] for p in root.iterdir() if p.is_file() and p.suffix=='.json']
    members += list(Path('runs/ensemble-public-export-v1').glob('*.json'))
    members += list(Path('runs/deployment-cli-controls-v1').glob('*.json'))
    members += [output/p for p in ['reserve_protocol_v1.json','real_sensor_protocol_v1.json','polar_reference_protocol_v1.json',
        'RESEARCH_ITERATION_5.md','validation_hardening.json','validation_hardening_checks.json','validation_source_snapshot_v1.json',
        'validation_environment_v1.json','ensemble_public_export_audit.json']]
    manifest={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(set(members))}
    capsule=output/'validation_hardening_v1.zip'
    with zipfile.ZipFile(capsule,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in manifest:
            archive.write(name,name)
        archive.writestr('CAPSULE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(capsule) as archive:
        if archive.testzip() is not None:
            raise ValueError('Measurement capsule CRC failure')
        import hashlib
        for name,entry in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=entry['sha256']:
                raise ValueError('Capsule byte verification failed')
    (output/'validation_hardening_v1.sha256').write_text(sha256(capsule)+'  '+capsule.name+'\n')
    print(json.dumps(dict(report=str(output/'RESEARCH_ITERATION_5.md'),capsule_bytes=capsule.stat().st_size,
        capsule_sha256=sha256(capsule),capsule_members=len(manifest)+1,every_member_verified=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports')
    p.add_argument('--logs',required=True)
    args=p.parse_args()
    generate(args.output,args.logs)
