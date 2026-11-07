"""Three predefined seeds of coupled-window training using the frozen v18 loop.

The temporary objective binding exists only inside this dedicated serial
worker. It changes no source file and is restored even on failure. Its exact
binding and all fitting dependencies are public/source-pinned before fitting.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import prepare_supervision
from dataset import sha256
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle
import train_search_window_v18 as engine
from train_mutual_pilot_v23 import (CONFIGURATION, MANIFEST, SOURCES as PILOT_SOURCES,
    fit_step, initialized_model, reusable_gates)

SEEDS = (20261101, 20261102, 20261103)
SOURCES = list(dict.fromkeys(PILOT_SOURCES+['verify_mutual_pilot_v23.py',
    'train_mutual_cohort_v24.py', 'verify_mutual_cohort_v24.py', 'control_mutual_cohort_v24.py']))


@contextmanager
def objective_binding():
    """Reuse the exact saved-state/selection loop, explicitly binding the new loss."""
    previous_step, previous_sources = engine.fitting_step, engine.SOURCES
    def coupled_step(model, optimizer, tiles, prepared, rng, step, configuration):
        if configuration != CONFIGURATION:raise ValueError('Only the declared coarse-adaptation configuration is allowed')
        return fit_step(model, optimizer, tiles, prepared, rng, step)
    try:
        engine.fitting_step, engine.SOURCES = coupled_step, SOURCES
        yield
    finally:
        engine.fitting_step, engine.SOURCES = previous_step, previous_sources


def published(commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):raise ValueError('Full public source freeze required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Full fitting source differs from public freeze: '+name)


def pilot_gate():
    reusable_gates(); root = Path('runs/localization-v23-mutual-cost-pilot')
    path = Path('reports/native_mutual_pilot_audit_v23.json'); gate = json.loads(path.read_text())
    protocol = json.loads((root/'protocol.json').read_text()); summary = json.loads((root/'summary.json').read_text())
    if (not gate['passed'] or not gate['exact_full_retraining'] or gate['completed_steps'] != 100
            or gate['independent_assignment_scalar_matrices'] != 8 or not gate['fine_encoder_unchanged']):
        raise ValueError('Measured exact-replay/scalar-audited new-objective pilot required')
    if (gate['pilot_summary_sha256'] != sha256(root/'summary.json')
            or gate['pilot_protocol_sha256'] != sha256(root/'protocol.json')
            or gate['source_sha256'] != sha256('verify_mutual_pilot_v23.py')):
        raise ValueError('New-objective pilot audit changed')
    for name, want in protocol['source_sha256'].items():
        if sha256(name) != want:raise ValueError('Audited fitting source changed')
    for name, want in json.loads((root/'COMPLETE.json').read_text())['sha256'].items():
        if sha256(root/name) != want:raise ValueError('Audited pilot artifacts changed')
    if summary['fitting_seconds']*60 > 3*3600:
        raise TimeoutError('Measured pilot projects over3 fitting hours for6000updates; no full cohort launched')
    return path, protocol, summary


def frozen_protocol(root, commit):
    root = Path(root); protocol = json.loads((root/'protocol.json').read_text())
    seal = json.loads((root/'PROTOCOL_SEAL.json').read_text())
    if seal != dict(protocol_sha256=sha256(root/'protocol.json')):raise ValueError('Frozen protocol seal differs')
    required = dict(publication_commit=commit, objective='dual_window_mass_v23', steps=2000, interval=500,
        pilot=False, variants=[CONFIGURATION], seeds=list(SEEDS), cohort_wall_cap_seconds=6*3600,
        per_invocation_wall_cap_seconds=6*3600, temperature=.1, assignment_correction_weight=1.)
    if any(protocol.get(k) != v for k, v in required.items()):raise ValueError('Frozen cohort configuration differs')
    return protocol


def run(output, commit):
    published(commit); engine.configure(); gate_path, pp, ps = pilot_gate()
    bundle = load_bundle(MANIFEST); manifest, tiles, data = bundle
    if sha256(MANIFEST) != pp['manifest_sha256']:raise ValueError('Pilot/full manifest differs')
    if not manifest['train_selection_disjoint'] or manifest['calibration_or_test_pixels_loaded']:
        raise ValueError('Audited development-only groups required')
    prepared, support = prepare_supervision(tiles, data)
    if support != pp['training_support']:raise ValueError('Pilot/full training support differs')
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    if sha256(Path(manifest['coarse_model'])/'selected.pt') != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen coarse source checkpoint changed')
    initialized_model(coarse)  # Fresh seeded primary weights equal controlled graph.
    protected = {k: v.clone() for k, v in coarse.model.state_dict().items()}
    root = Path(output)
    if root.exists():
        protocol = frozen_protocol(root, commit); engine.check_sources(protocol)
        if protocol['pilot_audit_sha256'] != sha256(gate_path):raise ValueError('Audited pilot changed after cohort freeze')
    else:
        root.mkdir(parents=True)
        protocol = dict(scope='Exposed-development three-seed coupled-window training. No fresh final, acceptance or SOTA claim.',
            objective='dual_window_mass_v23', publication_commit=commit, source_sha256={p: sha256(p) for p in SOURCES},
            manifest_sha256=sha256(MANIFEST), input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
            coarse_checkpoint_sha256=manifest['coarse_checkpoint_sha256'], training_support=support,
            steps=2000, interval=500, pilot=False, variants=[CONFIGURATION], seeds=list(SEEDS),
            context_learning_rate=.0001, coarse_learning_rate=.00001, weight_decay=.0001, gradient_clip=5.,
            optimizer='AdamW', pairs_per_update=1, tokens_per_image=676, negative_pair_every=4, cpu_threads=2,
            settings={CONFIGURATION: engine.SETTINGS[CONFIGURATION]}, temperature=.1, assignment_correction_weight=1.,
            training_loop='Unchanged v18 train/checkpoint/resume/evaluation loop; temporary dedicated-process fitting-step binding to train_mutual_pilot_v23.fit_step, restored on every exit. Source files unchanged.',
            supervision='Only positive-set normalization changes to dual-softmax reachable mass in both directions. Same exact3D labels, margin losses, genuine-anchor0.5, data draws and sensor augmentation.',
            inference='Unchanged raw-score MNN threshold0.5, four orientation proposals, frozen native fine CNN and two reference-surface updates; same3nativepixel AP3P. Query truth scoring only.',
            matched_historical_control='v18 coarse_adaptation/20261101, same architecture/optimizer/2000updates/sampler, selected0=54/72, final53/72; source and full raw history retained.',
            selection='All72 exposed pairs: unfiltered correct poses, then2pixel correspondence count, then earlier step. Initialization eligible. Report all three seeds; never headline best seed.',
            source_freshness='All groups already exposed development; original fresh-final data gate remains failed.',
            pilot_audit_sha256=sha256(gate_path), pilot_fitting_seconds=ps['fitting_seconds'],
            initial_gate_sha256=sha256('runs/localization-v18-window-initial-control/summary.json'),
            matched_control_protocol_sha256=sha256('runs/localization-v18-window-development/protocol.json'),
            calibration_or_test_pixels_loaded=0, fresh_data_gate_passed=False,
            cohort_wall_cap_seconds=6*3600, per_invocation_wall_cap_seconds=6*3600,
            environment=pp['environment'], created_utc=datetime.now(timezone.utc).isoformat())
        write_new(root/'protocol.json', protocol)
        write_new(root/'PROTOCOL_SEAL.json', dict(protocol_sha256=sha256(root/'protocol.json')))
    if support != protocol['training_support']:raise ValueError('Frozen training support differs')
    results = []; features = {}
    with objective_binding():
        for seed in SEEDS:
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(protocol['created_utc'])).total_seconds()
            if elapsed > protocol['cohort_wall_cap_seconds'] and not (root/f'{CONFIGURATION}-{seed}'/'COMPLETE.json').exists():
                raise TimeoutError('Original cohort deadline exhausted; no additional fitting/evaluation may start')
            results.append(engine.train(root, CONFIGURATION, seed, bundle, coarse, prepared, protocol, features))
    if any(not torch.equal(v, protected[k]) for k, v in coarse.model.state_dict().items()):
        raise ValueError('Original proposal checkpoint mutated')
    engine.check_sources(protocol)
    summary = dict(scope=protocol['scope'], objective=protocol['objective'], winner=CONFIGURATION,
        primary_seed=SEEDS[0], completed_optimizer_steps=sum(r['completed_steps'] for r in results),
        frozen_backbone_unchanged=True,
        results=[{k: r[k] for k in ('variant', 'seed', 'completed_steps', 'selected_step', 'evaluation', 'selection_key', 'fitting_seconds')} for r in results])
    engine.record_once(root/'summary.json', summary)
    engine.record_once(root/'COMPLETE.json', dict(protocol_sha256=sha256(root/'protocol.json'),
        protocol_seal_sha256=sha256(root/'PROTOCOL_SEAL.json'), summary_sha256=sha256(root/'summary.json')))
    print(json.dumps(summary, indent=2)); return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--source-commit', required=True)
    a = p.parse_args(); run(a.output, a.source_commit)
