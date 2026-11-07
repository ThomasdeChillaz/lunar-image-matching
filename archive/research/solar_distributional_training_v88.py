"""SOURCE-ONLY 100-update training/replay library for the v86 coarse model.

No import-time Torch, bank/model decode, job admission, reservation or launch.
An external, source-published controller must authorize and bound every call.
This pilot diagnoses optimization/reproducibility; it never selects a model.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


EVEN = tuple(range(0, 24, 2))
BANK_FORMAT = 'distributional-coarse128-bank-v88'
PILOT_FORMAT = 'distributional-coarse128-pilot-v88'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Pinned repository-relative bank inputs required')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Bank input escapes the declared repository')
    return resolved


@dataclass(frozen=True)
class PilotSettings:
    seed: int = 20261101
    updates: int = 100
    batch_points: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    conditional_weight: float = 1.0
    marginal_weight: float = 0.25
    discrimination_weight: float = 0.10
    discrimination_temperature: float = 0.10
    cpu_threads: int = 2

    def validate(self):
        # This library implements one predeclared pilot, not an open sweep.
        if asdict(self) != asdict(PilotSettings()):
            raise ValueError('Only the fixed seed20261101 / 100-update pilot is implemented')
        return self


ARRAY_NAMES = {
    'reference_images', 'reference_normal_xy', 'reference_geometry_valid',
    'native_xy', 'coarse_xy', 'world', 'point_label_ids', 'observed_conditions',
    'target_suns', 'target_view_basis', 'teacher', 'available',
    'correspondence_visible', 'coarse_support', 'reference_input_available',
}


def validate_unit(arrays, samples):
    """Reject label/support corruption without recoding visibility from RGB."""
    if set(arrays) != ARRAY_NAMES or type(samples) is not int or samples < 2:
        raise ValueError('Exact coarse128 unit schema and retained point count required')
    n = samples
    shapes = {
        'reference_images': (12, 128, 128), 'reference_normal_xy': (2, 128, 128),
        'reference_geometry_valid': (128, 128), 'native_xy': (n, 2, 2),
        'coarse_xy': (n, 2, 2), 'world': (n, 3), 'point_label_ids': (n,),
        'observed_conditions': (n, 12, 10), 'target_suns': (24, 3),
        'target_view_basis': (n, 2, 7), 'teacher': (n, 2, 24, 32),
        'available': (n, 2, 24), 'correspondence_visible': (n, 2),
        'coarse_support': (n, 2), 'reference_input_available': (n,),
    }
    for name, shape in shapes.items():
        a = arrays[name]
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype.hasobject:
            raise ValueError('Invalid coarse128 unit field: ' + name)
    if arrays['reference_images'].dtype != np.uint8:
        raise ValueError('Reference fields must preserve uint8 area-resized intensities')
    for name in ('reference_geometry_valid', 'available', 'correspondence_visible',
                 'coarse_support', 'reference_input_available'):
        if arrays[name].dtype != bool:
            raise ValueError('Explicit bool mask required: ' + name)
    if arrays['point_label_ids'].dtype != np.int64 or len(np.unique(arrays['point_label_ids'])) != n:
        raise ValueError('Unique retained physical-point label IDs required')
    for name in ('reference_normal_xy', 'coarse_xy', 'observed_conditions',
                 'target_suns', 'target_view_basis', 'teacher'):
        if arrays[name].dtype != np.float32:
            raise ValueError('Original float32 teacher/model inputs required: ' + name)
    for name in ('native_xy', 'world'):
        if arrays[name].dtype != np.float64:
            raise ValueError('Original float64 geometry labels required: ' + name)
    for name in ARRAY_NAMES - {'teacher'}:
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite bank field: ' + name)
    teacher, available = arrays['teacher'], arrays['available']
    if available[:, :, 1::2].any() or not np.isnan(teacher[:, :, 1::2]).all():
        raise ValueError('Odd-light targets must remain unavailable and NaN')
    if not np.isfinite(teacher[available]).all() or not np.isnan(teacher[~available]).all():
        raise ValueError('Available target vectors must be finite; unavailable vectors all NaN')
    if not np.allclose(np.linalg.norm(teacher[available], axis=-1), 1., atol=2e-4, rtol=0):
        raise ValueError('Precomputed teacher features must be unit32D vectors')
    if not np.array_equal(arrays['reference_input_available'], arrays['coarse_support'][:, 0]):
        raise ValueError('Reference-input support must remain separate from target availability')
    if (available & ~arrays['coarse_support'][:, :, None]).any():
        raise ValueError('Teacher availability exceeds declared receptive-field support')
    if not arrays['correspondence_visible'].all():
        raise ValueError('This adapter retains legacy co-visible centre labels only')
    centres = ((arrays['native_xy'] + .5) / 4 - .5).astype(np.float32)
    if not np.array_equal(centres, arrays['coarse_xy']):
        raise ValueError('Native/coarse half-pixel conversion changed')
    conditions = arrays['observed_conditions']
    for sl in (slice(0, 3), slice(3, 6), slice(6, 9)):
        if not np.allclose(np.linalg.norm(conditions[:, :, sl], axis=-1), 1., atol=1e-5, rtol=0):
            raise ValueError('Known reference Sun/normal/view directions must be unit')
    if not np.allclose(conditions[:, :, :3], arrays['target_suns'][list(EVEN)][None], atol=1e-7, rtol=0):
        raise ValueError('Observed reference-light slot differs from physical Sun metadata')
    views = arrays['target_view_basis']
    if not np.allclose(views[:, :, 0], 1., atol=1e-6, rtol=0):
        raise ValueError('Training target-view constant must equal one')
    for sl in (slice(1, 4), slice(4, 7)):
        if not np.allclose(np.linalg.norm(views[:, :, sl], axis=-1), 1., atol=1e-5, rtol=0):
            raise ValueError('True training target LOS/up directions must be unit')
    if not np.allclose(conditions[:, :, 6:9], views[:, 0, None, 1:4], atol=1e-7, rtol=0):
        raise ValueError('Observed condition must use known reference view, never query truth')
    normals = arrays['reference_normal_xy']
    if (np.sum(normals * normals, axis=0) > 1 + 1e-5).any():
        raise ValueError('Invalid estimated reference normal components')
    if (normals[:, ~arrays['reference_geometry_valid']] != 0).any():
        raise ValueError('Unknown reference geometry must retain zero inputs and explicit mask')
    # Each camera and common target cell remains in the sampling population.
    # Failure is explicit; we never silently reroll/drop a camera or target.
    counts = {}
    for role in range(2):
        for light in EVEN:
            mask = arrays['reference_input_available'] & available[:, role, light]
            count = int(mask.sum())
            if count < 2:
                raise ValueError('Insufficient valid distinct points for target role/light: '
                                 + str((role, light, count)))
            counts[str(role) + ':' + str(light)] = count
    for a in arrays.values():
        a.setflags(write=False)
    return counts


class CoarseTrainingBank:
    """Pinned new bank; bounded cache, no old canonical33 target substitution."""
    def __init__(self, repo_root, manifest_path, manifest_sha256, *,
                 allowed_training_groups, eligible_unit_indices,
                 expected_teacher_sha256, cache_units=4, clock=lambda: None):
        self.root = Path(repo_root).resolve()
        if Path.cwd().resolve() != self.root:
            raise ValueError('Run completed-bank validation from the declared repository root')
        self.path = _path(self.root, manifest_path)
        if sha(self.path) != manifest_sha256:
            raise ValueError('Coarse128 bank manifest changed')
        from solar_coarse_bank_v88 import validate_completed_bank
        completed = validate_completed_bank(str(manifest_path))
        if completed['manifest_sha256'] != manifest_sha256:
            raise ValueError('Completed external bank receipt belongs to different bytes')
        self.manifest_sha256 = manifest_sha256
        self.manifest = _json(self.path)
        m = self.manifest
        if (m.get('format') != BANK_FORMAT or m.get('split') != 'training'
                or m.get('array_schema') != 'distributional-coarse128-teacher-v86'
                or m.get('phase') != 'full_bank' or m.get('complete') is not True
                or m.get('heldout_image_reads') != 0
                or m.get('all_historical_rows_retained') is not True
                or m.get('source_lights') != list(EVEN)
                or m.get('teacher_descriptor_sha256') != expected_teacher_sha256
                or m.get('calibration_or_final_access') is not False):
            raise ValueError('Complete training-only new coarse128 bank required')
        if type(cache_units) is not int or not 1 <= cache_units <= 4:
            raise ValueError('Bounded one-to-four unit cache required')
        self.cache_units, self.clock, self.cache = cache_units, clock, {}
        self.groups = tuple(sorted(allowed_training_groups))
        self.records = {r['unit_index']: r for r in m['records']}
        if (len(self.records) != len(m['records'])
                or set(self.records) != set(eligible_unit_indices)
                or len(set(self.groups)) != len(self.groups) or not self.groups):
            raise ValueError('All original independently eligible training cameras required')
        if (sorted(m['training_groups']) != list(self.groups)
                or set(m['eligible_unit_indices']) != set(eligible_unit_indices)):
            raise ValueError('Full-bank camera/group inventory differs from the original contract')
        self.by_group = {g: [] for g in self.groups}
        for index, r in self.records.items():
            self.clock()
            if r.get('split') != 'training' or r.get('component') not in self.by_group:
                raise ValueError('Foreign split/terrain in coarse128 bank')
            self.by_group[r['component']].append(index)
            for field in ('path', 'metadata_path', 'seal_path'):
                if sha(_path(self.root, r[field])) != r[field.replace('path', 'sha256')]:
                    raise ValueError('Pinned bank record changed: ' + field)
            metadata = _json(_path(self.root, r['metadata_path']))
            seal = _json(_path(self.root, r['seal_path']))
            if (metadata.get('unit_index') != index or metadata.get('component') != r['component']
                    or metadata.get('split') != 'training' or metadata.get('samples') != r['samples']
                    or metadata.get('heldout_image_reads') != 0
                    or metadata.get('input_key') != r['input_key']
                    or seal.get('input_key') != r['input_key']
                    or seal.get('npz_sha256') != r['sha256']
                    or seal.get('metadata_sha256') != r['metadata_sha256']):
                raise ValueError('Camera/point metadata or sealed bank hashes disagree')
            sealed, deadline = (datetime.fromisoformat(seal[name]) for name in
                                ('sealed_utc', 'original_worker_deadline_utc'))
            if sealed.tzinfo is None or deadline.tzinfo is None or sealed >= deadline:
                raise ValueError('Bank unit was not sealed before its original work deadline')
        if any(not units for units in self.by_group.values()):
            raise ValueError('Every original training group must remain represented')
        self.by_group = {g: tuple(sorted(ids)) for g, ids in self.by_group.items()}
        self.expected_teacher_sha256 = expected_teacher_sha256
        from solar_distributional_response_v86 import TrainingViewPrior
        prior_path = _path(self.root, m['view_prior'])
        if sha(prior_path) != m['view_prior_sha256']:
            raise ValueError('Frozen full-training-bank prior changed')
        self.prior = TrainingViewPrior.from_frozen_payload(_json(prior_path))
        self.prior_path, self.prior_sha256 = prior_path, m['view_prior_sha256']
        provenance = self.prior.provenance
        if (set(provenance['allowed_training_groups']) != set(self.groups)
                or set(provenance['fitted_groups']) != set(self.groups)
                or provenance.get('sample_count') != sum(r['samples'] for r in self.records.values())
                or provenance.get('camera_units') != len(self.records)
                or provenance.get('weighting') != 'equal_group_then_camera_then_retained_point_query_role_only'
                or provenance.get('lighting_duplicates_counted') is not False):
            raise ValueError('Prior must use all retained training query-view labels with balanced geometry weights')

    def validate_model(self, model, expected_coarse_digest):
        import torch
        self.prior.validate()
        if expected_coarse_digest != self.expected_teacher_sha256:
            raise ValueError('New pilot foundation differs from the qualified full-bank teacher')
        if sha(self.prior_path) != self.prior_sha256 or model.prior_provenance != self.prior.provenance:
            raise ValueError('Model prior provenance differs from the frozen full-training-bank prior')
        factor = torch.tensor(self.prior.factor.copy(), dtype=torch.float32)
        mean = torch.tensor(self.prior.mean.copy(), dtype=torch.float32)
        if (not torch.equal(model.view_mean, mean) or not torch.equal(model.view_factor, factor)
                or not torch.equal(model.view_covariance, factor @ factor.T)):
            raise ValueError('Model prior buffers differ from the frozen training moments')

    def load(self, unit_index):
        self.clock()
        r = self.records[unit_index]
        path = _path(self.root, r['path'])
        if sha(self.path) != self.manifest_sha256 or sha(path) != r['sha256']:
            raise ValueError('Frozen manifest or unit bytes changed during training')
        if unit_index not in self.cache:
            with np.load(path, allow_pickle=False) as z:
                arrays = {name: z[name].copy() for name in z.files}
            validate_unit(arrays, r['samples'])
            if len(self.cache) >= self.cache_units:
                del self.cache[next(iter(self.cache))]
            self.cache[unit_index] = arrays
        return self.cache[unit_index]

    def preflight(self):
        counts = {}
        for index in sorted(self.records):
            counts[str(index)] = validate_unit(self.load(index), self.records[index]['samples'])
        return counts


def sample_training_batch(bank, rng, settings=PilotSettings()):
    """One camera/common target Sun per batch; no availability-driven rerolls."""
    settings.validate()
    group = bank.groups[int(rng.integers(len(bank.groups)))]
    units = bank.by_group[group]
    unit_index = units[int(rng.integers(len(units)))]
    observed_slot = int(rng.integers(12))
    role, target_slot = int(rng.integers(2)), int(rng.integers(12))
    observed_light, target_light = EVEN[observed_slot], EVEN[target_slot]
    a = bank.load(unit_index)
    eligible = np.flatnonzero(a['reference_input_available'] & a['available'][:, role, target_light])
    if len(eligible) < 2:
        raise ValueError('Selected original camera/target has fewer than two usable points')
    ids = rng.choice(eligible, size=min(settings.batch_points, len(eligible)), replace=False)
    field = np.concatenate([a['reference_images'][observed_slot][None].astype(np.float32) / 255,
                            a['reference_normal_xy']], axis=0)
    trace = dict(group=group, unit_index=int(unit_index), observed_reference_light=observed_light,
                 target_role=role, target_light=target_light, eligible_points=len(eligible),
                 point_indices=ids.tolist(), point_label_ids=a['point_label_ids'][ids].tolist(),
                 all_negatives_share_target_sun=True, query_view_used_as_training_label_only=True)
    return dict(field=field, points=a['coarse_xy'][ids, 0].copy(),
                condition=a['observed_conditions'][ids, observed_slot].copy(),
                target_sun=np.broadcast_to(a['target_suns'][target_light], (len(ids), 3)).copy(),
                target_view=a['target_view_basis'][ids, role].copy(),
                target_teacher=a['teacher'][ids, role, target_light].copy(), trace=trace)


def initialize(frozen_descriptor, prior, settings=PilotSettings()):
    """Call only inside a separately admitted job; actual model init is lazy."""
    settings.validate()
    import torch
    from solar_distributional_response_v86 import build_distributional_response
    from solar_native_v63 import state_digest
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings.seed)
    descriptor = frozen_descriptor.eval().requires_grad_(False)
    original_digest = state_digest(descriptor)
    model = build_distributional_response(descriptor, prior).train()
    if model.parameter_report['actual_parameters'] != 95116:
        raise ValueError('Declared 95,116-parameter model changed')
    if model.parameter_report['actual_trainable_parameters'] != 66844:
        raise ValueError('Declared 66,844 trainable parameters changed')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=settings.learning_rate, weight_decay=settings.weight_decay)
    return model, optimizer, np.random.default_rng(settings.seed), original_digest


def pilot_objective(model, batch, settings=PilotSettings()):
    """Proper full-covariance conditional/marginal NLL plus correspondence CE."""
    settings.validate()
    import torch
    from solar_distributional_response_v86 import gaussian_nll_torch
    x = {name: torch.from_numpy(np.asarray(batch[name]).copy()) for name in
         ('field', 'points', 'condition', 'target_sun', 'target_view', 'target_teacher')}
    base = model.coarse_reference_features(x['field'][None], x['points'][None])[0]
    conditional = model.distribution(base, x['condition'], x['target_sun'],
                                     target_view_basis=x['target_view'], diagonal_only=False)
    # The primary pre-pose likelihood has no target-view argument. Training
    # query truth contributes only to the separate conditional supervision.
    marginal = model.marginal_distribution(base, x['condition'], x['target_sun'],
                                            diagonal_only=False)
    c = gaussian_nll_torch(x['target_teacher'], conditional['mean'],
                           covariance=conditional['covariance']).mean() / 32
    m = gaussian_nll_torch(x['target_teacher'], marginal['mean'],
                           covariance=marginal['covariance']).mean() / 32
    n = len(base)
    if n < 2:
        raise ValueError('At least two distinct physical points required for discrimination')
    pair_nll = gaussian_nll_torch(x['target_teacher'][None].expand(n, -1, -1),
                                 marginal['mean'][:, None].expand(-1, n, -1),
                                 covariance=marginal['covariance'][:, None].expand(-1, n, -1, -1)) / 32
    # All columns use the SAME role and Sun. Query-view labels never condition
    # a row/column of this pre-pose matching score.
    logits = -pair_nll / settings.discrimination_temperature
    labels = torch.arange(n)
    discrimination = .5 * (torch.nn.functional.cross_entropy(logits, labels)
                            + torch.nn.functional.cross_entropy(logits.T, labels))
    loss = (settings.conditional_weight * c + settings.marginal_weight * m
            + settings.discrimination_weight * discrimination)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite distributional training objective')
    metrics = dict(loss=float(loss.detach()), conditional_nll_per_dimension=float(c.detach()),
                   marginal_nll_per_dimension=float(m.detach()),
                   correspondence_cross_entropy=float(discrimination.detach()),
                   marginal_covariance_trace=float(marginal['diagonal'].sum(-1).mean().detach()),
                   marginal_mean_norm=float(marginal['mean'].norm(dim=-1).mean().detach()),
                   marginal_top1_correct=int((logits.detach().argmax(-1) == labels).sum()),
                   batch_points=n, primary_matching_view_mode=marginal['view_mode'],
                   covariance_mode=marginal['covariance_mode'])
    return loss, metrics


def fitting_step(model, optimizer, bank, rng, *, expected_coarse_digest,
                 expected_prior_digest=None, settings=PilotSettings(), clock=lambda: None):
    import torch
    from solar_native_v63 import state_digest
    clock()
    batch = sample_training_batch(bank, rng, settings)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = pilot_objective(model, batch, settings)
    loss.backward()
    if any(p.grad is not None for p in model.coarse.parameters()):
        raise ValueError('Frozen original query/coarse encoder received a gradient')
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
        raise ValueError('Nonfinite trainable gradient')
    norm = torch.nn.utils.clip_grad_norm_(trainable, settings.gradient_clip, error_if_nonfinite=True)
    metrics['gradient_norm_before_clip'] = float(norm.detach())
    optimizer.step()
    clock()
    if state_digest(model.coarse) != expected_coarse_digest or model.coarse.training:
        raise ValueError('Frozen coarse encoder state or eval mode changed')
    if expected_prior_digest is not None and prior_digest(model) != expected_prior_digest:
        raise ValueError('Frozen training-view prior buffers or provenance changed')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model parameter after optimizer update')
    return dict(metrics, sampling=batch['trace'])


def prior_digest(model):
    payload = b''.join(getattr(model, name).detach().cpu().contiguous().numpy().tobytes()
                       for name in ('view_mean', 'view_covariance', 'view_factor'))
    return hashlib.sha256(payload + json.dumps(model.prior_provenance, sort_keys=True,
                         separators=(',', ':')).encode()).hexdigest()


def snapshot(model, optimizer, rng, step, settings=PilotSettings()):
    import torch
    return dict(format=PILOT_FORMAT, step=step, settings=asdict(settings.validate()),
                model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=deepcopy(optimizer.state_dict()), numpy_rng=deepcopy(rng.bit_generator.state),
                torch_rng=torch.get_rng_state().clone(), prior_provenance=deepcopy(model.prior_provenance),
                parameter_report=deepcopy(model.parameter_report))


def identical(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def train_pilot(model, optimizer, rng, bank, output, *, expected_coarse_digest,
                provenance, settings=PilotSettings(), clock=lambda: None):
    """No admission authority: external controller MUST cover startup+cleanup.

    Run only after successful source, full-bank, training-prior and budget
    admission. No retry/resume or checkpoint selection is implemented here.
    """
    import torch
    settings.validate()
    if not provenance or provenance.get('split') != 'training':
        raise ValueError('External published training-only provenance required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    try:
        clock()
        bank.validate_model(model, expected_coarse_digest)
        prior_seal = prior_digest(model)
        counts = bank.preflight()
        _write_new(output / 'execution.json', dict(format=PILOT_FORMAT, settings=asdict(settings),
                   bank_manifest_sha256=bank.manifest_sha256, provenance=provenance,
                   coarse_descriptor_sha256=expected_coarse_digest, target_usable_counts=counts,
                   parameter_report=model.parameter_report, model_selection_performed=False,
                   calibration_or_final_access=False))
        torch.save(snapshot(model, optimizer, rng, 0, settings), output / 'step_0.pt')
        with (output / 'trace.jsonl').open('x') as trace:
            for step in range(1, 101):
                clock()
                record = dict(step=step, **fitting_step(model, optimizer, bank, rng,
                    expected_coarse_digest=expected_coarse_digest, expected_prior_digest=prior_seal,
                    settings=settings, clock=clock))
                history.append(record)
                trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                trace.flush()
        clock()
        torch.save(snapshot(model, optimizer, rng, 100, settings), output / 'step_100.pt')
        _write_new(output / 'history_100.json', history)
        summary = dict(complete=True, format=PILOT_FORMAT, updates=100, seed=settings.seed,
                       bank_manifest_sha256=bank.manifest_sha256, model_selection_performed=False,
                       calibration_or_final_access=False, model_promoted=False,
                       artifacts={p.name: sha(p) for p in (output / 'step_0.pt', output / 'step_100.pt',
                                  output / 'history_100.json', output / 'trace.jsonl', output / 'execution.json')})
        clock()
        _write_new(output / 'summary.json', summary)
        return summary
    except BaseException as exc:
        _write_new(output / 'FAILED.json', dict(type=type(exc).__name__, message=str(exc),
                   completed_updates=len(history), all_partial_artifacts_preserved=True))
        raise


def replay_pilot(model, optimizer, rng, bank, original, *, expected_coarse_digest,
                 settings=PilotSettings(), clock=lambda: None, record_callback=None):
    """Exactly replay all 100 updates after a separate charged admission."""
    import torch
    original = Path(original)
    summary = _json(original / 'summary.json')
    if (summary.get('complete') is not True or summary.get('format') != PILOT_FORMAT
            or summary.get('bank_manifest_sha256') != bank.manifest_sha256
            or summary.get('updates') != 100):
        raise ValueError('Completed matching 100-update pilot required')
    for name, digest in summary['artifacts'].items():
        if sha(original / name) != digest:
            raise ValueError('Original training evidence changed before replay')
    bank.validate_model(model, expected_coarse_digest)
    prior_seal = prior_digest(model)
    bank.preflight()
    initial = torch.load(original / 'step_0.pt', map_location='cpu', weights_only=True)
    if not identical(snapshot(model, optimizer, rng, 0, settings), initial):
        raise ValueError('Initialization/model/optimizer/prior/RNG not exactly replayed')
    expected = _json(original / 'history_100.json')
    if len(expected) != 100:
        raise ValueError('All sampling/loss trace rows required')
    for step in range(1, 101):
        actual = dict(step=step, **fitting_step(model, optimizer, bank, rng,
            expected_coarse_digest=expected_coarse_digest, expected_prior_digest=prior_seal,
            settings=settings, clock=clock))
        if record_callback is not None:
            # Evidence observer only: save the ACTUAL replay row, including a
            # mismatch, before its comparison. It receives no model/RNG object.
            record_callback(deepcopy(actual))
            clock()
        if not identical(actual, expected[step - 1]):
            raise ValueError('Sampling/loss/gradient replay differs at step ' + str(step))
    clock()
    final = torch.load(original / 'step_100.pt', map_location='cpu', weights_only=True)
    if not identical(snapshot(model, optimizer, rng, 100, settings), final):
        raise ValueError('Final model/optimizer/RNG not exactly replayed')
    return dict(passed=True, updates=100, exact_model_optimizer_rng_sampling_losses_gradients=True,
                original_summary_sha256=sha(original / 'summary.json'),
                bank_manifest_sha256=bank.manifest_sha256,
                scope='Optimization reproducibility only; no selection or localization gain.')
