"""Audit amended completion; never claim the original time protocol passed."""
import argparse
import json
from pathlib import Path
import subprocess

from north_completion_v52 import (ORIGINAL, INVENTORY, read, sha, snapshot_tree,
                                  verify_copies, original_ended)


def check_amended_summary(summary):
    if (not summary['complete'] or summary['total_updates'] != 6000
            or summary['original_time_protocol_completed'] is not False
            or summary['completion_only_amendment'] is not True
            or summary['additional_fitting_updates'] != 0
            or summary['calibration_or_test_pixels_loaded'] != 0
            or not 0 <= summary['recovery_calendar_seconds'] <= 1800):
        raise ValueError('Explicit bounded no-fitting amendment required; original timeout stays failed')


def audit(run, output, numeric_output):
    root = Path(run); out = Path(output)
    if out.exists() or Path(numeric_output).exists(): raise FileExistsError('Preserve previous replay attempts')
    inventory = read(INVENTORY); original_ended(inventory)
    recovery = read(root / 'recovery_protocol.json'); summary = read(root / 'summary.json')
    check_amended_summary(summary)
    if (recovery['original_inventory_sha256'] != sha(INVENTORY)
            or summary['recovery_protocol_sha256'] != sha(root / 'recovery_protocol.json')
            or recovery['original_time_protocol_completed'] is not False
            or recovery['additional_fitting_updates'] != 0):
        raise ValueError('Completion amendment provenance differs')
    if snapshot_tree(ORIGINAL) != recovery['original_file_sha256']:
        raise ValueError('Original failed attempt was changed')
    verify_copies(ORIGINAL, root, recovery['original_file_sha256'])
    sealed = read(root / 'COMPLETION_COMPLETE.json')['sha256']
    if set(snapshot_tree(root)) != set(sealed) | {'COMPLETION_COMPLETE.json'}:
        raise ValueError('Amended completion file inventory differs')
    for name, digest in sealed.items():
        if sha(root / name) != digest: raise ValueError('Sealed completion changed')
    for name, digest in recovery['source_sha256'].items():
        if sha(name) != digest or subprocess.check_output(['git', 'show', recovery['publication_commit'] + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Executed/public completion source differs')
    # The old numerical checker is deliberately unchanged. Its result verifies
    # metrics/updates only; this wrapper supplies the failed-time/amendment scope.
    from verify_north_results_v39 import cohort
    numeric = cohort(root, numeric_output)
    if not numeric['passed'] or numeric['replayed_rows'] != 1728 or numeric['recorded_updates'] != 6000:
        raise ValueError('Full unchanged numerical cohort replay required')
    result = dict(passed=True, completion_only_amendment=True, original_time_protocol_completed=False,
        additional_fitting_updates=0, replayed_rows=1728, recorded_original_fitting_updates=6000,
        original_failed_attempt_byte_unchanged=True, original_copied_files_verified=len(recovery['original_file_sha256']),
        copied_file_proof='Every pre-existing source/destination file SHA matches the pre-inference original-file freeze; no hard links.',
        numeric_replay_sha256=sha(numeric_output), summary_sha256=sha(root / 'summary.json'),
        recovery_protocol_sha256=sha(root / 'recovery_protocol.json'), source_sha256=sha(__file__),
        original_inventory_sha256=sha(INVENTORY), neural_forward_repeated=False,
        full_optimizer_retraining=False, calibration_or_test_pixels_loaded=0,
        scope='Amended completion and full saved numerical replay. The original six-hour protocol remains FAILED. No extra training, fresh test, calibrated acceptance, adopted model or SOTA claim.')
    out.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2), flush=True); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', default='runs/north-v52-completion-only')
    p.add_argument('--output', default='reports/north_amended_completion_audit_v52.json')
    p.add_argument('--numeric-output', default='reports/north_amended_numeric_metric_replay_v52.json')
    a = p.parse_args(); audit(a.run, a.output, a.numeric_output)
