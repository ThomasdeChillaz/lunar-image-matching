"""Reuse independent geometry/native-pose replay for the new three-seed cohort."""
import argparse
import json
from pathlib import Path

from dataset import sha256
from lunargeo_dataset import write_new
from train_mutual_cohort_v24 import CONFIGURATION, SEEDS, frozen_protocol, pilot_gate, published
from verify_window_cohort_v18 import audit as common_audit


def audit(run, output):
    root = Path(run); protocol = json.loads((root/'protocol.json').read_text())
    published(protocol['publication_commit']); pilot_gate()
    frozen_protocol(root, protocol['publication_commit'])
    if (protocol['objective'], protocol['variants'], protocol['seeds'], protocol['steps']) != (
            'dual_window_mass_v23', [CONFIGURATION], list(SEEDS), 2000):raise ValueError('Declared cohort differs')
    complete = json.loads((root/'COMPLETE.json').read_text())
    if complete != dict(protocol_sha256=sha256(root/'protocol.json'), protocol_seal_sha256=sha256(root/'PROTOCOL_SEAL.json'),
            summary_sha256=sha256(root/'summary.json')):
        raise ValueError('Completed cohort seal differs')
    # This common auditor checks all native score/pose rows, source snapshots,
    # exact geometric/window labels, optimizer-step counts and sampler draws.
    # Its old training loss is never called: full optimizer retraining is false.
    core_path = root/'independent-common-audit.json'
    core = common_audit(root, core_path, partial=False, retrain=False)
    if (not core['passed'] or core['completed_runs'] != 3 or core['recorded_updates'] != 6000
            or core['replayed_pose_rows'] != 1080 or core['exact_retrained_steps'] != 0):
        raise ValueError('Complete new-cohort replay differs')
    proof = dict(core, base_auditor_source_sha256=core['source_sha256'], source_sha256=sha256(__file__),
        objective='dual_window_mass_v23', independent_new_loss_pilot_retraining_steps=100,
        full_optimizer_retraining_performed=False, core_audit_sha256=sha256(core_path),
        scope='All three exposed-development seeds/1080native checkpoint rows and6000sampler/update records replay. New loss exactly retrained only on100-update pilot; no fresh terrain or model-accuracy claim from replay alone.')
    write_new(output, proof); print(json.dumps(proof, indent=2)); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.run, a.output)
