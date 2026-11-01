"""Verify original rows are immutable, then replay every amended native pose."""
import argparse
import json
from pathlib import Path

from dataset import sha256
from lunargeo_dataset import write_new
from verify_native_moonst3r_v19 import audit as common_audit


def audit(run, output):
    root = Path(run); protocol = json.loads((root/'protocol.json').read_text())
    contract = json.loads(Path(protocol['recovery_contract_path']).read_text()); original = Path(contract['original_run'])
    if sha256(protocol['recovery_contract_path']) != protocol['recovery_contract_sha256']:raise ValueError('Public recovery contract changed')
    if (original/'summary.json').exists():raise ValueError('Expired original run was retrospectively completed')
    for name, want in contract['original_case_sha256'].items():
        if sha256(original/'cases'/name) != want or sha256(root/'cases'/name) != want:raise ValueError('Original case was replaced or rewritten')
    summary = json.loads((root/'summary.json').read_text())
    if (summary['original_cases_preserved'], summary['supplemental_cases'], summary['original_calendar_protocol_completed']) != (71, 1, False):
        raise ValueError('Recovery scope/count differs')
    if summary['supplemental_calendar_seconds'] > 600:raise ValueError('Supplemental calendar cap breached')
    core_path = root/'independent-native-audit.json'; core = common_audit(root, core_path)
    proof = dict(core, source_sha256=sha256(__file__), base_auditor_source_sha256=core['source_sha256'],
        original_calendar_protocol_completed=False, amended_complete_cases=72, original_cases_preserved=71, supplemental_cases=1,
        recovery_contract_sha256=sha256(protocol['recovery_contract_path']), core_audit_sha256=sha256(core_path),
        scope='Explicit amended two-phase development comparison. All72 metrics/common poses replay; original calendar-expired run remains incomplete. No fresh test, paired latency or SOTA claim.')
    write_new(output, proof); print(json.dumps(proof, indent=2)); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    audit(a.run, a.output)
