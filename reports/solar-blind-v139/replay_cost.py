"""Read-only replay of source/data hashes and measured admission arithmetic.

No image decoding, matching, training, private ledger or execution ticket is used.
The timing record attests the live pose controls; it is not a saved pose replay.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_timing(protocol, timing):
    samples = timing['samples']
    require(len(samples) == 2 and {r['layered'] for r in samples} == {False, True},
            'Both complete generated geometries required')
    for row in samples:
        require(row['all_six_pose_controls_pass'] is True and row['cameras'] == 2048
                and row['lights'] == 24 and row['extra_searches'] == 32
                and isinstance(row['elapsed_seconds'], (int, float))
                and 0 < row['elapsed_seconds'] < 480, 'Full measured search budget required')
    require(timing['full_pipeline'] is True and timing['lunar_images_opened'] is False
            and timing['physical_sunlight_simulation'] is False, 'Preserve control scope')
    forecast = 12*max(row['elapsed_seconds'] for row in samples)+30
    require(abs(forecast-timing['comparison_forecast_seconds']) < 1e-8,
            'Saved forecast differs from independent arithmetic')
    require(protocol['cap_seconds'] == 480 and forecast > protocol['cap_seconds'],
            'Measured forecast must exceed even the full original cap')
    return forecast


def replay(root):
    root = Path(root).resolve()
    home = root/'reports/solar-blind-v139'
    read = lambda p: json.loads(p.read_bytes())
    protocol, timing, result = (read(home/name) for name in ('protocol.json', 'timing.json', 'result.json'))
    pins = dict(protocol['sources'], **protocol['inputs'])
    for name, expected in pins.items():
        p = (root/name).resolve()
        require(p.is_relative_to(root) and p.is_file(), 'Missing/nonportable payload: '+name)
        require(hashlib.sha256(p.read_bytes()).hexdigest() == expected, 'Payload hash mismatch: '+name)
    require(hashlib.sha256((home/'protocol.json').read_bytes()).hexdigest() == result['protocol_sha256'],
            'Public receipt binds a different protocol')
    forecast = audit_timing(protocol, timing)
    require(len(protocol['case_ids']) == 8 and protocol['comparison_cells'] == 48
            and protocol['methods'] == ['direct', 'joint_no_extra', 'uniform', 'appearance', 'geometry', 'active'],
            'Original full study grid required')
    require(result['status'] == 'stopped_at_full_cost_gate' and result['lunar_inference_run'] is False
            and result['new_lunar_results'] == 0 and result['superiority_claim'] is False
            and result['supplement_fixed_charges_seconds'] == result['supplement_authorized_seconds'] == 2400
            and result['supplement_unreserved_seconds'] == 0, 'Preserve result and fixed-charge scope')
    return dict(passed=True, verified_sources=len(protocol['sources']), verified_inputs=len(protocol['inputs']),
        independently_recomputed_forecast_seconds=forecast, original_cap_seconds=480,
        forecast_exceeds_full_original_cap=True, new_lunar_accuracy_results=0,
        inference_performed=False, private_authority_used=False,
        scope='Read-only hash/admission replay; generated live pose controls are not independently replayed saved poses')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(replay(args.root), indent=2))
