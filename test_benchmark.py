import json
import pytest
from benchmark import run, summarize


def test_failure_counts_in_denominator():
    rows = [dict(method='test', case='identity', success=True, error_px=0, runtime_ms=1),
            dict(method='test', case='identity', success=False, error_px=None, runtime_ms=1)]
    result = summarize(rows)[0]
    assert result['success_rate'] == .5
    assert result['failed_registrations'] == 1


def test_end_to_end_and_no_overwrite(tmp_path):
    config = dict(seed=42, seeds=1, size=128, success_threshold_px=3,
                  cases=[dict(name='identity', angle=0, dx=0, dy=0, gain=1, gamma=1, noise=0)])
    output = tmp_path/'run'
    run(config, output)
    for name in ['results.csv', 'summary.json', 'config.json', 'environment.json', 'success.png', 'error.png', 'runtime.png', 'gallery.png']:
        assert (output/name).stat().st_size > 0
    assert all(r['success_rate']==1 for r in json.loads((output/'summary.json').read_text()))
    with pytest.raises(FileExistsError): run(config, output)
