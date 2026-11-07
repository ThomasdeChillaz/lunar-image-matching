import json
import pytest

from dataset import sha256
from evaluate_blend_dev import Rows
from evaluate_modern_blend_dev import bind_terminal


def terminal_run(root, log_support=1):
    root.mkdir()
    row = dict(pair_id='software-fixture', method='fixture', support=1)
    (root/'frozen_inputs.json').write_text(json.dumps(dict(split='dev', cases=[])))
    (root/'results.json').write_text(json.dumps([row]))
    (root/'results.jsonl').write_text(json.dumps(dict(row, support=log_support))+'\n')
    complete = dict(rows=1, results_sha256=sha256(root/'results.json'), frozen_inputs_sha256=sha256(root/'frozen_inputs.json'))
    (root/'COMPLETE.json').write_text(json.dumps(complete))
    return complete


def test_terminal_binding_checks_final_records_against_the_consumed_log(tmp_path):
    root = tmp_path/'run'
    expected = terminal_run(root)
    assert bind_terminal(Rows(root)) == expected
    changed = tmp_path/'changed'
    terminal_run(changed, log_support=2)
    with pytest.raises(ValueError, match='Streamed predictions differ'):
        bind_terminal(Rows(changed))


def test_partial_append_line_cannot_satisfy_a_completed_source(tmp_path):
    root = tmp_path/'run'
    terminal_run(root)
    path = root/'results.jsonl'
    path.write_text(path.read_text().rstrip('\n'))
    with pytest.raises(ValueError, match='Streamed predictions differ'):
        bind_terminal(Rows(root))
