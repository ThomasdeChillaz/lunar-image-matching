import json
from pathlib import Path
import pytest

from evaluate_blend_dev import Rows


def source(tmp_path):
    (tmp_path/'frozen_inputs.json').write_text(json.dumps(dict(methods=['ncc'])))
    return Rows(tmp_path)


def test_upstream_tail_waits_for_complete_line_without_losing_or_repeating_rows(tmp_path):
    cursor = source(tmp_path)
    path = tmp_path/'results.jsonl'
    path.write_bytes(b'{"pair_id":"case","method":"ncc"')
    cursor.refresh()
    assert cursor.offset == 0 and cursor.rows == {}
    with path.open('ab') as stream:
        stream.write(b'}\n')
    cursor.refresh()
    assert len(cursor.rows) == 1
    cursor.refresh()
    assert len(cursor.rows) == 1
    with path.open('ab') as stream:
        stream.write(b'{"pair_id":"case","method":"ncc"}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        cursor.refresh()


def test_stream_rejects_replaced_design_and_truncated_live_predictions(tmp_path):
    cursor = source(tmp_path)
    path = tmp_path/'results.jsonl'
    path.write_text('{"pair_id":"case","method":"ncc"}\n')
    cursor.refresh()
    path.write_text('')
    with pytest.raises(ValueError, match='replaced/truncated'):
        cursor.refresh()
    (tmp_path/'frozen_inputs.json').write_text('{}')
    with pytest.raises(ValueError, match='design changed'):
        cursor.refresh()
