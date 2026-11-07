"""Procedural file/seal controls only; no renderer, matcher or teacher."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,json
import pytest
import solar_completion_v70 as m


def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')


def fixture(tmp_path):
    old=tmp_path/'failed_v68';unit=old/'terrain'/'camera';unit.mkdir(parents=True)
    original=tmp_path/'old_v67';oldcontract=tmp_path/'base.json';failed=tmp_path/'failed.json'
    base={k:'unchanged-'+k for k in m.UNCHANGED_BASE_KEYS}
    base.update(new_output_root=str(old),state_path=str(failed),completion_absolute_deadline_utc=(datetime.now(timezone.utc)+timedelta(seconds=180)).isoformat(),
                original_root=str(original),missing_unit_indices=list(range(10,96)),reused_unit_indices=list(range(10)),render_source_sha256={},
                original_deadline_met=False,completion_cap_seconds=50400,cpu_threads=2)
    save(oldcontract,base);save(failed,{'state':'failed','completion_absolute_deadline_utc':base['completion_absolute_deadline_utc']})
    images=[]
    for role in('reference','query'):
        for light in range(24):
            path=unit/role/f'light-{light}.png';path.parent.mkdir(exist_ok=True);path.write_bytes(b'PROCEDURAL-DUMMY-IMAGE')
            images.append(dict(role=role,light=light,path=str(path.relative_to(unit)),sha256=m.sha(path)))
        (unit/role/'geometry.npz').write_bytes(b'PROCEDURAL-DUMMY-GEOMETRY')
    for name in('reference-to-query.npz','query-to-reference.npz'):(unit/name).write_bytes(b'PROCEDURAL-DUMMY-LABELS')
    save(unit/'execution.json',{'procedural':True})
    save(unit/'summary.json',dict(complete=True,rendered_images=48,images=images,original_unit_index=10,completion_contract_sha256=m.sha(oldcontract)))
    payload=m.payload_hashes(unit);assert len(payload)==54
    save(unit/'COMPLETE.json',{'sha256':dict(payload,**{'COMPLETE.json':m.EMPTY_SHA256})})
    snapshot=tmp_path/'amendment_snapshot.json'
    data=dict(v68_contract=str(oldcontract),v68_contract_sha256=m.sha(oldcontract),v68_failed_state=str(failed),v68_failed_state_sha256=m.sha(failed),
              v68_output_files_sha256={str(p.relative_to(old)):m.sha(p)for p in old.rglob('*')if p.is_file()},unit10_output=str(unit),unit10_payload_sha256=payload)
    save(snapshot,data)
    c=dict(base,new_output_root=str(tmp_path/'new_v70'),state_path=str(tmp_path/'v70_state.json'),
           continuation_remaining_unit_indices=list(range(11,96)),repaired_payload_unit_indices=[10],
           v68_failure_snapshot=str(snapshot),v68_failure_snapshot_sha256=m.sha(snapshot))
    return c,data,unit


def test_payload_seal_never_includes_itself(tmp_path):
    (tmp_path/'file.txt').write_text('payload')
    entries=m.seal_payload(tmp_path)
    assert entries=={'file.txt':m.sha(tmp_path/'file.txt')}
    assert 'COMPLETE.json'not in entries
    with pytest.raises(FileExistsError):m.seal_payload(tmp_path)


def test_diagnosed_snapshot_accepts_exact_complete_payload(tmp_path):
    c,data,unit=fixture(tmp_path)
    assert m.verify_amendment(c)==data
    with pytest.raises(ValueError):m.verify_seal(unit)


def test_no_deadline_extension_or_unit10_rerender(tmp_path):
    c,_,_=fixture(tmp_path)
    c['completion_absolute_deadline_utc']=(datetime.now(timezone.utc)+timedelta(hours=14)).isoformat()
    with pytest.raises(ValueError,match='completion_absolute_deadline'):m.verify_amendment(c)
    c,_,_=fixture(tmp_path/'second');c['continuation_remaining_unit_indices']=list(range(10,96))
    with pytest.raises(ValueError,match='render only11'):m.verify_amendment(c)


def test_payload_tamper_never_repaired(tmp_path):
    c,_,unit=fixture(tmp_path);(unit/'reference'/'light-0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='file changed'):m.verify_amendment(c)


def test_copy_repair_preserves_all_old_bytes_and_uses_correct_seal(tmp_path,monkeypatch):
    c,data,unit=fixture(tmp_path)
    original={str(p.relative_to(unit)):p.read_bytes()for p in unit.rglob('*')if p.is_file()}
    master={'units':[{}for _ in range(11)]};master['units'][10]={'output':str(Path(c['original_root'])/'terrain'/'camera')}
    monkeypatch.setattr(m,'verify_original',lambda contract:(master,{}))
    contract_path=tmp_path/'continuation.json';save(contract_path,c)
    provenance=m.repair_copy_unit10(c,str(contract_path),'PROCEDURAL-PUBLICATION')
    destination=Path(c['new_output_root'])/'terrain'/'camera'
    assert m.verify_seal(destination)['rendered_images']==48
    assert provenance['rendered_again']is False
    assert {str(p.relative_to(unit)):p.read_bytes()for p in unit.rglob('*')if p.is_file()}==original
    copied=m.payload_hashes(destination);copied.pop('REPAIRED_FROM_V68.json')
    assert copied==data['unit10_payload_sha256']
    assert 'COMPLETE.json'not in json.loads((destination/'COMPLETE.json').read_text())['sha256']
    assert m.repair_copy_unit10(c,str(contract_path),'PROCEDURAL-PUBLICATION')==provenance


def test_new_controller_only_loops_11_to_95_and_uses_safe_sealer():
    source=Path(__file__).with_name('run_solar_completion_v70.py').read_text()
    assert "for index in contract['continuation_remaining_unit_indices']"in source
    assert 'repair_copy_unit10(contract,contract_path,commit)'in source
    assert 'seal_payload(folder)'in source
    assert 'seal.open'not in source
