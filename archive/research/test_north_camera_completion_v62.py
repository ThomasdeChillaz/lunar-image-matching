"""Toy artifact/lineage and render-computation controls, not a geometry result."""
import ast,json
from pathlib import Path
import pytest
from north_camera_completion_v62 import (sha,read,inventory,completed_pairs,copy_pair,reused_images,verify_completion_lineage)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def pair(root,index,eligible=True):
    component='toy';pair_id=f'training_motion-{index}';key=component+'/'+pair_id;folder=Path(root)/key
    views={}
    for role in ('reference','query'):
        geometry=folder/role/'geometry.npz';geometry.parent.mkdir(parents=True,exist_ok=True);geometry.write_bytes(b'toy geometry')
        views[role]=dict(path=str(geometry.relative_to(root)),sha256=sha(geometry))
        for light in range(6):
            p=folder/role/f'light-{light}'/'image.png';p.parent.mkdir();p.write_bytes(b'toy image')
    row=dict(component=component,pair_id=pair_id,family='training_motion',pair_index=index,views=views,labels=[],eligible=eligible)
    write(folder/'summary.json',row)
    return row


def fixture(tmp_path,n=3):
    prior=tmp_path/'prior';prior.mkdir();schedule=tmp_path/'schedule.json'
    write(schedule,dict(pairs=[dict(component='toy',family='training_motion',pair_index=i)for i in range(n)]))
    write(prior/'protocol.json',dict(max_calendar_seconds=1800,schedule_sha256=sha(schedule),publication_commit='a'*40,created_utc='2026-11-02T00:00:03+00:00'))
    pair(prior,0,eligible=False);pair(prior,1)
    # Unfinished images exist, but no pair summary: never reuse these files.
    unfinished=prior/'toy/training_motion-2/reference/light-0/image.png';unfinished.parent.mkdir(parents=True);unfinished.write_bytes(b'incomplete')
    state=dict(state='failed',error_type='TimeoutError',error='Original fixed camera phase allowance exhausted:render',updated_utc='2026-11-02T00:30:00+00:00')
    snapshot=dict(partial_files_sha256=inventory(prior),terminal_controller_state=state,
        original_protocol_sha256=sha(prior/'protocol.json'),source_publication_commit='a'*40,
        completed_camera_pairs=2,completed_pair_ids=['toy/training_motion-0','toy/training_motion-1'])
    return prior,schedule,snapshot


def test_reuses_every_completed_pair_including_failed_geometry_but_no_partial(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path)
    records=completed_pairs(prior,snapshot,schedule)
    assert sorted(records)==snapshot['completed_pair_ids']
    destination=tmp_path/'copy';row=copy_pair(prior/'toy/training_motion-0',destination,records['toy/training_motion-0'])
    assert row['eligible']is False
    assert inventory(destination)==inventory(prior/'toy/training_motion-0')
    assert (destination/'summary.json').stat().st_ino!=(prior/'toy/training_motion-0/summary.json').stat().st_ino
    with pytest.raises(FileExistsError):copy_pair(prior/'toy/training_motion-0',destination,records['toy/training_motion-0'])


def test_mutated_original_data_are_rejected_before_copying(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path)
    (prior/'toy/training_motion-0/reference/light-0/image.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='terminal snapshot'):completed_pairs(prior,snapshot,schedule)


def test_unchanged_synthetic_pair_images_keep_old_timing_unknown(tmp_path):
    root=tmp_path/'root';root.mkdir();row=pair(root,0)
    images=reused_images(root,row)
    assert len(images)==12 and all(i['seconds']is None and i['reused_from_prior_attempt']for i in images)
    assert all(sha(root/i['image'])==i['sha256']for i in images)


def test_completed_manifest_cannot_hide_a_missing_image(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path)
    (prior/'toy/training_motion-1/reference/light-0/image.png').unlink()
    snapshot['partial_files_sha256']=inventory(prior)
    with pytest.raises(ValueError,match='lacks a declared image'):completed_pairs(prior,snapshot,schedule)


def test_original_success_is_not_relabelled_as_a_timeout(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path);snapshot['terminal_controller_state']['state']='complete'
    with pytest.raises(ValueError,match='Actual original rendering timeout'):completed_pairs(prior,snapshot,schedule)


def test_cannot_select_only_favorable_completed_pairs(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path);snapshot['completed_camera_pairs']=1;snapshot['completed_pair_ids']=snapshot['completed_pair_ids'][1:]
    with pytest.raises(ValueError,match='without selection'):completed_pairs(prior,snapshot,schedule)


def full_lineage(tmp_path):
    prior,schedule,snapshot=fixture(tmp_path,n=48);records=completed_pairs(prior,snapshot,schedule)
    snapshot_path=tmp_path/'snapshot.json';state_path=tmp_path/'state.json';write(snapshot_path,snapshot);write(state_path,snapshot['terminal_controller_state'])
    contract=tmp_path/'contract.json';root=tmp_path/'completion';root.mkdir()
    write(contract,dict(snapshot_path=str(snapshot_path),snapshot_sha256=sha(snapshot_path),prior_output=str(prior),
        prior_failed_state=str(state_path),prior_failed_state_sha256=sha(state_path),
        schedule_path=str(schedule),complete_pairs=records,completion_cap_seconds=900,new_image_count=552,reused_image_count=24))
    images=[]
    for i in range(48):
        key=f'toy/training_motion-{i}'
        if key in records:
            row=copy_pair(prior/key,root/key,records[key]);images.extend(reused_images(root,row))
        else:
            row=pair(root,i)
            for item in reused_images(root,row):
                item['seconds']=1.;item['reused_from_prior_attempt']=False;images.append(item)
    write(root/'protocol.json',dict(completion_contract_sha256=sha(contract),original_v59_deadline_met=False,max_calendar_seconds=900))
    write(root/'summary.json',dict(images=images,calendar_seconds=100.,original_render_program_observed_seconds=1797.,
        sum_observed_render_program_intervals_seconds=1897.,reused_images=24,newly_rendered_images=552))
    return root,contract


def test_complete_lineage_separates_original_new_and_combined_costs(tmp_path):
    root,contract=full_lineage(tmp_path);result=verify_completion_lineage(root,contract)
    assert result['passed']and result['newly_rendered_images']==552 and result['reused_images']==24
    assert result['original_attempt_deadline_met']is False
    summary=read(root/'summary.json');summary['images'][0]['seconds']=0.;write(root/'summary.json',summary)
    with pytest.raises(ValueError,match='Unknown old render timing'):verify_completion_lineage(root,contract)


def test_both_missing_endpoint_images_and_false_reuse_are_detected(tmp_path):
    root,contract=full_lineage(tmp_path);s=read(root/'summary.json');s['images'][-1]['reused_from_prior_attempt']=True;write(root/'summary.json',s)
    with pytest.raises(ValueError,match='must be reused'):verify_completion_lineage(root,contract)


def test_render_camera_light_and_exact_label_computations_are_unchanged():
    old=ast.parse(Path('render_north_camera_v59.py').read_text());new=ast.parse(Path(__file__).with_name('render_north_camera_completion_v62.py').read_text())
    def role_loop(t):return next(n for n in ast.walk(t)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='role')
    assert ast.dump(role_loop(old),include_attributes=False)==ast.dump(role_loop(new),include_attributes=False)
    def label_loop(t):return next(n for n in ast.walk(t)if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and [x.id for x in n.target.elts]==['role','other'])
    assert ast.dump(label_loop(old),include_attributes=False)==ast.dump(label_loop(new),include_attributes=False)
