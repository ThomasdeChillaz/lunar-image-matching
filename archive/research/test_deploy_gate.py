import copy

from deploy_gate import public_alignment


def test_public_ensemble_transform_is_rejected_without_consulting_truth():
    policy=dict(min_global=.4,min_patch=-1.,min_lit=0.,min_support=.2,min_agreement=0.,max_cycle=8.)
    row=dict(pair_id='example',estimated=True,reverse_estimated=True,cycle_p95_px=.1,
        transform=[[1.,0.,8.],[0.,1.,-4.]],correct=False,error_px=999.,
        selection_quality=dict(global_correlation=.8,patch_median_correlation=.7,
            joint_nondark_fraction=.5,geometric_support_fraction=.8,agreement_fraction=.25))
    public=public_alignment(row,policy)
    assert public['accepted'] and public['transform']==row['transform']
    assert 'correct' not in public and 'error_px' not in public
    changed=copy.deepcopy(row)
    changed.update(correct=True,error_px=0.)
    assert public_alignment(changed,policy)==public
    changed['selection_quality']['global_correlation']=.2
    rejected=public_alignment(changed,policy)
    assert not rejected['accepted'] and rejected['transform'] is None
