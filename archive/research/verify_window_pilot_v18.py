"""Independent geometry/window labels and exact optimizer replay of the v18 pilot."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compact_lunar import CompactMatcher
from context_supervision_v15 import prepare_supervision,proposal_grid
from dataset import sha256
from lunargeo_dataset import write_new
from rotation_bank_pose_v4 import quarter_turn
from search_window_learning_v18 import window_mask
from train_context_pilot_v15 import load_training
from train_search_window_v18 import configure,fitting_step,make_model,make_optimizer,published,verify_frozen
from verify_axial_results_v5 import projected


def independent_labels(tiles,data,prepared):
    grid=proposal_grid();native=(grid+.5)*4-.5;centers=np.rint(native);checked=0;windows=0
    for key,tile in tiles.items():
        z=data[key];a,b=tile['cameras'];forward,visible=projected(native,z['depth_a'],z['depth_b'],a,b)
        for k in range(4):
            labels=prepared[key]['rotations'][k]
            original_query=quarter_turn(native,k,512,inverse=True)
            reverse,rv=projected(original_query,z['depth_b'],z['depth_a'],b,a)
            for direction,truth,mask in [('forward',quarter_turn((forward+.5)/4-.5,k,128),visible),
                                          ('reverse',(reverse+.5)/4-.5,rv)]:
                if not np.array_equal(labels[direction+'_visible'],mask):raise ValueError('Independent visibility differs')
                if not np.allclose(labels[direction][mask],truth[mask],atol=4e-5,rtol=0):raise ValueError('Independent3D projection differs')
                # Reimplement candidate/native rounding and inclusive window bounds
                # in NumPy, separately from the Torch supervision implementation.
                target=labels[direction].astype(np.float32)*4+1.5
                expected=mask[:,None]&(np.abs(target[:,None]-centers[None])<=16).all(-1)
                actual=window_mask(torch.from_numpy(labels[direction]),torch.from_numpy(mask),torch.from_numpy(grid)).numpy()
                if not np.array_equal(expected,actual):raise ValueError('Independent native-window membership differs')
                checked+=len(grid);windows+=expected.size
    return dict(independently_projected_label_points=checked,independently_checked_window_memberships=windows)


def audit(run,output):
    root=Path(run);configure();protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'summary.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
    published(protocol['publication_commit'])
    for source,digest in protocol['source_sha256'].items():
        if sha256(source)!=digest or sha256(root/'source'/source)!=digest:raise ValueError('Pilot source changed')
    for path,digest in complete['sha256'].items():
        if sha256(root/path)!=digest:raise ValueError('Completed pilot changed')
    if protocol['selection_pixels_loaded'] or protocol['selection_inference'] or protocol['calibration_or_test_pixels_loaded']:
        raise ValueError('Train-only pilot scope differs')
    manifest,tiles,data=load_training('reports/localization_training_data_v5.json')
    if sha256('reports/localization_training_data_v5.json')!=protocol['manifest_sha256']:raise ValueError('Manifest changed')
    prepared,support=prepare_supervision(tiles,data)
    if support!=protocol['training_support']:raise ValueError('Training support differs')
    label_audit=independent_labels(tiles,data,prepared)
    coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
    torch.manual_seed(protocol['seed']);rng=np.random.default_rng(protocol['seed']);model=make_model(coarse,protocol['variant'])
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
    final=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
    latest=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    if any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()):raise ValueError('Seeded initialization differs')
    if len(summary['history'])!=100 or protocol['steps']!=100 or latest['step']!=100:raise ValueError('Pilot budget differs')
    optimizer=make_optimizer(model)
    for step,saved in enumerate(summary['history'],1):
        actual=fitting_step(model,optimizer,tiles,prepared,rng,step,protocol['variant'])
        if actual!=saved:raise ValueError('Exact gradient/loss/sampler replay differs at step '+str(step))
    for key,value in model.state_dict().items():
        if not torch.equal(value,final[key]) or not torch.equal(value,latest['model'][key]):raise ValueError('Exactly retrained weights differ')
    verify_frozen(model,initial)
    if rng.bit_generator.state!=latest['numpy_rng'] or not torch.equal(torch.get_rng_state(),latest['torch_rng']):raise ValueError('Exact RNG differs')
    if optimizer.state_dict()['param_groups']!=latest['optimizer']['param_groups']:raise ValueError('Optimizer groups differ')
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():
            if not torch.equal(value,latest['optimizer']['state'][key][name]):raise ValueError('Exactly retrained optimizer differs')
    if not any(not torch.equal(v,initial['coarse.'+k]) for k,v in model.coarse.state_dict().items()):raise ValueError('Adapted coarse weights never changed')
    result=dict(passed=True,completed_steps=100,exact_full_retraining=True,exact_model_optimizer_rng_replay=True,
        fine_encoder_unchanged=True,total_parameters=protocol['total_parameters'],**label_audit,
        pilot_root=str(root),pilot_summary_sha256=sha256(root/'summary.json'),pilot_protocol_sha256=sha256(root/'protocol.json'),
        fitting_source_sha256=protocol['source_sha256'],source_sha256=sha256(__file__),
        scope='Independent geometry/window membership plus exact same-code gradient/model/optimizer replay, not independent loss implementation or localization gain.')
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.run,args.output)
