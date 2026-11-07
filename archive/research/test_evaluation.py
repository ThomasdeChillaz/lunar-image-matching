import copy,json
from pathlib import Path
import numpy as np
import pytest
from evaluate import endpoint_statistics,summarize,validate_config
from apply_checkpoints import apply


def test_checkpoint_metric_uses_independent_target_positions():
    identity=np.array([[1,0,0],[0,1,0]],float)
    ref=np.array([[0,0],[10,20]],float);query=ref+[3,4]
    median,rmse=endpoint_statistics(identity,ref,query)
    assert median==5 and rmse==5
    assert endpoint_statistics(None,ref,query)==(None,None)


def row(kind,error,estimated=True):
    return dict(dataset_kind=kind,split='test',error_kind='map_reference_disagreement' if kind=='real_repeat_observation' else 'known_transform_error',method='sift',preprocessing='none',applicability='in_model',error_px=error,estimated=estimated,runtime_ms=1)


def test_reference_types_never_aggregate_and_failures_count():
    rows=[row('real_repeat_observation',None,False),row('real_repeat_observation',1),row('real_image_synthetic_transform',0)]
    out=summarize(rows,[3]);assert len(out)==2
    real=next(r for r in out if r['dataset_kind']=='real_repeat_observation')
    assert real['within_3px_rate']==.5 and real['failed_registrations']==1


def test_invalid_method_does_not_silently_fallback():
    config=json.loads(Path('evaluation_config.json').read_text());config['methods']=['imaginary']
    with pytest.raises(ValueError,match='methods'):validate_config(config)


def annotations():
    points=[dict(reference=[40+(i%5)*80,50+(i//5)*200],query=[42+(i%5)*80,53+(i//5)*200]) for i in range(10)]
    return dict(coordinate_system='512px common-grid pixel centers',reviewer='test fixture',uncertainty_px=3,pairs={'S1_01':points})


def test_import_requires_real_annotation_metadata():
    manifest=json.loads(Path('lunar_manifest.json').read_text());data=annotations()
    result=apply(manifest,data);pair=result['pairs'][0]
    assert pair['reference_type']=='independent_checkpoints' and pair['reference_uncertainty_px']==3
    assert pair['audit_status']=='independent_annotation_imported'


def test_duplicate_and_insufficient_checkpoints_rejected():
    manifest=json.loads(Path('lunar_manifest.json').read_text());data=annotations();data['pairs']['S1_01'][1]=data['pairs']['S1_01'][0]
    with pytest.raises(ValueError,match='Repeated'):apply(manifest,data)
    data=annotations();data['pairs']['S1_01']=data['pairs']['S1_01'][:5]
    with pytest.raises(ValueError,match='Ten finite'):apply(manifest,data)
