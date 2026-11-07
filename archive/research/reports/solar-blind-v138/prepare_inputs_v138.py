"""Metadata/hash-only clean-catalogue census and separated future experiment inputs.

No image decoding, matching, camera fitting, rendering, training or query scoring.
"""
import hashlib
import json
from pathlib import Path
import struct

HOME=Path('reports/solar-blind-v138')
COHORT=Path('reports/solar-clean-rerender-cohort-v106.json')
RECEIPT=Path('runs/solar-clean-radiance-v106/render/COMPLETE.json')
MASTER=Path('runs/solar-reference-v80-full/case-assets/manifest.json')
MODERN=Path('reports/solar-five-arm-v115/comparison-protocol.json')


def read(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def png_header(record):
    path=Path(record['path'])
    if sha(path)!=record['sha256']:
        raise ValueError('Changed completed clean image: '+str(path))
    raw=path.read_bytes()[:33]
    if raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[12:16]!=b'IHDR':
        raise ValueError('Expected a sealed PNG header')
    width,height,bit_depth,colour=struct.unpack('>IIBB',raw[16:26])
    if (width,height,bit_depth,colour)!=(512,512,16,2):
        raise ValueError('Native512 RGB16 clean source required')
    return dict(record,bytes=path.stat().st_size,native_shape=[512,512],bit_depth=16)


def prepare():
    cohort,receipt,master,modern=map(read,(COHORT,RECEIPT,MASTER,MODERN))
    if (not receipt['complete'] or receipt['cohort_sha256']!=sha(COHORT)
            or receipt['calibration_or_final_access'] or cohort['split']!='model_selection'):
        raise ValueError('Completed, source-bound development-only clean library required')
    rows={r['key']:r for r in receipt['rows']}
    declared={r['key']:r for r in cohort['renders']}
    if set(rows)!=set(declared): raise ValueError('Missing original completed render rows')
    master_rows={r['case_id']:r for r in master['cases']}
    inference=[]; labels=[]; files={str(p):sha(p) for p in (COHORT,RECEIPT,MASTER,MODERN)}
    for i,case in enumerate(cohort['cases']):
        origin=master_rows[case['case_id']]
        catalog=[]
        for key in case['reference_keys']:
            row=rows[key]; original=declared[key]
            if (row['role']!='reference' or row['unit_index']!=case['unit_index']
                    or row['clean_dither']!=0. or not row['geometry']['exact_camera_depth_and_rays']
                    or '/reference/' not in row['clean_rgb16']['path']):
                raise ValueError('Only qualified clean REFERENCE-camera images form the catalogue')
            image=png_header(row['clean_rgb16']); files[image['path']]=image['sha256']
            catalog.append(dict(light_id=row['light_index'],sun_world=original['sun_enu'],image=image))
        catalog.sort(key=lambda r:r['light_id'])
        if [r['light_id'] for r in catalog]!=list(range(24)):
            raise ValueError('All24 declared reference lights are required')
        reference=rows[case['reference_keys'][0]]
        geometry=dict(path=reference['archive_geometry_path'],sha256=reference['archive_geometry_sha256'])
        if sha(geometry['path'])!=geometry['sha256']: raise ValueError('Changed reference geometry')
        files[geometry['path']]=geometry['sha256']
        query=png_header(rows[case['query_key']]['clean_rgb16']);files[query['path']]=query['sha256']
        identifier=f'blind138-{i:02d}'
        inference.append(dict(id=identifier,reference_geometry=geometry,catalogue=catalog,
            query_image=query,query_intrinsics=origin['query_intrinsics']))
        query_geometry=rows[case['query_key']]
        scoring_geometry=dict(path=query_geometry['archive_geometry_path'],sha256=query_geometry['archive_geometry_sha256'])
        if sha(scoring_geometry['path'])!=scoring_geometry['sha256']: raise ValueError('Changed scoring geometry')
        files[scoring_geometry['path']]=scoring_geometry['sha256']
        labels.append(dict(id=identifier,source_case=case['case_id'],positive=case['positive'],
            component=origin['component'],query_component=origin['query_component'],
            query_geometry=scoring_geometry,query_sun_scoring_only=declared[case['query_key']]['sun_enu']))
    selected=set(modern['comparison_case_ids'])
    original_cases=[master_rows[k] for k in selected]
    available_refs={r['unit_index'] for r in receipt['rows'] if r['role']=='reference'}
    modern_refs={r['unit_index'] for r in original_cases}
    report=dict(schema='blind_catalogue_metadata_preparation_v138',prepared_only=True,
        lunar_matching_or_pose_fitting=False,photometric_images_decoded=False,scientific_execution_authorized=False,
        clean_source= str(COHORT),clean_source_sha256=sha(COHORT),source_files=files,
        pilot_cases=len(inference),pilot_positive_pairs=sum(r['positive'] for r in labels),
        positive_terrain_components=sorted({r['component'] for r in labels if r['positive']}),
        negative_endpoint_components=sorted({r[k] for r in labels if not r['positive'] for k in ('component','query_component')}),
        clean_reference_catalogues=len(available_refs),lights_per_reference=24,clean_completed_images=len(receipt['rows']),
        clean_png_dither=0.,clean_png_depth=16,native_pixels=512,
        historical_modern_reference_cameras=len(modern_refs),
        historical_modern_reference_cameras_with_clean_catalogue=sorted(modern_refs&available_refs),
        historical_modern_reference_cameras_missing_clean_catalogue=sorted(modern_refs-available_refs),
        historical_modern_exact_case_overlap=[r['source_case'] for r in labels if r['source_case'] in selected],
        historical_modern_query_encoding='Archived default-dither RGB8; not the clean pilot inputs',
        data_gate='Clean small pilot available, broad clean-terrain/modern comparison not yet available',
        no_fresh_geography_claim=True,calibration_or_final_access=False)
    return inference,labels,report


if __name__=='__main__':
    inference,labels,report=prepare()
    for name,document in [('inference-inputs.json',inference),('scoring-inputs.json',labels),('data-preparation.json',report)]:
        with (HOME/name).open('x') as f:json.dump(document,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_files'},indent=2))
