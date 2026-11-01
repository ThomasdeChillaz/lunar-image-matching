"""Prospective independent northern-terrain simulation plan; no raster reads."""
from dataclasses import dataclass,asdict
import hashlib
import json
import math
from pathlib import Path

SOURCE='https://pgda.gsfc.nasa.gov/data/LOLA_20mpp_NP/LDEM_83N_10MPP_ADJ.TIF'
RADIUS=1737400.
SIZE=512
SUNS=((2.,0.),(2.,120.),(2.,240.),(5.,0.),(5.,120.),(5.,240.))


def inverse_north_stereo(x,y,radius=RADIUS):
    if not all(math.isfinite(v)for v in (x,y,radius)) or radius<=0:raise ValueError('Finite coordinates and positive radius required')
    rho=math.hypot(x,y)
    return math.pi/2-2*math.atan(rho/(2*radius)), 0. if rho==0 else math.atan2(x,-y)


def body_point(x,y,height):
    if not math.isfinite(height) or RADIUS+height<=0:raise ValueError('Finite height above reference sphere required')
    lat,lon=inverse_north_stereo(x,y);r=RADIUS+height
    return [r*math.cos(lat)*math.cos(lon),r*math.cos(lat)*math.sin(lon),r*math.sin(lat)]


def enu_frame(x,y,height):
    lat,lon=inverse_north_stereo(x,y);sl,cl=math.sin(lon),math.cos(lon);sp,cp=math.sin(lat),math.cos(lat)
    return body_point(x,y,height),[[-sl,cl,0.],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]]


def local_point(x,y,height,origin,enu_rows):
    point=body_point(x,y,height);delta=[a-b for a,b in zip(point,origin)]
    return [sum(a*b for a,b in zip(row,delta))for row in enu_rows]


def camera_intrinsics(size=SIZE,fov_degrees=60.):
    if type(size)is not int or size<=0 or not(0<fov_degrees<180):raise ValueError('Positive integer image size and finite field of view required')
    focal=size/(2*math.tan(math.radians(fov_degrees)/2))
    return [[focal,0.,(size-1)/2],[0.,focal,(size-1)/2],[0.,0.,1.]]


def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def unit(v):
    if not all(math.isfinite(x)for x in v):raise ValueError('Finite vector required')
    length=math.sqrt(sum(x*x for x in v))
    if length<1e-12:raise ValueError('Degenerate camera direction')
    return [x/length for x in v]


def camera_to_world(position,target,up=(0.,1.,0.)):
    if len(position)!=3 or len(target)!=3 or not all(math.isfinite(x)for x in [*position,*target]):raise ValueError('Finite3D camera/target required')
    forward=unit([t-c for t,c in zip(target,position)]);right=unit(cross(forward,up));down=unit(cross(forward,right))
    return [[right[i],down[i],forward[i],position[i]]for i in range(3)]+[[0.,0.,0.,1.]]


def support_overlap(a,b):
    return abs(a['x']-b['x'])<20480 and abs(a['y']-b['y'])<20480


def plan():
    sites=[]
    for row,y in enumerate((-60000.,-30000.,0.,30000.,60000.)):
        for col,x in enumerate((-105000.,-75000.,-45000.,-15000.,15000.,45000.,75000.,105000.)):
            # Snap to declared native pixel centres, independent of all heights.
            native_col=math.floor((x+216400)/10);native_row=math.floor((216400-y)/10)
            sites.append(dict(component=f'np31-r{row}-c{col}',sector=f'projected-strip-{col//2}',
                x=-216400+(native_col+.5)*10,y=216400-(native_row+.5)*10,
                centre_native_row=native_row,centre_native_column=native_col))
    for sector in sorted({s['sector']for s in sites}):
        ordered=sorted([s for s in sites if s['sector']==sector],key=lambda s:hashlib.sha256(('20261101/'+s['component']).encode()).hexdigest())
        for s,split in zip(ordered,['final_test']*3+['confidence_calibration']+['model_selection']*2+['training']*4):s['split']=split
    for i,(x,y)in enumerate(((-135000.,-75000.),(-135000.,75000.),(135000.,-75000.),(135000.,75000.))):
        c=math.floor((x+216400)/10);r=math.floor((216400-y)/10)
        sites.append(dict(component=f'np31-pilot-{i}',sector='pilot_only',x=-216400+(c+.5)*10,
            y=216400-(r+.5)*10,centre_native_row=r,centre_native_column=c,split='renderer_cost_pilot'))
    for i,a in enumerate(sites):
        if not (1024<=a['centre_native_row']<43280-1024 and 1024<=a['centre_native_column']<43280-1024):raise ValueError('Terrain halo outside native source')
        if any(support_overlap(a,b)for b in sites[i+1:]):raise ValueError('Declared full terrain/occluder supports overlap')
        a['native_window']=[a['centre_native_row']-1024,a['centre_native_column']-1024,2049,2049]
        lat,lon=inverse_north_stereo(a['x'],a['y']);a.update(centre_latitude_deg=math.degrees(lat),centre_longitude_deg=math.degrees(lon)%360)
    counts={split:sum(s['split']==split for s in sites)for split in sorted({s['split']for s in sites})}
    if counts!=dict(confidence_calibration=4,final_test=12,model_selection=8,renderer_cost_pilot=4,training=16):raise ValueError('Predeclared group counts differ')
    return dict(scope='PREPARED additional self-rendered northern-terrain study; not released LunarGeo data, a passed freshness gate or a model result.',
        source_url=SOURCE,source_product_page='https://pgda.gsfc.nasa.gov/products/98',
        native_source_shape=[43280,43280],source_pixel_scale_m=10.,source_height_unit='metres above1737400m reference sphere',
        source_frame='MOON_ME DE421; renderer uses per-component explicit local ENU frame with body transform retained',
        source_grid_transform=[10.,0.,-216400.,0.,-10.,216400.],source_nodata='nan; no inpainting or replacement terrain',
        sites=sites,component_counts=counts,full_support_width_m=20480.,
        mesh_native_offset_indices=list(range(-1024,-256,4))+list(range(-256,257))+list(range(260,1025,4)),
        image_size=SIZE,intrinsics=camera_intrinsics(),pixel_convention='Integer native pixel centres; principal((W-1)/2,(H-1)/2). Blender edge coordinates subtract0.5; vertical image flip explicit.',
        camera_views=4,rendered_sun_elevation_azimuth_degrees=[list(s)for s in SUNS],images_per_component=24,
        camera_specs=[dict(view=0,position_offset=[0.,0.,1000.],target=[0.,0.,0.]),
            dict(view=1,position_offset=[250.,150.,1000.],target=[0.,0.,0.]),
            dict(view=2,position_offset=[450.,-200.,800.],target=[0.,0.,0.]),
            dict(view=3,position_offset=[-350.,300.,1200.],target=[100.,150.,0.])],
        camera_height_rule='Add maximum local core-mesh height to each declared cameraZoffset; target coordinates relative to source-centre terrain point. Deterministic geometry-only clearance; no image/matcher selection.',
        material=dict(type='Diffuse BSDF',constant_color=[.5,.5,.5],roughness=0.,external_albedo_texture=False),
        renderer=dict(engine='Cycles',device='CPU',threads=2,samples=16,denoising=False,
            max_bounces=2,pixel_filter='BOX',filter_size=.01,world_background_strength=.001,
            sun_strength=3.,sun_angular_size_radians=0.,view_transform='Standard',exposure=0.,gamma=1.,
            resolution_percentage=100,motion_blur=False,depth_of_field=False,
            package='Official Blender4.5.9Intel macOS',package_sha256='00c8a433504291374bfa045c0c2d708a779f8abc8400b4718fdd11c117486fa4',build_hash='8bf95cbd38d1'),
        primary_lighting_pairs=[[0,0],[0,3],[0,4],[1,5]],primary_query_views=[1,2,3],
        primary_pose_pairs_per_component=12,primary_final_positive_pairs=144,
        primary_final_negative_pairs=144,training_lighting_pairs='All6x6lighting pairs for referenceview0and3other queryviews:108positive pairs pertrainingcomponent.',
        negative_pair_policy='Deterministic next-component cycle within eachsplit sorted by componentID; same12declared lighting/view patterns. Freeze explicit both-endpoint case list before inference. Allterrain halos disjoint; poseGTexpressed in referenceENUvia retained body transforms, scoring only.',
        depth_rule='Keep unnormalized renderer depth/position outputs and explicit validated convention. Ray ranges and camera-axisZare distinct; analytic plane/box and independent projected-correspondence gates must pass first.',
        split_before_any_source_height_decode=True,split_before_patch_extraction=True,
        neural_inference_performed=False,calibration_or_test_images_rendered=False,source_height_pixels_decoded=0,
        gates_required=['Official runtime signature/checksum and actual analytic camera/depth/occlusion controls.',
            'Actual cost pilot on four declared pilot components; no reserve-model inference.',
            'All historical optical/training footprint exposure audit; northern candidates are not yet certified fresh.',
            'Source byte/version/array hashes, valid-terrain checks, mesh/camera/lighting hashes.',
            'Independent exact-correspondence oracle >=95%valid development cases before fitting.',
            'Publish complete source/case/metric freeze before any final matcher inference.'],
        generation_budget=dict(cpu_threads=2,pilot_calendar_cap_seconds=1800,full_render_calendar_cap_seconds=6*3600,
            policy='No renderer while existing fitting/baseline/regression jobs are live. Measure pilot cost before allocating full generation.'),
        limits=['Shared DEM/renderer/reflectance remain dependence; spatially separate supports do not create independent missions.',
            'Simplified diffuse reflectance and finite terrain support do not reproduce all lunar optics/remote occluders.',
            'Original missing12fresh released LunarGeo groups remain a recorded shortfall.',
            'No paid compute,human landmark review or verified orbital navigation claim.'])


if __name__=='__main__':
    output=Path('reports/north_render_prospective_plan_v31.json')
    if output.exists():raise FileExistsError('Do not overwrite the source/footprint split freeze')
    p=plan();p['planner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(dict(component_counts=p['component_counts'],source_height_pixels_decoded=0,prospective_only=True)))
