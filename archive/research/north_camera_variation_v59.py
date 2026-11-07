"""Draft geometry-only camera schedules; no image loading or renderer result.

Each component has independent deterministic camera draws. Both families
vary reference cameras, query positions and targets; diagnostic holdout uses
roll magnitudes absent from the training family. Roll is not geometric
occlusion, and neither camera metadata nor a view ID may enter a matcher.
"""
import hashlib,math,random
from pathlib import Path
from north_render_plan_v31 import camera_to_world,camera_intrinsics

FAMILIES=('training_motion','unseen_roll','occlusion_stress')
ALLOWED_SPLITS=('training','train','model_selection','renderer_cost_pilot')


def rolled_pose(position,target,roll_degrees):
 if not math.isfinite(roll_degrees):raise ValueError('Finite optical-axis roll required')
 pose=camera_to_world(position,target);angle=math.radians(roll_degrees);c,s=math.cos(angle),math.sin(angle)
 # Right multiplication rotates the camera right/down basis; forward unchanged.
 for row in pose[:3]:
  right,down=row[0],row[1];row[0]=c*right+s*down;row[1]=-s*right+c*down
 return pose


def schedule(site,clearance,count=4):
 if site['split']not in ALLOWED_SPLITS:raise ValueError('Calibration/final camera construction remains outside this development control')
 if not isinstance(count,int)or isinstance(count,bool)or not 1<=count<=8:raise ValueError('One to eight predeclared pairs per family required')
 if not math.isfinite(clearance):raise ValueError('Finite acquired terrain clearance required')
 component=site['component'];result=[]
 for family in FAMILIES:
  digest=hashlib.sha256(('north-camera-v59/20261102/'+component+'/'+family).encode()).hexdigest()
  rng=random.Random(int(digest,16))
  for i in range(count):
   reference_position=[rng.uniform(-150,150),rng.uniform(-150,150),clearance+rng.uniform(1000,1400)]
   reference_target=[rng.uniform(-100,100),rng.uniform(-100,100),0.]
   query_position=[reference_position[0]+rng.uniform(-500,500),reference_position[1]+rng.uniform(-500,500),clearance+rng.uniform(900,1500)]
   query_target=[rng.uniform(-150,150),rng.uniform(-150,150),0.]
   # Family difference is explicit; geometry distribution shift is reported,
   # not reinterpreted as a new independent terrain or physical sensor test.
   reference_roll=rng.uniform(-20,20)
   relative_roll=rng.choice((-1,1))*rng.uniform(35,55)if family=='unseen_roll'else rng.uniform(-20,20)
   query_roll=reference_roll+relative_roll
   query_fov=60.
   if family=='occlusion_stress':
    # Geometry-only stress design, not a claim that any point is occluded.
    # Narrower10deg queryFOV keeps the oblique ray cone below the horizon.
    height=clearance+rng.uniform(800,1000);zenith=math.radians(rng.uniform(65,75));azimuth=rng.uniform(-math.pi,math.pi)
    radius=height*math.tan(zenith)
    query_position=[query_target[0]+radius*math.cos(azimuth),query_target[1]+radius*math.sin(azimuth),height]
    query_fov=10.
   def spec(position,target,roll,fov=60.):
    return dict(position=position,target=target,roll_degrees=roll,cam2world=rolled_pose(position,target,roll),
     fov_degrees=fov,intrinsics=camera_intrinsics(512,fov))
   result.append(dict(component=component,split=site['split'],family=family,pair_index=i,
    seed_sha256=digest,relative_roll_parameter_degrees=relative_roll,
    reference=spec(reference_position,reference_target,reference_roll),
    query=spec(query_position,query_target,query_roll,query_fov)))
 return result


def protocol(sites,clearances):
 if not sites or len({s['component']for s in sites})!=len(sites):raise ValueError('Nonempty unique declared component list required')
 if set(clearances)!={s['component']for s in sites}:raise ValueError('Exact acquired clearance inventory required')
 pairs=[p for site in sites for p in schedule(site,clearances[site['component']])]
 return dict(schema_version=59,prepared_only=True,rendered_images=0,matcher_forwards=0,
  families=list(FAMILIES),pairs=pairs,cpu_threads=2,
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  roll_definition='Difference of camera optical-axis roll parameters relative to each look-at basis; not the full relative SO3 rotation angle.',
  image_size=512,lighting='Retain actual per-site six-light solar contract; no brightness-based sunlight or visibility labels.',
  occlusion_definition='Source-valid and target-in-frame, but target mesh first-hit differs by more than0.05m. Shadows and out-of-frame points are separate.',
  occlusion_training_gate=dict(minimum_geometrically_occluded_points=8,minimum_components_with_occlusion=2,minimum_independently_verified_occluded_query_rays=2),
  required_before_inference=['Publish this exact camera schedule and source hashes.',
   'Pilot on previously excluded renderer-cost terrains; never read calibration/final supports.',
   'Measure actual full-image finite surface support, BVH visibility and overlap; preserve invalid candidates and declare eligibility geometrically before matching.',
   'Exact projected-correspondence pose control must pass at least95% of valid pilot geometries.',
   'Freeze unseen-roll cases before model inference; no adaptation to heldout camera families.',
   'Use image-only matchers and common reference-only geometry/pose estimator.'],
  scope='Draft camera-distribution control on existing development/pilot terrains. Not new independent terrain, a rendered/audited dataset, a final test, or a model improvement.')
