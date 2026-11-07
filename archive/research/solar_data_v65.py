"""New development cameras; audited v59 camera mathematics with a new fixed seed.

Exactly two motion and two roll-varied pairs per original development group.
No new geography, calibration/final access or image-based camera selection.
"""
import hashlib,math,random
from north_camera_variation_v59 import rolled_pose
from north_render_plan_v31 import camera_intrinsics
FAMILIES=('training_motion','unseen_roll')
ALLOWED_SPLITS=('training','model_selection','renderer_cost_pilot')
def schedule(site, clearance, count=4):
    if site['split'] not in ALLOWED_SPLITS:
        raise ValueError('Calibration/final camera construction remains outside this development control')
    if not isinstance(count, int) or isinstance(count, bool) or (not 1 <= count <= 8):
        raise ValueError('One to eight predeclared pairs per family required')
    if not math.isfinite(clearance):
        raise ValueError('Finite acquired terrain clearance required')
    component = site['component']
    result = []
    for family in FAMILIES:
        digest = hashlib.sha256(('solar-camera-v65/20261104/' + component + '/' + family).encode()).hexdigest()
        rng = random.Random(int(digest, 16))
        for i in range(count):
            reference_position = [rng.uniform(-150, 150), rng.uniform(-150, 150), clearance + rng.uniform(1000, 1400)]
            reference_target = [rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0]
            query_position = [reference_position[0] + rng.uniform(-500, 500), reference_position[1] + rng.uniform(-500, 500), clearance + rng.uniform(900, 1500)]
            query_target = [rng.uniform(-150, 150), rng.uniform(-150, 150), 0.0]
            reference_roll = rng.uniform(-20, 20)
            relative_roll = rng.choice((-1, 1)) * rng.uniform(35, 55) if family == 'unseen_roll' else rng.uniform(-20, 20)
            query_roll = reference_roll + relative_roll
            query_fov = 60.0
            if family == 'occlusion_stress':
                height = clearance + rng.uniform(800, 1000)
                zenith = math.radians(rng.uniform(65, 75))
                azimuth = rng.uniform(-math.pi, math.pi)
                radius = height * math.tan(zenith)
                query_position = [query_target[0] + radius * math.cos(azimuth), query_target[1] + radius * math.sin(azimuth), height]
                query_fov = 10.0

            def spec(position, target, roll, fov=60.0):
                return dict(position=position, target=target, roll_degrees=roll, cam2world=rolled_pose(position, target, roll), fov_degrees=fov, intrinsics=camera_intrinsics(512, fov))
            result.append(dict(component=component, split=site['split'], family=family, pair_index=i, seed_sha256=digest, relative_roll_parameter_degrees=relative_roll, reference=spec(reference_position, reference_target, reference_roll), query=spec(query_position, query_target, query_roll, query_fov)))
    return result
