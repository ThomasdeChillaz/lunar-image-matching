"""Text-metadata camera/conic consistency for the CRESENT-365 candidate source.

No image decoding or matcher inference. Frame0 identified a convention and
60degree camera hypothesis; different declared frames must validate it before
using these fields. This is not a dense-depth or fresh-pose benchmark gate.
"""
import csv
import math

SIZE=1024
FOCAL=512/math.tan(math.pi/6)
CONTROL_IDS=(1,7,17,101,1000,7000,15282)


def multiply(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def transpose(a):
    return list(map(list,zip(*a)))


def rotation_wxyz(q):
    if len(q)!=4 or not all(math.isfinite(v) for v in q) or abs(sum(v*v for v in q)-1)>1e-6:
        raise ValueError('Finite unit wxyz quaternion required; no silent normalization')
    w,x,y,z=q
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]


def annotations(text,kind):
    number={'world':6,'ellipse':5}[kind];rows={}
    # The published world header omits the rotation field. Validate actual row
    # width rather than silently shifting its final numeric angle into the ID.
    for fields in csv.reader(text.splitlines()[1:],skipinitialspace=True):
        if not fields:continue
        if len(fields)!=number+1:raise ValueError('Published annotation row width differs')
        name=fields[-1].strip()
        if name in rows:raise ValueError('Duplicate source crater ID')
        rows[name]=[float(v) for v in fields[:-1]]
    return rows


def project_conic(world,position,world_to_camera,focal=FOCAL,principal=(512.,512.)):
    x,y,z,a,b,phi=world
    if not all(math.isfinite(v) for v in world) or min(a,b)<=0:raise ValueError('Finite positive crater ellipse required')
    radius=math.sqrt(x*x+y*y+z*z);rho=math.hypot(x,y)
    if radius==0:raise ValueError('World crater centre cannot be zero')
    east=[-y/rho,x/rho,0] if rho else [1.,0.,0.]
    up=[x/radius,y/radius,z/radius]
    north=[up[1]*east[2]-up[2]*east[1],up[2]*east[0]-up[0]*east[2],up[0]*east[1]-up[1]*east[0]]
    basis=transpose([east,north,[v-p for v,p in zip((x,y,z),position)]])
    camera=multiply(world_to_camera,basis);c,s=math.cos(phi),math.sin(phi)
    local=[[a*a*c*c+b*b*s*s,(a*a-b*b)*c*s,0],[(a*a-b*b)*c*s,a*a*s*s+b*b*c*c,0],[0,0,-1]]
    k=[[focal,0,principal[0]],[0,focal,principal[1]],[0,0,1.]]
    homography=multiply(k,camera);dual=multiply(multiply(homography,local),transpose(homography))
    if dual[2][2]==0:raise ValueError('Degenerate projected conic')
    centre=[dual[0][2]/dual[2][2],dual[1][2]/dual[2][2]]
    normalized=[[v/(-dual[2][2]) for v in row] for row in dual]
    shape=[[normalized[i][j]+centre[i]*centre[j] for j in range(2)] for i in range(2)]
    return centre,shape,camera[2][2]


def ellipse_shape(ellipse):
    _,_,a,b,phi=ellipse;c,s=math.cos(phi),math.sin(phi)
    return [[a*a*c*c+b*b*s*s,(a*a-b*b)*c*s],[(a*a-b*b)*c*s,a*a*s*s+b*b*c*c]]


def audit_frame(pose,world,ellipses):
    if set(world)!=set(ellipses):raise ValueError('World/image annotation IDs differ')
    if len(pose)!=7 or not all(math.isfinite(v) for v in pose):raise ValueError('Finite seven-number pose record required')
    rotation=rotation_wxyz(pose[3:]);errors=[];shape_errors=[];excluded={}
    for name,truth in ellipses.items():
        reason=None
        if not all(math.isfinite(v) for v in truth+world[name]) or min(truth[2:4])<=0:reason='invalid_label'
        elif not (0<=truth[0]<SIZE and 0<=truth[1]<SIZE):reason='ellipse_centre_outside_image'
        if reason:
            excluded[reason]=excluded.get(reason,0)+1;continue
        centre,shape,depth=project_conic(world[name],pose[:3],rotation)
        if depth<=0:
            excluded['behind_camera']=excluded.get('behind_camera',0)+1;continue
        expected=ellipse_shape(truth)
        errors.append(math.hypot(centre[0]-truth[0],centre[1]-truth[1]))
        shape_errors.append(max(abs(shape[i][j]-expected[i][j]) for i in range(2) for j in range(2))/max(1,max(abs(v) for row in expected for v in row)))
    if not errors:raise ValueError('No eligible finite front-facing annotation ellipses')
    return dict(annotations=len(world),eligible_ellipses=len(errors),excluded=excluded,
        centre_rmse_px=math.sqrt(sum(v*v for v in errors)/len(errors)),max_centre_error_px=max(errors),
        max_shape_relative_error=max(shape_errors),
        passed=len(errors)>=8 and max(errors)<1e-4 and max(shape_errors)<1e-5)
