"""Narrow, resolved-ellipse source-camera check after the broad audit failed.

The original all-annotation gate remains failed and immutable. The resolution
predicate is geometric and must be checked on different declared metadata
frames. Neither this function nor its source labels certify renderer depth,
terrain visibility, benchmark splits or matching accuracy.
"""
import math
from cresent_geometry_v20 import audit_frame,SIZE

MINIMUM_MINOR_PX=2.
VALIDATION_IDS=(2,33,333,3333,9999)


def resolved_camera_check(pose,world,ellipses):
    if set(world)!=set(ellipses):raise ValueError('Source crater identities differ')
    use={};excluded={}
    for name,label in ellipses.items():
        w=world[name];reason=None
        if not all(math.isfinite(v) for v in label+w) or min(label[2:4])<=0:reason='invalid_label'
        elif not (0<=label[0]<SIZE and 0<=label[1]<SIZE):reason='centre_outside_image'
        elif min(label[2:4])<MINIMUM_MINOR_PX:reason='unresolved_ellipse'
        elif sum(c*(p-c) for c,p in zip(w[:3],pose[:3]))<=0:reason='surface_disc_faces_away'
        if reason:excluded[reason]=excluded.get(reason,0)+1
        else:use[name]=label
    if use:
        measured=audit_frame(pose,{name:world[name] for name in use},use)
    else:
        measured=dict(passed=False,eligible_ellipses=0)
    return dict(camera_convention_on_resolved_annotations_passed=measured['passed'],
        input_annotations=len(world),geometric_resolution_exclusions=excluded,measurement=measured,
        condition='Finite image-centre labels, both semiaxes>=2px, and surface disc facing camera. This is not full terrain visibility.',
        scope='Narrow source-camera annotation consistency only. Original broad annotation audit remains failed; no dense-depth, pose or fresh-final gate passes from this check.')
