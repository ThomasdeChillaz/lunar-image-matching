"""Predeclared coarse-to-fine lunar Sun search; no query truth inputs."""
import numpy as np
from solar_physics_v63 import solar_direction


def sun_search_catalogue(latitude_deg):
    """A priori fine catalogue; primary initialization uses only5deg/.5deg.

    The1.25deg/.25deg lattice is fixed without consulting any query label and
    contains all24 experimental hour angles. Physical columns can therefore
    also serve a secondary supplied-Sun track without interpolation of states.
    """
    rows=[]
    for di in range(-6,7):
        for hi in range(-72,73):
            h,d=1.25*hi,.25*di;s=solar_direction(latitude_deg,h,d)
            if s[2]>0:
                rows.append(dict(hour_angle_deg=h,subsolar_latitude_deg=d,
                    sun_enu=s.tolist(),coarse=(hi%4==0 and di%2==0)))
    if not rows or not any(r['coarse']for r in rows):raise ValueError('No above-horizon coarse Sun configurations')
    return rows


def bounded_sun_indices(parameters,centre,round_index):
    """Local refinement window around a surviving hypothesis, never query truth."""
    p=np.asarray(parameters,float);c=np.asarray(centre,float)
    if p.ndim!=2 or p.shape[1]!=2 or c.shape!=(2,)or not np.isfinite(p).all()or not np.isfinite(c).all()or round_index not in range(3):raise ValueError('Finite hour/decl catalogue and declared refinement round required')
    hour_radius=2.5 if round_index==0 else 1.25
    use=(np.abs(p[:,0]-c[0])<=hour_radius+1e-9)&(np.abs(p[:,1]-c[1])<=.25+1e-9)
    ids=np.flatnonzero(use)
    if not len(ids):raise ValueError('Precomputed catalogue has no bounded refinement support')
    return ids

