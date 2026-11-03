"""Lunar illumination states and coverage-aware Sun-ray evidence.

This is independent of camera visibility and of image brightness. Unit world
rays point TOWARD the Sun. Reference-map ceiling bounds are explicit assumptions,
not assertions about unobserved real lunar terrain.
"""
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np


class SolarState(IntEnum):
    UNKNOWN=0
    CLEAR=1
    SHADOW=2


@dataclass(frozen=True)
class TerrainEnvelope:
    xmin:float
    xmax:float
    ymin:float
    ymax:float
    maximum_height:float
    ceiling_valid:bool=True

    def __post_init__(self):
        if not np.isfinite([self.xmin,self.xmax,self.ymin,self.ymax,self.maximum_height]).all()or self.xmin>=self.xmax or self.ymin>=self.ymax:
            raise ValueError('Finite nonempty reference-map envelope required')

    def supported_distance(self,origin,ray):
        """Distance until a ray exits the explicitly supported XY rectangle."""
        if not(self.xmin<=origin[0]<=self.xmax and self.ymin<=origin[1]<=self.ymax):return 0.
        limit=float('inf')
        for x,v,lo,hi in zip(origin[:2],ray[:2],(self.xmin,self.ymin),(self.xmax,self.ymax)):
            if v>0:limit=min(limit,(hi-x)/v)
            elif v<0:limit=min(limit,(lo-x)/v)
        return float(limit)


def unit_vector(v):
    v=np.asarray(v,float)
    if v.shape!=(3,)or not np.isfinite(v).all()or np.linalg.norm(v)<1e-12:raise ValueError('Finite nonzero three-vector required')
    return v/np.linalg.norm(v)


def solar_direction(latitude_deg,hour_angle_deg,subsolar_latitude_deg=1.5):
    """Same ENU convention and coupled lunar solar geometry as the v37 source."""
    if not np.isfinite([latitude_deg,hour_angle_deg,subsolar_latitude_deg]).all()or not(-90<=latitude_deg<=90)or abs(subsolar_latitude_deg)>1.5+1e-10:
        raise ValueError('Finite lunar latitude/hour angle and solar latitude within +/-1.5deg required')
    p,h,d=map(math.radians,(latitude_deg,hour_angle_deg,subsolar_latitude_deg))
    return np.array([-math.cos(d)*math.sin(h),math.cos(p)*math.sin(d)-math.sin(p)*math.cos(d)*math.cos(h),math.sin(p)*math.sin(d)+math.cos(p)*math.cos(d)*math.cos(h)])


def lighting_schedule(latitude_deg):
    rows=[]
    for i in range(24):
        h=-86.25+7.5*i;s=solar_direction(latitude_deg,h)
        if s[2]<=0:raise ValueError('Declared daylight schedule is below the mathematical horizon')
        rows.append(dict(index=i,hour_angle_deg=h,subsolar_latitude_deg=1.5,sun_enu=s.tolist(),training_allowed=i%2==0))
    return rows


def sun_candidates(latitude_deg):
    result=[]
    for d in np.arange(-1.5,1.50001,.5):
        for h in np.arange(-90.,90.00001,5.):
            s=solar_direction(latitude_deg,float(h),float(d))
            if s[2]>0:result.append(dict(hour_angle_deg=float(h),subsolar_latitude_deg=float(d),sun_enu=s.tolist()))
    return result


def classify_solar_ray(point,normal,sun,envelope,first_hit,epsilon_m=.001,coverage_limit=None):
    """first_hit(origin,unit_ray,max_distance) returns positive distance or None.

    A hit outside supported coverage cannot prove a shadow. No-hit is clear
    only after reaching the trusted terrain ceiling before losing coverage.
    coverage_limit optionally clips support at the first hole along the ray.
    """
    p=np.asarray(point,float);n=unit_vector(normal);s=unit_vector(sun)
    if p.shape!=(3,)or not np.isfinite(p).all()or not np.isfinite(epsilon_m)or epsilon_m<=0:raise ValueError('Finite surface point and positive self-hit offset required')
    if p[2]>envelope.maximum_height+1e-6:raise ValueError('Surface point exceeds declared terrain ceiling')
    if not(envelope.xmin<=p[0]<=envelope.xmax and envelope.ymin<=p[1]<=envelope.ymax):
        return dict(state=int(SolarState.UNKNOWN),reason='source_outside_coverage')
    if np.dot(n,s)<=0:return dict(state=int(SolarState.SHADOW),reason='surface_faces_away_from_sun')
    origin=p+epsilon_m*(n+s);limit=envelope.supported_distance(origin,s)
    if coverage_limit is not None:
        v=float(coverage_limit(origin,s))
        if math.isnan(v)or v<0:raise ValueError('Nonnegative first-coverage-loss distance required')
        limit=min(limit,v)
    if limit<=0:return dict(state=int(SolarState.UNKNOWN),reason='immediate_coverage_loss')
    ceiling=(envelope.maximum_height-origin[2]+epsilon_m)/s[2]if s[2]>0 else float('inf')
    certified=envelope.ceiling_valid and s[2]>0 and ceiling<=limit
    maximum=min(limit,max(ceiling,epsilon_m))if certified else limit
    if not np.isfinite(maximum):
        return dict(state=int(SolarState.UNKNOWN),reason='unbounded_uncertified_coverage')
    hit=first_hit(origin,s,maximum)
    if hit is not None:
        hit=float(hit)
        if not np.isfinite(hit)or hit<=0:raise ValueError('Ray backend must return positive finite hit distance or None')
        if hit<=maximum:return dict(state=int(SolarState.SHADOW),reason='terrain_intersection',first_hit_m=hit)
    if certified:return dict(state=int(SolarState.CLEAR),reason='cleared_supported_terrain_ceiling')
    return dict(state=int(SolarState.UNKNOWN),reason='coverage_lost_before_clearance')


def solar_log_evidence(states,shadow_probability):
    """Log likelihood ratio to uninformative p=.5; unknown contributes zero.

    This is acquisition/inference evidence, not an output acceptance policy.
    """
    states=np.asarray(states)
    if not np.isin(states,[0,1,2]).all():raise ValueError('Explicit unknown/clear/shadow states required')
    p=np.clip(np.asarray(shadow_probability,float),.05,.95)
    if not np.isfinite(p).all():raise ValueError('Finite observed shadow probability required')
    observed=np.where(states==SolarState.SHADOW,p,1-p)
    return np.where(states==SolarState.UNKNOWN,0.,np.log(2*observed))


def soft_shadow_observation(image):
    import cv2
    image=np.asarray(image)
    if image.ndim!=2 or image.dtype!=np.uint8:raise ValueError('Native grayscale uint8 image required')
    threshold,_=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    low,high=np.percentile(image,[5,95]);scale=max(3.,.05*(high-low))
    if high-low<1:return np.full(image.shape,.5),dict(threshold=float(threshold),informative=False)
    z=np.clip((image.astype(float)-threshold)/scale,-40,40)
    return 1/(1+np.exp(z)),dict(threshold=float(threshold),scale=scale,informative=True)
