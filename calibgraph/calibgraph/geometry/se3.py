"""SE(3) utilities.

Convention: T_A_B maps coordinates from frame B into frame A.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation
FloatArray = NDArray[np.float64]

def _arr(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)

def validate_transform(transform: ArrayLike, *, atol: float = 1e-8, raise_on_error: bool = True) -> bool:
    T = _arr(transform)
    reasons=[]
    if T.shape != (4,4):
        reasons.append(f"expected (4,4), got {T.shape}")
    else:
        R=T[:3,:3]
        if not np.allclose(T[3], [0,0,0,1], atol=atol): reasons.append("bad last row")
        if not np.allclose(R.T@R, np.eye(3), atol=atol): reasons.append("rotation not orthonormal")
        if not np.isclose(np.linalg.det(R),1.0,atol=atol): reasons.append("rotation determinant not +1")
    if reasons and raise_on_error: raise ValueError("invalid SE(3): "+"; ".join(reasons))
    return not reasons

def make_transform(rotation: ArrayLike, translation: ArrayLike) -> FloatArray:
    R=_arr(rotation); t=_arr(translation).reshape(-1)
    if R.shape != (3,3): raise ValueError(f"rotation shape {R.shape}")
    if t.shape != (3,): raise ValueError(f"translation shape {t.shape}")
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=t; validate_transform(T); return T

def inverse(transform: ArrayLike) -> FloatArray:
    T=_arr(transform); validate_transform(T); R=T[:3,:3]; t=T[:3,3]
    out=np.eye(4); out[:3,:3]=R.T; out[:3,3]=-(R.T@t); return out

def compose(*transforms: ArrayLike) -> FloatArray:
    out=np.eye(4)
    for transform in transforms:
        T=_arr(transform); validate_transform(T); out=out@T
    validate_transform(out, atol=1e-7); return out

def transform_points(transform: ArrayLike, points: ArrayLike) -> FloatArray:
    T=_arr(transform); validate_transform(T); pts=_arr(points); single=pts.ndim==1
    if single:
        if pts.shape != (3,): raise ValueError(f"point shape {pts.shape}")
        pts=pts[None,:]
    elif pts.ndim != 2 or pts.shape[1] != 3: raise ValueError(f"points shape {pts.shape}")
    h=np.c_[pts, np.ones(len(pts))]
    ans=(T@h.T).T[:,:3]
    return ans[0] if single else ans

def random_transform(rng: np.random.Generator, *, max_rotation_deg: float=180.0, max_translation: float=1.0) -> FloatArray:
    axis=rng.normal(size=3); n=np.linalg.norm(axis); axis=np.array([1.,0.,0.]) if n<1e-12 else axis/n
    angle=np.deg2rad(rng.uniform(-max_rotation_deg,max_rotation_deg))
    R=Rotation.from_rotvec(axis*angle).as_matrix(); t=rng.uniform(-max_translation,max_translation,size=3)
    return make_transform(R,t)

def rotation_error_deg(estimate: ArrayLike, reference: ArrayLike) -> float:
    E=_arr(estimate); G=_arr(reference); validate_transform(E); validate_transform(G)
    d=E[:3,:3]@G[:3,:3].T
    return float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(d).as_rotvec())))

def translation_error(estimate: ArrayLike, reference: ArrayLike) -> float:
    E=_arr(estimate); G=_arr(reference); validate_transform(E); validate_transform(G)
    return float(np.linalg.norm(E[:3,3]-G[:3,3]))
