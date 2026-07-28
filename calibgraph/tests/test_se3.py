import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from calibgraph.geometry.se3 import compose,inverse,make_transform,random_transform,rotation_error_deg,transform_points,translation_error,validate_transform

def test_identity_is_valid(): assert validate_transform(np.eye(4))

def test_inverse_round_trip():
    T=random_transform(np.random.default_rng(7))
    np.testing.assert_allclose(compose(T,inverse(T)),np.eye(4),atol=1e-10)

def test_transform_chaining():
    T_A_B=make_transform(Rotation.from_euler("z",90,degrees=True).as_matrix(),[1,0,0])
    T_B_C=make_transform(np.eye(3),[1,0,0])
    np.testing.assert_allclose(transform_points(compose(T_A_B,T_B_C),[0,0,0]),[1,1,0],atol=1e-10)

def test_point_round_trip():
    rng=np.random.default_rng(11); T=random_transform(rng); pts=rng.normal(size=(20,3))
    np.testing.assert_allclose(transform_points(inverse(T),transform_points(T,pts)),pts,atol=1e-10)

def test_error_metrics():
    E=make_transform(Rotation.from_euler("x",5,degrees=True).as_matrix(),[.003,.004,0])
    assert rotation_error_deg(E,np.eye(4)) == pytest.approx(5.0)
    assert translation_error(E,np.eye(4)) == pytest.approx(.005)

def test_invalid_transform_rejected():
    T=np.eye(4); T[0,0]=2
    with pytest.raises(ValueError): validate_transform(T)
