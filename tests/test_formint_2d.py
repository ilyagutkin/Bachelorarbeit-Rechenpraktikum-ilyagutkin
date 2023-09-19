import pytest
import import_hack
from methodsnm.fe_2d import *
from methodsnm.intrule_2d import *
from methodsnm.formint import *
from methodsnm.mesh import Mesh1D
from methodsnm.meshfct import *
from numpy.linalg import norm

try:
    from methodsnm.solution import *
except ImportError:
    pass

@pytest.mark.parametrize("h", [1/5,1/10,1/20])
def test_source_2d(h):
    triangle = TriangleTransformation(points = array([[0,0],[-h,0],[0,h]]))
    p1 = P1_Triangle_FE()
    c = ConstantFunction(1) 
    s = SourceIntegral(c)
    elvec = s.compute_element_vector(p1, triangle, intrule = EdgeMidPointRule())
    elvec -= array([h**2/6,h**2/6,h**2/6])
    assert norm(elvec) < 1e-14
