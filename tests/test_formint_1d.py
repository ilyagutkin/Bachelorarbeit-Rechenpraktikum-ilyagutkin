import pytest
import import_hack
from methodsnm.fe_1d import *
from methodsnm.formint import *
from methodsnm.mesh import Mesh1D
from methodsnm.meshfct import *
from numpy.linalg import norm

try:
    from methodsnm.solution import *
except ImportError:
    pass

@pytest.mark.parametrize("h", [1/5,1/10,1/20])
def test_source_1d(h):
    m = Mesh1D((0,h),1)
    p1 = P1_Segment_FE()
    c = GlobalFunction(lambda x: 1, mesh = m)
    s = SourceIntegral(c)
    elvec = s.compute_element_vector(p1, m.trafo(0))
    elvec -= array([0.5*h,0.5*h])
    assert norm(elvec) < 1e-14

@pytest.mark.parametrize("h", [1/5,1/10,1/20])
def test_mass_1d(h):
    mesh = Mesh1D((0,h),1)
    p1 = P1_Segment_FE()
    c = GlobalFunction(lambda x: 1, mesh = mesh)
    m = MassIntegral(c)
    elmat = m.compute_element_matrix(p1, p1, mesh.trafo(0))
    elmat -= array([[h/3,h/6],[h/6,h/3]])
    assert norm(elmat) < 1e-14

@pytest.mark.parametrize("h", [1/5,1/10,1/20])
def test_lap_1d(h):
    mesh = Mesh1D((0,h),1)
    p1 = P1_Segment_FE()
    c = GlobalFunction(lambda x: 1, mesh = mesh)
    m = LaplaceIntegral(c)
    elmat = m.compute_element_matrix(p1, p1, mesh.trafo(0))
    print(elmat)
    elmat -= array([[1/h,-1/h],[-1/h,1/h]])
    assert norm(elmat) < 1e-10
