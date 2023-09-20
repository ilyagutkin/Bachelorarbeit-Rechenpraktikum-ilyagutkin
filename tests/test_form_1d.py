import pytest
import import_hack
from methodsnm.fe_1d import *
from methodsnm.fes import *
#from methodsnm.intrule_1d import *
from methodsnm.formint import *
from methodsnm.forms import *
from methodsnm.mesh_1d import Mesh1D
from methodsnm.meshfct import *
from numpy.linalg import norm
from scipy import sparse

try:
    from methodsnm.solution import *
except ImportError:
    pass

@pytest.mark.parametrize("ne", [5,10,20])
def test_linearform_1d(ne):
    m = Mesh1D((0,1),ne)
    h = 1/ne    
    p1fes = P1_Segments_Space(m)
    lf = LinearForm(p1fes)
    c = GlobalFunction(lambda x: 1, mesh = m)
    s = SourceIntegral(c)
    lf += s
    lf.assemble()
    print(lf.vector)
    ref = array([0.5*h]+[h for i in range(ne-1)]+[0.5*h])
    lf.vector -= ref
    print(lf.vector)
    assert norm(lf.vector) < 1e-14

@pytest.mark.parametrize("ne", [5,10,20])
def test_bilinearform_mass_1d(ne):
    mesh = Mesh1D((0,1),ne)
    h = 1/ne    
    p1fes = P1_Segments_Space(mesh)
    blf = BilinearForm(p1fes)
    c = GlobalFunction(lambda x: 1, mesh = mesh)
    m = MassIntegral(c)
    blf += m
    blf.assemble()

    ref = sparse.lil_matrix((p1fes.ndof, p1fes.ndof))
    for i in range(1,p1fes.ndof-1):
        ref[i,i-1] = h/6
        ref[i,i] = 2*h/3
        ref[i,i+1] = h/6
    ref[0,0] = h/3
    ref[0,1] = h/6
    ref[p1fes.ndof-1,p1fes.ndof-1] = h/3
    ref[p1fes.ndof-1,p1fes.ndof-2] = h/6

    blf.matrix -= ref
    assert norm(blf.matrix.data) < 1e-14

@pytest.mark.parametrize("ne", [5,10,20])
def test_bilinearform_lap_1d(ne):
    mesh = Mesh1D((0,1),ne)
    h = 1/ne    
    p1fes = P1_Segments_Space(mesh)
    blf = BilinearForm(p1fes)
    c = GlobalFunction(lambda x: 1, mesh = mesh)
    l = LaplaceIntegral(c)
    blf += l
    blf.assemble()

    ref = sparse.lil_matrix((p1fes.ndof, p1fes.ndof))
    for i in range(1,p1fes.ndof-1):
        ref[i,i-1] = -1/h
        ref[i,i] = 2/h
        ref[i,i+1] = -1/h
    ref[0,0] = 1/h
    ref[0,1] = -1/h
    ref[p1fes.ndof-1,p1fes.ndof-1] = 1/h
    ref[p1fes.ndof-1,p1fes.ndof-2] = -1/h

    blf.matrix -= ref
    assert norm(blf.matrix.data) < 1e-10
