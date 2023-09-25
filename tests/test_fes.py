import pytest
import import_hack
from methodsnm.fes import *
from numpy.linalg import norm
from methodsnm.mesh_1d import *
from methodsnm.mesh_2d import *

@pytest.mark.parametrize("ne", [1,2,10])
def test_P1_FES_1d(ne):
    mesh = Mesh1D((0,1),ne)
    p1fes = P1_Segments_Space(mesh)
    dofs = p1fes.element_dofs(0, bndry=True)
    assert dofs == [0]
    nbe = len(mesh.elements(bndry=True))
    dofs = p1fes.element_dofs(nbe-1, bndry=True)
    assert dofs == [ne]

def test_P1_FES_2d():
    mesh = StructuredRectangleMesh(2, 2)
    p1fes = P1_Triangle_Space(mesh)
    dof_marked = np.zeros(9, dtype=int)
    for belnr, verts in enumerate(mesh.elements(bndry=True)):
        dofs = p1fes.element_dofs(belnr, bndry=True)
        assert len(dofs) == 2
        dof_marked[dofs] += 1
    assert all(d == 2 for i,d in enumerate(dof_marked) if i != 4)

def test_P3_FES_2d():
    mesh = StructuredRectangleMesh(2, 2)
    p3fes = P3_Triangle_Space(mesh)
    dof_marked = np.zeros(p3fes.ndof, dtype=int)
    for belnr, verts in enumerate(mesh.elements(bndry=True)):
        dofs = p3fes.element_dofs(belnr, bndry=True)
        assert len(dofs) == 4
        dof_marked[dofs] += 1
    dof_marked_ref = np.array([2,2,2,2,0,2,2,2,2,
                               1,1,1,1,0,0,0,0,1,1,1,1,
                               1,1,1,1,0,0,0,0,1,1,1,1,
                               0,0,0,0,0,0,0,0])
    assert (dof_marked_ref == dof_marked_ref).all()