import pytest
import import_hack
from methodsnm.fes import *
from numpy.linalg import norm
from methodsnm.mesh_1d import *
from methodsnm.mesh_2d import *
from methodsnm.mesh_4d import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf

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

@pytest.fixture
def mesh():
    T = 2
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    mesh = UnstructuredHypertriangleMesh(T, ngmesh)
    return mesh


def test_P1_Hypertriangle_dofs_per_element(mesh):
    """
    For each 4D simplex, the P1 space must assign exactly
    the 5 vertex DOFs = global vertex numbers.
    """
    fes = P1_Hypertriangle_Space(mesh)

    for elnr, verts in enumerate(mesh.elements()):
        dofs = fes.element_dofs(elnr)
        assert len(dofs) == 5
        assert np.allclose(dofs, verts)

def test_fespace_boundary_sets_consistent(mesh):
    """
    Testet, dass sich der Rand korrekt in three disjoint pieces zerlegt:
    boundary = initial ∪ top ∪ side
    """
    fes = P1_Hypertriangle_Space(mesh)

    I = fes.initial_vertices()
    T = fes.top_vertices()
    B = fes.boundary_vertices()

    S = B - I - T

    assert I.isdisjoint(T)
    assert I.isdisjoint(S)
    assert T.isdisjoint(S)
    assert I | T | S == B

def test_P2_Hypertriangle_element_dofs_structure(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    for elnr, verts in enumerate(mesh.elements()):
        dofs = fes.element_dofs(elnr)

        # Länge muss 15 sein
        assert len(dofs) == 15

        # Erste 5 = Vertices
        assert np.allclose(dofs[:5], verts)

        # Letzte 10 = edge DOFs → müssen >= nv sein
        edge_dofs = dofs[5:]
        assert np.all(edge_dofs >= fes.nv)

        # Keine Duplikate
        assert len(set(dofs)) == 15

def test_P2_Hypertriangle_edge_connectivity(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    # Alle globalen Edges sollten in mesh.edges liegen
    global_edges = set(tuple(sorted(e)) for e in mesh.edges)

    for elnr, verts in enumerate(mesh.elements()):
        el_edges = mesh.hypercells2edges[elnr]

        # Jede lokale Edge muss existieren
        for gid in el_edges:
            v1, v2 = mesh.edges[gid]
            assert tuple(sorted((v1, v2))) in global_edges

def test_P2_Hypertriangle_boundary_dofs(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    Bv = fes.boundary_vertices()
    bdofs = fes.boundary_dofs()

    # Vertex DOFs müssen enthalten sein
    for v in Bv:
        assert v in bdofs

    # Edge DOFs müssen enthalten sein, wenn beide Vertices boundary sind
    for eid, (v1, v2) in enumerate(mesh.edges):
        if v1 in Bv and v2 in Bv:
            assert fes.nv + eid in bdofs
        else:
            assert (fes.nv + eid) not in bdofs

def test_P2_Hypertriangle_boundary_sets_consistent(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    I = fes.initial_vertices()
    T = fes.top_vertices()
    B = fes.boundary_vertices()

    S = B - I - T

    assert I.isdisjoint(T)
    assert I.isdisjoint(S)
    assert T.isdisjoint(S)

    assert I | T | S == B


def test_P2_Hypertriangle_global_dofs_complete(mesh):
    fes = P2_Hypertriangle_Space(mesh)

    # Anzahl DOFs muss stimmen = nv + ne
    assert fes.ndof == fes.nv + fes.ne

    # Alle DOFs liegen im Bereich
    assert set(range(fes.ndof)) == \
           set(range(fes.nv)) | set(range(fes.nv, fes.nv + fes.ne))
