import numpy as np
import pytest
from methodsnm.mesh_4d import UnstructuredHypertriangleMesh, StructuredTesseraktMesh
from methodsnm.mesh_2d import StructuredRectangleMesh
from methodsnm.fe_4d import P1_Hypertriangle_FE, P2_Hypertriangle_FE
from methodsnm.fes import *
from methodsnm.meshfct import *
from netgen.csg import unit_cube
from ngsolve import Mesh,VOL,specialcf
from methodsnm.forms import compute_difference_L2


# ----------------------------------------------------------
#  Basic Mesh Tests
# ----------------------------------------------------------
ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))

def test_vertices_unique_and_sorted():
    mesh = UnstructuredHypertriangleMesh(T=2,ngmesh=ngmesh)
    verts = mesh.vertices
    assert len(verts) == len(np.unique(verts))
    assert np.all(np.diff(verts) >= 0)


def test_points_match_vertices():
    mesh = UnstructuredHypertriangleMesh(T=1,ngmesh=ngmesh)
    assert mesh.points.shape[0] == mesh.vertices.shape[0]


def test_find_element_contains_vertex():
    mesh = UnstructuredHypertriangleMesh(T=1,ngmesh=ngmesh)
    for i in range(5):
        ip = mesh.points[mesh.hypercells[0, i]]
        el = mesh.find_element(ip)
        assert el == 0


def test_jacobian_invertible():
    mesh = UnstructuredHypertriangleMesh(T=1,ngmesh=ngmesh)
    trafo = mesh.trafo(0)
    F = trafo.jacobian(np.zeros((1,4)))[0]
    assert np.linalg.det(F) != 0


# ----------------------------------------------------------
#  FEFunction: consistency tests
# ----------------------------------------------------------

def test_p1_set_global_constant():
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)
    fe = P1_Hypertriangle_FE()
    class DummyFES:
        def __init__(self, mesh):
            self.mesh = mesh
            self.ndof = mesh.points.shape[0]
            self.fe = fe
        def element_dofs(self, el):
            return mesh.hypercells[el]
        def finite_element(self, el):
            return fe

    fes = DummyFES(mesh)
    u = FEFunction(fes)
    u._set(lambda x: 3.14)

    assert np.allclose(u.vector, 3.14)


def test_p1_set_boundary_only_changes_boundary():
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)
    fe = P1_Hypertriangle_FE()

    class DummyFES:
        def __init__(self, mesh):
            self.mesh = mesh
            self.ndof = mesh.points.shape[0]
            self.fe = fe
        def element_dofs(self, el):
            return mesh.hypercells[el]
        def finite_element(self, el):
            return fe

    fes = DummyFES(mesh)
    u = FEFunction(fes)
    u._set(lambda x: 7.0)

    # overwrite boundary
    u._set(lambda x: 99.0, boundary=True)

    for i in range(fes.ndof):
        if i in mesh.bndry_vertices:
            assert u.vector[i] == 99.0
        else:
            assert u.vector[i] == 7.0


def test_p1_evaluate_matches_exact_linear_function():
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)

    fe = P1_Hypertriangle_FE()

    class DummyFES:
        def __init__(self, mesh):
            self.mesh = mesh
            self.ndof = mesh.points.shape[0]
            self.fe = fe
        def element_dofs(self, el):
            return mesh.hypercells[el]
        def finite_element(self, el):
            return fe

    fes = DummyFES(mesh)
    u = FEFunction(fes)

    # Linear exact solution
    f_exact = lambda x: 1 + 2*x[0] - x[1] + 0.5*x[2] + 3*x[3]

    u._set(f_exact)

    # check random points in each element
    for el in range(len(mesh.hypercells)):
        pts = mesh.points[mesh.hypercells[el]]
        # draw random point inside simplex
        lamb = np.random.rand(5)
        lamb /= np.sum(lamb)
        ip = np.dot(lamb, pts)

        val = u._evaluate(ip)
        val_exact = f_exact(ip)
        assert abs(val - val_exact) < 1e-12

# ----------------------------------------------------------
#  P2 tests
# ----------------------------------------------------------

def test_p2_midpoint_setting():
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)
    p2 = P2_Hypertriangle_FE()
    class DummyFES:
        def __init__(self, mesh):
            self.mesh = mesh
            self.fe = p2
            self.ndof = mesh.points.shape[0] + len(mesh.edges)
        def element_dofs(self, el):
            return mesh.p2_dofs[el]
        def finite_element(self, el):
            return p2

    fes = DummyFES(mesh)
    u = FEFunction(fes)

    f = lambda x: np.sum(x)
    u._set_P2(f)

    # check vertex DOFs
    for v in range(len(mesh.points)):
        assert abs(u.vector[v] - f(mesh.points[v])) < 1e-12

    # check edge midpoint DOFs
    offset = len(mesh.points)
    for e, (v1, v2) in enumerate(mesh.edges):
        midpoint = 0.5*(mesh.points[v1] + mesh.points[v2])
        assert abs(u.vector[offset+e] - f(midpoint)) < 1e-12


# ----------------------------------------------------------
#  GLOBAL PATCH TEST
# ----------------------------------------------------------

def test_global_patch_linear():
    """
    Patch-test: eine globale lineare Funktion MUSS exakt reproduziert werden.
    """
    mesh = UnstructuredHypertriangleMesh(T=1, ngmesh=ngmesh)
    fe = P1_Hypertriangle_FE()

    class DummyFES:
        def __init__(self, mesh):
            self.mesh = mesh
            self.ndof = mesh.points.shape[0]
            self.fe = fe
        def element_dofs(self, el):
            return mesh.hypercells[el]
        def finite_element(self, el):
            return fe

    fes = DummyFES(mesh)
    u = FEFunction(fes)

    f = lambda x: x[0] + 2*x[1] - x[2] + 0.25*x[3]
    u._set(f)

    # check all centroids
    for el in range(len(mesh.hypercells)):
        pts = mesh.points[mesh.hypercells[el]]
        centroid = np.mean(pts, axis=0)
        assert abs(u._evaluate(centroid) - f(centroid)) < 1e-12

mesh1 = UnstructuredHypertriangleMesh(T=2,ngmesh=ngmesh)
mesh2 = StructuredTesseraktMesh(2,2,2,2)
mesh3 = StructuredRectangleMesh(5,5)
fes = []
fes.append(P1_Hypertriangle_Space(mesh1))
fes.append(P1_Tesserakt_Space(mesh2)) 
fes.append(P1_Triangle_Space(mesh3))

@pytest.mark.parametrize("fes", fes)
def test_constant_vs_FE(fes):
    """
    Checks that a FEFunction filled with a constant value matches a
    ConstantFunction exactly in the L2 sense.
    """
    mesh = fes.mesh
    c = 2.0

    u_ex = ConstantFunction(c, mesh=mesh)

    u_h = FEFunction(fes)
    u_h._set(lambda x: c +x[0]*0 +x[1]*0)

    for _ in range(5):
            el = np.random.randint(0, mesh.ne)
            trafo = mesh.trafo(el)

            # random reference point (barycentric but normalized)
            ref = np.random.rand(mesh.dimension)
            # sum(ref) <= 1 erzwingen für Simplexe
            s = np.sum(ref)
            if s > 1:
                ref = ref / (s + 1e-14)
            x = trafo(ref)   # global point

            uh_val = u_h.evaluate(x, trafo)
            uex_val = u_ex.evaluate(x, trafo)

            assert abs(uh_val - uex_val) < 1e-12

@pytest.mark.parametrize("fes", fes)
def test_global_vs_FE(fes):
    """
    Ensures that a FEFunction initialized from a smooth GlobalFunction
    reproduces the same field with negligible L2 error.
    """
    mesh = fes.mesh

    u_exact =lambda x: np.sin(x[0]) + x[1]**2

    u_ex = GlobalFunction(u_exact, mesh=mesh)

    u_h = FEFunction(fes)
    u_h._set(u_exact)

    err = compute_difference_L2(u_h, u_ex, mesh, intorder=3)
    assert err < 1e-1

@pytest.mark.parametrize("fes", fes)
def test_constant_vs_global(fes):
    """
    Confirms that ConstantFunction and an equivalent GlobalFunction
    (returning the same constant) are numerically identical in L2.
    """
    mesh = fes.mesh
    c = 3.5

    u1 = ConstantFunction(c, mesh=mesh)
    u2 = GlobalFunction(lambda x: c, mesh=mesh)

    # FE needed only to get mesh for error integration
    dummy = FEFunction(fes)

    err = compute_difference_L2(u1, u2, mesh, intorder=3)
    assert err < 1e-12