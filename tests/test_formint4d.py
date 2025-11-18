import numpy as np
import pytest
from numpy.linalg import norm, det, inv
from methodsnm.intrule import select_integration_rule
from methodsnm.fe_4d import P1_Hypertriangle_FE
from methodsnm.formint import SourceIntegral, MassIntegral, LaplaceIntegral
from methodsnm.meshfct import GlobalFunction
from methodsnm.mesh_4d import UnstructuredHypertriangleMesh

from ngsolve import Mesh, unit_cube


# ------------------------------------------------------------
# Fixture: echtes 4D-Hypertriangle-Mesh
# ------------------------------------------------------------
@pytest.fixture
def mesh():
    T = 2
    ngmesh = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    return UnstructuredHypertriangleMesh(T, ngmesh)


# ------------------------------------------------------------
# Geometrische Hilfsfunktionen
# ------------------------------------------------------------

def simplex_volume(verts):
    """
    Volumen eines 4D-Simplex:
    |K| = det([v1-v0, ..., v4-v0]) / 24
    """
    A = np.array([
        verts[1] - verts[0],
        verts[2] - verts[0],
        verts[3] - verts[0],
        verts[4] - verts[0],
    ]).T
    return abs(det(A)) / 24.0


def reference_gradients(verts):
    """
    Liefert baryzentrische Gradienten ∇λ_i in R^4 für i=0..4
    """
    x0, x1, x2, x3, x4 = verts

    M = np.array([
        x1 - x0,
        x2 - x0,
        x3 - x0,
        x4 - x0,
    ]).T

    Minv = inv(M)

    grads = np.zeros((5, 4))
    grads[0] = -Minv.sum(axis=1)

    for i in range(4):
        grads[i+1] = Minv[:, i]

    return grads


# ------------------------------------------------------------
#                TESTS: FormIntegrals in 4D
# ------------------------------------------------------------

def test_source_4d(mesh):
    """
    Test für SourceIntegral im 4D-P1-Simplex.
    """
    fe = P1_Hypertriangle_FE()
    gf = GlobalFunction(lambda x: 1.0, mesh=mesh)
    S = SourceIntegral(gf)

    el = 0
    verts = mesh.points[mesh.elements()[el]]
    trafo = mesh.trafo(el)

    vol = simplex_volume(verts)
    elem_vec = S.compute_element_vector(fe, trafo)
    expected = np.full(5, vol / 5)

    assert norm(elem_vec - expected) < 1e-10


def test_mass_4d(mesh):
    """
    Test der Mass-Matrix in 4D.
    """
    fe = P1_Hypertriangle_FE()
    gf = GlobalFunction(lambda x: 1.0, mesh=mesh)
    M = MassIntegral(gf)

    el = 0
    verts = mesh.points[mesh.elements()[el]]
    trafo = mesh.trafo(el)

    vol = simplex_volume(verts)
    elem_mat = M.compute_element_matrix(fe, fe, trafo)
    expected = np.full((5, 5), vol / 30)
    np.fill_diagonal(expected, 2 * vol / 30)

    assert norm(elem_mat - expected) < 1e-10


def test_laplace_4d(mesh):
    from methodsnm.fe_4d import P1_Hypertriangle_FE
    from methodsnm.meshfct import GlobalFunction
    from methodsnm.formint import LaplaceIntegral

    fe = P1_Hypertriangle_FE()
    gf = GlobalFunction(lambda x: 1.0, mesh=mesh)
    L = LaplaceIntegral(gf)

    el = 0
    verts_idx = mesh.elements()[el]
    verts = mesh.points[verts_idx]
    trafo = mesh.trafo(el)

    # Matrix aus deiner Implementierung
    elem_mat = L.compute_element_matrix(fe, fe, trafo)

    x0,x1,x2,x3,x4 = verts
    A = np.column_stack([x1-x0, x2-x0, x3-x0, x4-x0])
    AinvT = np.linalg.inv(A)
    grads_ref = np.array([
        [-1, -1, -1, -1],
        [ 1,  0,  0,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
    ])
    grads_phys = grads_ref @ AinvT
    vol = abs(np.linalg.det(A))/24.0

    expected = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            expected[i,j] = vol * grads_phys[i].dot(grads_phys[j])
    
    intr = select_integration_rule(2, fe.eltype)
    F = trafo.jacobian(intr.nodes)[0]

    print("A    =\n", A)
    print("F    =\n", F)
    print("A.T  =\n", A.T)


    from numpy.linalg import norm
    assert norm(elem_mat - expected) < 1e-12
