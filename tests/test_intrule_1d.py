import pytest
import import_hack
from methodsnm.intrule_1d import *
from numpy.linalg import norm

try:
    from methodsnm.solution import *
except ImportError:
    pass

@pytest.mark.parametrize("K", [1,2,3,4,5,6,7,8,9,10])
def test_newtoncotes(K):
    nc = NewtonCotesRule(n=K)
    for k in range(K):
        assert np.isclose(nc.integrate(lambda x: x**k),1/(k+1))
        assert nc.exactness_degree >= k

@pytest.mark.parametrize("K", [1,2,3,4,5,6,7,8,9,10])
def test_gausslegendre(K):
    gl = GaussLegendreRule(n=K)
    for k in range(2*K):
        assert np.isclose(gl.integrate(lambda x: x**k),1/(k+1))
        assert gl.exactness_degree >= k

@pytest.mark.parametrize("K", [1,2,3,4,5,6,7,8,9,10])
def test_gausslegendre_vs_numpy(K):
    glnp = NP_GaussLegendreRule(n=K)
    gl = GaussLegendreRule(n=K)
    for k in range(2*K):
        assert np.isclose(gl.integrate(lambda x: x**k),glnp.integrate(lambda x: x**k))
    assert gl.exactness_degree == glnp.exactness_degree
    assert np.isclose(gl.integrate(lambda x: sin(x)),glnp.integrate(lambda x: sin(x)))

@pytest.mark.parametrize("K", [1,2,3,4,5,6,7,8,9,10])
@pytest.mark.parametrize("alpha", [0])
@pytest.mark.parametrize("beta", [0])
#TODO: expand the test to non-zero alpha and beta
def test_gaussjacobi(K,alpha,beta):
    gl = GaussJacobiRule(n=K,alpha=alpha,beta=beta)
    for k in range(2*K):
        assert np.isclose(gl.integrate(lambda x: x**k,1/(k+1)))
