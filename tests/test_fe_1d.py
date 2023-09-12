import pytest
import import_hack
from methodsnm.fe_1d import *
from numpy.linalg import norm

try:
    from methodsnm.solution import *
except ImportError:
    pass

@pytest.mark.parametrize("sample", [[0.25, [0.75,0.25]],
                                    [0.75, [0.25,0.75]]])
def test_P1(sample):
    xval, yvals = sample
    p1 = P1_Segment_FE()
    x = np.array([xval])
    yvals = np.array(yvals)
    yvals -= p1.evaluate(x)
    assert norm(yvals) < 1e-14

@pytest.mark.parametrize("sample", [[0.00, [1.0,0.0,0.0]],
                                    [0.50, [0.0,1.0,0.0]],
                                    [0.75, [-1.0/8,3.0/4,3.0/8]],
                                    [1.00, [0.0,0.0,1.0]],])
def test_LagrangeP2(sample):
    xval, yvals = sample
    p1 = Lagrange_Segment_FE(2)
    x = np.array([xval])
    yvals = np.array(yvals)
    yvals -= p1.evaluate(x)
    assert norm(yvals) < 1e-14
