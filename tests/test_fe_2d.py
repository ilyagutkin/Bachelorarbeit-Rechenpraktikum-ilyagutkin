import pytest
import import_hack
from methodsnm.fe_2d import *
from numpy.linalg import norm

try:
    from methodsnm.solution import *
except ImportError:
    pass


@pytest.mark.parametrize(
    "sample",
    [
        [[0, 0], [1, 0, 0, 0, 0, 0]],
        [[1, 0], [0, 1, 0, 0, 0, 0]],
        [[0, 1], [0, 0, 1, 0, 0, 0]],
        [[0.5, 0.5], [0, 0, 0, 1, 0, 0]],
        [[0, 0.5], [0, 0, 0, 0, 1, 0]],
        [[0.5, 0], [0, 0, 0, 0, 0, 1]],
    ],
)
def test_LagrangeP2(sample):
    xval, yvals = sample
    p1 = P2_Triangle_FE()
    x = np.array([xval])
    yvals = np.array(yvals, dtype="float64").reshape((1, 6))
    yvals -= p1.evaluate(x)
    assert norm(yvals) < 1e-14


@pytest.mark.parametrize(
    "sample",
    [
        [[0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
        [[2 / 3, 1 / 3], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        [[1 / 3, 2 / 3], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        [[0, 1 / 3], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
        [[0, 2 / 3], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
        [[1 / 3, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
        [[2 / 3, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
        [[1 / 3, 1 / 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    ],
)
def test_LagrangeP3(sample):
    xval, yvals = sample
    p1 = P3_Triangle_FE()
    x = np.array([xval])
    yvals = np.array(yvals, dtype="float64").reshape((1, 10))
    yvals -= p1.evaluate(x)
    assert norm(yvals) < 1e-14


@pytest.mark.parametrize(
    "sample",
    [
        [[[0, 1], [1, 0]], [1, 0, 0]],
        [[[0, 0], [0, 1]], [0, 1, 0]],
        [[[0, 0], [1, 0]], [0, 0, 1]],
    ],
)
def test_P1Edge(sample):
    xval, yvals = sample
    p1 = P1Edge_Triangle_FE()
    x = np.array(xval, dtype="float64")
    yvals = np.array(yvals, dtype="float64")
    calc_y = p1.evaluate(x)
    integral = 0.5 * (calc_y[0, :] + calc_y[1, :])
    yvals -= integral
    assert norm(yvals) < 1e-14
