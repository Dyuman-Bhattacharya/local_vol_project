# tests/unit/test_utils_interpolation.py
import numpy as np
import pytest

from utils.interpolation import Interp1D, Interp2DGrid


def test_interp1d_linear_exact():
    x = np.linspace(0.0, 10.0, 11)
    y = 2.0 * x - 1.0

    interp = Interp1D(x, y, kind="linear")

    xq = np.linspace(0.0, 10.0, 101)
    yq = interp(xq)

    assert np.allclose(yq, 2.0 * xq - 1.0)


def test_interp1d_linear_extrapolation():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    interp = Interp1D(x, y, kind="linear", fill_value="extrapolate")

    xq = np.array([-1.0, 2.0])
    yq = interp(xq)

    assert np.allclose(yq, [-1.0, 2.0])


def test_interp1d_pchip_monotone():
    # Monotone data → pchip preserves monotonicity
    x = np.linspace(0.0, 5.0, 20)
    y = np.log1p(x)

    interp = Interp1D(x, y, kind="pchip")

    xq = np.linspace(0.0, 5.0, 200)
    yq = interp(xq)

    assert np.all(np.diff(yq) >= 0.0)


def test_interp2d_grid_bilinear_exact_plane():
    # z(x,y) = x + 2y → bilinear interpolation exact
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 13)

    X, Y = np.meshgrid(x, y)
    Z = X + 2.0 * Y

    interp = Interp2DGrid(x, y, Z, method="linear")

    xq = np.random.rand(100)
    yq = np.random.rand(100)

    zq = interp(xq, yq)

    assert np.allclose(zq, xq + 2.0 * yq, atol=1e-12)


def test_interp2d_grid_shape_preservation():
    x = np.linspace(0.0, 2.0, 5)
    y = np.linspace(0.0, 3.0, 7)

    Z = np.random.randn(len(y), len(x))
    interp = Interp2DGrid(x, y, Z)

    xq = np.linspace(0.0, 2.0, 10)
    yq = np.linspace(0.0, 3.0, 10)

    Zq = interp(xq, yq)

    assert Zq.shape == (10,)
