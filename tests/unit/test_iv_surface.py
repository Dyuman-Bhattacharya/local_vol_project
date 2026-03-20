# tests/unit/test_iv_surface.py
import numpy as np
import pytest

from implied_volatility.black_scholes import bs_price
from implied_volatility.surface import IVSurface, ArbitrageFreeIVSurface
from arbitrage.static_checks import check_static_arbitrage_calls


def test_surface_interpolation_exact_on_grid():
    x = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 1.0])
    iv_grid = np.array([
        [0.2, 0.25, 0.3],
        [0.22, 0.27, 0.32],
    ])

    surf = IVSurface(x_grid=x, t_grid=T, iv_grid=iv_grid)

    for i, Ti in enumerate(T):
        for j, xj in enumerate(x):
            iv = surf.iv(xj, Ti)
            assert abs(iv - iv_grid[i, j]) < 1e-12


def test_surface_total_variance():
    x = np.array([100.0])
    T = np.array([0.5, 1.0])
    iv_grid = np.array([
        [0.2],
        [0.3],
    ])

    surf = IVSurface(x_grid=x, t_grid=T, iv_grid=iv_grid)
    w = surf.total_variance(x=np.array([100.0, 100.0]), T=np.array([0.5, 1.0]))

    expected = np.array([0.2**2 * 0.5, 0.3**2 * 1.0])
    assert np.allclose(w, expected)


def test_surface_monotone_grids_required():
    x_bad = np.array([100.0, 90.0])
    T = np.array([0.5, 1.0])
    iv_grid = np.ones((2, 2))

    with pytest.raises(ValueError):
        IVSurface(x_grid=x_bad, t_grid=T, iv_grid=iv_grid)


def test_surface_dataframe_shape():
    x = np.array([90.0, 100.0])
    T = np.array([0.5, 1.0])
    iv_grid = np.array([
        [0.2, 0.25],
        [0.22, 0.27],
    ])

    surf = IVSurface(x_grid=x, t_grid=T, iv_grid=iv_grid)
    df = surf.to_dataframe()

    assert len(df) == 4
    assert set(df.columns) >= {"T", "X", "iv", "w"}


def test_arbitrage_free_iv_surface_recovers_grid_node_vols():
    S0 = 100.0
    r = 0.01
    q = 0.0
    K = np.array([80.0, 100.0, 120.0])
    T = np.array([0.25, 0.5, 1.0])
    sigma = 0.24
    C = np.vstack([bs_price(S0, K, t, r=r, q=q, sigma=sigma, option_type="call") for t in T])

    surf = ArbitrageFreeIVSurface(K_grid=K, T_grid=T, C_grid=C, S0=S0, r=r, q=q)
    iv = surf.iv(K[None, :], T[:, None])

    assert np.allclose(iv, sigma, atol=1e-8)


def test_arbitrage_free_iv_surface_preserves_static_arbitrage_on_dense_grid():
    S0 = 100.0
    r = 0.01
    q = 0.0
    K = np.linspace(70.0, 130.0, 13)
    T = np.array([0.25, 0.5, 1.0])
    sigma = 0.22
    C = np.vstack([bs_price(S0, K, t, r=r, q=q, sigma=sigma, option_type="call") for t in T])

    surf = ArbitrageFreeIVSurface(K_grid=K, T_grid=T, C_grid=C, S0=S0, r=r, q=q)
    K_dense = np.linspace(K[0], K[-1], 41)
    T_dense = np.linspace(T[0], T[-1], 21)
    C_dense = surf.call_price(K_dense[None, :], T_dense[:, None])

    rep = check_static_arbitrage_calls(C_dense, K_dense, T_dense, S0=S0, r=r, q=q, tol=1e-8)
    assert rep.counts["n_fail"] == 0
