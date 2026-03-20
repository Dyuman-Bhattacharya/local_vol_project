import numpy as np

from implied_volatility.black_scholes import bs_price
from local_volatility.surface import LocalVolSurface
from pricing.pde_solver import PDEConfig, solve_european_pde_local_vol_surface


def test_surface_solver_returns_calendar_time_grid_and_terminal_payoff():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2

    S_nodes = np.linspace(20.0, 300.0, 301)
    T_nodes = np.linspace(0.0, T, 41)
    sigma_grid = np.full((T_nodes.size, S_nodes.size), sigma)
    lv = LocalVolSurface.from_dupire_grid(sigma_grid, S_nodes, T_nodes)

    t_grid, S_grid, V_grid = solve_european_pde_local_vol_surface(
        S_ref=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call",
        lv_surface=lv,
        cfg=PDEConfig(n_space=300, n_time=140),
        S_min=20.0,
        S_max=300.0,
    )

    assert np.all(np.diff(t_grid) > 0.0)
    assert np.all(np.diff(S_grid) > 0.0)
    assert V_grid.shape == (t_grid.size, S_grid.size)
    assert np.allclose(V_grid[-1, :], np.maximum(S_grid - K, 0.0), atol=1e-10)

    p_surface = float(np.interp(S0, S_grid, V_grid[0, :]))
    p_bs = float(bs_price(S0, K, T, r, q, sigma, "call"))
    assert abs(p_surface - p_bs) / p_bs < 0.02
