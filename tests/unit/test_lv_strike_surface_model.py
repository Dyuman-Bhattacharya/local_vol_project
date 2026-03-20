import numpy as np

from hedging.delta_hedger import LocalVolPDEDeltaModelStrikeSurface
from implied_volatility.black_scholes import bs_delta, bs_price
from local_volatility.surface import LocalVolSurface
from pricing.pde_solver import PDEConfig, solve_european_pde_local_vol_surface


def _build_constant_surface(*, sigma: float, T: float) -> LocalVolSurface:
    S_grid = np.linspace(20.0, 300.0, 301)
    T_grid = np.linspace(0.0, T, 41)
    sigma_grid = np.full((T_grid.size, S_grid.size), sigma)
    return LocalVolSurface.from_dupire_grid(sigma_grid, S_grid, T_grid)


def test_strike_surface_model_prices_and_deltas_two_strikes():
    S0 = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2
    K_nodes = np.array([90.0, 110.0])

    lv = _build_constant_surface(sigma=sigma, T=T)
    cfg = PDEConfig(n_space=280, n_time=140)

    t_grid = None
    S_grid = None
    V_grid = None
    D_grid = None

    for idx, K in enumerate(K_nodes):
        t_surface, S_surface, V_surface = solve_european_pde_local_vol_surface(
            S_ref=S0,
            K=K,
            T=T,
            r=r,
            q=q,
            option_type="call",
            lv_surface=lv,
            cfg=cfg,
            S_min=20.0,
            S_max=300.0,
        )
        D_surface = np.gradient(V_surface, S_surface, axis=1, edge_order=2)

        if t_grid is None:
            t_grid = t_surface
            S_grid = S_surface
            V_grid = np.empty((t_grid.size, K_nodes.size, S_grid.size), dtype=float)
            D_grid = np.empty_like(V_grid)

        V_grid[:, idx, :] = V_surface
        D_grid[:, idx, :] = D_surface

    for K in K_nodes:
        model = LocalVolPDEDeltaModelStrikeSurface(
            K=K,
            T=T,
            r=r,
            q=q,
            option_type="call",
            K_grid=K_nodes,
            S_grid=S_grid,
            t_grid=t_grid,
            V_grid=V_grid,
            D_grid=D_grid,
        )

        p_model = model.price(S0, 0.0)
        p_bs = float(bs_price(S0, K, T, r, q, sigma, "call"))
        assert abs(p_model - p_bs) / p_bs < 0.02

        d_model = model.delta(S0, 0.0)
        d_bs = float(bs_delta(np.array([S0]), np.array([K]), np.array([T]), r, q, np.array([sigma]), "call")[0])
        assert abs(d_model - d_bs) < 0.05
