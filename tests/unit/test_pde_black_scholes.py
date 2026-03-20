import numpy as np

from local_volatility.surface import LocalVolSurface
from pricing.pde_solver import price_european_pde_local_vol, PDEConfig
from implied_volatility.black_scholes import bs_price


def test_pde_matches_black_scholes_constant_vol():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2

    # Constant local vol surface
    K_grid = np.linspace(20.0, 300.0, 401)
    T_grid = np.linspace(0.0, T, 61)
    sigma_grid = np.full((T_grid.size, K_grid.size), sigma)

    lv = LocalVolSurface.from_dupire_grid(sigma_grid, K_grid, T_grid)

    cfg = PDEConfig(
        n_space=500,
        n_time=250,
        theta=0.5,
        S_max_mult=5.0,
    )

    p_pde = price_european_pde_local_vol(
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call",
        lv_surface=lv,
        cfg=cfg,
    )

    p_bs = float(bs_price(S0, K, T, r, q, sigma, "call"))

    # PDE discretization error is O(dx^2 + dt^2)
    assert abs(p_pde - p_bs) / p_bs < 5e-3
