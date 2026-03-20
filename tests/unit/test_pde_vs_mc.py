import numpy as np

from local_volatility.surface import LocalVolSurface
from pricing.pde_solver import price_european_pde_local_vol, PDEConfig
from pricing.monte_carlo import price_european_mc_local_vol, MCConfig


def test_pde_and_mc_agree_constant_local_vol():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    q = 0.0
    sigma = 0.25

    K_grid = np.linspace(30.0, 250.0, 401)
    T_grid = np.linspace(0.0, T, 41)
    sigma_grid = np.full((T_grid.size, K_grid.size), sigma)

    lv = LocalVolSurface.from_dupire_grid(sigma_grid, K_grid, T_grid)

    p_pde = price_european_pde_local_vol(
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call",
        lv_surface=lv,
        cfg=PDEConfig(n_space=500, n_time=250),
    )

    price_mc, stderr = price_european_mc_local_vol(
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call",
        lv_surface=lv,
        cfg=MCConfig(n_paths=120_000, n_steps=252),
    )

    # PDE should lie within MC confidence band
    assert abs(p_pde - price_mc) < 3.0 * stderr
