import numpy as np

from local_volatility.surface import LocalVolSurface
from pricing.monte_carlo import price_european_mc_local_vol, MCConfig
from implied_volatility.black_scholes import bs_price


def test_mc_matches_black_scholes_constant_vol():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2

    # Constant local vol surface
    K_grid = np.linspace(50.0, 150.0, 51)
    T_grid = np.linspace(0.0, T, 11)
    sigma_grid = np.full((T_grid.size, K_grid.size), sigma)

    lv = LocalVolSurface.from_dupire_grid(sigma_grid, K_grid, T_grid)

    cfg = MCConfig(
        n_paths=150_000,
        n_steps=252,
        seed=123,
        variance_reduction=("antithetic",),
    )

    price_mc, stderr = price_european_mc_local_vol(
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

    # MC error should be within ~3 sigma
    assert abs(price_mc - p_bs) < 3.0 * stderr
