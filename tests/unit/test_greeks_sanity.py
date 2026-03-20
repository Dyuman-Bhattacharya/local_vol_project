import numpy as np

from local_volatility.surface import LocalVolSurface
from pricing.greeks import delta_gamma_local_vol_pde
from pricing.pde_solver import PDEConfig


def test_delta_gamma_signs_and_magnitude():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma = 0.2

    K_grid = np.linspace(40.0, 200.0, 301)
    T_grid = np.linspace(0.0, T, 31)
    sigma_grid = np.full((T_grid.size, K_grid.size), sigma)

    lv = LocalVolSurface.from_dupire_grid(sigma_grid, K_grid, T_grid)

    delta, gamma = delta_gamma_local_vol_pde(
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type="call",
        lv_surface=lv,
        cfg=None,
    )

    # Call option sanity
    assert 0.0 < delta < 1.0
    assert gamma > 0.0

    # Order-of-magnitude sanity
    assert gamma < 0.1
