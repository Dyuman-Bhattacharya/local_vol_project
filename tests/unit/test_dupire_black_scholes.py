import numpy as np

from implied_volatility.black_scholes import bs_price
from local_volatility.dupire import dupire_local_vol_from_call_grid, DupireConfig


def test_dupire_recovers_constant_black_scholes_vol():
    """
    For a sufficiently wide and dense strike grid, Dupire applied
    to Black–Scholes prices should recover sigma (away from boundaries).
    """
    S0 = 100.0
    r = 0.03
    q = 0.0
    sigma_true = 0.22

    # Wide, dense grid is essential for stability
    K = np.linspace(20.0, 300.0, 801)
    T = np.array([0.25, 0.5, 1.0, 2.0])

    C = np.vstack([
        bs_price(S0, K, t, r, q, sigma_true, "call")
        for t in T
    ])

    cfg = DupireConfig(
        floor_density=1e-10,
        cap_sigma=None,
    )

    sigma_loc = dupire_local_vol_from_call_grid(
        C, K, T,
        r=r,
        q=q,
        cfg=cfg,
    )

    # Evaluate near ATM, away from boundaries
    k_atm_idx = np.argmin(np.abs(K - S0))
    interior_slice = sigma_loc[:, k_atm_idx]

    # Dupire is noisy near short maturities and wings;
    # allow a conservative tolerance
    assert np.all(np.isfinite(interior_slice))
    assert np.allclose(
        interior_slice,
        sigma_true,
        rtol=0.08,
        atol=0.01,
    )
