import numpy as np

from implied_volatility.black_scholes import bs_price
from surface_fitting.spline import fit_spline_surface
from implied_volatility.surface import call_price_grid_from_iv_surface
from arbitrage.projection import project_call_surface_static_arbitrage
from arbitrage.static_checks import check_static_arbitrage_calls


def test_interpolated_total_variance_surface_projects_to_arbitrage_free_call_grid():
    S0 = 100.0
    r = 0.02
    q = 0.0
    sigma = 0.25

    # Grid
    K = np.linspace(60, 140, 41)
    T = np.array([0.25, 0.5, 1.0])

    # Generate BS prices → implied total variance slices
    x_slices = []
    w_slices = []

    for t in T:
        C = bs_price(S0, K, t, r, q, sigma, "call")
        iv = np.full_like(K, sigma)
        x = np.log(K / (S0 * np.exp((r - q) * t)))
        w = iv * iv * t
        x_slices.append(x)
        w_slices.append(w)

    variance_surface = fit_spline_surface(x_slices, w_slices, T, kind_x="pchip", kind_T="pchip", method_2d="linear")

    C_raw = call_price_grid_from_iv_surface(
        variance_surface,
        S0=S0,
        K_grid=K,
        T_grid=T,
        r=r,
        q=q,
        coord="x_T",
    )
    C_fit = project_call_surface_static_arbitrage(
        C_raw,
        K,
        T,
        S0=S0,
        r=r,
        q=q,
        tol=1e-8,
    ).C_projected

    rep = check_static_arbitrage_calls(
        C_fit,
        K,
        T,
        S0=S0,
        r=r,
        q=q,
        tol=1e-8,
    )

    assert rep.counts["n_fail"] == 0
