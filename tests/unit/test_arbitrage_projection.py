import numpy as np

from implied_volatility.black_scholes import bs_price
from arbitrage.projection import (
    project_call_surface_static_arbitrage,
    summarize_calendar_violations,
)
from arbitrage.static_checks import check_static_arbitrage_calls


def make_bs_surface(
    *,
    S0=100.0,
    r=0.01,
    q=0.0,
    sigma=0.25,
):
    K = np.linspace(70.0, 130.0, 25)
    T = np.array([0.10, 0.25, 0.50, 1.00], dtype=float)
    C = np.vstack([bs_price(S0, K, t, r=r, q=q, sigma=sigma, option_type="call") for t in T])
    return C, K, T


def test_projection_repairs_calendar_and_convexity_violations():
    S0 = 100.0
    r = 0.01
    q = 0.0
    C, K, T = make_bs_surface(S0=S0, r=r, q=q)

    # Inject a calendar violation at a single strike.
    i_mid = len(K) // 2
    C[2, i_mid] = C[1, i_mid] - 2.0

    rep_before = check_static_arbitrage_calls(C, K, T, S0=S0, r=r, q=q)
    assert rep_before.counts["fail_calendar_monotone"] > 0

    result = project_call_surface_static_arbitrage(C, K, T, S0=S0, r=r, q=q)
    rep_after = check_static_arbitrage_calls(result.C_projected, K, T, S0=S0, r=r, q=q)

    assert result.counts_before["fail_calendar_monotone"] > 0
    assert result.counts_after["n_fail"] == 0
    assert rep_after.counts["n_fail"] == 0
    assert result.rmse_adjustment > 0.0


def test_calendar_violation_summary_flags_adjacent_pair():
    S0 = 100.0
    r = 0.01
    C, K, T = make_bs_surface(S0=S0, r=r)

    C[2, 5] = C[1, 5] - 0.25
    pairs = summarize_calendar_violations(C, K, T)

    assert len(pairs) == len(T) - 1
    assert any(p["n_negative_adjacent"] > 0 for p in pairs)
