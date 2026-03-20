# tests/unit/test_arbitrage_static_checks.py
import numpy as np
import pytest

from implied_volatility.black_scholes import bs_price
from arbitrage.static_checks import check_static_arbitrage_calls


def make_bs_surface(
    *,
    S0=100.0,
    r=0.02,
    q=0.0,
    sigma=0.2,
    K=None,
    T=None,
):
    if K is None:
        K = np.linspace(40.0, 200.0, 161)
    if T is None:
        T = np.array([0.1, 0.25, 0.5, 1.0, 2.0])

    C = np.vstack(
        [bs_price(S0, K, t, r=r, q=q, sigma=sigma, option_type="call") for t in T]
    )
    return C, K, T


def test_black_scholes_surface_has_no_static_arbitrage():
    """
    A smooth Black–Scholes surface should pass:
      - bounds
      - strike monotonicity
      - strike convexity
      - calendar monotonicity
    """
    S0 = 100.0
    r = 0.01
    q = 0.0

    C, K, T = make_bs_surface(S0=S0, r=r, q=q, sigma=0.25)

    report = check_static_arbitrage_calls(
        C,
        K,
        T,
        S0=S0,
        r=r,
        q=q,
        tol=1e-8,
        return_derivatives=True,
    )

    assert report.counts["n_fail"] == 0
    assert report.bounds_ok.all()
    assert report.strike_monotone_ok.all()
    assert report.strike_convex_ok.all()
    assert report.calendar_monotone_ok.all()


def test_strike_monotonicity_violation_is_detected():
    """
    Artificially increase price at a higher strike to violate ∂C/∂K <= 0.
    """
    S0 = 100.0
    r = 0.01
    C, K, T = make_bs_surface(S0=S0, r=r)

    # Inject violation at one maturity slice
    j = 2  # some maturity
    i = len(K) // 2
    C[j, i + 1] = C[j, i] + 5.0  # impossible: higher strike more expensive

    report = check_static_arbitrage_calls(C, K, T, S0=S0, r=r)

    assert report.counts["fail_strike_monotone"] > 0
    assert report.counts["n_fail"] > 0


def test_strike_convexity_violation_is_detected():
    """
    Break convexity by pushing the middle strike price above linear interpolation.
    """
    S0 = 100.0
    r = 0.01
    C, K, T = make_bs_surface(S0=S0, r=r)

    j = 1
    i = len(K) // 2

    # Violate butterfly condition
    C[j, i] += 10.0

    report = check_static_arbitrage_calls(C, K, T, S0=S0, r=r)

    assert report.counts["fail_strike_convex"] > 0
    assert report.counts["n_fail"] > 0


def test_calendar_monotonicity_violation_is_detected():
    """
    Force a longer maturity option to be cheaper than a shorter one at same strike.
    """
    S0 = 100.0
    r = 0.01
    C, K, T = make_bs_surface(S0=S0, r=r)

    i = len(K) // 2

    # Force C(T2) < C(T1)
    C[3, i] = C[2, i] - 5.0

    report = check_static_arbitrage_calls(C, K, T, S0=S0, r=r)

    assert report.counts["fail_calendar_monotone"] > 0
    assert report.counts["n_fail"] > 0


def test_call_bounds_violation_is_detected():
    """
    Force call price to exceed upper bound S e^{-qT}.
    """
    S0 = 100.0
    r = 0.01
    q = 0.0

    C, K, T = make_bs_surface(S0=S0, r=r, q=q)

    j = 0
    i = 0
    C[j, i] = S0 * np.exp(-q * T[j]) + 1.0  # violates upper bound

    report = check_static_arbitrage_calls(C, K, T, S0=S0, r=r, q=q)

    assert report.counts["fail_bounds"] > 0
    assert report.counts["n_fail"] > 0
