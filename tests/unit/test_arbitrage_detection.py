import numpy as np

from implied_volatility.black_scholes import bs_price
from arbitrage.static_checks import check_static_arbitrage_calls


def test_butterfly_arbitrage_is_detected():
    S0 = 100.0
    r = 0.01
    q = 0.0
    sigma = 0.2

    K = np.linspace(80, 120, 21)
    T = np.array([1.0])

    C = bs_price(S0, K, T[0], r, q, sigma, "call")[None, :]

    # Inject butterfly arbitrage:
    # artificially bump the middle strike
    mid = K.size // 2
    C[0, mid] += 5.0

    rep = check_static_arbitrage_calls(
        C,
        K,
        T,
        S0=S0,
        r=r,
        q=q,
        tol=1e-8,
    )

    assert rep.counts["fail_strike_convex"] > 0
