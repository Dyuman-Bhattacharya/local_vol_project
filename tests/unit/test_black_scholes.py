# tests/unit/test_black_scholes.py
import numpy as np
import pytest

from implied_volatility.black_scholes import (
    bs_price,
    bs_vega,
    put_call_parity_put_from_call,
    put_call_parity_call_from_put,
    no_arbitrage_bounds_call,
    no_arbitrage_bounds_put,
)


def test_put_call_parity_consistency():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.2

    C = bs_price(S, K, T, r, q, sigma, option_type="call")
    P = bs_price(S, K, T, r, q, sigma, option_type="put")

    P_from_C = put_call_parity_put_from_call(C, S, K, T, r, q)
    C_from_P = put_call_parity_call_from_put(P, S, K, T, r, q)

    assert np.allclose(P, P_from_C, atol=1e-12)
    assert np.allclose(C, C_from_P, atol=1e-12)


def test_no_arbitrage_bounds_call():
    S = 100.0
    K = np.array([80.0, 100.0, 120.0])
    T = 1.0
    r = 0.03
    q = 0.01
    sigma = 0.25

    C = bs_price(S, K, T, r, q, sigma, option_type="call")
    lo, hi = no_arbitrage_bounds_call(S, K, T, r, q)

    assert np.all(C >= lo - 1e-12)
    assert np.all(C <= hi + 1e-12)


def test_no_arbitrage_bounds_put():
    S = 100.0
    K = np.array([80.0, 100.0, 120.0])
    T = 1.0
    r = 0.03
    q = 0.01
    sigma = 0.25

    P = bs_price(S, K, T, r, q, sigma, option_type="put")
    lo, hi = no_arbitrage_bounds_put(S, K, T, r, q)

    assert np.all(P >= lo - 1e-12)
    assert np.all(P <= hi + 1e-12)


def test_vega_positive_and_zero_at_expiry():
    S = 100.0
    K = 100.0
    r = 0.01
    q = 0.0
    sigma = 0.2

    T = np.array([0.0, 0.5, 1.0])
    v = bs_vega(S, K, T, r, q, sigma)

    assert v[0] == 0.0
    assert v[1] > 0.0
    assert v[2] > 0.0


def test_intrinsic_value_at_expiry():
    S = np.array([90.0, 100.0, 110.0])
    K = 100.0
    T = 0.0
    r = 0.0
    q = 0.0
    sigma = 0.3

    C = bs_price(S, K, T, r, q, sigma, option_type="call")
    P = bs_price(S, K, T, r, q, sigma, option_type="put")

    assert np.allclose(C, np.maximum(S - K, 0.0))
    assert np.allclose(P, np.maximum(K - S, 0.0))
