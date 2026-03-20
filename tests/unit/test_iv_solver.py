# tests/unit/test_iv_solver.py
import numpy as np
import pytest

from implied_volatility.black_scholes import bs_price
from implied_volatility.iv_solver import (
    implied_vol,
    implied_vol_one,
    IVSolverConfig,
)


def test_iv_round_trip_call():
    S = 100.0
    r = 0.03
    q = 0.01
    T = np.array([0.25, 0.5, 1.0])
    K = np.array([90.0, 100.0, 110.0])
    sigma_true = 0.2

    prices = bs_price(S, K, T, r, q, sigma_true, option_type="call")
    iv = implied_vol(prices, S, K, T, r, q, option_type="call")

    assert np.allclose(iv, sigma_true, atol=1e-6)


def test_iv_round_trip_put():
    S = 100.0
    r = 0.03
    q = 0.01
    T = 1.0
    K = np.array([80.0, 100.0, 120.0])
    sigma_true = 0.35

    prices = bs_price(S, K, T, r, q, sigma_true, option_type="put")
    iv = implied_vol(prices, S, K, T, r, q, option_type="put")

    assert np.allclose(iv, sigma_true, atol=1e-6)


def test_iv_returns_nan_for_zero_maturity():
    cfg = IVSolverConfig()
    iv = implied_vol_one(
        price=1.0,
        S=100.0,
        K=100.0,
        T=0.0,
        r=0.01,
        q=0.0,
        option_type="call",
        cfg=cfg,
    )
    assert np.isnan(iv)


def test_iv_outside_arbitrage_bounds_nan():
    cfg = IVSolverConfig(nan_on_bound_violation=True)
    iv = implied_vol_one(
        price=1000.0,  # absurd price
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.01,
        q=0.0,
        option_type="call",
        cfg=cfg,
    )
    assert np.isnan(iv)


def test_newton_and_brent_consistency():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.02
    q = 0.0
    sigma_true = 0.25

    price = bs_price(S, K, T, r, q, sigma_true, option_type="call")

    cfg_newton = IVSolverConfig(method="newton")
    cfg_brent = IVSolverConfig(method="brent")

    iv_n = implied_vol_one(price, S, K, T, r, q, "call", cfg_newton)
    iv_b = implied_vol_one(price, S, K, T, r, q, "call", cfg_brent)

    assert abs(iv_n - sigma_true) < 1e-6
    assert abs(iv_b - sigma_true) < 1e-6