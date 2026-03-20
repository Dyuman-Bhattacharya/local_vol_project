# tests/unit/test_arbitrage_density.py
import numpy as np
import pytest

from implied_volatility.black_scholes import bs_price
from arbitrage.density import breeden_litzenberger_density


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
        K = np.linspace(1e-2, 400.0, 2001)  # wide grid for density mass check
    if T is None:
        T = np.array([0.25, 0.5, 1.0])

    C = np.vstack(
        [bs_price(S0, K, t, r=r, q=q, sigma=sigma, option_type="call") for t in T]
    )
    return C, K, T


def test_density_non_negative_for_black_scholes():
    """
    Breeden–Litzenberger density for BS should be non-negative (up to numerical noise).
    """
    r = 0.01
    C, K, T = make_bs_surface(r=r)

    report = breeden_litzenberger_density(C, K, T, r=r, tol=1e-10)

    assert report.counts["n_negative"] == 0
    assert not np.isnan(report.density).any()


def test_density_mass_is_approximately_one_for_wide_grid():
    """
    With a sufficiently wide K grid, ∫ p(K,T) dK ≈ 1.
    This is a diagnostic, not an exact theorem.
    """
    r = 0.01
    C, K, T = make_bs_surface(r=r)

    report = breeden_litzenberger_density(C, K, T, r=r)

    for m in report.mass:
        assert abs(m - 1.0) < 5e-2  # loose tolerance; numerical differentiation
