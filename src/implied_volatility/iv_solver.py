# src/implied_volatility/iv_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from utils.platform_compat import apply_windows_platform_fastpath

apply_windows_platform_fastpath()
from scipy.optimize import brentq

from implied_volatility.black_scholes import (
    bs_price,
    bs_vega,
    no_arbitrage_bounds_call,
    no_arbitrage_bounds_put,
)

OptionType = Literal["call", "put"]


class IVSolverError(RuntimeError):
    """Raised when IV inversion fails unexpectedly (not just infeasible inputs)."""


@dataclass(frozen=True)
class IVSolverConfig:
    """
    Configuration for implied vol inversion.

    newton:
      - fast, but can fail for low vega / extreme options
    brent:
      - robust, but slower (needs bracket)
    """
    method: Literal["newton", "brent", "hybrid"] = "hybrid"
    max_iter: int = 100
    tol: float = 1e-10

    # Newton safeguards
    vega_floor: float = 1e-10
    step_cap: float = 1.0  # cap Newton step magnitude in vol units

    # Vol bounds for bracketing
    sigma_min: float = 1e-6
    sigma_max: float = 5.0  # 500% vol; widen if needed

    # If True: return NaN for prices outside no-arbitrage bounds instead of raising.
    nan_on_bound_violation: bool = True


def _arb_bounds(
    option_type: OptionType,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
):
    if option_type == "call":
        return no_arbitrage_bounds_call(S, K, T, r=r, q=q)
    if option_type == "put":
        return no_arbitrage_bounds_put(S, K, T, r=r, q=q)
    raise ValueError("option_type must be 'call' or 'put'")


def implied_vol_one(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    cfg: IVSolverConfig,
    sigma0: Optional[float] = None,
) -> float:
    """
    Invert Black-Scholes to get implied volatility for a single option quote.

    Returns NaN if:
      - T==0 (IV undefined)
      - price violates no-arbitrage bounds and cfg.nan_on_bound_violation is True
      - solver fails to converge (in hybrid it will try fallback)
    """
    price = float(price)
    S = float(S)
    K = float(K)
    T = float(T)

    if T <= 0.0:
        return float("nan")

    lo, hi = _arb_bounds(option_type, np.array([S]), np.array([K]), np.array([T]), r=r, q=q)
    lo = float(lo[0])
    hi = float(hi[0])
    if not (lo - 1e-12 <= price <= hi + 1e-12):
        if cfg.nan_on_bound_violation:
            return float("nan")
        raise ValueError(f"Price violates no-arbitrage bounds: {price} not in [{lo}, {hi}]")

    # Initial guess
    if sigma0 is None:
        # crude but reliable-ish initial guess:
        # ATM-ish guess based on Brenner-Subrahmanyam style scaling
        # sigma ~ sqrt(2*pi/T) * (C / (S e^{-qT})) for near-ATM calls
        disc_q = np.exp(-q * T)
        sigma0 = np.sqrt(2.0 * np.pi / T) * max(price, 1e-12) / (S * disc_q)
        sigma0 = float(np.clip(sigma0, 0.05, 1.0))  # clamp initial guess
    sigma0 = float(sigma0)

    def f(sig: float) -> float:
        return float(bs_price(S, K, T, r=r, q=q, sigma=sig, option_type=option_type) - price)

    # Choose method
    method = cfg.method

    if method in ("newton", "hybrid"):
        sig = sigma0
        for _ in range(cfg.max_iter):
            val = f(sig)
            if abs(val) < cfg.tol:
                return float(sig)
            vega = float(bs_vega(S, K, T, r=r, q=q, sigma=sig))
            if vega < cfg.vega_floor:
                break  # fallback
            step = val / vega
            step = float(np.clip(step, -cfg.step_cap, cfg.step_cap))
            sig_new = sig - step
            # keep within reasonable bounds
            sig = float(np.clip(sig_new, cfg.sigma_min, cfg.sigma_max))
        if method == "newton":
            return float("nan")

    # Brent fallback (robust)
    try:
        # Ensure bracket produces opposite signs:
        a = cfg.sigma_min
        b = cfg.sigma_max
        fa = f(a)
        fb = f(b)

        # If bracket doesn't work, expand b a bit (rare, but can happen for weird inputs)
        if fa * fb > 0.0:
            b2 = max(10.0, b * 2.0)
            fb2 = f(b2)
            if fa * fb2 > 0.0:
                return float("nan")
            b = b2
            fb = fb2

        root = brentq(lambda s: f(s), a=a, b=b, xtol=cfg.tol, maxiter=cfg.max_iter)
        return float(root)
    except Exception:
        return float("nan")


def implied_vol(
    prices,
    S,
    K,
    T,
    r: float,
    q: float,
    option_type: OptionType,
    cfg: Optional[IVSolverConfig] = None,
    sigma0: Optional[float] = None,
) -> np.ndarray:
    """
    Vectorized implied vol inversion for arrays.

    Uses a safe per-element scalar routine (brute but robust).
    Later you can optimize (Numba / vectorized brent) if needed.
    """
    if cfg is None:
        cfg = IVSolverConfig()

    prices = np.asarray(prices, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    prices, S, K, T = np.broadcast_arrays(prices, S, K, T)

    out = np.empty_like(prices, dtype=float)
    it = np.nditer(prices, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        out[idx] = implied_vol_one(
            price=float(prices[idx]),
            S=float(S[idx]),
            K=float(K[idx]),
            T=float(T[idx]),
            r=r,
            q=q,
            option_type=option_type,
            cfg=cfg,
            sigma0=sigma0,
        )
        it.iternext()

    return out
