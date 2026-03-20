# src/implied_volatility/black_scholes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from utils.platform_compat import apply_windows_platform_fastpath

apply_windows_platform_fastpath()
from scipy.stats import norm


OptionType = Literal["call", "put"]


class BlackScholesError(RuntimeError):
    """Raised when Black-Scholes computations fail due to invalid inputs."""


def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _validate_common_inputs(S, K, T):
    S = _to_float_array(S)
    K = _to_float_array(K)
    T = _to_float_array(T)

    if np.any(S <= 0.0):
        raise ValueError("Spot S must be strictly positive.")
    if np.any(K <= 0.0):
        raise ValueError("Strike K must be strictly positive.")
    if np.any(T < 0.0):
        raise ValueError("Time to expiry T must be non-negative.")

    return S, K, T


def _d1_d2(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute d1, d2 for Black-Scholes with continuous dividend yield q.

    d1 = [ln(S/K) + (r - q + 0.5 sigma^2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

    Convention:
    - If T == 0, d1/d2 aren't meaningful; caller should handle payoff directly.
    """
    sigma = _to_float_array(sigma)
    if np.any(sigma <= 0.0):
        raise ValueError("Volatility sigma must be strictly positive.")

    # Broadcast
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)

    sqrtT = np.sqrt(np.maximum(T, 0.0))
    denom = sigma * sqrtT

    # Avoid division by zero for T=0: d1/d2 will be overwritten/unused if caller handles T=0
    denom_safe = np.where(denom == 0.0, 1.0, denom)

    num = np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    d1 = num / denom_safe
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(
    S,
    K,
    T,
    r: float,
    q: float,
    sigma,
    option_type: OptionType = "call",
) -> np.ndarray:
    """
    Black-Scholes price for European call/put with continuous dividend yield q.

    Parameters are array-broadcastable.
    """
    S, K, T = _validate_common_inputs(S, K, T)
    sigma = _to_float_array(sigma)

    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)

    # Handle T=0 exactly: price is intrinsic (no discounting ambiguity at T=0)
    out = np.empty_like(S, dtype=float)
    t0 = (T == 0.0)
    if np.any(t0):
        if option_type == "call":
            out[t0] = np.maximum(S[t0] - K[t0], 0.0)
        elif option_type == "put":
            out[t0] = np.maximum(K[t0] - S[t0], 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # T>0 standard formula
    mask = ~t0
    if np.any(mask):
        d1, d2 = _d1_d2(S[mask], K[mask], T[mask], r=r, q=q, sigma=sigma[mask])
        disc_q = np.exp(-q * T[mask])
        disc_r = np.exp(-r * T[mask])

        if option_type == "call":
            out[mask] = S[mask] * disc_q * norm.cdf(d1) - K[mask] * disc_r * norm.cdf(d2)
        elif option_type == "put":
            out[mask] = K[mask] * disc_r * norm.cdf(-d2) - S[mask] * disc_q * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    return out


def bs_vega(
    S,
    K,
    T,
    r: float,
    q: float,
    sigma,
) -> np.ndarray:
    """
    Black-Scholes vega: ∂Price/∂sigma.

    Returned in price units per 1.0 vol (i.e., not per 1% vol).
    Vega -> 0 as T -> 0 or deep ITM/OTM.
    """
    S, K, T = _validate_common_inputs(S, K, T)
    sigma = _to_float_array(sigma)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)

    v = np.zeros_like(S, dtype=float)
    mask = (T > 0.0)
    if np.any(mask):
        d1, _ = _d1_d2(S[mask], K[mask], T[mask], r=r, q=q, sigma=sigma[mask])
        v[mask] = S[mask] * np.exp(-q * T[mask]) * norm.pdf(d1) * np.sqrt(T[mask])
    return v


def no_arbitrage_bounds_call(
    S,
    K,
    T,
    r: float,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model-independent no-arbitrage bounds for European call with dividend yield q:

      lower = max(S e^{-qT} - K e^{-rT}, 0)
      upper = S e^{-qT}
    """
    S, K, T = _validate_common_inputs(S, K, T)
    S, K, T = np.broadcast_arrays(S, K, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    lower = np.maximum(S * disc_q - K * disc_r, 0.0)
    upper = S * disc_q
    return lower, upper


def no_arbitrage_bounds_put(
    S,
    K,
    T,
    r: float,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model-independent no-arbitrage bounds for European put with dividend yield q:

      lower = max(K e^{-rT} - S e^{-qT}, 0)
      upper = K e^{-rT}
    """
    S, K, T = _validate_common_inputs(S, K, T)
    S, K, T = np.broadcast_arrays(S, K, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    lower = np.maximum(K * disc_r - S * disc_q, 0.0)
    upper = K * disc_r
    return lower, upper


def put_call_parity_call_from_put(
    P,
    S,
    K,
    T,
    r: float,
    q: float,
) -> np.ndarray:
    """
    C = P + S e^{-qT} - K e^{-rT}
    """
    P = _to_float_array(P)
    S, K, T = _validate_common_inputs(S, K, T)
    P, S, K, T = np.broadcast_arrays(P, S, K, T)
    return P + S * np.exp(-q * T) - K * np.exp(-r * T)


def put_call_parity_put_from_call(
    C,
    S,
    K,
    T,
    r: float,
    q: float,
) -> np.ndarray:
    """
    P = C - S e^{-qT} + K e^{-rT}
    """
    C = _to_float_array(C)
    S, K, T = _validate_common_inputs(S, K, T)
    C, S, K, T = np.broadcast_arrays(C, S, K, T)
    return C - S * np.exp(-q * T) + K * np.exp(-r * T)

def bs_delta(
    S,
    K,
    T,
    r,
    q,
    sigma,
    option_type="call",
):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle expiry
    out = np.zeros_like(S, dtype=float)
    mask = T > 0
    if not np.any(mask):
        return out

    d1 = (np.log(S[mask] / K[mask]) + (r - q + 0.5 * sigma[mask]**2) * T[mask]) / (
        sigma[mask] * np.sqrt(T[mask])
    )

    if option_type == "call":
        out[mask] = np.exp(-q * T[mask]) * norm.cdf(d1)
    else:
        out[mask] = np.exp(-q * T[mask]) * (norm.cdf(d1) - 1.0)

    return out
