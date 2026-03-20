# src/arbitrage/static_checks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from utils.numerical import require_strictly_increasing


class ArbitrageCheckError(RuntimeError):
    """Raised when arbitrage checks cannot be performed due to invalid inputs."""


@dataclass(frozen=True)
class StaticArbitrageReport:
    """
    Report for static arbitrage checks on a call price surface C(K, T).

    All boolean masks have shape (nT, nK) unless otherwise stated.
    """
    pass_mask: np.ndarray
    bounds_ok: np.ndarray
    strike_monotone_ok: np.ndarray
    strike_convex_ok: np.ndarray
    calendar_monotone_ok: np.ndarray

    # Convenience summary counts
    counts: Dict[str, int]

    # Optional derivative arrays for debugging
    dC_dK: Optional[np.ndarray] = None
    d2C_dK2: Optional[np.ndarray] = None
    dC_dT: Optional[np.ndarray] = None


def _bounds_call(S0: float, K: np.ndarray, T: np.ndarray, r: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call bounds on a grid given spot S0 and rates:
      lower = max(S0 e^{-qT} - K e^{-rT}, 0)
      upper = S0 e^{-qT}
    """
    disc_q = np.exp(-q * T)[:, None]  # (nT,1)
    disc_r = np.exp(-r * T)[:, None]
    lower = np.maximum(S0 * disc_q - K[None, :] * disc_r, 0.0)
    upper = S0 * disc_q * np.ones((T.size, K.size), dtype=float)
    return lower, upper


def finite_diff_dK(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    ∂C/∂K using central differences on a non-uniform K grid, per maturity slice.
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    nT, nK = C.shape
    if nK != K.size:
        raise ValueError("C.shape[1] must equal len(K)")
    d = np.empty_like(C)

    # per row
    for j in range(nT):
        f = C[j, :]
        # central interior
        d[j, 1:-1] = (f[2:] - f[:-2]) / (K[2:] - K[:-2])
        # one-sided
        d[j, 0] = (f[1] - f[0]) / (K[1] - K[0])
        d[j, -1] = (f[-1] - f[-2]) / (K[-1] - K[-2])
    return d


def finite_diff_d2K(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    ∂²C/∂K² using the standard 3-pt non-uniform stencil per maturity slice:
      f''_i ≈ 2[(f_{i+1}-f_i)/h_i - (f_i-f_{i-1})/h_{i-1}] / (h_{i-1}+h_i)
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    nT, nK = C.shape
    if nK != K.size:
        raise ValueError("C.shape[1] must equal len(K)")

    d2 = np.empty_like(C)
    h = np.diff(K)  # (nK-1,)

    for j in range(nT):
        f = C[j, :]
        hm = h[:-1]
        hp = h[1:]
        num = (f[2:] - f[1:-1]) / hp - (f[1:-1] - f[:-2]) / hm
        d2[j, 1:-1] = 2.0 * num / (hm + hp)
        d2[j, 0] = d2[j, 1]
        d2[j, -1] = d2[j, -2]
    return d2

def finite_diff_dT(C: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    ∂C/∂T using finite differences along maturity axis.

    If only a single maturity is present (nT == 1),
    the derivative is undefined; return zeros so that
    calendar monotonicity is treated as trivially satisfied.
    """
    C = np.asarray(C, dtype=float)
    T = require_strictly_increasing(T, "T")
    nT, nK = C.shape
    if nT != T.size:
        raise ValueError("C.shape[0] must equal len(T)")

    d = np.zeros_like(C)

    # No calendar arbitrage can be defined with a single maturity
    if nT == 1:
        return d

    for i in range(nK):
        f = C[:, i]
        d[1:-1, i] = (f[2:] - f[:-2]) / (T[2:] - T[:-2])
        d[0, i] = (f[1] - f[0]) / (T[1] - T[0])
        d[-1, i] = (f[-1] - f[-2]) / (T[-1] - T[-2])

    return d


def check_static_arbitrage_calls(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    S0: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-10,
    return_derivatives: bool = False,
) -> StaticArbitrageReport:
    """
    Check basic static no-arbitrage for European call price surface C(K, T).

    Inputs
    ------
    C : array (nT, nK) call prices on a rectilinear grid
    K : array (nK,) strikes, strictly increasing
    T : array (nT,) maturities (year fractions), strictly increasing
    S0: spot at valuation date (assumed same across grid)
    r : continuous risk-free rate
    q : continuous dividend yield
    tol: numerical tolerance
    return_derivatives: include derivative arrays for debugging

    Checks
    ------
    1) Bounds: lower <= C <= upper
    2) Strike monotonicity: ∂C/∂K <= 0
    3) Strike convexity: ∂²C/∂K² >= 0
    4) Calendar monotonicity: ∂C/∂T >= 0
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    T = require_strictly_increasing(T, "T")

    if C.ndim != 2:
        raise ArbitrageCheckError("C must be a 2D array with shape (nT, nK)")
    nT, nK = C.shape
    if nK != K.size or nT != T.size:
        raise ArbitrageCheckError(f"Shape mismatch: C {C.shape}, K {K.shape}, T {T.shape}")
    if S0 <= 0.0:
        raise ArbitrageCheckError("S0 must be positive")

    # 1) Bounds
    lower, upper = _bounds_call(S0=float(S0), K=K, T=T, r=float(r), q=float(q))
    bounds_ok = (C >= lower - tol) & (C <= upper + tol)

    # 2) Strike monotonicity
    dC_dK = finite_diff_dK(C, K)
    strike_monotone_ok = dC_dK <= tol

    # 3) Strike convexity
    d2C_dK2 = finite_diff_d2K(C, K)
    strike_convex_ok = d2C_dK2 >= -tol

    # 4) Calendar monotonicity
    dC_dT = finite_diff_dT(C, T)
    calendar_monotone_ok = dC_dT >= -tol

    pass_mask = bounds_ok & strike_monotone_ok & strike_convex_ok & calendar_monotone_ok

    counts = {
        "n_total": int(pass_mask.size),
        "n_fail": int((~pass_mask).sum()),
        "fail_bounds": int((~bounds_ok).sum()),
        "fail_strike_monotone": int((~strike_monotone_ok).sum()),
        "fail_strike_convex": int((~strike_convex_ok).sum()),
        "fail_calendar_monotone": int((~calendar_monotone_ok).sum()),
    }

    if return_derivatives:
        return StaticArbitrageReport(
            pass_mask=pass_mask,
            bounds_ok=bounds_ok,
            strike_monotone_ok=strike_monotone_ok,
            strike_convex_ok=strike_convex_ok,
            calendar_monotone_ok=calendar_monotone_ok,
            counts=counts,
            dC_dK=dC_dK,
            d2C_dK2=d2C_dK2,
            dC_dT=dC_dT,
        )

    return StaticArbitrageReport(
        pass_mask=pass_mask,
        bounds_ok=bounds_ok,
        strike_monotone_ok=strike_monotone_ok,
        strike_convex_ok=strike_convex_ok,
        calendar_monotone_ok=calendar_monotone_ok,
        counts=counts,
        dC_dK=None,
        d2C_dK2=None,
        dC_dT=None,
    )