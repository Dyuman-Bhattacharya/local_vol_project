# src/local_volatility/dupire.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from utils.numerical import require_strictly_increasing, safe_divide
from arbitrage.static_checks import finite_diff_dK, finite_diff_d2K, finite_diff_dT


class DupireError(RuntimeError):
    """Raised when Dupire local volatility extraction cannot be performed."""


@dataclass(frozen=True)
class DupireConfig:
    """
    Configuration for Dupire extraction on a (T, K) call price grid.

    Notes
    -----
    - floor_density: floors the effective density (denominator) to avoid division by ~0.
    - cap_sigma: optional cap on returned sigma_loc to avoid numerical blowups.
    """
    floor_density: float = 1e-8
    cap_sigma: Optional[float] = None
    eps_div: float = 1e-14


def dupire_local_var_from_call_grid(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    r: float,
    q: float = 0.0,
    cfg: DupireConfig = DupireConfig(),
) -> np.ndarray:
    """
    Compute Dupire local variance sigma_loc^2(K,T) on the same (T,K) grid
    from a call surface C(K,T) using the generalized Dupire formula (with carry q):

        sigma_loc^2(K,T) =
            [ ∂C/∂T + (r - q) K ∂C/∂K + q C ] / [ 0.5 K^2 ∂²C/∂K² ]

    Inputs
    ------
    C : (nT, nK) call prices
    K : (nK,) strikes (strictly increasing)
    T : (nT,) maturities in years (strictly increasing)
    r : risk-free rate (cont. comp.)
    q : dividend yield (cont. comp.)

    Returns
    -------
    sig2 : (nT, nK) local variance on (T,K) grid

    Notes
    -----
    - This is numerically ill-posed if the input surface is noisy. You should feed
      a *smoothed, arbitrage-checked* call grid (post-fitting).
    - Near wings and very short maturities, denominator gets tiny and can blow up.
      Use regularization in local_volatility/regularization.py.
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    T = require_strictly_increasing(T, "T")

    if C.ndim != 2:
        raise DupireError("C must be 2D array with shape (nT,nK)")
    nT, nK = C.shape
    if nK != K.size or nT != T.size:
        raise DupireError(f"Shape mismatch: C {C.shape}, K {K.shape}, T {T.shape}")

    # Derivatives
    dC_dT = finite_diff_dT(C, T)
    dC_dK = finite_diff_dK(C, K)
    d2C_dK2 = finite_diff_d2K(C, K)

    KK = K[None, :]  # (1,nK)

    numer = dC_dT + (r - q) * KK * dC_dK + q * C
    denom = 0.5 * (KK * KK) * d2C_dK2

    # Floor denominator via floor on density: ∂²C/∂K² is proportional to density * e^{-rT}
    # We floor denom directly for simplicity.
    denom_floored = np.where(np.abs(denom) < cfg.floor_density, np.sign(denom) * cfg.floor_density, denom)

    sig2 = safe_divide(numer, denom_floored, eps=cfg.eps_div)

    # Negative local variance is non-physical; occurs due to noise/violations.
    sig2 = np.maximum(sig2, 0.0)

    if cfg.cap_sigma is not None:
        cap2 = float(cfg.cap_sigma) ** 2
        sig2 = np.minimum(sig2, cap2)

    return sig2


def dupire_local_vol_from_call_grid(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    r: float,
    q: float = 0.0,
    cfg: DupireConfig = DupireConfig(),
) -> np.ndarray:
    """
    Convenience wrapper: returns sigma_loc (not squared).
    """
    sig2 = dupire_local_var_from_call_grid(C, K, T, r=r, q=q, cfg=cfg)
    return np.sqrt(np.maximum(sig2, 0.0))