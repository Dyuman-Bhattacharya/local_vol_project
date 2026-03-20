# src/arbitrage/density.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from utils.numerical import require_strictly_increasing
from arbitrage.static_checks import finite_diff_d2K



class DensityError(RuntimeError):
    """Raised when density computation fails or inputs are inconsistent."""


@dataclass(frozen=True)
class DensityReport:
    """
    Results of Breeden–Litzenberger density extraction.

    density: p(K, T) on the same (T, K) grid as prices.
    mass: ∫ p(K,T) dK per T (approx by trapezoid rule)
    negative_mask: where density < -tol
    """
    density: np.ndarray
    mass: np.ndarray
    negative_mask: np.ndarray
    counts: Dict[str, int]


def breeden_litzenberger_density(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    r: float,
    tol: float = 1e-12,
) -> DensityReport:
    """
    Compute risk-neutral density from a call price surface via Breeden–Litzenberger:

      p(K, T) = e^{rT} * ∂²C/∂K²

    Inputs
    ------
    C : array (nT, nK) call prices C(K, T)
    K : array (nK,) strikes, strictly increasing
    T : array (nT,) maturities, strictly increasing
    r : continuous risk-free rate

    Notes
    -----
    - Requires sufficient smoothness of C in K (in practice, interpolated/fitted surface).
    - Density is only meaningful where the grid is sufficiently dense (especially near ATM).
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    T = require_strictly_increasing(T, "T")

    if C.ndim != 2:
        raise DensityError("C must be 2D array shape (nT, nK)")
    nT, nK = C.shape
    if nK != K.size or nT != T.size:
        raise DensityError(f"Shape mismatch: C {C.shape}, K {K.shape}, T {T.shape}")

    d2 = finite_diff_d2K(C, K)  # (nT, nK)
    disc = np.exp(r * T)[:, None]  # (nT,1)
    p = disc * d2

    negative_mask = p < -tol

    # Approx mass check: ∫ p(K,T) dK ≈ 1 for each T (over full K range if it spans [0,∞) well)
    # In practice, you won't cover [0,∞), so this is a *diagnostic* not a strict condition.
    mass = np.trapezoid(p, K, axis=1)

    counts = {
        "n_total": int(p.size),
        "n_negative": int(negative_mask.sum()),
        "n_nan": int(np.isnan(p).sum()),
    }

    return DensityReport(
        density=p,
        mass=mass,
        negative_mask=negative_mask,
        counts=counts,
    )


def density_normalization_diagnostic(
    density: np.ndarray,
    K: np.ndarray,
    *,
    tol: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience diagnostic:
      returns (mass, ok_mask) where mass_j = ∫ p(K,T_j) dK

    ok_mask is True when |mass - 1| <= tol.

    This is NOT a theorem unless K-grid covers the relevant support.
    """
    density = np.asarray(density, dtype=float)
    K = require_strictly_increasing(K, "K")
    if density.ndim != 2 or density.shape[1] != K.size:
        raise DensityError("density must have shape (nT, nK) matching K")
    mass = np.trapezoid(density, K, axis=1)
    ok = np.abs(mass - 1.0) <= tol
    return mass, ok
