# src/local_volatility/surface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from utils.numerical import require_strictly_increasing
from utils.interpolation import Interp2DGrid


class LocalVolSurfaceError(RuntimeError):
    """Raised when LocalVolSurface is mis-specified or queried incorrectly."""


@dataclass(frozen=True)
class LocalVolSurface:
    """
    Container for local volatility surface sigma_loc(S,t) represented on a rectilinear grid.

    IMPORTANT:
      - Dupire produces sigma_loc on a (K,T) grid (strike,maturity).
      - For practical pricing, we treat this as sigma_loc(S,t) by identifying S~K
        on the grid and interpolating.

    Convention:
      - S_grid: (nS,) strictly increasing
      - t_grid: (nt,) strictly increasing (years)
      - sigma_grid: shape (nt, nS) with sigma_grid[j,i] = sigma(t_j, S_i)

    This is a *query object*: it doesn't compute Dupire; it only interpolates.
    """
    S_grid: np.ndarray
    t_grid: np.ndarray
    sigma_grid: np.ndarray
    domain_mask: Optional[np.ndarray] = None
    method: Literal["linear", "nearest"] = "linear"
    fill_value: Optional[float] = None

    def __post_init__(self) -> None:
        require_strictly_increasing(self.S_grid, "S_grid")
        require_strictly_increasing(self.t_grid, "t_grid")
        sg = np.asarray(self.sigma_grid, dtype=float)
        if sg.ndim != 2:
            raise ValueError("sigma_grid must be 2D (nt,nS)")
        if sg.shape != (self.t_grid.size, self.S_grid.size):
            raise ValueError(
                f"sigma_grid must have shape (nt,nS)=({self.t_grid.size},{self.S_grid.size}), got {sg.shape}"
            )
        if self.domain_mask is not None:
            dm = np.asarray(self.domain_mask, dtype=np.int8)
            if dm.shape != sg.shape:
                raise ValueError(
                    f"domain_mask must have shape (nt,nS)=({self.t_grid.size},{self.S_grid.size}), got {dm.shape}"
                )

    def sigma(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Interpolate sigma_loc at (S,t) points.

        CRITICAL: Do not extrapolate outside the calibrated grid.
        - Clamp t into [t_min, t_max]
        - Clamp S into [S_min, S_max]
        This prevents pathological local-vol extrapolation that can blow up PDE prices
        and induce large systematic hedging bias.
        """
        S = np.asarray(S, dtype=float)
        t = np.asarray(t, dtype=float)

        # Clamp to domain
        t_clamped = np.clip(t, self.t_grid[0], self.t_grid[-1])
        S_clamped = np.clip(S, self.S_grid[0], self.S_grid[-1])

        interp = Interp2DGrid(
            x_grid=self.S_grid,
            y_grid=self.t_grid,
            z=self.sigma_grid,
            method=self.method,
            fill_value=None,  # irrelevant now, we never query outside bounds
        )
        return interp(S_clamped, t_clamped)

    def domain_state(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Discrete domain classification at query points.

        Returns:
        - 0: extrapolation / unsupported
        - 1: weak support
        - 2: trusted interior
        """
        S = np.asarray(S, dtype=float)
        t = np.asarray(t, dtype=float)
        X, Y = np.broadcast_arrays(S, t)
        out = np.zeros_like(X, dtype=np.int8)

        if self.domain_mask is None:
            inside = (
                (X >= self.S_grid[0]) & (X <= self.S_grid[-1]) &
                (Y >= self.t_grid[0]) & (Y <= self.t_grid[-1])
            )
            out[inside] = 2
            return out

        inside = (
            (X >= self.S_grid[0]) & (X <= self.S_grid[-1]) &
            (Y >= self.t_grid[0]) & (Y <= self.t_grid[-1])
        )
        if not np.any(inside):
            return out

        S_inside = X[inside]
        t_inside = Y[inside]
        iS = np.searchsorted(self.S_grid, S_inside, side="left")
        iT = np.searchsorted(self.t_grid, t_inside, side="left")
        iS = np.clip(iS, 0, self.S_grid.size - 1)
        iT = np.clip(iT, 0, self.t_grid.size - 1)

        left_S = np.maximum(iS - 1, 0)
        choose_left_S = np.abs(S_inside - self.S_grid[left_S]) <= np.abs(self.S_grid[iS] - S_inside)
        iS = np.where((iS > 0) & choose_left_S, left_S, iS)

        left_T = np.maximum(iT - 1, 0)
        choose_left_T = np.abs(t_inside - self.t_grid[left_T]) <= np.abs(self.t_grid[iT] - t_inside)
        iT = np.where((iT > 0) & choose_left_T, left_T, iT)

        out_inside = np.asarray(self.domain_mask, dtype=np.int8)[iT, iS]
        out[inside] = out_inside
        return out

    def is_trusted(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.domain_state(S, t) == 2

    def is_supported(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.domain_state(S, t) >= 1

    

    @staticmethod
    def from_dupire_grid(
        sigma_loc_grid: np.ndarray,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        *,
        domain_mask: Optional[np.ndarray] = None,
        method: Literal["linear", "nearest"] = "linear",
        fill_value: Optional[float] = None,
    ) -> "LocalVolSurface":
        """
        Construct LocalVolSurface from Dupire output sigma_loc(K,T) by re-labelling K as S and T as t.
        """
        return LocalVolSurface(
            S_grid=np.asarray(K_grid, dtype=float),
            t_grid=np.asarray(T_grid, dtype=float),
            sigma_grid=np.asarray(sigma_loc_grid, dtype=float),
            domain_mask=np.asarray(domain_mask, dtype=np.int8) if domain_mask is not None else None,
            method=method,
            fill_value=fill_value,
        )
