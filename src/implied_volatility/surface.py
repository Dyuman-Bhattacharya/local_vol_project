# src/implied_volatility/surface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from utils.interpolation import Interp2DGrid, Interp1D
from utils.numerical import require_strictly_increasing
from .iv_solver import implied_vol, IVSolverConfig


SurfaceCoord = Literal["K_T", "x_T"]


@dataclass(frozen=True)
class IVSurface:
    """
    Container for implied volatility surface defined on a rectilinear grid.

    Convention:
      - x_axis: either strike K or log-moneyness x
      - t_axis: maturity T in years
      - iv_grid: shape (nT, nX) where iv_grid[j, i] = iv(T_j, X_i)
    """
    x_grid: np.ndarray
    t_grid: np.ndarray
    iv_grid: np.ndarray
    coord: SurfaceCoord = "K_T"
    fill_value: Optional[float] = None
    method: Literal["linear", "nearest"] = "linear"

    def __post_init__(self) -> None:
        require_strictly_increasing(self.x_grid, "x_grid")
        require_strictly_increasing(self.t_grid, "t_grid")
        iv = np.asarray(self.iv_grid, dtype=float)
        if iv.ndim != 2:
            raise ValueError("iv_grid must be 2D with shape (nT, nX)")
        nT, nX = iv.shape
        if nT != self.t_grid.size or nX != self.x_grid.size:
            raise ValueError(
                f"iv_grid shape must be (nT, nX)=({self.t_grid.size},{self.x_grid.size}), got {iv.shape}"
            )

    @property
    def total_variance_grid(self) -> np.ndarray:
        """w = sigma^2 * T (with T broadcast along x dimension)."""
        T = self.t_grid[:, None]  # (nT,1)
        return (self.iv_grid ** 2) * T

    def iv(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Interpolate implied vol at (x, T) points.
        """
        interp = Interp2DGrid(
            x_grid=self.x_grid,
            y_grid=self.t_grid,
            z=self.iv_grid,
            method=self.method,
            fill_value=self.fill_value,
        )
        return interp(x, T)

    def total_variance(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Interpolate total variance w(x,T) = sigma^2(x,T) * T.
        """
        # Interpolate iv then compute w; if you later store w directly, swap this.
        sig = self.iv(x, T)
        return (sig ** 2) * np.asarray(T, dtype=float)

    def to_dataframe(self):
        """
        Return long-form DataFrame (requires pandas).
        Columns: T, X, iv, w
        """
        import pandas as pd  # local import to keep base deps light

        TT, XX = np.meshgrid(self.t_grid, self.x_grid, indexing="ij")  # (nT,nX)
        iv = np.asarray(self.iv_grid, dtype=float)
        w = iv * iv * TT
        df = pd.DataFrame(
            {
                "T": TT.ravel(),
                "X": XX.ravel(),
                "iv": iv.ravel(),
                "w": w.ravel(),
                "coord": self.coord,
            }
        )
        return df

    def plot_slice(self, T: float):
        """
        Quick plotting helper for a single maturity slice.
        """
        import matplotlib.pyplot as plt

        # nearest slice
        j = int(np.argmin(np.abs(self.t_grid - T)))
        plt.figure()
        plt.plot(self.x_grid, self.iv_grid[j, :])
        plt.xlabel("K" if self.coord == "K_T" else "log-moneyness x")
        plt.ylabel("implied vol")
        plt.title(f"IV slice at T≈{self.t_grid[j]:.4f}")
        plt.show()


@dataclass(frozen=True)
class ArbitrageFreeIVSurface:
    """
    Implied-vol surface backed by an arbitrage-free call-price grid.

    The no-arbitrage object is the call surface:
      - piecewise linear in strike K within each maturity slice,
      - linearly interpolated in maturity T between slices.

    Because convex, decreasing strike slices remain convex/decreasing under convex
    combinations, and because call values are interpolated monotonically in maturity,
    the interpolated call surface remains static-arbitrage-free inside the grid.
    IV values are obtained by Black-Scholes inversion of those interpolated call prices.
    """

    K_grid: np.ndarray
    T_grid: np.ndarray
    C_grid: np.ndarray
    S0: float
    r: float
    q: float = 0.0
    iv_solver_cfg: IVSolverConfig = IVSolverConfig(method="hybrid", max_iter=100, tol=1e-10)

    def __post_init__(self) -> None:
        require_strictly_increasing(self.K_grid, "K_grid")
        require_strictly_increasing(self.T_grid, "T_grid")
        C = np.asarray(self.C_grid, dtype=float)
        if C.ndim != 2:
            raise ValueError("C_grid must be 2D with shape (nT, nK)")
        if C.shape != (self.T_grid.size, self.K_grid.size):
            raise ValueError(
                f"C_grid shape must be (nT, nK)=({self.T_grid.size}, {self.K_grid.size}), got {C.shape}"
            )
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")

    def _interp_strike_all_maturities(self, K_new: np.ndarray) -> np.ndarray:
        """
        Interpolate call prices in strike for every maturity slice.
        """
        K_new = np.asarray(K_new, dtype=float)
        K_eval = np.clip(K_new, self.K_grid[0], self.K_grid[-1])
        out = np.empty((self.T_grid.size, K_eval.size), dtype=float)
        for j in range(self.T_grid.size):
            out[j, :] = np.interp(K_eval, self.K_grid, self.C_grid[j, :])
        return out

    def call_price(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Interpolate call prices on the arbitrage-free grid.
        """
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        KK, TT = np.broadcast_arrays(K, T)

        T_eval = np.clip(TT.ravel(), self.T_grid[0], self.T_grid[-1])
        K_eval = np.clip(KK.ravel(), self.K_grid[0], self.K_grid[-1])

        by_strike = self._interp_strike_all_maturities(K_eval)  # (nT, nPts)
        out = np.empty(K_eval.size, dtype=float)

        for p in range(K_eval.size):
            out[p] = float(np.interp(T_eval[p], self.T_grid, by_strike[:, p]))

        return out.reshape(KK.shape)

    def iv(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Return Black-Scholes implied vol of the arbitrage-free interpolated call surface.
        """
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        KK, TT = np.broadcast_arrays(K, T)
        C = self.call_price(KK, TT)
        S = np.full_like(C, float(self.S0), dtype=float)
        return implied_vol(
            prices=C,
            S=S,
            K=KK,
            T=TT,
            r=float(self.r),
            q=float(self.q),
            option_type="call",
            cfg=self.iv_solver_cfg,
        )

    def to_iv_surface(self) -> IVSurface:
        """
        Materialize the implied vols on the stored rectilinear grid.
        """
        TT, KK = np.meshgrid(self.T_grid, self.K_grid, indexing="ij")
        iv_grid = self.iv(KK, TT)
        return IVSurface(
            x_grid=np.asarray(self.K_grid, dtype=float),
            t_grid=np.asarray(self.T_grid, dtype=float),
            iv_grid=np.asarray(iv_grid, dtype=float),
            coord="K_T",
        )

    def plot_surface(self):
        """
        Quick 3D surface plot (matplotlib). Not for publication; just diagnostics.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        TT, XX = np.meshgrid(self.t_grid, self.x_grid, indexing="ij")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XX, TT, self.iv_grid, rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel("K" if self.coord == "K_T" else "x")
        ax.set_ylabel("T")
        ax.set_zlabel("iv")
        ax.set_title("Implied Volatility Surface")
        plt.show()

from typing import Callable
from .black_scholes import bs_price


def call_price_grid_from_iv_surface(
    iv_surface,
    *,
    S0: float,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r: float,
    q: float = 0.0,
    coord: Literal["K_T", "x_T"] = "K_T",
) -> np.ndarray:
    """
    Convert an implied-volatility surface into a call-price grid C(T, K).

    Parameters
    ----------
    iv_surface :
        Any object with method iv(x, T) → sigma.
        (e.g. IVSurface or SplineSurface)
    S0 : float
        Spot price at valuation time.
    K_grid : (nK,) array
        Strike grid (strictly increasing).
    T_grid : (nT,) array
        Maturity grid (strictly increasing).
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    coord : "K_T" or "x_T"
        How iv_surface expects its first argument:
          - "K_T": pass strike K directly
          - "x_T": pass log-moneyness x = log(K/F)

    Returns
    -------
    C : (nT, nK) array
        Call prices on the grid.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    if S0 <= 0.0:
        raise ValueError("S0 must be positive")

    C = np.empty((T_grid.size, K_grid.size), dtype=float)

    for j, T in enumerate(T_grid):
        if T <= 0.0:
            # exact intrinsic at T=0
            C[j, :] = np.maximum(S0 - K_grid, 0.0)
            continue

        F = S0 * np.exp((r - q) * T)

        if coord == "K_T":
            sig = iv_surface.iv(K_grid, T)
        elif coord == "x_T":
            x = np.log(K_grid / F)
            sig = iv_surface.iv(x, T)
        else:
            raise ValueError("coord must be 'K_T' or 'x_T'")
        
        # --- CRITICAL GUARDRAIL ---
        # Floor implied vol to avoid nonphysical values from extrapolation
        sig = np.asarray(sig, dtype=float)
        sig = np.maximum(sig, 1e-4)   # 1 bp vol floor; conservative and standard
        
        C[j, :] = bs_price(
            S=S0,
            K=K_grid,
            T=T,
            r=r,
            q=q,
            sigma=sig,
            option_type="call",
        )

    return C
