# src/surface_fitting/spline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from utils.interpolation import Interp1D, Interp2DGrid
from utils.numerical import require_strictly_increasing


def total_variance_from_iv(iv: np.ndarray, T: float | np.ndarray) -> np.ndarray:
    iv = np.asarray(iv, dtype=float)
    T = np.asarray(T, dtype=float)
    return (iv * iv) * T


def iv_from_total_variance(w: np.ndarray, T: float | np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    T = np.asarray(T, dtype=float)
    return np.sqrt(np.maximum(w / np.maximum(T, 1e-16), 0.0))



class SplineFitError(RuntimeError):
    """Raised when spline fitting cannot be performed."""


@dataclass(frozen=True)
class SplineSurface:
    """
    Fallback surface: interpolate total variance w(x, T) on a rectilinear grid,
    using 2D grid interpolation.

    Convention:
      - x_grid: (nX,)
      - T_grid: (nT,)
      - w_grid: shape (nT, nX) with w_grid[j,i] = w(T_j, x_i)
    """
    x_grid: np.ndarray
    T_grid: np.ndarray
    w_grid: np.ndarray
    method: Literal["linear", "nearest"] = "linear"
    fill_value: Optional[float] = None

    def __post_init__(self) -> None:
        require_strictly_increasing(self.x_grid, "x_grid")
        require_strictly_increasing(self.T_grid, "T_grid")
        w = np.asarray(self.w_grid, dtype=float)
        if w.ndim != 2:
            raise ValueError("w_grid must be 2D")
        if w.shape != (self.T_grid.size, self.x_grid.size):
            raise ValueError(f"w_grid must have shape (nT,nX)=({self.T_grid.size},{self.x_grid.size})")

    def total_variance(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        interp = Interp2DGrid(
            x_grid=self.x_grid,
            y_grid=self.T_grid,
            z=self.w_grid,
            method=self.method,
            fill_value=self.fill_value,
        )
        return interp(x, T)

    def iv(self, x: np.ndarray, T: np.ndarray) -> np.ndarray:
        w = self.total_variance(x, T)
        TT = np.asarray(T, dtype=float)
        return iv_from_total_variance(w, TT)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if values.size == 0 or weights.size != values.size:
        raise ValueError("weighted quantile requires non-empty aligned values and weights")
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(np.quantile(values, q))
    cdf = np.cumsum(weights) / total
    idx = int(np.searchsorted(cdf, float(np.clip(q, 0.0, 1.0)), side="left"))
    idx = min(max(idx, 0), values.size - 1)
    return float(values[idx])


def _smooth_slice_to_x_grid(
    x_slice: np.ndarray,
    w_slice: np.ndarray,
    weight_slice: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    x_slice = np.asarray(x_slice, dtype=float).ravel()
    w_slice = np.asarray(w_slice, dtype=float).ravel()
    weight_slice = np.asarray(weight_slice, dtype=float).ravel()
    if x_slice.size != w_slice.size or x_slice.size != weight_slice.size:
        raise ValueError("Slice arrays must have the same length")

    if x_slice.size < 2:
        return np.full_like(x_grid, float(w_slice[0]) if x_slice.size == 1 else np.nan, dtype=float)

    positive_weights = np.maximum(weight_slice, 1e-8)
    bandwidth = _weighted_quantile(np.abs(np.diff(np.sort(x_slice))), np.ones(max(x_slice.size - 1, 1)), 0.5)
    span = float(np.max(x_slice) - np.min(x_slice))
    bandwidth = max(float(bandwidth), span / 18.0, 5e-3)

    out = np.empty_like(x_grid, dtype=float)
    for i, x0 in enumerate(np.asarray(x_grid, dtype=float)):
        z = (x_slice - x0) / bandwidth
        kernel = np.exp(-0.5 * z * z)
        weights = positive_weights * kernel
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 1e-12:
            nearest = int(np.argmin(np.abs(x_slice - x0)))
            out[i] = float(w_slice[nearest])
        else:
            out[i] = float(np.sum(weights * w_slice) / total)
    return out


def build_total_variance_grid_from_slices(
    x_slices: Sequence[np.ndarray],
    w_slices: Sequence[np.ndarray],
    T_slices: Sequence[float],
    *,
    weights_slices: Optional[Sequence[np.ndarray]] = None,
    x_grid: Optional[np.ndarray] = None,
    T_grid: Optional[np.ndarray] = None,
    kind_x: Literal["linear", "pchip"] = "pchip",
    kind_T: Literal["linear", "pchip"] = "pchip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a rectilinear grid (T_grid, x_grid, w_grid) from sparse slice data by:
      1) Interpolating each maturity slice w(x) onto a common x_grid.
      2) Interpolating w at each x across maturities onto a common T_grid.

    This is the canonical interpolation path for building a smooth total-variance surface.
    """
    T_arr = np.asarray(T_slices, dtype=float).ravel()
    if T_arr.size != len(x_slices) or T_arr.size != len(w_slices):
        raise ValueError("T_slices, x_slices, w_slices must have same length")

    order = np.argsort(T_arr)
    T_arr = T_arr[order]
    x_slices = [np.asarray(x_slices[i], dtype=float).ravel() for i in order]
    w_slices = [np.asarray(w_slices[i], dtype=float).ravel() for i in order]
    if weights_slices is not None:
        weights_slices = [np.asarray(weights_slices[i], dtype=float).ravel() for i in order]
    require_strictly_increasing(T_arr, "T_slices (sorted)")

    # Choose common grids
    if x_grid is None:
        # union-ish: take min/max across slices and use a moderate grid
        xmin = max(np.min(x) for x in x_slices)
        xmax = min(np.max(x) for x in x_slices)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            raise SplineFitError("Cannot infer a valid common x_grid from slices")
        x_grid = np.linspace(float(xmin), float(xmax), 81)
    else:
        x_grid = require_strictly_increasing(np.asarray(x_grid, dtype=float).ravel(), "x_grid")

    if T_grid is None:
        T_grid = T_arr.copy()
    else:
        T_grid = require_strictly_increasing(np.asarray(T_grid, dtype=float).ravel(), "T_grid")

    # Step 1: interpolate each slice onto x_grid
    W_on_x = np.empty((T_arr.size, x_grid.size), dtype=float)
    for j, (xj, wj) in enumerate(zip(x_slices, w_slices)):
        require_strictly_increasing(xj, f"x_slice[{j}]")
        if weights_slices is not None:
            weights_j = np.asarray(weights_slices[j], dtype=float).ravel()
            if weights_j.size != xj.size:
                raise ValueError(f"weights_slice[{j}] must align with x_slice[{j}]")
            W_on_x[j, :] = _smooth_slice_to_x_grid(xj, wj, weights_j, x_grid)
        else:
            fj = Interp1D(xj, wj, kind=kind_x, fill_value="extrapolate")
            W_on_x[j, :] = fj(x_grid)

    # Step 2: interpolate across T onto T_grid at each x
    W_grid = np.empty((T_grid.size, x_grid.size), dtype=float)
    for i in range(x_grid.size):
        gi = Interp1D(T_arr, W_on_x[:, i], kind=kind_T, fill_value="extrapolate")
        W_grid[:, i] = gi(T_grid)

    # Floor at 0 to avoid nonsense
    W_grid = np.maximum(W_grid, 0.0)
    return x_grid, T_grid, W_grid


def fit_spline_surface(
    x_slices: Sequence[np.ndarray],
    w_slices: Sequence[np.ndarray],
    T_slices: Sequence[float],
    *,
    weights_slices: Optional[Sequence[np.ndarray]] = None,
    x_grid: Optional[np.ndarray] = None,
    T_grid: Optional[np.ndarray] = None,
    kind_x: Literal["linear", "pchip"] = "pchip",
    kind_T: Literal["linear", "pchip"] = "pchip",
    method_2d: Literal["linear", "nearest"] = "linear",
    fill_value: Optional[float] = None,
) -> SplineSurface:
    """
    Fit a fallback spline-based surface of total variance and expose via 2D interpolation.
    """
    xg, Tg, Wg = build_total_variance_grid_from_slices(
        x_slices, w_slices, T_slices,
        weights_slices=weights_slices,
        x_grid=x_grid, T_grid=T_grid,
        kind_x=kind_x, kind_T=kind_T,
    )
    return SplineSurface(x_grid=xg, T_grid=Tg, w_grid=Wg, method=method_2d, fill_value=fill_value)
