# src/utils/interpolation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from utils.numerical import as_1d, as_2d, require_strictly_increasing
from utils.platform_compat import apply_windows_platform_fastpath

try:
    apply_windows_platform_fastpath()
    from scipy.interpolate import interp1d, PchipInterpolator, RegularGridInterpolator
except Exception as e:  # pragma: no cover
    interp1d = None
    PchipInterpolator = None
    RegularGridInterpolator = None


class InterpolationError(RuntimeError):
    """Raised when an interpolation routine is unavailable or misused."""


@dataclass(frozen=True)
class Interp1D:
    """
    Lightweight 1D interpolator wrapper.

    kind:
      - "linear": piecewise linear
      - "cubic": cubic spline via scipy interp1d (requires SciPy)
      - "pchip": monotone shape-preserving cubic (recommended for vols/variances)
    """
    x: np.ndarray
    y: np.ndarray
    kind: Literal["linear", "cubic", "pchip"] = "linear"
    fill_value: float | Tuple[float, float] | str = "extrapolate"

    def __post_init__(self) -> None:
        require_strictly_increasing(self.x, "x")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y must have the same length")

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)

        if self.kind in ("cubic", "pchip") and (interp1d is None or PchipInterpolator is None):
            raise InterpolationError("SciPy is required for cubic/pchip interpolation.")

        if self.kind == "linear":
            # np.interp does linear interpolation, and extrapolates using end values.
            # We want optional extrapolation behavior to match SciPy's default: linear extrapolation.
            # We'll implement linear extrapolation explicitly.
            x = self.x
            y = self.y
            y_new = np.interp(x_new, x, y)

            if self.fill_value == "extrapolate":
                # Linear extrapolation on both ends
                left_slope = (y[1] - y[0]) / (x[1] - x[0])
                right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

                left_mask = x_new < x[0]
                right_mask = x_new > x[-1]
                y_new[left_mask] = y[0] + left_slope * (x_new[left_mask] - x[0])
                y_new[right_mask] = y[-1] + right_slope * (x_new[right_mask] - x[-1])
                return y_new

            if isinstance(self.fill_value, tuple):
                lo, hi = self.fill_value
                y_new[x_new < x[0]] = float(lo)
                y_new[x_new > x[-1]] = float(hi)
                return y_new

            if isinstance(self.fill_value, (int, float)):
                y_new[(x_new < x[0]) | (x_new > x[-1])] = float(self.fill_value)
                return y_new

            raise ValueError("Unsupported fill_value for linear interpolation.")

        if self.kind == "cubic":
            f = interp1d(
                self.x,
                self.y,
                kind="cubic",
                fill_value=self.fill_value,
                bounds_error=False,
                assume_sorted=True,
            )
            return np.asarray(f(x_new), dtype=float)

        # pchip
        f = PchipInterpolator(self.x, self.y, extrapolate=(self.fill_value == "extrapolate"))
        y_new = np.asarray(f(x_new), dtype=float)
        if self.fill_value != "extrapolate":
            # If user asked for constant fill, apply it outside bounds.
            if isinstance(self.fill_value, tuple):
                lo, hi = self.fill_value
                y_new[x_new < self.x[0]] = float(lo)
                y_new[x_new > self.x[-1]] = float(hi)
            elif isinstance(self.fill_value, (int, float)):
                y_new[(x_new < self.x[0]) | (x_new > self.x[-1])] = float(self.fill_value)
            else:
                raise ValueError("Unsupported fill_value for pchip.")
        return y_new


@dataclass(frozen=True)
class Interp2DGrid:
    """
    2D interpolator on a rectilinear grid.

    axes: x (e.g. strike/log-moneyness), t (e.g. maturity)
    values: z[t_index, x_index] or z[x_index, t_index]?

    We enforce a consistent convention:
      - x_grid has shape (nx,)
      - y_grid has shape (ny,)
      - z has shape (ny, nx) where z[j, i] = z(y_grid[j], x_grid[i])

    method:
      - "linear": bilinear
      - "nearest": nearest neighbor
    """
    x_grid: np.ndarray
    y_grid: np.ndarray
    z: np.ndarray
    method: Literal["linear", "nearest"] = "linear"
    fill_value: Optional[float] = None

    def __post_init__(self) -> None:
        require_strictly_increasing(self.x_grid, "x_grid")
        require_strictly_increasing(self.y_grid, "y_grid")
        z = as_2d(self.z, "z")
        ny, nx = z.shape
        if nx != self.x_grid.size or ny != self.y_grid.size:
            raise ValueError(
                f"z must have shape (ny, nx) = ({self.y_grid.size}, {self.x_grid.size}), got {z.shape}"
            )
        if RegularGridInterpolator is None:
            raise InterpolationError("SciPy is required for 2D grid interpolation (RegularGridInterpolator).")

    def __call__(self, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)
        X, Y = np.broadcast_arrays(x_new, y_new)
        pts = np.column_stack([Y.ravel(), X.ravel()])  # note order: (y, x)

        rgi = RegularGridInterpolator(
            (self.y_grid, self.x_grid),
            self.z,
            method=self.method,
            bounds_error=False,
            fill_value=self.fill_value,
        )
        out = np.asarray(rgi(pts), dtype=float).reshape(X.shape)
        return out


def make_iv_time_interp(
    maturities: np.ndarray,
    values: np.ndarray,
    *,
    kind: Literal["linear", "pchip"] = "pchip",
) -> Interp1D:
    """
    Convenience helper commonly used for term-structure interpolation.

    For vol/variance-like quantities, pchip is typically the safest default.
    """
    maturities = require_strictly_increasing(maturities, "maturities")
    values = as_1d(values, "values")
    if maturities.size != values.size:
        raise ValueError("maturities and values must have same length")
    return Interp1D(maturities, values, kind=kind, fill_value="extrapolate")
