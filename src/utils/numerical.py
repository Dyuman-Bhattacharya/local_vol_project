# src/utils/numerical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


ArrayLike = np.ndarray


class NumericalError(RuntimeError):
    """Raised when a numerical routine fails due to invalid inputs or instability."""


def as_1d(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Ensure x is a 1D numpy array (float64)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    return x


def as_2d(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Ensure x is a 2D numpy array (float64)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")
    return x


def safe_divide(num: ArrayLike, den: ArrayLike, eps: float = 1e-14) -> np.ndarray:
    """
    Elementwise division with denominator flooring.
    This is *not* a substitute for proper regularization; it prevents crashes.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    den_safe = np.where(np.abs(den) < eps, np.sign(den) * eps + (den == 0) * eps, den)
    return num / den_safe


def clip(x: ArrayLike, lo: float, hi: float) -> np.ndarray:
    """Convenience wrapper around np.clip with float casting."""
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def is_strictly_increasing(x: ArrayLike) -> bool:
    x = as_1d(x, "x")
    return bool(np.all(np.diff(x) > 0.0))


def require_strictly_increasing(x: ArrayLike, name: str = "x") -> np.ndarray:
    x = as_1d(x, name)
    if not np.all(np.diff(x) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return x


def finite_diff_first(
    f: ArrayLike,
    x: ArrayLike,
    scheme: Literal["central", "forward", "backward"] = "central",
) -> np.ndarray:
    """
    Compute first derivative df/dx on a 1D grid.
    - central: central in interior, one-sided at boundaries
    - forward/backward: one-sided everywhere
    """
    f = as_1d(f, "f")
    x = require_strictly_increasing(x, "x")
    if f.shape[0] != x.shape[0]:
        raise ValueError("f and x must have the same length")

    n = f.size
    d = np.empty_like(f)

    if n < 2:
        raise ValueError("Need at least 2 points for first derivative")

    dx = np.diff(x)

    if scheme == "forward":
        d[:-1] = (f[1:] - f[:-1]) / dx
        d[-1] = d[-2]
        return d

    if scheme == "backward":
        d[1:] = (f[1:] - f[:-1]) / dx
        d[0] = d[1]
        return d

    # central: interior central, boundaries one-sided
    if n == 2:
        d[:] = (f[1] - f[0]) / (x[1] - x[0])
        return d

    d[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    d[0] = (f[1] - f[0]) / (x[1] - x[0])
    d[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    return d


def finite_diff_second(
    f: ArrayLike,
    x: ArrayLike,
) -> np.ndarray:
    """
    Compute second derivative d^2 f / dx^2 on a (possibly non-uniform) 1D grid.

    Uses the standard three-point formula for non-uniform grids in the interior:
      f''(x_i) ≈ 2 * [ (f_{i+1}-f_i)/h_i - (f_i-f_{i-1})/h_{i-1} ] / (h_{i-1}+h_i)

    Boundaries use a simple one-sided fallback based on nearest interior stencil.
    """
    f = as_1d(f, "f")
    x = require_strictly_increasing(x, "x")
    if f.shape[0] != x.shape[0]:
        raise ValueError("f and x must have the same length")

    n = f.size
    if n < 3:
        raise ValueError("Need at least 3 points for second derivative")

    d2 = np.empty_like(f)
    h = np.diff(x)

    # interior
    hm = h[:-1]   # h_{i-1}
    hp = h[1:]    # h_i
    num = (f[2:] - f[1:-1]) / hp - (f[1:-1] - f[:-2]) / hm
    d2[1:-1] = 2.0 * num / (hm + hp)

    # boundary: copy nearest interior (stable, boring default)
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2


@dataclass(frozen=True)
class TridiagonalSystem:
    """
    Represents a tridiagonal linear system A x = d with:
      - a: subdiagonal (length n-1)
      - b: diagonal     (length n)
      - c: superdiagonal(length n-1)
      - d: RHS          (length n)
    """
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def validate(self) -> None:
        a = as_1d(self.a, "a")
        b = as_1d(self.b, "b")
        c = as_1d(self.c, "c")
        d = as_1d(self.d, "d")
        n = b.size
        if a.size != n - 1 or c.size != n - 1 or d.size != n:
            raise ValueError(f"Invalid sizes: a={a.size}, b={b.size}, c={c.size}, d={d.size}")


def solve_tridiagonal(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    *,
    check_diagonal: bool = True,
    eps: float = 1e-14,
) -> np.ndarray:
    """
    Solve tridiagonal system using the Thomas algorithm (O(n)).

    Parameters
    ----------
    a : (n-1,) subdiagonal
    b : (n,) diagonal
    c : (n-1,) superdiagonal
    d : (n,) RHS
    check_diagonal : if True, raise if a pivot is too small
    eps : pivot threshold

    Returns
    -------
    x : (n,) solution
    """
    a = as_1d(a, "a").copy()
    b = as_1d(b, "b").copy()
    c = as_1d(c, "c").copy()
    d = as_1d(d, "d").copy()

    n = b.size
    if a.size != n - 1 or c.size != n - 1 or d.size != n:
        raise ValueError("Tridiagonal arrays have inconsistent sizes")

    # Forward elimination
    for i in range(1, n):
        pivot = b[i - 1]
        if check_diagonal and abs(pivot) < eps:
            raise NumericalError(f"Near-zero pivot encountered at i={i-1}: {pivot}")
        w = a[i - 1] / pivot
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x = np.empty(n, dtype=float)
    pivot = b[-1]
    if check_diagonal and abs(pivot) < eps:
        raise NumericalError(f"Near-zero pivot encountered at last index: {pivot}")
    x[-1] = d[-1] / pivot

    for i in range(n - 2, -1, -1):
        pivot = b[i]
        if check_diagonal and abs(pivot) < eps:
            raise NumericalError(f"Near-zero pivot encountered at i={i}: {pivot}")
        x[i] = (d[i] - c[i] * x[i + 1]) / pivot

    return x


def central_diff_2d_x(u: ArrayLike, dx: float) -> np.ndarray:
    """
    First derivative along axis=1 (x-direction) for a 2D array u[t, x] using central differences.
    Boundaries use one-sided.
    """
    u = as_2d(u, "u")
    if dx <= 0:
        raise ValueError("dx must be positive")

    du = np.empty_like(u)
    du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    du[:, 0] = (u[:, 1] - u[:, 0]) / dx
    du[:, -1] = (u[:, -1] - u[:, -2]) / dx
    return du


def central_diff2_2d_x(u: ArrayLike, dx: float) -> np.ndarray:
    """
    Second derivative along axis=1 (x-direction) for a 2D array u[t, x] using central differences.
    Boundaries copy nearest interior value.
    """
    u = as_2d(u, "u")
    if dx <= 0:
        raise ValueError("dx must be positive")

    d2 = np.empty_like(u)
    d2[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    d2[:, 0] = d2[:, 1]
    d2[:, -1] = d2[:, -2]
    return d2
