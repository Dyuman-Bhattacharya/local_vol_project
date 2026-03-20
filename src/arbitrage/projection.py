from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from utils.platform_compat import apply_windows_platform_fastpath

apply_windows_platform_fastpath()
from scipy.optimize import Bounds, LinearConstraint, minimize

from utils.numerical import require_strictly_increasing
from .static_checks import check_static_arbitrage_calls


class ArbitrageProjectionError(RuntimeError):
    """Raised when static-arbitrage projection cannot be completed."""


@dataclass(frozen=True)
class StaticArbitrageProjectionResult:
    """
    Result of projecting a call-price surface onto a discretely static-arbitrage-free grid.
    """

    C_projected: np.ndarray
    counts_before: Dict[str, int]
    counts_after: Dict[str, int]
    iterations: int
    rmse_adjustment: float
    max_abs_adjustment: float
    history: List[Dict[str, object]]


def _column_pava_nondecreasing(y: np.ndarray) -> np.ndarray:
    """
    Pool-adjacent-violators algorithm for an L2 projection onto nondecreasing sequences.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        return y.copy()

    vals = list(y.copy())
    wts = [1.0] * n
    starts = list(range(n))
    ends = list(range(n))

    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-14:
            i += 1
            continue

        vals[i] = (wts[i] * vals[i] + wts[i + 1] * vals[i + 1]) / (wts[i] + wts[i + 1])
        wts[i] += wts[i + 1]
        ends[i] = ends[i + 1]

        del vals[i + 1]
        del wts[i + 1]
        del starts[i + 1]
        del ends[i + 1]

        if i > 0:
            i -= 1

    out = np.empty(n, dtype=float)
    for v, s, e in zip(vals, starts, ends):
        out[s : e + 1] = v
    return out


def _repair_row_convexity_and_monotonicity(
    row: np.ndarray,
    K: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tol: float,
    max_passes: int,
) -> np.ndarray:
    """
    L2-project a maturity row onto the discretely decreasing, convex call-price cone.

    The row is solved as a small convex quadratic program with linear inequality
    constraints, which is materially more robust on noisy live surfaces than the
    earlier local "push the midpoint down" heuristic.
    """
    row = np.clip(np.asarray(row, dtype=float).copy(), lower, upper)
    K = np.asarray(K, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = K.size

    if row.size != n or lower.size != n or upper.size != n:
        raise ArbitrageProjectionError("Row/bounds/K size mismatch in row projection")
    if n < 3:
        return np.clip(row, lower, upper)

    h = np.diff(K)
    if np.any(h <= 0.0):
        raise ArbitrageProjectionError("K must be strictly increasing in row projection")

    # Start from the previous heuristic output as a good feasible-ish initial guess.
    x0 = row.copy()
    for _ in range(max_passes):
        changed = False

        for i in range(1, n - 1):
            hm = float(h[i - 1])
            hp = float(h[i])
            cap = (hm * x0[i + 1] + hp * x0[i - 1]) / (hm + hp)

            new_val = min(x0[i], cap)
            new_val = float(np.clip(new_val, lower[i], upper[i]))
            if new_val < x0[i] - tol:
                x0[i] = new_val
                changed = True

        x0 = np.minimum.accumulate(x0)
        x0 = np.clip(x0, lower, upper)
        if not changed:
            break

    A_rows: list[np.ndarray] = []
    lb_rows: list[float] = []
    ub_rows: list[float] = []

    # Monotone decrease in strike: x_i - x_{i+1} >= 0
    for i in range(n - 1):
        a = np.zeros(n, dtype=float)
        a[i] = 1.0
        a[i + 1] = -1.0
        A_rows.append(a)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)

    # Discrete convexity on a nonuniform strike grid.
    for i in range(1, n - 1):
        hm = float(h[i - 1])
        hp = float(h[i])
        a = np.zeros(n, dtype=float)
        a[i - 1] = hp
        a[i] = -(hm + hp)
        a[i + 1] = hm
        A_rows.append(a)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)

    linear_constraint = LinearConstraint(np.vstack(A_rows), np.asarray(lb_rows), np.asarray(ub_rows))
    bounds = Bounds(lower, upper)

    def objective(x: np.ndarray) -> float:
        diff = x - row
        return 0.5 * float(np.dot(diff, diff))

    def gradient(x: np.ndarray) -> np.ndarray:
        return x - row

    result = minimize(
        objective,
        x0=x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[linear_constraint],
        options={"ftol": max(float(tol), 1e-12), "maxiter": max(200, 20 * n), "disp": False},
    )

    projected = np.asarray(result.x if result.success else x0, dtype=float)
    projected = np.clip(projected, lower, upper)

    # Final numerical cleanup to absorb tiny solver tolerance noise.
    projected = np.minimum.accumulate(projected)
    for i in range(1, n - 1):
        hm = float(h[i - 1])
        hp = float(h[i])
        cap = (hm * projected[i + 1] + hp * projected[i - 1]) / (hm + hp)
        if projected[i] > cap:
            projected[i] = cap
    projected = np.minimum.accumulate(np.clip(projected, lower, upper))

    return projected


def summarize_calendar_violations(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Summarize adjacent maturity-pair calendar violations on a call grid.
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    T = require_strictly_increasing(T, "T")

    if C.shape != (T.size, K.size):
        raise ValueError(f"Shape mismatch: C {C.shape}, K {K.shape}, T {T.shape}")

    out: List[Dict[str, float]] = []
    for j in range(T.size - 1):
        diff = C[j + 1, :] - C[j, :]
        worst_idx = int(np.argmin(diff))
        min_diff = float(diff[worst_idx])
        n_negative = int(np.sum(diff < 0.0))
        out.append(
            {
                "T_left": float(T[j]),
                "T_right": float(T[j + 1]),
                "n_negative_adjacent": n_negative,
                "worst_adjacent_diff": min_diff,
                "worst_strike": float(K[worst_idx]),
            }
        )
    return out


def project_call_surface_static_arbitrage(
    C: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    *,
    S0: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-10,
    max_outer_iter: int = 60,
    max_row_passes: int = 24,
) -> StaticArbitrageProjectionResult:
    """
    Enforce discrete static no-arbitrage on a call-price surface.

    Strategy:
      1. Project each strike column onto a nondecreasing term structure (calendar monotonicity).
      2. Repair each maturity row so it is decreasing and convex in strike.
      3. Repeat until static checks pass or max iterations is reached.
    """
    C = np.asarray(C, dtype=float)
    K = require_strictly_increasing(K, "K")
    T = require_strictly_increasing(T, "T")

    if C.ndim != 2:
        raise ArbitrageProjectionError("C must be a 2D array with shape (nT, nK)")
    if C.shape != (T.size, K.size):
        raise ArbitrageProjectionError(f"Shape mismatch: C {C.shape}, K {K.shape}, T {T.shape}")
    if S0 <= 0.0:
        raise ArbitrageProjectionError("S0 must be positive")

    C0 = C.copy()
    out = C.copy()
    rep0 = check_static_arbitrage_calls(out, K, T, S0=S0, r=r, q=q, tol=tol)
    history: List[Dict[str, object]] = [
        {
            "stage": "raw",
            "iteration": 0,
            "counts": dict(rep0.counts),
            "calendar_pairs": summarize_calendar_violations(out, K, T),
        }
    ]

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    for outer in range(1, max_outer_iter + 1):
        # Step 1: enforce calendar monotonicity per strike column with bounds.
        for i in range(K.size):
            lower_col = np.maximum(S0 * disc_q - float(K[i]) * disc_r, 0.0)
            upper_col = S0 * disc_q
            y = np.clip(out[:, i], lower_col, upper_col)
            y = _column_pava_nondecreasing(y)
            y = np.clip(y, lower_col, upper_col)
            y = _column_pava_nondecreasing(y)
            out[:, i] = y

        rep_after_calendar = check_static_arbitrage_calls(out, K, T, S0=S0, r=r, q=q, tol=tol)
        history.append(
            {
                "stage": "after_calendar_projection",
                "iteration": outer,
                "counts": dict(rep_after_calendar.counts),
                "calendar_pairs": summarize_calendar_violations(out, K, T),
            }
        )

        # Step 2: repair each maturity row in strike.
        for j in range(T.size):
            lower_row = np.maximum(S0 * np.exp(-q * T[j]) - K * np.exp(-r * T[j]), 0.0)
            upper_row = np.full(K.size, S0 * np.exp(-q * T[j]), dtype=float)
            out[j, :] = _repair_row_convexity_and_monotonicity(
                out[j, :],
                K,
                lower_row,
                upper_row,
                tol=tol,
                max_passes=max_row_passes,
            )

        rep = check_static_arbitrage_calls(out, K, T, S0=S0, r=r, q=q, tol=tol)
        history.append(
            {
                "stage": "after_row_repair",
                "iteration": outer,
                "counts": dict(rep.counts),
                "calendar_pairs": summarize_calendar_violations(out, K, T),
            }
        )

        if rep.counts["n_fail"] == 0:
            return StaticArbitrageProjectionResult(
                C_projected=out,
                counts_before=dict(rep0.counts),
                counts_after=dict(rep.counts),
                iterations=outer,
                rmse_adjustment=float(np.sqrt(np.mean((out - C0) ** 2))),
                max_abs_adjustment=float(np.max(np.abs(out - C0))),
                history=history,
            )

    rep_final = check_static_arbitrage_calls(out, K, T, S0=S0, r=r, q=q, tol=tol)
    if rep_final.counts["n_fail"] != 0:
        raise ArbitrageProjectionError(
            "Static-arbitrage projection did not converge to a fully feasible grid. "
            f"Remaining counts: {rep_final.counts}"
        )

    return StaticArbitrageProjectionResult(
        C_projected=out,
        counts_before=dict(rep0.counts),
        counts_after=dict(rep_final.counts),
        iterations=max_outer_iter,
        rmse_adjustment=float(np.sqrt(np.mean((out - C0) ** 2))),
        max_abs_adjustment=float(np.max(np.abs(out - C0))),
        history=history,
    )
