from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from utils.numerical import solve_tridiagonal, require_strictly_increasing
from local_volatility.surface import LocalVolSurface
from .boundary_conditions import payoff, left_boundary_value, right_boundary_value


OptionType = Literal["call", "put"]


class PDESolverError(RuntimeError):
    """Raised when PDE solver encounters invalid inputs or instability."""


@dataclass(frozen=True)
class PDEConfig:
    """
    Crank-Nicolson solver configuration in log-space x=ln S, tau=T-t.
    """

    n_space: int = 400
    n_time: int = 200
    theta: float = 0.5
    S_max_mult: float = 5.0
    S_min_mult: float = 1e-3
    enforce_positive: bool = True


def _resolve_spot_grid(
    *,
    S_ref: float,
    K: float,
    cfg: PDEConfig,
    S_min: Optional[float] = None,
    S_max: Optional[float] = None,
    S_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    if S_grid is not None:
        grid = require_strictly_increasing(np.asarray(S_grid, dtype=float), "S_grid")
        if np.any(grid <= 0.0):
            raise PDESolverError("S_grid must be strictly positive")
        return grid

    base = max(float(S_ref), float(K), 1.0)
    lo = float(S_min) if S_min is not None else max(cfg.S_min_mult * base, 1e-10)
    hi = float(S_max) if S_max is not None else cfg.S_max_mult * base

    if lo <= 0.0 or hi <= lo:
        raise PDESolverError("Invalid spot grid bounds")

    n_space = int(cfg.n_space)
    if n_space < 50:
        raise PDESolverError("n_space too small for stable PDE solution")

    x = np.linspace(np.log(lo), np.log(hi), n_space)
    return np.exp(x)


def _check_cfg(cfg: PDEConfig) -> None:
    if int(cfg.n_time) < 20:
        raise PDESolverError("n_time too small for stable PDE solution")
    if not (0.0 <= float(cfg.theta) <= 1.0):
        raise PDESolverError("theta must be in [0,1]")


def _solve_call_surface(
    *,
    S_ref: float,
    K: float,
    T: float,
    r: float,
    q: float,
    lv_surface: LocalVolSurface,
    cfg: PDEConfig,
    t0: float = 0.0,
    S_min: Optional[float] = None,
    S_max: Optional[float] = None,
    S_grid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if S_ref <= 0.0 or K <= 0.0:
        raise PDESolverError("S_ref and K must be positive")
    if T <= 0.0:
        grid = _resolve_spot_grid(S_ref=S_ref, K=K, cfg=cfg, S_min=S_min, S_max=S_max, S_grid=S_grid)
        t_grid = np.array([0.0], dtype=float)
        V_grid = payoff(grid, K, "call")[None, :]
        return t_grid, grid, V_grid

    _check_cfg(cfg)

    S_nodes = _resolve_spot_grid(S_ref=S_ref, K=K, cfg=cfg, S_min=S_min, S_max=S_max, S_grid=S_grid)
    x = np.log(S_nodes)
    dx = float(x[1] - x[0])
    if dx <= 0.0:
        raise PDESolverError("Spot grid must map to a strictly increasing log grid")

    n_time = int(cfg.n_time)
    theta = float(cfg.theta)

    tau_grid = np.linspace(0.0, float(T), n_time)
    t_grid = np.linspace(0.0, float(T), n_time)
    dt = float(tau_grid[1] - tau_grid[0])

    V_tau = np.empty((n_time, S_nodes.size), dtype=float)
    u = payoff(S_nodes, K, "call")
    V_tau[0, :] = u

    def build_tridiag_coeffs(tau_k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_calendar = float(t0 + (T - tau_k))
        sig = lv_surface.sigma(S_nodes, t_calendar)
        sig2 = sig * sig
        A = 0.5 * sig2
        B = (r - q) - 0.5 * sig2

        alpha = A / (dx * dx)
        beta = B / (2.0 * dx)

        a = alpha - beta
        b = -2.0 * alpha - r
        c = alpha + beta
        return a, b, c

    n_x = S_nodes.size
    i0 = 1
    i1 = n_x - 2
    n_interior = i1 - i0 + 1

    if n_interior < 1:
        raise PDESolverError("Spot grid is too small to form interior PDE nodes")

    for k in range(n_time - 1):
        tau_k = float(tau_grid[k])
        tau_k1 = float(tau_grid[k + 1])

        a_k, b_k, c_k = build_tridiag_coeffs(tau_k)
        a_k1, b_k1, c_k1 = build_tridiag_coeffs(tau_k1)

        u_old = u.copy()

        u_L_next = left_boundary_value(tau=tau_k1, K=K, r=r, q=q, option_type="call")
        u_R_next = right_boundary_value(S_max=float(S_nodes[-1]), tau=tau_k1, K=K, r=r, q=q, option_type="call")

        a0 = a_k[i0 : i1 + 1]
        b0 = b_k[i0 : i1 + 1]
        c0 = c_k[i0 : i1 + 1]
        a1 = a_k1[i0 : i1 + 1]
        b1 = b_k1[i0 : i1 + 1]
        c1 = c_k1[i0 : i1 + 1]

        uI = u_old[i0 : i1 + 1]
        rhs = uI + (1.0 - theta) * dt * (
            a0 * u_old[i0 - 1 : i1] + b0 * uI + c0 * u_old[i0 + 1 : i1 + 2]
        )

        sub = -theta * dt * a1[1:]
        diag = 1.0 - theta * dt * b1
        sup = -theta * dt * c1[:-1]

        rhs[0] += theta * dt * a1[0] * u_L_next
        rhs[-1] += theta * dt * c1[-1] * u_R_next

        u_new_interior = solve_tridiagonal(sub, diag, sup, rhs, check_diagonal=True)

        u[i0 : i1 + 1] = u_new_interior
        u[0] = u_L_next
        u[-1] = u_R_next

        if cfg.enforce_positive:
            u = np.maximum(u, 0.0)

        V_tau[k + 1, :] = u

    # tau increases from expiry to start, so reverse to align with calendar t in [0, T].
    V_grid = V_tau[::-1, :]
    return t_grid, S_nodes, V_grid


def solve_european_pde_local_vol_surface(
    *,
    S_ref: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    lv_surface: LocalVolSurface,
    cfg: PDEConfig = PDEConfig(),
    t0: float = 0.0,
    S_min: Optional[float] = None,
    S_max: Optional[float] = None,
    S_grid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve a full option value surface on an absolute-spot grid.

    Returns
    -------
    t_grid : (nT,)
        Calendar time since hedge start, increasing from 0 to T.
    S_grid : (nS,)
        Absolute spot grid used by the PDE.
    V_grid : (nT, nS)
        Option values V(t, S) for the supplied strike K.
    """

    t_grid, S_nodes, call_grid = _solve_call_surface(
        S_ref=S_ref,
        K=K,
        T=T,
        r=r,
        q=q,
        lv_surface=lv_surface,
        cfg=cfg,
        t0=t0,
        S_min=S_min,
        S_max=S_max,
        S_grid=S_grid,
    )

    if option_type == "call":
        return t_grid, S_nodes, call_grid

    if option_type != "put":
        raise PDESolverError("option_type must be 'call' or 'put'")

    tau = float(T) - t_grid[:, None]
    put_grid = call_grid - S_nodes[None, :] * np.exp(-q * tau) + float(K) * np.exp(-r * tau)
    return t_grid, S_nodes, put_grid


def price_european_pde_local_vol(
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    lv_surface: LocalVolSurface,
    cfg: PDEConfig = PDEConfig(),
    t0: float = 0.0,
) -> float:
    """
    Price a European option under local volatility via backward PDE solved in log-space.
    """

    if S0 <= 0.0 or K <= 0.0:
        raise PDESolverError("S0 and K must be positive")
    if T <= 0.0:
        return float(max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0))

    spot_floor = max(cfg.S_min_mult * max(S0, K, 1.0), 1e-10)
    spot_cap = cfg.S_max_mult * max(S0, K, 1.0)

    t_grid, S_grid, V_grid = solve_european_pde_local_vol_surface(
        S_ref=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        option_type=option_type,
        lv_surface=lv_surface,
        cfg=cfg,
        t0=t0,
        S_min=spot_floor,
        S_max=spot_cap,
    )

    return float(np.interp(S0, S_grid, V_grid[0, :]))
