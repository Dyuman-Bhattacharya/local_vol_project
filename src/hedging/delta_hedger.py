# src/hedging/delta_hedger.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from utils.platform_compat import apply_windows_platform_fastpath

apply_windows_platform_fastpath()
from scipy.interpolate import RegularGridInterpolator

import numpy as np

from implied_volatility.black_scholes import bs_price
from .transaction_costs import TransactionCostModel


OptionType = Literal["call", "put"]


class DeltaModel(Protocol):
    def price(self, S: float, t: float) -> float: ...
    def delta(self, S: float, t: float) -> float: ...


@dataclass(frozen=True)
class BSDeltaModel:
    """
    Black-Scholes delta model with constant sigma.

    Uses analytic BS price and a central finite-difference delta.
    """

    S0_ref: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    option_type: OptionType
    bump_rel: float = 1e-4
    bump_abs_min: float = 1e-4

    def price(self, S: float, t: float) -> float:
        tau = max(self.T - t, 0.0)
        return float(bs_price(S, self.K, tau, self.r, self.q, self.sigma, self.option_type))

    def delta(self, S: float, t: float) -> float:
        tau = max(self.T - t, 0.0)
        if tau <= 0.0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            return -1.0 if S < self.K else 0.0

        h = max(self.bump_abs_min, self.bump_rel * max(S, 1.0))
        p_up = bs_price(S + h, self.K, tau, self.r, self.q, self.sigma, self.option_type)
        p_dn = bs_price(max(S - h, 1e-12), self.K, tau, self.r, self.q, self.sigma, self.option_type)
        return float((p_up - p_dn) / (2.0 * h))


@dataclass(frozen=True)
class LocalVolPDEDeltaModel:
    """
    Delta model using precomputed local-vol PDE grids.

    No PDE solves are performed during hedging.
    """

    K: float
    T: float
    r: float
    q: float
    option_type: OptionType

    S_grid: np.ndarray
    t_grid: np.ndarray
    V_grid: np.ndarray
    D_grid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_V_interp",
            RegularGridInterpolator(
                (self.t_grid, self.S_grid),
                self.V_grid,
                bounds_error=False,
                fill_value=None,
            ),
        )
        object.__setattr__(
            self,
            "_D_interp",
            RegularGridInterpolator(
                (self.t_grid, self.S_grid),
                self.D_grid,
                bounds_error=False,
                fill_value=None,
            ),
        )

    def price(self, S: float, t: float) -> float:
        t_eval = np.clip(t, self.t_grid[0], self.t_grid[-1])
        S_eval = np.clip(S, self.S_grid[0], self.S_grid[-1])
        return float(self._V_interp((t_eval, S_eval)))

    def delta(self, S: float, t: float) -> float:
        tau = self.T - t
        if tau <= 0.0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            return -1.0 if S < self.K else 0.0

        t_eval = np.clip(t, self.t_grid[0], self.t_grid[-1])
        S_eval = np.clip(S, self.S_grid[0], self.S_grid[-1])
        return float(self._D_interp((t_eval, S_eval)))


@dataclass(frozen=True)
class LocalVolPDEDeltaModelStrikeSurface:
    """
    LV delta model using precomputed absolute-spot PDE grids over (t, K, S).

    This is the active spot-consistent design. Strike variation is handled
    explicitly as a grid axis rather than via a reference-strike shortcut.
    """

    K: float
    T: float
    r: float
    q: float
    option_type: OptionType

    K_grid: np.ndarray
    S_grid: np.ndarray
    t_grid: np.ndarray
    V_grid: np.ndarray
    D_grid: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_V_interp",
            RegularGridInterpolator(
                (self.t_grid, self.K_grid, self.S_grid),
                self.V_grid,
                bounds_error=False,
                fill_value=None,
            ),
        )
        object.__setattr__(
            self,
            "_D_interp",
            RegularGridInterpolator(
                (self.t_grid, self.K_grid, self.S_grid),
                self.D_grid,
                bounds_error=False,
                fill_value=None,
            ),
        )

    def price(self, S: float, t: float) -> float:
        t_eval = np.clip(t, self.t_grid[0], self.t_grid[-1])
        K_eval = np.clip(self.K, self.K_grid[0], self.K_grid[-1])
        S_eval = np.clip(S, self.S_grid[0], self.S_grid[-1])
        return float(self._V_interp((t_eval, K_eval, S_eval)))

    def delta(self, S: float, t: float) -> float:
        tau = self.T - t
        if tau <= 0.0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            return -1.0 if S < self.K else 0.0

        t_eval = np.clip(t, self.t_grid[0], self.t_grid[-1])
        K_eval = np.clip(self.K, self.K_grid[0], self.K_grid[-1])
        S_eval = np.clip(S, self.S_grid[0], self.S_grid[-1])
        return float(self._D_interp((t_eval, K_eval, S_eval)))


@dataclass
class HedgeState:
    shares: float
    cash: float


def evolve_cash(cash: float, r: float, dt: float) -> float:
    return float(cash * np.exp(r * dt))


def option_payoff(ST: float, K: float, option_type: OptionType) -> float:
    if option_type == "call":
        return float(max(ST - K, 0.0))
    return float(max(K - ST, 0.0))


@dataclass(frozen=True)
class DeltaHedger:
    """
    Simulates discrete-time delta hedging for a single option along a realized price path.
    """

    model: DeltaModel
    K: float
    T: float
    r: float
    option_type: OptionType
    tx_costs: TransactionCostModel = TransactionCostModel(kind="none")

    def run_path(
        self,
        times: np.ndarray,
        spots: np.ndarray,
        *,
        initial_premium: Optional[float] = None,
    ):
        times = np.asarray(times, dtype=float)
        spots = np.asarray(spots, dtype=float)
        if times.ndim != 1 or spots.ndim != 1 or times.size != spots.size:
            raise ValueError("times and spots must be 1D arrays of same length")
        if not np.all(np.diff(times) >= 0.0):
            raise ValueError("times must be non-decreasing")
        if np.any(spots <= 0.0):
            raise ValueError("spots must be positive")

        mask = times <= self.T + 1e-15
        times = times[mask]
        spots = spots[mask]
        if times.size < 2:
            raise ValueError("Need at least two time points to hedge")

        S0 = float(spots[0])
        t0 = float(times[0])
        premium = float(initial_premium) if initial_premium is not None else float(self.model.price(S0, t0))

        delta0 = float(self.model.delta(S0, t0))
        shares = delta0
        cash = premium - shares * S0

        cost0 = self.tx_costs.cost(d_shares=shares, S=S0)
        cash -= cost0
        state = HedgeState(shares=shares, cash=cash)

        shares_hist = [state.shares]
        cash_hist = [state.cash]
        port_hist = [state.shares * S0 + state.cash]
        delta_hist = [delta0]
        cost_hist = [cost0]

        for i in range(1, times.size):
            if self.model.__class__.__name__ == "LocalVolPDEDeltaModel":
                if i % 10 == 0 or i == times.size - 1:
                    print(f"    LV rebalance {i}/{times.size - 1}", end="\r", flush=True)

            t_prev = float(times[i - 1])
            t_cur = float(times[i])
            S_cur = float(spots[i])
            dt = t_cur - t_prev

            state.cash = evolve_cash(state.cash, self.r, dt)
            new_delta = float(self.model.delta(S_cur, t_cur))
            d_shares = new_delta - state.shares
            tc = self.tx_costs.cost(d_shares=d_shares, S=S_cur)
            state.cash -= d_shares * S_cur
            state.cash -= tc
            state.shares = new_delta

            V = state.shares * S_cur + state.cash
            shares_hist.append(state.shares)
            cash_hist.append(state.cash)
            port_hist.append(V)
            delta_hist.append(new_delta)
            cost_hist.append(tc)

        t_end = float(times[-1])
        S_end = float(spots[-1])
        if t_end < self.T:
            state.cash = evolve_cash(state.cash, self.r, self.T - t_end)
            t_end = self.T

        V_T = state.shares * S_end + state.cash
        payoff = option_payoff(S_end, self.K, self.option_type)
        error = float(V_T - payoff)

        if self.model.__class__.__name__ == "LocalVolPDEDeltaModel":
            print()

        return {
            "times": times,
            "spots": spots,
            "shares": np.array(shares_hist, dtype=float),
            "cash": np.array(cash_hist, dtype=float),
            "portfolio": np.array(port_hist, dtype=float),
            "delta": np.array(delta_hist, dtype=float),
            "tx_costs": np.array(cost_hist, dtype=float),
            "premium": premium,
            "terminal_spot": S_end,
            "terminal_portfolio": float(V_T),
            "terminal_payoff": float(payoff),
            "hedge_error": error,
        }
