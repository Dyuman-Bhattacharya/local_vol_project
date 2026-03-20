# src/hedging/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .delta_hedger import DeltaHedger


class BacktestError(RuntimeError):
    """Raised when backtest execution fails."""


@dataclass(frozen=True)
class BacktestResult:
    """
    Aggregated results from running many hedges.
    """
    per_trade: pd.DataFrame
    summary: dict


def run_delta_hedge_backtest(
    *,
    hedger: DeltaHedger,
    price_paths: Sequence[pd.DataFrame],
    # each path DataFrame must have columns: "t" (years), "S"
    initial_premium: Optional[float] = None,
) -> BacktestResult:
    """
    Run hedging over multiple realized paths.

    Parameters
    ----------
    hedger : DeltaHedger
    price_paths : list of DataFrames with columns ['t','S'] for each path
    initial_premium : optional fixed premium for all paths; else model.price at t=0 per path.

    Returns
    -------
    BacktestResult
    """
    rows = []
    for idx, df in enumerate(price_paths):
        if not isinstance(df, pd.DataFrame):
            raise BacktestError("Each path must be a pandas DataFrame")
        if "t" not in df.columns or "S" not in df.columns:
            raise BacktestError("Each path DataFrame must have columns ['t','S']")

        times = df["t"].to_numpy(dtype=float)
        spots = df["S"].to_numpy(dtype=float)

        out = hedger.run_path(times, spots, initial_premium=initial_premium)

        total_cost = float(np.sum(out["tx_costs"]))
        error_net = float(out["hedge_error"])
        error_gross = float(error_net + total_cost)

        n_reb = int(len(out["times"]) - 1)

        rows.append(
            {
                "path_id": idx,
                "premium": float(out["premium"]),
                "terminal_spot": float(out["terminal_spot"]),
                "terminal_payoff": float(out["terminal_payoff"]),
                "terminal_portfolio": float(out["terminal_portfolio"]),
                "hedge_error_net": error_net,
                "hedge_error_gross": error_gross,
                "total_tx_cost": total_cost,
                "n_rebalances": n_reb,
            }
        )


    per_trade = pd.DataFrame(rows)

    summary = {
        "n_paths": int(len(per_trade)),
        "mean_hedge_error_net": float(per_trade["hedge_error_net"].mean()),
        "std_hedge_error_net": (
            float(per_trade["hedge_error_net"].std(ddof=1))
            if len(per_trade) > 1
            else float("nan")
        ),
        "rmse_hedge_error_net": float(
            np.sqrt(np.mean(per_trade["hedge_error_net"].to_numpy() ** 2))
        ),
        "mean_hedge_error_gross": float(per_trade["hedge_error_gross"].mean()),
        "rmse_hedge_error_gross": float(
            np.sqrt(np.mean(per_trade["hedge_error_gross"].to_numpy() ** 2))
        ),
        "mean_total_tx_cost": float(per_trade["total_tx_cost"].mean()),
        "mean_n_rebalances": float(per_trade["n_rebalances"].mean()),
    }

    return BacktestResult(per_trade=per_trade, summary=summary)
