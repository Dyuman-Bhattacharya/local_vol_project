from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


class MetricsError(RuntimeError):
    """Raised when metrics cannot be computed."""


def _error_stats(e: np.ndarray) -> Dict[str, float]:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        raise MetricsError("No finite hedge errors")

    mean = float(np.mean(e))
    std = float(np.std(e, ddof=1)) if e.size > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(e * e)))
    mad = float(np.median(np.abs(e - np.median(e))))
    q05 = float(np.quantile(e, 0.05))
    q50 = float(np.quantile(e, 0.50))
    q95 = float(np.quantile(e, 0.95))

    return {
        "n": int(e.size),
        "mean": mean,
        "std": std,
        "rmse": rmse,
        "median_abs_dev": mad,
        "q05": q05,
        "q50": q50,
        "q95": q95,
    }


def summarize_backtest(per_trade: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize hedging performance.

    Required columns:
      - hedge_error_net

    Optional columns:
      - hedge_error_gross
      - total_tx_cost
      - n_rebalances
    """
    if "hedge_error_net" not in per_trade.columns:
        raise MetricsError("per_trade must include hedge_error_net")

    out: Dict[str, float] = {}

    # --- Net error (what you actually realize)
    net_stats = _error_stats(per_trade["hedge_error_net"].to_numpy())
    for k, v in net_stats.items():
        out[f"net_{k}"] = v

    # --- Gross error (replication quality ignoring costs)
    if "hedge_error_gross" in per_trade.columns:
        gross_stats = _error_stats(per_trade["hedge_error_gross"].to_numpy())
        for k, v in gross_stats.items():
            out[f"gross_{k}"] = v

    # --- Transaction costs
    if "total_tx_cost" in per_trade.columns:
        out["mean_total_tx_cost"] = float(per_trade["total_tx_cost"].mean())
        out["median_total_tx_cost"] = float(per_trade["total_tx_cost"].median())

    # --- Rebalancing intensity
    if "n_rebalances" in per_trade.columns:
        out["mean_n_rebalances"] = float(per_trade["n_rebalances"].mean())

    return out


def summarize_market_panel_backtest(per_trade: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize a daily market-panel hedge study.

    Required columns:
      - entry_pricing_error
      - replication_error_net
      - market_pnl_net

    Optional columns:
      - replication_error_gross
      - market_pnl_gross
      - total_tx_cost
      - n_rebalances
      - mean_abs_mtm_error
    """
    required = ["entry_pricing_error", "replication_error_net", "market_pnl_net"]
    missing = [c for c in required if c not in per_trade.columns]
    if missing:
        raise MetricsError(f"per_trade is missing required columns: {missing}")

    out: Dict[str, float] = {}

    for prefix, col in [
        ("pricing_error", "entry_pricing_error"),
        ("replication_net", "replication_error_net"),
        ("market_pnl_net", "market_pnl_net"),
    ]:
        stats = _error_stats(per_trade[col].to_numpy())
        for k, v in stats.items():
            out[f"{prefix}_{k}"] = v

    for prefix, col in [
        ("replication_gross", "replication_error_gross"),
        ("market_pnl_gross", "market_pnl_gross"),
    ]:
        if col in per_trade.columns:
            stats = _error_stats(per_trade[col].to_numpy())
            for k, v in stats.items():
                out[f"{prefix}_{k}"] = v

    if "total_tx_cost" in per_trade.columns:
        out["mean_total_tx_cost"] = float(per_trade["total_tx_cost"].mean())
        out["median_total_tx_cost"] = float(per_trade["total_tx_cost"].median())

    if "n_rebalances" in per_trade.columns:
        out["mean_n_rebalances"] = float(per_trade["n_rebalances"].mean())

    if "mean_abs_mtm_error" in per_trade.columns:
        mtm = pd.to_numeric(per_trade["mean_abs_mtm_error"], errors="coerce")
        mtm = mtm[np.isfinite(mtm)]
        out["mean_abs_mtm_error"] = float(mtm.mean()) if len(mtm) else float("nan")

    return out
