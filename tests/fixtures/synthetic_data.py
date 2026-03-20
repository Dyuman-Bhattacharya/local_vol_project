# tests/fixtures/synthetic_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from implied_volatility.black_scholes import bs_price


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSSyntheticChainConfig:
    """
    Generate a synthetic option chain from Black–Scholes with constant sigma.

    This is the cleanest end-to-end fixture because:
      - it's arbitrage-free by construction,
      - Dupire should recover constant local vol ~ sigma (away from boundaries),
      - PDE pricing under local vol should match BS pricing for vanillas.
    """
    S0: float = 100.0
    r: float = 0.02
    q: float = 0.0
    sigma: float = 0.25

    # grids
    K_min: float = 50.0
    K_max: float = 200.0
    nK: int = 151

    T_list: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    option_type: OptionType = "call"

    # liquidity fields (optional)
    volume: float = 1000.0
    open_interest: float = 5000.0


def generate_bs_chain(cfg: BSSyntheticChainConfig) -> pd.DataFrame:
    """
    Returns a DataFrame in the *standardized schema* expected by your market_data module.
    """
    K = np.linspace(cfg.K_min, cfg.K_max, cfg.nK)
    T = np.array(cfg.T_list, dtype=float)

    rows = []
    for t in T:
        prices = bs_price(cfg.S0, K, t, cfg.r, cfg.q, cfg.sigma, cfg.option_type)
        for k, c in zip(K, prices):
            rows.append(
                {
                    "date": pd.Timestamp("2025-01-01"),
                    "maturity_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(round(365 * t))),
                    "time_to_expiry": float(t),
                    "underlying": "SYNTH",
                    "underlying_price": float(cfg.S0),
                    "strike": float(k),
                    "option_type": cfg.option_type,
                    "bid": float(max(c * 0.999, 0.0)),
                    "ask": float(c * 1.001),
                    "mid": float(c),
                    "volume": float(cfg.volume),
                    "open_interest": float(cfg.open_interest),
                    "risk_free_rate": float(cfg.r),
                    "dividend_yield": float(cfg.q),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BSDailyPanelConfig:
    ticker: str = "SYNTH"
    start_date: str = "2025-01-02"
    spot_path: Tuple[float, ...] = (100.0, 101.0, 100.5, 102.0, 101.5, 103.0, 104.0)
    r: float = 0.01
    q: float = 0.0
    sigma: float = 0.20
    option_type: OptionType = "call"
    expiry_days: Tuple[int, ...] = (8, 15, 29)
    rel_strikes: Tuple[float, ...] = (0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15)
    volume: float = 500.0
    open_interest: float = 2500.0


def generate_bs_daily_panel(cfg: BSDailyPanelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(cfg.start_date)
    dates = pd.bdate_range(start=start, periods=len(cfg.spot_path))
    history = pd.DataFrame({"date": dates, "S": np.asarray(cfg.spot_path, dtype=float)})

    rows = []
    for date, S0 in zip(dates, cfg.spot_path):
        for days in cfg.expiry_days:
            maturity = date + pd.Timedelta(days=int(days))
            T = max((maturity - date).days / 365.0, 0.0)
            if T <= 0.0:
                continue
            for rel_k in cfg.rel_strikes:
                K = float(S0) * float(rel_k)
                mid = float(
                    bs_price(
                        S=float(S0),
                        K=float(K),
                        T=float(T),
                        r=float(cfg.r),
                        q=float(cfg.q),
                        sigma=float(cfg.sigma),
                        option_type=cfg.option_type,
                    )
                )
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "maturity_date": pd.Timestamp(maturity),
                        "time_to_expiry": float(T),
                        "underlying": cfg.ticker,
                        "underlying_price": float(S0),
                        "strike": float(round(K, 4)),
                        "option_type": cfg.option_type,
                        "bid": float(max(mid * 0.999, 0.0)),
                        "ask": float(mid * 1.001),
                        "mid": float(mid),
                        "volume": float(cfg.volume),
                        "open_interest": float(cfg.open_interest),
                        "risk_free_rate": float(cfg.r),
                        "dividend_yield": float(cfg.q),
                    }
                )
    panel = pd.DataFrame(rows)
    return panel, history
