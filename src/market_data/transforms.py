# src/market_data/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


class MarketDataTransformError(RuntimeError):
    """Raised when transforms cannot be applied due to missing columns or invalid values."""


DayCount = Literal["ACT/365", "ACT/252"]


@dataclass(frozen=True)
class TransformConfig:
    """
    Transform settings.

    day_count:
      - ACT/365: calendar-day basis (simple)
      - ACT/252: trading-day basis (approx; uses business days)
    """
    day_count: DayCount = "ACT/365"
    add_forward: bool = True
    add_log_moneyness: bool = True

    # If True and time_to_expiry is missing/NaN, compute from date/maturity_date.
    compute_time_to_expiry_if_missing: bool = True


def compute_time_to_expiry_years(
    date: pd.Series,
    maturity_date: pd.Series,
    *,
    day_count: DayCount = "ACT/365",
) -> pd.Series:
    """
    Compute time to expiry in years from date and maturity_date.

    - ACT/365: (maturity - date).days / 365
    - ACT/252: business days / 252 (approx)
    """
    date = pd.to_datetime(date, errors="coerce")
    maturity_date = pd.to_datetime(maturity_date, errors="coerce")

    if day_count == "ACT/365":
        delta_days = (maturity_date - date).dt.total_seconds() / (24.0 * 3600.0)
        return delta_days / 365.0

    if day_count == "ACT/252":
        # Approx: count business days between dates (vectorized via numpy busday_count)
        d = date.dt.date
        m = maturity_date.dt.date
        # Convert NaT to NaN after busday_count by masking
        mask = d.notna() & m.notna()
        out = pd.Series(np.nan, index=date.index, dtype=float)
        if mask.any():
            d_np = d[mask].astype("datetime64[D]").to_numpy()
            m_np = m[mask].astype("datetime64[D]").to_numpy()
            bdays = np.busday_count(d_np, m_np)
            out.loc[mask] = bdays / 252.0
        return out

    raise ValueError(f"Unknown day_count: {day_count}")


def add_derived_columns(
    df: pd.DataFrame,
    *,
    cfg: TransformConfig = TransformConfig(),
) -> pd.DataFrame:
    """
    Add derived quantities needed downstream:

    - time_to_expiry (years), if missing or NaN and date + maturity_date exist
    - forward price F = S * exp((r - q) T)
    - log-moneyness x = log(K / F)

    Returns a new DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise MarketDataTransformError("Input must be a pandas.DataFrame")

    out = df.copy()

    # Ensure required base columns exist (some may be NaN; validation should run before)
    required = ["underlying_price", "strike", "risk_free_rate", "dividend_yield"]
    for c in required:
        if c not in out.columns:
            raise MarketDataTransformError(f"Missing required column: {c}")

    # Compute time_to_expiry if requested
    if cfg.compute_time_to_expiry_if_missing:
        if "time_to_expiry" not in out.columns:
            out["time_to_expiry"] = np.nan

        need_T = out["time_to_expiry"].isna()
        if need_T.any():
            if "date" in out.columns and "maturity_date" in out.columns:
                T_new = compute_time_to_expiry_years(
                    out.loc[need_T, "date"],
                    out.loc[need_T, "maturity_date"],
                    day_count=cfg.day_count,
                )
                out.loc[need_T, "time_to_expiry"] = T_new
            # else: leave as NaN; later stages will drop or skip

    # Forward and log-moneyness
    S = pd.to_numeric(out["underlying_price"], errors="coerce")
    K = pd.to_numeric(out["strike"], errors="coerce")
    r = pd.to_numeric(out["risk_free_rate"], errors="coerce").fillna(0.0)
    q = pd.to_numeric(out["dividend_yield"], errors="coerce").fillna(0.0)
    T = pd.to_numeric(out.get("time_to_expiry", np.nan), errors="coerce")

    if cfg.add_forward:
        out["forward"] = S * np.exp((r - q) * T)

    if cfg.add_log_moneyness:
        # Use forward if available, else compute on the fly
        F = out["forward"] if "forward" in out.columns else (S * np.exp((r - q) * T))
        # Guard: log(K/F) requires positive K, F; produce NaN otherwise
        valid = (K > 0.0) & (F > 0.0)
        x = pd.Series(np.nan, index=out.index, dtype=float)
        x.loc[valid] = np.log((K.loc[valid] / F.loc[valid]).astype(float))
        out["log_moneyness"] = x

    return out


def enforce_year_fractions(
    df: pd.DataFrame,
    *,
    min_T: float = 0.0,
) -> pd.DataFrame:
    """
    Simple helper to zero-floor tiny negative times produced by timestamp quirks.
    """
    out = df.copy()
    if "time_to_expiry" in out.columns:
        T = pd.to_numeric(out["time_to_expiry"], errors="coerce")
        out["time_to_expiry"] = T.where(T >= min_T, other=min_T)
    return out